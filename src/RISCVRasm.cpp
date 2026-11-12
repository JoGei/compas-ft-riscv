/*
 * Copyright 2021 Chair of EDA, Technical University of Munich
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "RISCVRasm.h"
#include "RISCVSubtarget.h"

#include "common.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#define N_MIN_INTRA_INSTRUCTION_COUNT 3

namespace {

llvm::cl::opt<std::string> enable_rasm_predicate{
    "RASM-predicate",
    llvm::cl::desc("RV32 equality-predicate signature selection; requires RASM for the same functions"),
    llvm::cl::value_desc("functions")};

llvm::cl::opt<std::string> enable_rasm_predicate_unpadded{
    "RASM-predicate-unpadded",
    llvm::cl::desc("Omit predicate-to-branch and successor-entry NOP spacing; requires RASM-predicate for the same functions"),
    llvm::cl::value_desc("functions")};

llvm::cl::opt<std::string> enable_rasm_predicate_shadow{
    "RASM-predicate-shadow",
    llvm::cl::desc("Derive predicates from NZDC shadow operands; requires RASM-predicate and plain NZDC with CGS for the same functions"),
    llvm::cl::value_desc("functions")};

llvm::cl::opt<std::string> enable_rasm_predicate_xpred{
    "RASM-predicate-xpred",
    llvm::cl::desc("Use XPred sacc.i for signed-six-bit predicate adjustments; requires RASM-predicate and +xpred"),
    llvm::cl::value_desc("functions")};

llvm::cl::opt<std::string> enable_rasm_predicate_psp_simm6{
    "RASM-predicate-psp-simm6",
    llvm::cl::desc("Try bounded joint RASM label allocation for signed-six-bit predicate adjustments; independent of XPred lowering"),
    llvm::cl::value_desc("functions")};

llvm::cl::opt<unsigned> rasm_predicate_psp_seed{
    "RASM-predicate-psp-seed",
    llvm::cl::desc("Seed for the bounded RASM predicate signed-six-bit allocator"),
    llvm::cl::init(1)};

llvm::cl::opt<unsigned> rasm_predicate_psp_max_attempts{
    "RASM-predicate-psp-max-attempts",
    llvm::cl::desc("Bound on signed-six-bit allocation attempts (maximum 32; zero selects the legacy fallback)"),
    llvm::cl::init(32)};

constexpr unsigned PredicateGapNops = 3;

// LLVM describes JAL as a call even when COMPAS uses rd=x0 to transfer to a
// local block. Treat these transfers like ordinary direct branches in both
// the inter-block and intra-block signature calculations.
llvm::MachineBasicBlock *direct_branch_target(const llvm::MachineInstr &MI) {
  const bool local_jal =
      MI.getOpcode() == llvm::RISCV::JAL && MI.getNumOperands() >= 2 &&
      MI.getOperand(0).isReg() &&
      MI.getOperand(0).getReg() == llvm::RISCV::X0 &&
      MI.getOperand(1).isMBB();
  if (!MI.isConditionalBranch() && !MI.isUnconditionalBranch() && !local_jal)
    return nullptr;

  for (const auto &MO : MI.operands()) {
    if (MO.isMBB())
      return MO.getMBB();
  }
  return nullptr;
}

bool is_local_unconditional_jump(const llvm::MachineInstr &MI) {
  return !MI.isConditionalBranch() && direct_branch_target(MI) != nullptr;
}

bool is_error_block(const llvm::MachineBasicBlock &MBB) {
  return MBB.succ_size() == 1 && *MBB.succ_begin() == &MBB;
}

void validate_predicate_cfg(const llvm::MachineFunction &MF,
                            bool allow_detector_blocks,
                            const riscv_common::RsmRegions *regions = nullptr) {
  for (const auto &MBB : MF) {
    // Original self-loop blocks are rejected before DMR. Plain NZDC only
    // creates its terminal error block; branches to it are detector checks,
    // not application predicates requiring signature selection.
    if (allow_detector_blocks && is_error_block(MBB))
      continue;
    if (regions && regions->isTransparent(&MBB))
      continue;
    if (MBB.empty() || (&MBB != &MF.front() && MBB.pred_empty()) ||
        is_error_block(MBB))
      llvm::report_fatal_error("COMPAS: RASM-predicate does not support empty, unreachable or self-loop application blocks");
    unsigned conditional_count = 0;
    for (const auto &MI : MBB) {
      if (MI.isIndirectBranch() && !MI.isReturn())
        llvm::report_fatal_error("COMPAS: RASM-predicate does not support indirect branches");
      if (!MI.isConditionalBranch())
        continue;
      auto *target = direct_branch_target(MI);
      if (allow_detector_blocks && target && is_error_block(*target))
        continue;
      if (regions && !regions->monitorsBranch(MI))
        continue;
      auto predicate = riscv_common::getSoftBranchPredicate(MI);
      if (!predicate)
        llvm::report_fatal_error("COMPAS: RASM-predicate supports only direct BEQ/BNE register branches");
      if (++conditional_count != 1)
        llvm::report_fatal_error("COMPAS: RASM-predicate requires one application conditional branch per block");
      for (auto next = std::next(MI.getIterator()); next != MBB.end(); ++next)
        if (!next->isDebugInstr())
          llvm::report_fatal_error("COMPAS: RASM-predicate requires a terminal conditional branch with implicit fallthrough");
      const auto next_block = std::next(MBB.getIterator());
      const auto *fallthrough = next_block == MF.end() ? nullptr : &*next_block;
      if (!fallthrough || !MBB.isSuccessor(fallthrough) ||
          fallthrough == predicate->taken)
        llvm::report_fatal_error("COMPAS: RASM-predicate requires distinct taken and fallthrough successors");
      if (predicate->lhs == llvm::RISCV::X5 ||
          predicate->rhs == llvm::RISCV::X5 ||
          predicate->lhs == llvm::RISCV::X29 ||
          predicate->rhs == llvm::RISCV::X29)
        llvm::report_fatal_error("COMPAS: RASM-predicate branch operands alias reserved signature registers");
    }
  }
}

} // namespace

llvm::FunctionPass *llvm::createRISCVRasm() { return new RISCVRasm(); }

llvm::FunctionPass *llvm::createRISCVRacfed() { return new RISCVRacfed(); }

RISCVRasm::RISCVRasm() : RISCVDmr{} {}

RISCVRacfed::RISCVRacfed() : RISCVRasm{} {}

void validateRasmPredicateOptions(const llvm::MachineFunction &MF,
                                  bool after_dmr) {
  const std::string name{MF.getName()};
  const auto selected = [&](const std::string &functions) {
    return riscv_common::inCSString(functions, name);
  };
  const bool predicate = selected(enable_rasm_predicate);
  const bool unpadded = selected(enable_rasm_predicate_unpadded);
  const bool shadow = selected(enable_rasm_predicate_shadow);
  const bool xpred = selected(enable_rasm_predicate_xpred);
  const bool psp_simm6 = selected(enable_rasm_predicate_psp_simm6);
  if (!predicate && (unpadded || shadow || xpred || psp_simm6))
    llvm::report_fatal_error(
        "COMPAS: RASM predicate modifiers (unpadded, shadow, xpred, psp-simm6) require "
        "RASM-predicate for the same function");
  if (!predicate)
    return;

  if (!selected(llvm::cl::enable_rasm))
    llvm::report_fatal_error(
        "COMPAS: RASM-predicate requires RASM for the same function");
  const auto &ST = MF.getSubtarget<llvm::RISCVSubtarget>();
  if (ST.is64Bit() || ST.hasStdExtCOrZca())
    llvm::report_fatal_error(
        "COMPAS: RASM-predicate requires RV32 without compressed instructions");
  if (xpred && !riscv_common::getXPredSelectiveAccumulateOpcode(MF))
    llvm::report_fatal_error(
        "COMPAS: RASM-predicate-xpred requires an XPred-enabled LLVM build "
        "and the +xpred function feature");
  if (selected(llvm::cl::enable_nzdc_nemesis) ||
      selected(llvm::cl::enable_nzdc_nemesec) ||
      selected(llvm::cl::enable_swift) || selected(llvm::cl::enable_eddi) ||
      llvm::cl::enable_repair || selected(llvm::cl::enable_cfcss) ||
      selected(llvm::cl::enable_racfed))
    llvm::report_fatal_error(
        "COMPAS: RASM-predicate only supports plain NZDC composition; "
        "NEMESIS, NEMESEC, SWIFT, EDDI, REPAIR, CFCSS and RACFED are unsupported");
  if (selected(llvm::cl::enable_nzdc) && selected(llvm::cl::enable_fgs))
    llvm::report_fatal_error(
        "COMPAS: RASM-predicate with NZDC requires CGS; FGS is unsupported");
  if (shadow && !selected(llvm::cl::enable_nzdc))
    llvm::report_fatal_error(
        "COMPAS: RASM-predicate-shadow requires plain NZDC with CGS for the "
        "same function");

  // The first COMPAS hook validates original code. DMR-generated detector
  // branches and shadow operands are validated separately after that pass.
  if (after_dmr && selected(llvm::cl::enable_nzdc))
    return;

  const auto *TII = ST.getInstrInfo();
  for (const auto &MBB : MF) {
    for (const auto &MI : MBB) {
      if (MI.isInlineAsm())
        llvm::report_fatal_error(
            "COMPAS: RASM-predicate prototype does not support inline assembly");
      if ((MI.isCall() && !is_local_unconditional_jump(MI)) ||
          TII->isTailCall(MI))
        llvm::report_fatal_error(
            "COMPAS: RASM-predicate prototype requires leaf functions, "
            "including no tail calls");
      for (const auto &operand : MI.operands()) {
        if (operand.isReg() &&
            (operand.getReg() == llvm::RISCV::X5 ||
             operand.getReg() == llvm::RISCV::X29))
          llvm::report_fatal_error(
              "COMPAS: RASM-predicate input uses reserved signature or "
              "predicate register x5/x29");
      }
    }
  }
  validate_predicate_cfg(MF, false);
}

bool RISCVRasm::runOnMachineFunction(llvm::MachineFunction &MF) {
  validateRasmPredicateOptions(MF, true);
  MF_ = &MF;
  fname_ = std::string{MF_->getName()};
  TII_ = MF_->getSubtarget().getInstrInfo();
  predicate_mode_ = riscv_common::inCSString(enable_rasm_predicate, fname_);
  predicate_unpadded_ =
      riscv_common::inCSString(enable_rasm_predicate_unpadded, fname_);
  predicate_shadow_mode_ =
      riscv_common::inCSString(enable_rasm_predicate_shadow, fname_);
  predicate_xpred_mode_ =
      riscv_common::inCSString(enable_rasm_predicate_xpred, fname_);
  predicate_psp_simm6_ =
      riscv_common::inCSString(enable_rasm_predicate_psp_simm6, fname_);

  if (!riscv_common::inCSString(llvm::cl::enable_rasm,
                                std::string{MF_->getName()})) {
    return false;
  }

  llvm::outs() << "Running RASM pass on " << MF_->getName() << "\n";

  regions_ = std::make_unique<riscv_common::RsmRegions>(
      *MF_, llvm::cl::enable_rasm.getValue());
  if (predicate_mode_)
    validate_predicate_mode();

  init();
  harden();

  return true;
}

void RISCVRasm::validate_predicate_mode() {
  validate_predicate_cfg(*MF_,
      riscv_common::inCSString(llvm::cl::enable_nzdc, fname_), regions_.get());
}

void RISCVRasm::finalizePredicateLayout() {
  if (!predicate_mode_)
    return;
  generate_predicate_entry_spacing();
  validate_predicate_branch_span();
}

void RISCVRasm::validate_predicate_branch_span() {
  // COMPAS executes after LLVM's branch-relaxation pass. Conservatively bound
  // the whole final function, including its error block and worst-case block
  // alignment padding, so every internal branch displacement fits signed 13
  // bits without a later expansion changing the predicate/spacing template.
  constexpr uint64_t MaxSpan = 4092;
  uint64_t span = 0;
  const auto add_bytes = [&](uint64_t bytes) {
    if (bytes > MaxSpan - span)
      llvm::report_fatal_error(
          "COMPAS: RASM-predicate final conservative function span exceeds "
          "4092 bytes; late branch relaxation is unsupported");
    span += bytes;
  };
  for (const auto &MBB : *MF_) {
    add_bytes(MBB.getAlignment().value() - 1);
    for (const auto &MI : MBB) {
      const unsigned size = TII_->getInstSizeInBytes(MI);
      if ((!size && !MI.isMetaInstruction()) || size % 4)
        llvm::report_fatal_error(
            "COMPAS: RASM-predicate requires known fixed-width instruction "
            "sizes for branch-span validation");
      add_bytes(size);
    }
  }
}

void RISCVRasm::generate_predicate_entry_spacing() {
  if (predicate_unpadded_)
    return;
  // Put padding inside the destination block rather than after a terminator.
  // This retains valid MBB structure and makes the cost explicit: these NOPs
  // execute on every normal nonentry block entry, unlike unreachable padding
  // in the handwritten local study. Error handling retains its own contract.
  for (auto &MBB : *MF_) {
    if (&MBB == &MF_->front() || err_bbs_.count(&MBB))
      continue;
    auto insert = MBB.begin();
    const auto DL = MBB.empty() ? llvm::DebugLoc{} : MBB.front().getDebugLoc();
    for (unsigned i = 0; i < PredicateGapNops; ++i)
      llvm::BuildMI(MBB, insert, DL, TII_->get(llvm::RISCV::ADDI))
          .addReg(riscv_common::k0, llvm::RegState::Define)
          .addReg(riscv_common::k0)
          .addImm(0);
  }
}

void RISCVRasm::init() {
  predicate_sites_ = predicate_simm6_sites_ = predicate_fused_sites_ =
      predicate_range_fallbacks_ = 0;
  mbb_sigs_.clear();
  mbb_sum_ii_sigs_.clear();
  mbb_signature_arbr_instrs_.clear();
  mbb_signature_check_instrs_.clear();
  cf_err_bb_ = nullptr;
  err_bbs_.clear();
  config_.eds = riscv_common::ErrorDetectionStrategy::ED3; // TODO: should be
                                                           // exposed to CLI
  if (config_.eds == riscv_common::ErrorDetectionStrategy::ED0 ||
      config_.eds == riscv_common::ErrorDetectionStrategy::ED1 ||
      config_.eds == riscv_common::ErrorDetectionStrategy::ED3) {
    // insert an error-BB in MF_
    insertErrorBB();
  } else {
    assert(0 && "TODO");
  }
}

// RASM hardening scheme
void RISCVRasm::harden() {

  //////////////////////////////////////////////////////////////////////////////
  // Step 1: Assign compile time signatues and random arbitration value
  // "subRanPrevVal" to each basic block of the function.
  generate_signatures();

  //////////////////////////////////////////////////////////////////////////////
  // Step 2: Insert runtime signature arbitration and signature check in the
  // beginning of each basic block except for machine function entry basic
  // block where we just assign the runtime signature with the block's compile
  // time signature
  generate_signature_checks();

  //////////////////////////////////////////////////////////////////////////////
  // Step 3: just sets all intra block to 0 for debugging
  generate_intrablock_signature_updates();

  //////////////////////////////////////////////////////////////////////////////
  // Step 4: Calculate traversal adjustment values for non-trivial
  // instructions
  generate_traversal_adjustments();

  if (predicate_xpred_mode_ || predicate_psp_simm6_)
    llvm::outs() << "COMPAS_XPRED function=" << fname_
                 << " requested=" << predicate_xpred_mode_
                 << " sites=" << predicate_sites_
                 << " simm6_fit=" << predicate_simm6_sites_
                 << " fused=" << predicate_fused_sites_
                 << " software=" << predicate_sites_ - predicate_fused_sites_
                 << " range_fallback=" << predicate_range_fallbacks_ << "\n";

  /*
    for (auto &MBB : *MF_) {
      // there could be err-BBs from other passes
      // the convention is if a BB keeps jumping to itself then this is also
      // an error-BB
      if (MBB.succ_size() == 1 && &MBB == *MBB.succ_begin()) {
        continue;
      }

      // special processing for some individual instrs:
      // we generate new instrs in the following loop which should not be
      // considered in future iterations. For this we use following flag
      bool ignore_this_mi{false};
      // in case we have a conditional and unconditional branch both in MBB then
      // we dont need to update signature before the final unconditional branch
      // we use the following flag to keep track of this
      bool ignore_jump{false};
      for (auto &MI : MBB) {
        if (ignore_this_mi) {
          ignore_this_mi = false;
          continue;
        }

      // in case we call another function then RTS would be corrupted
      // as we use t0 RISCV register for RTS, hence
      // preserving them across function call using the stack
      if (MI.isCall()) {
        // jumps are also considered as calls so filtering them out
        if (MI.getOperand(0).isReg() &&
            MI.getOperand(0).getReg() == llvm::RISCV::X0) {
          continue;
        }
        // filtering out tail calls
        if (TII_->isTailCall(MI)) {
          continue;
        }

        llvm::MachineBasicBlock::iterator insert{MI.getIterator()};
        insert++;

        // NOTE: stack would be used to pass args if they exceed arg regs hence
        //       we shouldn't use stack for CFC signatures

        if (MI.getOpcode() == llvm::RISCV::PseudoCALLIndirect ||
            !(riscv_common::inCSString(llvm::cl::enable_nzdc,
                                       std::string{MF_->getName()}) &&
              riscv_common::setmapContains(knownLibcalls2Duplicable_,
                                           getCalledFuncName(&MI)))) {
          // if we are in DMR func and we are going to libcall via this MI call
          // then no need to stack RTS explicitly as this is already done in DMR
          // else: we stack RTS as below

          riscv_common::saveRegs({kRTS}, &MBB, MI.getIterator());
          riscv_common::loadRegs({kRTS}, &MBB, insert);

          if (riscv_common::inCSString(llvm::cl::enable_nzdc,
                                       std::string{MF_->getName()})) {
            llvm::BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                          TII_->get(llvm::RISCV::ADDI))
                .addReg(P2S_.at(riscv_common::kSP))
                .addReg(P2S_.at(riscv_common::kSP))
                .addImm(-4);
            llvm::BuildMI(MBB, insert, MI.getDebugLoc(),
                          TII_->get(llvm::RISCV::ADDI))
                .addReg(P2S_.at(riscv_common::kSP))
                .addReg(P2S_.at(riscv_common::kSP))
                .addImm(4);
          }
        }

        // alternatively: for G, we simply move the expected value back into G
        //        llvm::BuildMI(MBB, insert, MI.getDebugLoc(),
        //                      TII_->get(llvm::RISCV::ADDI))
        //            .addReg(kRTS)
        //            .addReg(riscv_common::k0)
        //            .addImm(mbb_sigs_[&MBB].first);

        continue;
      }

        // RASM processing around the branch terminators
        if (MI.isConditionalBranch()) {
          assert(MI.getOperand(2).isMBB());
          auto SBB{MI.getOperand(2).getMBB()};
          // filtering error-handler jumps
          if (err_bbs_.find(SBB) != err_bbs_.end()) {
            // this could be the loadback check at the last
            // if thats the case then have to update RASM update
            if (SBB != cf_err_bb_ && &MBB.back() == &MI) {
              for (auto &op : MI.operands()) {
                if (op.isReg() && P2S_.find(op.getReg()) != P2S_.end()) {
                  assert(MBB.succ_size() == 1);
                  auto SBB{*std::begin(MBB.successors())};
                  auto adjust_value = mbb_sigs_.at(SBB).first +
                                      mbb_sigs_.at(SBB).second -
                                      mbb_sigs_.at(&MBB).first;
                  assert_imm(adjust_value);
                  llvm::BuildMI(MBB, std::end(MBB), MI.getDebugLoc(),
                                TII_->get(llvm::RISCV::ADDI))
                      .addReg(kRTS)
                      .addReg(kRTS)
                      .addImm(adjust_value);
                  ignore_this_mi = true;

                  break;
                }
              }
            }

            continue;
          }
          auto taken_adjust_value = mbb_sigs_.at(SBB).first +
                                    mbb_sigs_.at(SBB).second -
                                    mbb_sigs_.at(&MBB).first;
          assert_imm(taken_adjust_value);
          llvm::BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                        TII_->get(llvm::RISCV::ADDI))
              .addReg(kRTS)
              .addReg(kRTS)
              .addImm(taken_adjust_value);

          llvm::MachineBasicBlock *other_SBB{nullptr};
          for (auto SBBx : MBB.successors()) {
            if (SBBx != SBB) {
              other_SBB = SBBx;
              break;
            }
          }
          auto nottaken_adjust_value =
              -(mbb_sigs_.at(SBB).first + mbb_sigs_.at(SBB).second -
                mbb_sigs_.at(&MBB).first) +
              (mbb_sigs_.at(other_SBB).first + mbb_sigs_.at(other_SBB).second -
               mbb_sigs_.at(&MBB).first);
          assert_imm(nottaken_adjust_value);
          llvm::BuildMI(MBB, std::next(MI.getIterator()), MI.getDebugLoc(),
                        TII_->get(llvm::RISCV::ADDI))
              .addReg(kRTS)
              .addReg(kRTS)
              .addImm(nottaken_adjust_value);
          ignore_this_mi = true;
          if (MBB.back().isUnconditionalBranch()) {
            ignore_jump = true;
          } else {
            if (MBB.back().isCall() && MBB.back().getOperand(1).isMBB() &&
                err_bbs_.find(MBB.back().getOperand(1).getMBB()) !=
                    err_bbs_.end()) {
              ignore_jump = true;
            }
          }

          continue;
        }
        if (MI.isUnconditionalBranch()) {
          if (!ignore_jump) {
            assert(MI.getOperand(0).isMBB());
            auto SBB{MI.getOperand(0).getMBB()};
            auto adjust_value = mbb_sigs_.at(SBB).first +
                                mbb_sigs_.at(SBB).second -
                                mbb_sigs_.at(&MBB).first;
            assert_imm(adjust_value);
            llvm::BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                          TII_->get(llvm::RISCV::ADDI))
                .addReg(kRTS)
                .addReg(kRTS)
                .addImm(adjust_value);
          }

          continue;
        }

        // RASM processing before the return instruction
        if (MI.isReturn()) {
          short rand{unif_dist_(gen_)};
          llvm::BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                        TII_->get(llvm::RISCV::ADDI))
              .addReg(kRTS)
              .addReg(kRTS)
              .addImm(rand - mbb_sigs_.at(&MBB).first);
          llvm::BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                        TII_->get(llvm::RISCV::ADDI))
              .addReg(kC)
              .addReg(riscv_common::k0)
              .addImm(rand);
          llvm::BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(),
                        TII_->get(llvm::RISCV::BNE))
              .addReg(kRTS)
              .addReg(kC)
              .addMBB(cf_err_bb_);
          break;
        }

        // in case this MBB has no branch/ret terminator then we have to do RASM
        // update at end of MBB
        if (&MI == &MBB.back() && MBB.succ_size()) {
          assert(MBB.succ_size() == 1);
          auto SBB{*std::begin(MBB.successors())};

          auto adjust_value = mbb_sigs_.at(SBB).first + mbb_sigs_.at(SBB).second
    - mbb_sigs_.at(&MBB).first; assert_imm(adjust_value); llvm::BuildMI(MBB,
    std::end(MBB), MI.getDebugLoc(), TII_->get(llvm::RISCV::ADDI)) .addReg(kRTS)
              .addReg(kRTS)
              .addImm(adjust_value);
          break;
        }
      }
    } */
}

// for now we just write '1' to special memory address (0xfff8) to tell the
// simulator that this FI is covered in a separate error-BB. Further we
// stay in this loop to avoid further functional execution
void RISCVRasm::insertErrorBB() {
  cf_err_bb_ = MF_->CreateMachineBasicBlock();
  MF_->push_back(cf_err_bb_);

  auto DLL{MF_->front().front().getDebugLoc()};

  if (config_.eds == riscv_common::ErrorDetectionStrategy::ED0) {
    // storing '1' to addr : (0xfff8 = 0x10000 - 0x8):
    // lui t1, 16 -> makes t1 = 0x10000
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL,
                  TII_->get(llvm::RISCV::LUI))
        .addReg(llvm::RISCV::X6, llvm::RegState::Define)
        .addImm(16);
    // addi t2, zero, 1 -> makes t2 = 1
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL,
                  TII_->get(llvm::RISCV::ADDI))
        .addReg(llvm::RISCV::X7, llvm::RegState::Define)
        .addReg(riscv_common::k0)
        .addImm(1);
    // sw t2, -8(t1) -> stores t2 to (t1 - 8) i.e. store 1 to 0xfff8
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL,
                  TII_->get(isa_config_.store_opcode))
        .addReg(llvm::RISCV::X7)
        .addReg(llvm::RISCV::X6)
        .addImm(-8);

  } else if (config_.eds == riscv_common::ErrorDetectionStrategy::ED3) {
    // storing '93' to a7, aka SYS_exit code ECALL code:
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL,
                  TII_->get(llvm::RISCV::ADDI))
        .addReg(llvm::RISCV::X17, llvm::RegState::Define) // a7
        .addReg(llvm::RISCV::X0)  // zero
        .addImm(93);
    // storing '-512' to a0, aka exit return value when ecall forces newlib sys
    // exit:
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL,
                  TII_->get(llvm::RISCV::ADDI))
        .addReg(llvm::RISCV::X10, llvm::RegState::Define) // a7
        .addReg(llvm::RISCV::X0)  // zero
        .addImm(-256);
    // invoke 'ecall'
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL,
                  TII_->get(llvm::RISCV::ECALL))
        .addReg(llvm::RISCV::X17, llvm::RegState::Implicit)
        .addReg(llvm::RISCV::X10, llvm::RegState::Implicit);
  }

  if (config_.eds == riscv_common::ErrorDetectionStrategy::ED0 ||
      config_.eds == riscv_common::ErrorDetectionStrategy::ED3) {

    // to encode the error-BB in order to make it a label so as to be able to
    // jump to it from anywhere in asm
    cf_err_bb_->addSuccessor(cf_err_bb_);

    // keep on repeating this errBB as we dont want to execute code now
    // Use the branch pseudo so LLVM sees a terminator, rather than a call
    // with its destination register set to zero.
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL,
                  TII_->get(llvm::RISCV::PseudoBR))
        .addMBB(cf_err_bb_);
  } else {
    // quit early using ebreak
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL,
                  TII_->get(llvm::RISCV::EBREAK));
  }
}

bool RISCVRasm::branches_to_errbbs(const llvm::MachineInstr *mi) {
  auto *target = direct_branch_target(*mi);
  assert(target && "passed instruction is not a direct local branch!");
  return err_bbs_.find(target) != err_bbs_.end();
}

bool RISCVRasm::try_generate_psp_simm6_signatures() {
  // A bounded joint search: propose unique signed-ten-bit x labels, then
  // match unique q=x+y entry labels within all incoming conditional-edge
  // domains. Retrying x assignments makes this a joint allocation policy,
  // rather than independently shrinking the original label ranges. This is
  // RASM-only: RACFED's intra-block totals would change these equations.
  constexpr unsigned MaxAttempts = 32;
  constexpr unsigned MaxMatchingSteps = 200000;
  constexpr int MinQ = -1024;
  constexpr int MaxQ = 1022;
  std::vector<llvm::MachineBasicBlock *> blocks;
  std::map<const llvm::MachineBasicBlock *, unsigned> index;
  for (auto &MBB : *MF_) {
    if (err_bbs_.count(&MBB) || regions_->isTransparent(&MBB))
      continue;
    index.emplace(&MBB, blocks.size());
    blocks.push_back(&MBB);
  }
  const unsigned count = blocks.size();
  std::vector<std::set<unsigned>> normal_predecessors(count);
  std::vector<std::set<unsigned>> conditional_predecessors(count);
  unsigned sites = 0;
  for (unsigned source = 0; source < count; ++source) {
    auto &MBB = *blocks[source];
    for (auto *physical_target : MBB.successors()) {
      const auto *target = regions_->resolve(physical_target);
      const auto found = index.find(target);
      if (found != index.end())
        normal_predecessors[found->second].insert(source);
    }
    for (const auto &MI : MBB) {
      if (!MI.isConditionalBranch() || !regions_->monitorsBranch(MI))
        continue;
      const auto *taken = regions_->resolve(direct_branch_target(MI));
      if (!taken || index.find(taken) == index.end())
        continue; // NZDC detector branches do not select application intent.
      const auto *fallthrough = regions_->resolve(MBB.getFallThrough());
      if (!fallthrough || index.find(fallthrough) == index.end())
        llvm::report_fatal_error(
            "COMPAS: PSP allocation requires two normal predicate successors");
      conditional_predecessors[index.at(taken)].insert(source);
      conditional_predecessors[index.at(fallthrough)].insert(source);
      ++sites;
    }
  }

  unsigned attempts = 0;
  unsigned matching_steps = 0;
  const unsigned limit = std::min<unsigned>(rasm_predicate_psp_max_attempts, MaxAttempts);
  const auto report = [&](bool allocated, const char *reason) {
    llvm::outs() << "COMPAS_PSP function=" << fname_
                 << " status=" << (allocated ? "allocated" : "fallback")
                 << " reason=" << reason << " blocks=" << count
                 << " sites=" << sites << " seed=" << rasm_predicate_psp_seed
                 << " attempts=" << attempts << " max_attempts=" << limit
                 << " matching_steps=" << matching_steps
                 << " max_matching_steps=" << MaxMatchingSteps
                 << " require_nonzero_normal_edges=1\n";
  };
  if (count > 1024)
    llvm::report_fatal_error(
        "COMPAS: PSP allocation exceeds the 1024 distinct signed-ten-bit block labels");
  if (!count || !limit) {
    report(false, !limit ? "attempt_limit" : "label_capacity");
    return false;
  }

  // Use a separate engine so the selected lowering (software or XPred) does
  // not change allocations or consume the legacy signature/return RNG stream.
  std::mt19937 engine(rasm_predicate_psp_seed);
  std::vector<int> permutation(count);
  std::iota(permutation.begin(), permutation.end(), 0);
  for (; attempts < limit && matching_steps < MaxMatchingSteps; ) {
    ++attempts;
    if (attempts > 1)
      std::shuffle(permutation.begin(), permutation.end(), engine);
    const int origin = std::uniform_int_distribution<int>(
        -512, 512 - static_cast<int>(count))(engine);
    std::vector<int> x(count);
    for (unsigned i = 0; i < count; ++i)
      x[i] = origin + permutation[i];

    std::vector<std::vector<int>> domains(count);
    bool feasible = true;
    for (unsigned target = 0; target < count; ++target) {
      int lower = std::max(MinQ, x[target] - 512);
      int upper = std::min(MaxQ, x[target] + 511);
      for (unsigned source : conditional_predecessors[target]) {
        lower = std::max(lower, x[source] - 32);
        upper = std::min(upper, x[source] + 31);
      }
      std::set<int> forbidden{x[target]}; // y must be nonzero.
      for (unsigned source : normal_predecessors[target])
        forbidden.insert(x[source]); // No zero logical normal-edge update.
      for (int q = lower; q <= upper; ++q)
        if (!forbidden.count(q))
          domains[target].push_back(q);
      if (domains[target].empty()) {
        feasible = false;
        break;
      }
      std::shuffle(domains[target].begin(), domains[target].end(), engine);
    }
    if (!feasible)
      continue;

    std::vector<unsigned> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
      return domains[a].size() < domains[b].size();
    });
    std::vector<int> owner(MaxQ - MinQ + 1, -1);
    std::vector<int> q(count, 0);
    std::vector<bool> seen(owner.size());
    std::function<bool(unsigned)> augment = [&](unsigned target) {
      for (int candidate : domains[target]) {
        if (matching_steps == MaxMatchingSteps)
          return false;
        ++matching_steps;
        const unsigned slot = candidate - MinQ;
        if (seen[slot])
          continue;
        seen[slot] = true;
        if (owner[slot] < 0 || augment(static_cast<unsigned>(owner[slot]))) {
          owner[slot] = target;
          q[target] = candidate;
          return true;
        }
      }
      return false;
    };
    for (unsigned target : order) {
      std::fill(seen.begin(), seen.end(), false);
      if (!augment(target)) {
        feasible = false;
        break;
      }
    }
    if (!feasible)
      continue;

    // Validate the completed assignment independently of the search choices
    // before publishing any labels to the instrumentation pass.
    std::set<int> unique_x, unique_q;
    for (unsigned target = 0; target < count; ++target) {
      const int y = q[target] - x[target];
      if (!llvm::isInt<10>(x[target]) || !llvm::isInt<10>(y) || !y ||
          !unique_x.insert(x[target]).second || !unique_q.insert(q[target]).second)
        llvm::report_fatal_error("COMPAS: invalid constrained PSP label assignment");
      for (unsigned source : conditional_predecessors[target])
        if (!llvm::isInt<6>(q[target] - x[source]))
          llvm::report_fatal_error("COMPAS: constrained PSP edge exceeds signed-six-bit range");
      for (unsigned source : normal_predecessors[target])
        if (q[target] == x[source])
          llvm::report_fatal_error("COMPAS: constrained PSP produced a zero normal-edge update");
    }
    for (unsigned i = 0; i < count; ++i)
      mbb_sigs_[blocks[i]] = {static_cast<short>(x[i]), static_cast<short>(q[i] - x[i])};
    report(true, "matched");
    return true;
  }
  report(false, matching_steps == MaxMatchingSteps ? "matching_budget" : "search_exhausted");
  return false;
}

void RISCVRasm::generate_signatures() {
  //////////////////////////////////////////////////////////////////////////////
  // Step 1: Assign compile time signatues and random arbitration value
  // "subRanPrevVal" to each basic block of the function.
  llvm::outs() << "RACFED S1: Assigning compile time sigs to each MBB of "
               << MF_->getName() << "\n";

  if (predicate_psp_simm6_) {
    for (auto &MBB : *MF_)
      if (is_error_block(MBB))
        err_bbs_.insert(&MBB);
    if (try_generate_psp_simm6_signatures()) {
      for (const auto &MBB : *MF_) {
        const auto found = mbb_sigs_.find(&MBB);
        if (found != mbb_sigs_.end())
          llvm::outs() << "COMPAS_PSP_LABEL function=" << fname_
                       << " block=" << MBB.getNumber()
                       << " x=" << found->second.first
                       << " y=" << found->second.second
                       << " q=" << found->second.first + found->second.second << "\n";
      }
      return;
    }
  }

  // assigning compile time sigs to each MBB
  std::set<short> sigs_sofar{}, sum_sofar{};
  for (auto &MBB : *MF_) {
    // there could be err-BBs from other passes
    // the convention is if a BB keeps jumping to itself then this is also
    // an error-BB
    if (MBB.succ_size() == 1 && &MBB == *MBB.succ_begin()) {
      err_bbs_.emplace(&MBB);
      continue;
    }
    if (regions_->isTransparent(&MBB))
      continue;

    short y = 0;
    while (1) {
      std::uniform_int_distribution<short> unif_dist{-512, 511};
      auto x{unif_dist(gen_)};
      if (sigs_sofar.find(x) == std::end(sigs_sofar)) {
        sigs_sofar.emplace(x);
        while (1) { // make sure that y, aka, subRanPrevVal is not zero
          // because it is used for a signature arbirtation
          // instruction makiung it a NOP if 0
          y = unif_dist(gen_);
          if (sum_sofar.find(x + y) == std::end(sum_sofar) && y != 0) {
            sum_sofar.emplace(x + y);
            mbb_sigs_[&MBB] = {x, y};
            break;
          }
        }
        break;
      }
    }
  }
  if (predicate_psp_simm6_ || predicate_xpred_mode_)
    for (const auto &MBB : *MF_) {
      const auto found = mbb_sigs_.find(&MBB);
      if (found != mbb_sigs_.end())
        llvm::outs() << "COMPAS_PSP_LABEL function=" << fname_
                     << " block=" << MBB.getNumber()
                     << " x=" << found->second.first
                     << " y=" << found->second.second
                     << " q=" << found->second.first + found->second.second << "\n";
    }
}

void RISCVRasm::generate_signature_checks() {
  //////////////////////////////////////////////////////////////////////////////
  // Insert runtime signature arbitration and signature check in the
  // beginning of each basic block except for machine function entry basic
  // block where we just assign the runtime signature with the block's compile
  // time signature
  llvm::outs() << "Insert runtime signature arbitration and "
                  "signature check handling of traversals on "
               << MF_->getName() << "\n";
  for (auto &MBB : *MF_) {
    if (MBB.succ_size() == 1 && &MBB == *MBB.succ_begin()) {
      // filter out error handler basic blocks, here blocks that jump to
      // itself
      continue;
    }
    if (regions_->isTransparent(&MBB))
      continue;

    if (MBB.pred_empty()) {
      // for entryBB we dont check RTS
      llvm::BuildMI(MF_->front(), riscv_common::signatureEntryInsertion(*MF_),
                    std::begin(MF_->front())->getDebugLoc(),
                    TII_->get(llvm::RISCV::ADDI))
          .addReg(kRTS, llvm::RegState::Define)
          .addReg(riscv_common::k0)
          .addImm(mbb_sigs_.at(&MF_->front()).first);
    } else {
      auto insert{std::begin(MBB)};
      // updating RTS
      llvm::BuildMI(MBB, insert, insert->getDebugLoc(),
                    TII_->get(llvm::RISCV::ADDI))
          .addReg(kRTS, llvm::RegState::Define)
          .addReg(kRTS)
          .addImm(-mbb_sigs_.at(&MBB).second);
      mbb_signature_arbr_instrs_[&MBB] = &(*(std::prev(insert)));

      // comparing the RTS with compile-time sig
      llvm::BuildMI(MBB, insert, insert->getDebugLoc(),
                    TII_->get(llvm::RISCV::ADDI))
          .addReg(kC, llvm::RegState::Define)
          .addReg(riscv_common::k0)
          .addImm(mbb_sigs_.at(&MBB).first);

      mbb_signature_check_instrs_[&MBB] = &(*(std::prev(insert)));
      llvm::BuildMI(MBB, insert, insert->getDebugLoc(),
                    TII_->get(llvm::RISCV::BNE))
          .addReg(kRTS)
          .addReg(kC)
          .addMBB(cf_err_bb_);
    }
  }
}

short RISCVRasm::calculate_adjustment(
    const llvm::MachineBasicBlock *source_bb,
    const llvm::MachineBasicBlock *target_bb) {
  short ret = mbb_sigs_.at(target_bb).first + mbb_sigs_.at(target_bb).second -
              mbb_sigs_.at(source_bb).first;

  assert_imm(ret);
  return ret;
}

void RISCVRasm::save_restore_runtime_signature(llvm::MachineInstr *call_instr) {
  riscv_common::preserveSignatureAcrossCall(call_instr, {kRTS}, kC);
}

short RISCVRacfed::calculate_adjustment(
    const llvm::MachineBasicBlock *source_bb,
    const llvm::MachineBasicBlock *target_bb) {
  short ret = mbb_sigs_.at(target_bb).first + mbb_sigs_.at(target_bb).second -
              (mbb_sigs_.at(source_bb).first + mbb_sum_ii_sigs_.at(source_bb));

  assert_imm(ret);
  return ret;
}

void RISCVRasm::generate_traversal_adjustments() {
  llvm::outs() << "RACFED S3: Calculate traversal adjustment values for "
                  "non-trivial instructions on "
               << MF_->getName() << "\n";
  for (auto &MBB : *MF_) {
    if (err_bbs_.find(&MBB) != err_bbs_.end() || regions_->isTransparent(&MBB))
      continue;

    // Visit only instructions present before traversal instrumentation. In
    // particular, a merged fallthrough adjustment inserts no instruction to
    // skip, whereas a call may insert several save/restore instructions.
    std::vector<llvm::MachineInstr *> instructions;
    for (auto &MI : MBB)
      instructions.push_back(&MI);

    // A conditional branch's taken adjustment also executes when the branch
    // is not taken. Carry it forward so the following transfer (or implicit
    // fallthrough) adjusts from the signature that actually reaches it.
    short carried_adjustment = 0;
    bool falls_through = true;
    for (auto *MI : instructions) {
      auto *physical_branch_target = direct_branch_target(*MI);
      auto *target = regions_->resolve(physical_branch_target);
      const auto helper = riscv_common::frameHelper(*MI);
      if (helper == riscv_common::FrameHelper::Save)
        continue; // RTS starts after this helper, before FP setup.
      if (MI->isCall() && !is_local_unconditional_jump(*MI) &&
          helper != riscv_common::FrameHelper::Restore) {
        // Non-local no-link jumps and tail calls do not return to this block.
        if ((MI->getNumOperands() && MI->getOperand(0).isReg() &&
             MI->getOperand(0).getReg() == llvm::RISCV::X0) ||
            TII_->isTailCall(*MI)) {
          falls_through = false;
          break;
        }
        save_restore_runtime_signature(MI);
        continue;
      }

      if (MI->isConditionalBranch()) {
        assert(physical_branch_target &&
               "conditional branch has no local target!");
        // Error checks leave the normal path's signature untouched. Its
        // adjustment belongs at the eventual normal transfer, including an
        // implicit fallthrough after a final error check.
        if (branches_to_errbbs(MI))
          continue;
        // Unselected instructions still connect signed regions. Keep their
        // normal transfer arithmetic so each legal target receives its state.
        assert(target && "monitored branch has no normal destination!");

        const auto target_adjustment = calculate_adjustment(&MBB, target);
        if (predicate_mode_ && regions_->monitorsBranch(*MI)) {
          auto *fallthrough = regions_->resolve(MBB.getFallThrough());
          if (!fallthrough || fallthrough == target ||
              err_bbs_.count(fallthrough) || carried_adjustment)
            llvm::report_fatal_error("COMPAS: unsupported RASM-predicate successor adjustment");
          const auto false_adjustment = calculate_adjustment(&MBB, fallthrough);
          auto predicate = riscv_common::getSoftBranchPredicate(*MI);
          if (!predicate)
            llvm::report_fatal_error(
                "COMPAS: RASM-predicate lost its application BEQ/BNE");
          if (predicate_shadow_mode_) {
            const auto shadow_source = [&](llvm::Register source) {
              const auto pair = P2S_.find(source);
              if (pair == P2S_.end() || pair->second == kRTS ||
                  pair->second == kC)
                llvm::report_fatal_error(
                    "COMPAS: RASM-predicate-shadow requires a mapped NZDC "
                    "primary operand with an independent shadow register");
              return pair->second;
            };
            // Map x0 too: NZDC's shadow zero is the faultable x27 stream,
            // not the architectural zero used by the primary branch.
            predicate->lhs = shadow_source(predicate->lhs);
            predicate->rhs = shadow_source(predicate->rhs);
          }
          riscv_common::emitSoftBranchPredicate(*MI, *predicate, kC);
          const bool simm6_fit = llvm::isInt<6>(false_adjustment) &&
                                  llvm::isInt<6>(target_adjustment);
          ++predicate_sites_;
          predicate_simm6_sites_ += simm6_fit;
          const bool fused = predicate_xpred_mode_ &&
              riscv_common::tryEmitXPredPredicateSelectedAdd(
                  *MI, kRTS, kC, false_adjustment, target_adjustment);
          if (fused) {
            ++predicate_fused_sites_;
          } else {
            if (predicate_xpred_mode_)
              ++predicate_range_fallbacks_;
            riscv_common::emitPredicateSelectedAdd(*MI, kRTS, kC,
                                                 false_adjustment,
                                                 target_adjustment);
          }
          if (predicate_xpred_mode_ || predicate_psp_simm6_)
            llvm::outs() << "COMPAS_XPRED_SITE function=" << fname_
                         << " block=" << MBB.getNumber()
                         << " taken=" << target->getNumber()
                         << " fallthrough=" << fallthrough->getNumber()
                         << " true=" << target_adjustment
                         << " false=" << false_adjustment
                         << " fused=" << fused << "\n";
          const unsigned gap_nops = predicate_unpadded_ ? 0 : PredicateGapNops;
          for (unsigned i = 0; i < gap_nops; ++i)
            llvm::BuildMI(MBB, MI->getIterator(), MI->getDebugLoc(),
                          TII_->get(llvm::RISCV::ADDI))
                .addReg(riscv_common::k0, llvm::RegState::Define)
                .addReg(riscv_common::k0)
                .addImm(0);
          // On normal fallthrough the selected false adjustment has already
          // executed. Do not subtract a taken adjustment or overwrite it with
          // the old fallthrough compensation.
          carried_adjustment = false_adjustment;
          continue;
        }
        const short adjustment = target_adjustment - carried_adjustment;
        assert_imm(adjustment);
        llvm::BuildMI(MBB, MI->getIterator(), MI->getDebugLoc(),
                      TII_->get(llvm::RISCV::ADDI))
            .addReg(kRTS, llvm::RegState::Define)
            .addReg(kRTS)
            .addImm(adjustment);
        carried_adjustment = target_adjustment;
        continue;
      }

      if (is_local_unconditional_jump(*MI)) {
        // A duplicate jump to an error block has no signature, and any
        // instructions following an unconditional transfer are unreachable
        // on the fault-free path.
        if (!branches_to_errbbs(MI)) {
          assert(target && "normal jump has no normal destination!");
          const short adjustment =
              calculate_adjustment(&MBB, target) - carried_adjustment;
          assert_imm(adjustment);
          llvm::BuildMI(MBB, MI->getIterator(), MI->getDebugLoc(),
                        TII_->get(llvm::RISCV::ADDI))
              .addReg(kRTS, llvm::RegState::Define)
              .addReg(kRTS)
              .addImm(adjustment);
        }
        falls_through = false;
        break;
      }

      if (MI->isReturn() || helper == riscv_common::FrameHelper::Restore) {
        short adjust_value = 0, rand = 0;
        const short current_signature = mbb_sigs_.at(&MBB).first +
                                        mbb_sum_ii_sigs_.at(&MBB) +
                                        carried_adjustment;
        do {
          std::uniform_int_distribution<short> unif_dist{
              -1024 - current_signature, 1023 - current_signature};
          rand = unif_dist(gen_);
          adjust_value = rand - current_signature;
        } while (adjust_value < -2048 || adjust_value > 2047);
        assert_imm(adjust_value);

        llvm::BuildMI(MBB, MI->getIterator(), MI->getDebugLoc(),
                      TII_->get(llvm::RISCV::ADDI))
            .addReg(kRTS, llvm::RegState::Define)
            .addReg(kRTS)
            .addImm(adjust_value);
        llvm::BuildMI(MBB, MI->getIterator(), MI->getDebugLoc(),
                      TII_->get(llvm::RISCV::ADDI))
            .addReg(kC, llvm::RegState::Define)
            .addReg(riscv_common::k0)
            .addImm(rand);
        llvm::BuildMI(MBB, MI->getIterator(), MI->getDebugLoc(),
                      TII_->get(llvm::RISCV::BNE))
            .addReg(kRTS)
            .addReg(kC)
            .addMBB(cf_err_bb_);
        falls_through = false;
        break;
      }
    }

    if (!falls_through)
      continue;
    auto *physical_target = MBB.getFallThrough();
    auto *target = regions_->resolve(physical_target);
    if (!target || err_bbs_.find(target) != err_bbs_.end())
      continue;

    const short adjustment =
        calculate_adjustment(&MBB, target) - carried_adjustment;
    assert_imm(adjustment);
    if (predicate_mode_ && adjustment == 0)
      continue;
    // Preserve the existing one-instruction optimization when the layout
    // successor can only be entered by this fallthrough edge.
    if (target == physical_target &&
        regions_->predecessors(target).size() == 1) {
      const short merged_adjustment = adjustment - mbb_sigs_.at(target).second;
      assert_imm(merged_adjustment);
      mbb_signature_arbr_instrs_.at(target)->getOperand(2).setImm(
          merged_adjustment);
    } else {
      llvm::BuildMI(MBB, MBB.end(), MBB.back().getDebugLoc(),
                    TII_->get(llvm::RISCV::ADDI))
          .addReg(kRTS, llvm::RegState::Define)
          .addReg(kRTS)
          .addImm(adjustment);
    }
  }
}

// ------------------------------------------- RACFED --------------------------

bool RISCVRacfed::runOnMachineFunction(llvm::MachineFunction &MF) {
  MF_ = &MF;
  fname_ = std::string{MF_->getName()};
  TII_ = MF_->getSubtarget().getInstrInfo();

  if (!riscv_common::inCSString(llvm::cl::enable_racfed,
                                std::string{MF_->getName()})) {
    return false;
  }

  llvm::outs() << "Running RACFED pass on " << MF_->getName() << "\n";

  regions_ = std::make_unique<riscv_common::RsmRegions>(
      *MF_, llvm::cl::enable_racfed.getValue());
  init();

  llvm::outs() << "Running RACFED harden() on " << MF_->getName() << "\n";

  harden();

  return true;
}

void RISCVRacfed::init() {

  RISCVRasm::init();

  mi_random_value_.clear();
}

void RISCVRasm::generate_intrablock_signature_updates() {
  for (auto &MBB : *MF_) {
    if (regions_->isTransparent(&MBB) ||
        (MBB.succ_size() == 1 && &MBB == *MBB.succ_begin())) {
      continue;
    }
    mbb_sum_ii_sigs_[&MBB] = 0; // force to 0 -> RASM has no intra block updates
  }
}

void RISCVRacfed::generate_intrablock_signature_updates() {

  llvm::outs() << "RACFED S2: Intra-block runtime signature updates on "
               << MF_->getName() << "\n";

  for (auto &MBB : *MF_) {
    // there could be err-BBs from other passes
    // the convention is if a BB keeps jumping to itself then this is also
    // an error-BB
    if (regions_->isTransparent(&MBB) ||
        (MBB.succ_size() == 1 && &MBB == *MBB.succ_begin())) {
      continue;
    }
    llvm::outs() << "> mbb[" << &MBB << "] {X=" << mbb_sigs_.at(&MBB).first
                 << ", Y=" << mbb_sigs_.at(&MBB).second << "}: " << MBB << "\n";
    std::vector<llvm::MachineInstr *> orig_instrs;

    short sum_ii_rvs = 0;
    const auto start = &MBB == &MF_->front()
                           ? riscv_common::signatureEntryInsertion(*MF_)
                           : MBB.begin();
    bool monitoring = false;
    for (auto &MI : MBB) {
      if (MI.getIterator() == start)
        monitoring = true;
      if (!monitoring)
        continue; // Entry prefix through the save helper has no active RTS.
      llvm::outs() << "> mi[" << &MI << "]: " << MI
                   << "back?: " << (&MI == &(MBB.back())) << "mback["
                   << &(MBB.instr_back()) << "]" << MBB.instr_back() << "\n";

      if (!regions_->monitorsInstruction(MI) ||
          riscv_common::frameHelper(MI) != riscv_common::FrameHelper::None ||
          MI.isMetaInstruction() || // metadata emits no monitored instruction
          (MI.isBranch() && !MI.isConditionalBranch()) ||
          is_local_unconditional_jump(MI) ||
          (MI.isConditionalBranch() &&
           !branches_to_errbbs(&MI)) || // normal branches use traversal updates
          //((&MI == &(MBB.instr_back()) ) && MI.isConditionalBranch() &&
          // branches_to_errbbs(
          //     &MI)) || // ignore DMR not taken/fallthrough duplicate
          MI.isReturn() // returns have a special random arbitration such that
                        // we do not need to perform a random update for the
                        // instruction before return
      ) {
        llvm::outs() << "\>> not considered ui: [" << &MI << "]" << MI << "\n";
        continue;
      }

      short rv = 0;
      if (MI.getIterator() != MBB.begin()) {
        // first instruction, we will either the initial
        // signature function entry or the consecutive checks
        // here
        short cs = mbb_sigs_.at(&MBB).first +
                   sum_ii_rvs; // current signature, we need to
        // predict to avoid overflows

        std::uniform_int_distribution<short> unif_dist{-1024 - cs, 1023 - cs};
        while (rv == 0) {
          rv = unif_dist(gen_);
        }

        llvm::outs() << "> gen rv [" << rv << "] for  "
                     << "mi: [" << &MI << "]" << MI << "\n"
                     << "cs was " << cs << " now is " << cs + rv << "\n";

        assert_imm(cs + rv);
      }

      orig_instrs.push_back(&MI);
      llvm::outs() << ">> ui: [" << &MI << "]" << MI << "\n";

      mi_random_value_[&MBB][&MI] = rv;

      sum_ii_rvs += rv;
    }

    mbb_sum_ii_sigs_[&MBB] = 0; // force to 0

    if (orig_instrs.size() >= N_MIN_INTRA_INSTRUCTION_COUNT) {
      // There is no point in intra block protection for basic blocks with
      // only 2 or 1 instruction since all intra block errors are also legal
      // control flow
      for (auto &mi : orig_instrs) {
        auto rv = mi_random_value_.at(&MBB).at(mi);
        if (rv == 0)
          continue;

        mbb_sum_ii_sigs_.at(&MBB) += rv;

        llvm::outs() << ">adding µi update [" << rv << "] for  "
                     << "mi: [" << mi << "]" << *mi << "\n";
        llvm::MachineBasicBlock::iterator insert{mi->getIterator()};

        llvm::BuildMI(MBB, insert, mi->getDebugLoc(),
                      TII_->get(llvm::RISCV::ADDI))
            .addReg(kRTS, llvm::RegState::Define)
            .addReg(kRTS)
            .addImm(rv);
      }
    }
  }
}

// RACFED hardening scheme
void RISCVRacfed::harden() {
  //////////////////////////////////////////////////////////////////////////////
  // Step 1: Assign compile time signatues and random arbitration value
  // "subRanPrevVal" to each basic block of the function.
  generate_signatures();

  //////////////////////////////////////////////////////////////////////////////
  // Step 2: Now add for all original and trivial instruction the intra block
  // signature update with random value. Trivial means that the instructions
  // are neither branches nor returns. Only BBs with more than 2 trivial
  // instructions may be protected against intra block control flow errors
  generate_intrablock_signature_updates();
  //////////////////////////////////////////////////////////////////////////////
  // Step 3: Insert runtime signature arbitration and signature check in the
  // beginning of each basic block except for machine function entry basic
  // block where we just assign the runtime signature with the block's compile
  // time signature
  generate_signature_checks();

  //////////////////////////////////////////////////////////////////////////////
  // Step 4: Calculate traversal adjustment values for non-trivial
  // instructions
  generate_traversal_adjustments();
}
