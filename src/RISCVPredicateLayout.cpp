/*
 * Copyright 2026 Chair of EDA, Technical University of Munich
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RISCVPredicateLayout.h"

#include "RISCV.h"
#include "common.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <vector>

namespace {

struct Layout {
  std::map<const llvm::MachineBasicBlock *, int64_t> blocks;
  std::map<const llvm::MachineInstr *, int64_t> instructions;
};

Layout conservativeLayout(const llvm::MachineFunction &MF) {
  Layout layout;
  const auto *TII = MF.getSubtarget().getInstrInfo();
  int64_t offset = 0;
  for (const auto &MBB : MF) {
    // Any real alignment gap is no larger than this. Differences between
    // these offsets therefore bound each branch's actual displacement even
    // when the function's eventual address is not known yet.
    offset += MBB.getAlignment().value() - 1;
    layout.blocks.emplace(&MBB, offset);
    for (const auto &MI : MBB) {
      layout.instructions.emplace(&MI, offset);
      const unsigned size = TII->getInstSizeInBytes(MI);
      if ((!size && !MI.isMetaInstruction()) || size % 4)
        llvm::report_fatal_error(
            "COMPAS: predicate branch relaxation requires known fixed-width "
            "instruction sizes");
      offset += size;
    }
  }
  return layout;
}

llvm::MachineBasicBlock *localTarget(const llvm::MachineInstr &MI) {
  if (!MI.isBranch() && MI.getOpcode() != llvm::RISCV::JAL)
    return nullptr;
  for (const auto &operand : MI.operands())
    if (operand.isMBB())
      return operand.getMBB();
  return nullptr;
}

unsigned inverseCondition(unsigned opcode) {
  switch (opcode) {
  case llvm::RISCV::BEQ:
    return llvm::RISCV::BNE;
  case llvm::RISCV::BNE:
    return llvm::RISCV::BEQ;
  case llvm::RISCV::BLT:
    return llvm::RISCV::BGE;
  case llvm::RISCV::BGE:
    return llvm::RISCV::BLT;
  case llvm::RISCV::BLTU:
    return llvm::RISCV::BGEU;
  case llvm::RISCV::BGEU:
    return llvm::RISCV::BLTU;
  default:
    llvm::report_fatal_error("COMPAS: unsupported conditional opcode in "
                             "predicate branch relaxation");
  }
}

} // namespace

void riscv_common::relaxPredicateBranches(llvm::MachineFunction &MF) {
  const auto *TII = MF.getSubtarget().getInstrInfo();
  std::set<llvm::MachineInstr *> relaxed;
  for (;;) {
    const auto layout = conservativeLayout(MF);
    std::vector<llvm::MachineInstr *> pending;
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        if (!MI.isConditionalBranch())
          continue;
        auto *target = localTarget(MI);
        if (!target)
          llvm::report_fatal_error(
              "COMPAS: predicate branch relaxation requires a local target");
        const auto displacement =
            layout.blocks.at(target) - layout.instructions.at(&MI);
        // GNU's whole-function relaxation can retain an unnecessary long
        // form for a forward branch at the final 4092-byte boundary, even
        // with -mno-relax. Leave one instruction of forward headroom so that
        // all expansion remains explicit here. Backward branches retain the
        // architectural -4096-byte limit.
        if (llvm::isInt<13>(displacement) && displacement <= 4088)
          continue;
        if (relaxed.count(&MI))
          llvm::report_fatal_error(
              "COMPAS: predicate branch trampoline exceeds the conditional "
              "branch range (excessive block alignment)");
        pending.push_back(&MI);
      }
    }
    if (pending.empty())
      break;

    for (auto *branch : pending) {
      auto *source = branch->getParent();
      auto *taken = localTarget(*branch);
      auto *fallthrough = source->getNextNode();
      if (!fallthrough || !source->isSuccessor(fallthrough) ||
          fallthrough == taken ||
          std::next(branch->getIterator()) != source->end())
        llvm::report_fatal_error(
            "COMPAS: predicate branch relaxation requires normalized distinct "
            "taken and fallthrough edges");

      // The monitor has already selected the state for the original logical
      // edge. Preserve that selection while replacing the physical transfer:
      //   cond T; F:  ->  !cond F; J: j T; F:
      // Neither this trampoline nor the inversion resets/updates the state.
      auto *trampoline = MF.CreateMachineBasicBlock(source->getBasicBlock());
      MF.insert(std::next(source->getIterator()), trampoline);
      llvm::BuildMI(*trampoline, trampoline->end(), branch->getDebugLoc(),
                    TII->get(llvm::RISCV::PseudoBR))
          .addMBB(taken);
      trampoline->addSuccessor(taken);
      source->replaceSuccessor(taken, trampoline);
      source->normalizeSuccProbs();
      branch->setDesc(TII->get(inverseCondition(branch->getOpcode())));
      for (auto &operand : branch->operands())
        if (operand.isMBB())
          operand.setMBB(fallthrough);
      relaxed.insert(branch);
    }
  }

  // No silent assembler relaxation may alter the remaining transfer shape.
  // Conditional ranges were checked at the fixed point; check every local
  // direct jump too, including DMR and error-handler transfers.
  const auto layout = conservativeLayout(MF);
  unsigned direct_jumps = 0;
  for (const auto &MBB : MF) {
    for (const auto &MI : MBB) {
      auto *target = localTarget(MI);
      if (!target || MI.isConditionalBranch())
        continue;
      if (MI.getOpcode() != llvm::RISCV::JAL &&
          MI.getOpcode() != llvm::RISCV::PseudoBR)
        llvm::report_fatal_error(
            "COMPAS: unsupported local direct jump in predicate layout");
      const auto displacement =
          layout.blocks.at(target) - layout.instructions.at(&MI);
      if (!llvm::isInt<21>(displacement))
        llvm::report_fatal_error(
            "COMPAS: predicate local direct jump exceeds the JAL range; "
            "indirect long-jump relaxation is unsupported");
      ++direct_jumps;
    }
  }

  if (!relaxed.empty())
    normalizeControlFlow(MF); // Repair all physical-register live-ins/flags.
  llvm::outs() << "COMPAS_PREDICATE_LAYOUT function=" << MF.getName()
               << " relaxed=" << relaxed.size()
               << " direct_jumps=" << direct_jumps << "\n";
}
