/*
 * Copyright 2026 Chair of EDA, Technical University of Munich
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace llvm {
class MachineFunction;
}

namespace riscv_common {
// Relax branches after predicate instrumentation without changing its logical
// edges or signature arithmetic. The added direct-jump blocks are transfer
// infrastructure, not application regions. Require fixed-width RV32 code and
// reject local direct jumps outside the architectural JAL range.
void relaxPredicateBranches(llvm::MachineFunction &MF);
} // namespace riscv_common
