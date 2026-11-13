; A one-, two- and three-instruction region must each get post-instruction updates.
target triple = "riscv32-unknown-elf"

define i32 @one(i32 %a) nounwind {
  %v = add i32 %a, 1
  ret i32 %v
}
define i32 @two(i32 %a) nounwind {
  %v = add i32 %a, 1
  %w = xor i32 %v, 7
  ret i32 %w
}
define i32 @three(i32 %a) nounwind {
  %v = add i32 %a, 1
  %w = xor i32 %v, 7
  %x = and i32 %w, 31
  ret i32 %x
}
