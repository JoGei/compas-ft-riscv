target triple = "riscv32-unknown-elf"
@output = external global i32
declare i32 @ordinary_callee(i32)

; No target attributes: command-line RV32/C and RV64 negative cases must not
; inherit the RV32I frontend feature list of functional.c.
define i32 @target_check(i32 %a, i32 %b) nounwind {
  %condition = icmp eq i32 %a, %b
  br i1 %condition, label %taken, label %other
taken:
  store volatile i32 17, ptr @output
  ret i32 17
other:
  store volatile i32 29, ptr @output
  store volatile i32 31, ptr @output
  ret i32 29
}

define i32 @bad_relation(i32 %a, i32 %b) nounwind {
  %condition = icmp ult i32 %a, %b
  br i1 %condition, label %taken, label %other
taken:
  store volatile i32 17, ptr @output
  ret i32 17
other:
  store volatile i32 29, ptr @output
  store volatile i32 31, ptr @output
  ret i32 29
}

define i32 @bad_call(i32 %a) nounwind {
  %v = call i32 @ordinary_callee(i32 %a)
  %r = add i32 %v, 1
  ret i32 %r
}
