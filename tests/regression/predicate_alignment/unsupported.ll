target triple = "riscv32-unknown-elf"
@output = external global i32
declare i32 @callee(i32)

; No function target attributes: command-line RV64/C checks must take effect.
define i32 @target_check(i32 %a, i32 %b) nounwind {
  %test = icmp eq i32 %a, %b
  br i1 %test, label %yes, label %no
yes:
  store volatile i32 17, ptr @output
  ret i32 17
no:
  store volatile i32 29, ptr @output
  store volatile i32 31, ptr @output
  ret i32 29
}

define i32 @bad_relation(i32 %a, i32 %b) nounwind {
  %test = icmp ult i32 %a, %b
  br i1 %test, label %yes, label %no
yes:
  store volatile i32 17, ptr @output
  ret i32 17
no:
  store volatile i32 29, ptr @output
  store volatile i32 31, ptr @output
  ret i32 29
}

define i32 @bad_call(i32 %a) nounwind {
  %r = call i32 @callee(i32 %a)
  %s = add i32 %r, 1
  ret i32 %s
}
