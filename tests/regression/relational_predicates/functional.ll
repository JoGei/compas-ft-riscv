; Direct branch comparisons with observable, asymmetric successor effects.

; No target attributes: the runner controls RV32 and native SACC features.

target triple = "riscv32-unknown-elf"

@taken_output = global i32 0, align 4

@fallthrough_output = global i32 0, align 4



define i32 @eq(i32 %a, i32 %b) nounwind {
entry:
  %condition = icmp eq i32 %a, %b
  br i1 %condition, label %yes, label %no
yes:
  store volatile i32 17, ptr @taken_output, align 4
  ret i32 17
no:
  store volatile i32 29, ptr @fallthrough_output, align 4
  store volatile i32 31, ptr @fallthrough_output, align 4
  ret i32 29
}

define i32 @ne(i32 %a, i32 %b) nounwind {
entry:
  %condition = icmp ne i32 %a, %b
  br i1 %condition, label %yes, label %no
yes:
  store volatile i32 17, ptr @taken_output, align 4
  ret i32 17
no:
  store volatile i32 29, ptr @fallthrough_output, align 4
  store volatile i32 31, ptr @fallthrough_output, align 4
  ret i32 29
}

define i32 @slt(i32 %a, i32 %b) nounwind {
entry:
  %condition = icmp slt i32 %a, %b
  br i1 %condition, label %yes, label %no
yes:
  store volatile i32 17, ptr @taken_output, align 4
  ret i32 17
no:
  store volatile i32 29, ptr @fallthrough_output, align 4
  store volatile i32 31, ptr @fallthrough_output, align 4
  ret i32 29
}

define i32 @sge(i32 %a, i32 %b) nounwind {
entry:
  %condition = icmp sge i32 %a, %b
  br i1 %condition, label %yes, label %no
yes:
  store volatile i32 17, ptr @taken_output, align 4
  ret i32 17
no:
  store volatile i32 29, ptr @fallthrough_output, align 4
  store volatile i32 31, ptr @fallthrough_output, align 4
  ret i32 29
}

define i32 @ult(i32 %a, i32 %b) nounwind {
entry:
  %condition = icmp ult i32 %a, %b
  br i1 %condition, label %yes, label %no
yes:
  store volatile i32 17, ptr @taken_output, align 4
  ret i32 17
no:
  store volatile i32 29, ptr @fallthrough_output, align 4
  store volatile i32 31, ptr @fallthrough_output, align 4
  ret i32 29
}

define i32 @uge(i32 %a, i32 %b) nounwind {
entry:
  %condition = icmp uge i32 %a, %b
  br i1 %condition, label %yes, label %no
yes:
  store volatile i32 17, ptr @taken_output, align 4
  ret i32 17
no:
  store volatile i32 29, ptr @fallthrough_output, align 4
  store volatile i32 31, ptr @fallthrough_output, align 4
  ret i32 29
}

define i32 @slt_zero(i32 %a, i32 %b) nounwind {
entry:
  %condition = icmp slt i32 %a, 0
  br i1 %condition, label %yes, label %no
yes:
  store volatile i32 17, ptr @taken_output, align 4
  ret i32 17
no:
  store volatile i32 29, ptr @fallthrough_output, align 4
  store volatile i32 31, ptr @fallthrough_output, align 4
  ret i32 29
}

define i32 @sge_zero(i32 %a, i32 %b) nounwind {
entry:
  %condition = icmp sge i32 %a, 0
  br i1 %condition, label %yes, label %no
yes:
  store volatile i32 17, ptr @taken_output, align 4
  ret i32 17
no:
  store volatile i32 29, ptr @fallthrough_output, align 4
  store volatile i32 31, ptr @fallthrough_output, align 4
  ret i32 29
}

define i32 @ordinary(i32 %a, i32 %b) nounwind {
entry:
  %condition = icmp eq i32 %a, %b
  br i1 %condition, label %yes, label %no
yes:
  store volatile i32 17, ptr @taken_output, align 4
  ret i32 17
no:
  store volatile i32 29, ptr @fallthrough_output, align 4
  store volatile i32 31, ptr @fallthrough_output, align 4
  ret i32 29
}
