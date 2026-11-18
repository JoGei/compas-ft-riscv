target triple = "riscv32-unknown-elf"
declare i32 @tail_target(i32)
declare i32 @may_throw(i32)
declare i32 @__gxx_personality_v0(...)

define i32 @tail_owner(i32 %value) nounwind {
  %result = musttail call i32 @tail_target(i32 %value)
  ret i32 %result
}

define i32 @exception_owner(i32 %value) personality ptr @__gxx_personality_v0 {
  %result = invoke i32 @may_throw(i32 %value)
      to label %normal unwind label %exception
normal:
  ret i32 %result
exception:
  %landing = landingpad { ptr, i32 } cleanup
  ret i32 -1
}
