/* An ordinary ABI helper deliberately destroys both monitor registers. */
.option norvc
.option norelax
.text
.globl ordinary_helper
.type ordinary_helper,@function
ordinary_helper:
  li t0, 123
  li t4, 456
  slli t1, a0, 1
  add a0, a0, t1
  addi a0, a0, 7
  ret
.size ordinary_helper, .-ordinary_helper
