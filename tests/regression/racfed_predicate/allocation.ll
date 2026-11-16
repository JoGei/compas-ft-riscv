; Clone the function with distinct names to exercise deterministic signature
; draws across functions. Before nonzero-edge enforcement, clone 168 had a
; zero taken adjustment in the CompaSeC RACFED software profile.
target triple = "riscv32-unknown-unknown"
@port_a = global i32 0, align 4
@port_b = global i32 0, align 4

define dso_local i32 @allocation_case(i32 noundef %left, i32 noundef %right) local_unnamed_addr noinline nounwind {
entry:
  %add = add i32 %left, 7
  %xor = xor i32 %right, %left
  %cmp = icmp eq i32 %add, %xor
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store volatile i32 17, ptr @port_a, align 4
  br label %return

if.end:
  store volatile i32 29, ptr @port_b, align 4
  store volatile i32 31, ptr @port_b, align 4
  br label %return

return:
  %retval.0 = phi i32 [ 17, %if.then ], [ 29, %if.end ]
  ret i32 %retval.0
}
