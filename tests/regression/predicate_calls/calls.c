/* Returning-call coverage with observable branches and independent literal
 * expectations in run.py. All callback targets share the protected ABI. */
typedef unsigned int u32;
#define NOINLINE __attribute__((noinline))

volatile u32 event_a, event_b, event_c;
extern u32 ordinary_helper(u32);

NOINLINE u32 protected_leaf(u32 a, u32 b) {
  if (a == b) {
    event_a = 17;
    return a + 11;
  }
  event_b = 29;
  return a ^ b ^ 0x55U;
}

NOINLINE u32 callback_leaf(u32 a, u32 b) {
  if (a != b) {
    event_a = 47;
    return a + b + 9;
  }
  event_b = 53;
  return a - b + 13;
}

u32 (*volatile callback)(u32, u32) = protected_leaf;

NOINLINE u32 direct_owner(u32 a, u32 b) {
  volatile u32 local = a + 5;
  u32 first = protected_leaf(a, b);
  if (first == a + 11)
    event_c = 31;
  else
    event_c = 37;
  return first + protected_leaf(b, a) + local;
}

NOINLINE u32 ordinary_owner(u32 a, u32 b) {
  volatile u32 local = a ^ 0x123U;
  u32 value = ordinary_helper(a);
  if (a == b)
    event_c = 41;
  else
    event_c = 43;
  return value + local + b;
}

NOINLINE u32 callback_owner(u32 a, u32 b) {
  volatile u32 local = a + b;
  u32 result = callback(a, b);
  if (a == b)
    event_c = 89;
  else
    event_c = 97;
  return result + local;
}

NOINLINE u32 ten_arg_leaf(u32 a0, u32 a1, u32 a2, u32 a3, u32 a4,
                         u32 a5, u32 a6, u32 a7, u32 a8, u32 a9) {
  u32 total = a0 + 2*a1 + 3*a2 + 4*a3 + 5*a4 + 6*a5 + 7*a6 + 8*a7
              + 9*a8 + 10*a9;
  if (a0 == a9)
    event_a = 59;
  else
    event_b = 61;
  return total;
}

NOINLINE u32 stack_owner(u32 value, u32 last) {
  volatile u32 local = value ^ 0x145U;
  u32 result = ten_arg_leaf(value, value + 1, value + 2, value + 3,
                           value + 4, value + 5, value + 6, value + 7,
                           value + 8, last);
  if (value == last)
    event_c = 101;
  else
    event_c = 103;
  return result + local;
}

NOINLINE u32 large_owner(u32 value, u32 other) {
  volatile u32 local[1100];
  u32 index = value & 1U;
  local[index] = value + 7;
  local[1098 + index] = value ^ 0x33U;
  u32 result = protected_leaf(value, other);
  if (result == value + 11)
    event_c = 67;
  else
    event_c = 71;
  return result + local[index] + local[1098 + index];
}

NOINLINE u32 recursive_owner(u32 value, u32 count) {
  if (count == 0) {
    event_c = 73;
    return value + 3;
  }
  u32 result = recursive_owner(value + 2, count - 1);
  if (count == 1)
    event_c = 79;
  else
    event_c = 83;
  return result + count;
}
