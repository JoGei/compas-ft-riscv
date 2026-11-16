/* Independent literal oracle: both directions, wrapping operands, zero,
 * ordinary and ordered stores, and a shared successor after a diamond. */
typedef unsigned int u32;
#define NOINLINE __attribute__((noinline))

volatile u32 port_a;
volatile u32 port_b;
volatile u32 joined;
u32 ordinary;

#define EFFECTS(condition)                 \
  if (condition) {                       \
    port_a = 17;                         \
    return 17;                           \
  }                                      \
  port_b = 29;                           \
  port_b = 31;                           \
  return 29

NOINLINE u32 eq_rr(u32 left, u32 right) {
  EFFECTS((left + 7U) == (right ^ left));
}
NOINLINE u32 ne_rr(u32 left, u32 right) {
  EFFECTS((left + 7U) != (right ^ left));
}
NOINLINE u32 eq_zero(u32 left, u32 right) {
  EFFECTS(((left + 7U) & right) == 0U);
}
NOINLINE u32 ne_zero(u32 left, u32 right) {
  EFFECTS(((left + 7U) & right) != 0U);
}
NOINLINE u32 shared(u32 left, u32 right) {
  u32 value;
  if (left == right) {
    port_a = 41;
    value = left + 3U;
  } else {
    port_b = 43;
    value = right + 5U;
  }
  ordinary = value;
  joined = value;
  return value ^ 19U;
}

int main(void) {
  if (eq_rr(0xffffffffU, 0xfffffff9U) != 17 || port_a != 17)
    return 1;
  if (eq_rr(0xffffffffU, 0xfffffff8U) != 29 || port_b != 31)
    return 2;
  if (ne_rr(0xffffffffU, 0xfffffff9U) != 29 || port_b != 31)
    return 3;
  if (ne_rr(0xffffffffU, 0xfffffff8U) != 17 || port_a != 17)
    return 4;
  if (eq_zero(0xffffffffU, 0xfffffff9U) != 17 || port_a != 17)
    return 5;
  if (eq_zero(0xffffffffU, 6U) != 29 || port_b != 31)
    return 6;
  if (ne_zero(0xffffffffU, 0xfffffff9U) != 29 || port_b != 31)
    return 7;
  if (ne_zero(0xffffffffU, 6U) != 17 || port_a != 17)
    return 8;
  if (shared(7, 7) != 25 || ordinary != 10 || joined != 10 || port_a != 41)
    return 9;
  if (shared(7, 3) != 27 || ordinary != 8 || joined != 8 || port_b != 43)
    return 10;
  return 42;
}
