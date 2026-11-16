/* Literal fault-free oracle, including wrapping arithmetic and both outcomes.
 * ordinary is intentionally outside every protection selection. */
typedef unsigned int u32;
#define NOINLINE __attribute__((noinline))
volatile u32 output;

#define RESULT(condition) \
  if (condition) { output = 17; return 17; } \
  output = 29; output = 31; return 29

NOINLINE u32 eq_rr(u32 a, u32 b) { RESULT((a + 7U) == (b ^ a)); }
NOINLINE u32 ne_rr(u32 a, u32 b) { RESULT((a + 7U) != (b ^ a)); }
NOINLINE u32 eq_zero(u32 a, u32 b) { RESULT(((a + 7U) & b) == 0); }
NOINLINE u32 ne_zero(u32 a, u32 b) { RESULT(((a + 7U) & b) != 0); }
/* With an orig-only ledger, the entry's address setup stays below the legacy
 * three-instruction threshold. Predicate helpers must not force accounting. */
NOINLINE u32 tiny(u32 a, u32 b) {
  if (a == b) { output = a; return a; }
  output = b; output = b; return b;
}
NOINLINE u32 ordinary(u32 a, u32 b) { RESULT(a == b); }

int main(void) {
  if (eq_rr(0xffffffffU, 0xfffffff9U) != 17 || output != 17) return 1;
  if (eq_rr(0xffffffffU, 0xfffffff8U) != 29 || output != 31) return 2;
  if (ne_rr(0xffffffffU, 0xfffffff9U) != 29 || output != 31) return 3;
  if (ne_rr(0xffffffffU, 0xfffffff8U) != 17 || output != 17) return 4;
  if (eq_zero(0xffffffffU, 0xfffffff9U) != 17 || output != 17) return 5;
  if (eq_zero(0xffffffffU, 6U) != 29 || output != 31) return 6;
  if (ne_zero(0xffffffffU, 0xfffffff9U) != 29 || output != 31) return 7;
  if (ne_zero(0xffffffffU, 6U) != 17 || output != 17) return 8;
  if (ordinary(3, 3) != 17 || output != 17) return 9;
  if (ordinary(3, 4) != 29 || output != 31) return 10;
  if (tiny(3, 3) != 3 || output != 3) return 11;
  if (tiny(3, 4) != 4 || output != 4) return 12;
  return 42;
}
