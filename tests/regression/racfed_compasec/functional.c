/* Literal source oracle, including both outcomes, stores and an ordinary call. */
#define NOINLINE __attribute__((noinline))

unsigned ordinary_word;
volatile unsigned ordered_word;
unsigned char ordinary_byte;
unsigned short ordinary_half;

NOINLINE unsigned balance(unsigned a, unsigned b) { return (a + b) * 3 + 1; }
NOINLINE unsigned direction(unsigned a, unsigned b) {
  if (a == b) {
    ordered_word = 17;
    return a + 3;
  }
  ordered_word = 29;
  return b ^ 7;
}
NOINLINE unsigned stores(unsigned a) {
  ordinary_word = a;
  ordered_word = a + 1;
  ordinary_byte = (unsigned char)a;
  ordinary_half = (unsigned short)(a + 2);
  return a ^ 5;
}
NOINLINE unsigned zero_store(void) { ordinary_word = 0; return 11; }
NOINLINE unsigned ordinary_helper(unsigned a) { return a + 13; }
NOINLINE unsigned call_boundary(unsigned a) {
  unsigned v = ordinary_helper(a);
  ordered_word = v;
  return v + 7;
}

int main(void) {
  if (balance(2, 4) != 19 || balance(0, 3) != 10)
    return 1;
  if (direction(7, 7) != 10 || ordered_word != 17)
    return 2;
  if (direction(7, 3) != 4 || ordered_word != 29)
    return 3;
  if (stores(0x12345) != 0x12340 || ordinary_word != 0x12345 ||
      ordered_word != 0x12346 || ordinary_byte != 0x45 || ordinary_half != 0x2347)
    return 4;
  if (zero_store() != 11 || ordinary_word != 0)
    return 5;
  if (call_boundary(9) != 29 || ordered_word != 22)
    return 6;
  return 42;
}
