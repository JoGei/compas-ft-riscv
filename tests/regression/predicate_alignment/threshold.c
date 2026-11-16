/* The branch's input operands and memory addresses arrive in argument
 * registers. Thus exactly N volatile loads are original entry computations. */
typedef unsigned int u32;
volatile u32 source_value = 73;
volatile u32 destination;

#define BRANCH \
  if (a == b) { *port = a; return a; } \
  *port = b; *port = b; return b
#define FN(name, loads) \
  __attribute__((noinline)) u32 name(u32 a, u32 b, volatile u32 *src, \
                                     volatile u32 *port) { loads BRANCH; }

FN(threshold0, )
FN(threshold1, (void)*src;)
FN(threshold2, (void)*src; (void)*src;)
FN(threshold3, (void)*src; (void)*src; (void)*src;)

#define CHECK(name) \
  if (name(3, 3, &source_value, &destination) != 3 || destination != 3) return 1; \
  if (name(3, 4, &source_value, &destination) != 4 || destination != 4) return 2

int main(void) {
  CHECK(threshold0);
  CHECK(threshold1);
  CHECK(threshold2);
  CHECK(threshold3);
  return source_value == 73 ? 42 : 3;
}
