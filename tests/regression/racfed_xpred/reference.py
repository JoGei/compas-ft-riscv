"""Small, independent RV32I/sacc.i interpreter for the linked test fixtures.

This test-local reference admits only instructions needed by these fixtures.
It is not a platform model, timing model, or fault-coverage proof. In particular,
all memory is ordinary byte RAM and no peripheral side effects are modeled.
"""

import struct

MASK = 0xffffffff
REGISTERS = ("zero ra sp gp tp t0 t1 t2 s0 s1 a0 a1 a2 a3 a4 a5 "
             "a6 a7 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 t3 t4 t5 t6").split()
REGNO = {name: index for index, name in enumerate(REGISTERS)}
PRIMARY_TO_SHADOW = {
    "zero": "s11", "ra": "t2", "sp": "s1", "gp": "s6", "tp": "t3",
    "t0": "t4", "t1": "t5", "s0": "t6", "a0": "s8", "a1": "s9",
    "a2": "s10", "a3": "s5", "a4": "s4", "a5": "s7", "a6": "s3",
    "a7": "s2",
}


def signed(value, bits=32):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def elf_memory(path):
    raw = path.read_bytes()
    if raw[:7] != b"\x7fELF\x01\x01\x01":
        raise ValueError("expected little-endian ELF32")
    headers = struct.unpack_from("<HHIIIIIHHHHHH", raw, 16)
    if headers[0:2] != (2, 243):
        raise ValueError("expected a linked RISC-V executable")
    memory = {}
    for index in range(headers[9]):
        kind, offset, virtual, physical, filesz, memsz, _, _ = struct.unpack_from(
            "<8I", raw, headers[4] + index * headers[8])
        if kind != 1:
            continue
        if virtual != physical or filesz > memsz:
            raise ValueError("unsupported load segment")
        memory.update({physical + n: value for n, value in enumerate(raw[offset:offset + filesz])})
        memory.update({physical + n: 0 for n in range(filesz, memsz)})
    return memory


def step(word, pc, registers, memory):
    """Execute one fixed-width instruction; every source reads the old state."""
    if pc & 3 or word & 3 != 3:
        raise ValueError("expected aligned RV32I instruction")
    opcode, rd, rs1, rs2 = word & 127, (word >> 7) & 31, (word >> 15) & 31, (word >> 20) & 31
    funct3, funct7 = (word >> 12) & 7, word >> 25
    left, right = registers[rs1], registers[rs2]
    immediate = signed(word >> 20, 12)
    next_pc, value, store, load, name, written = (pc + 4) & MASK, None, None, None, None, None
    if opcode == 0x33:
        operations = {
            (0, 0): ("add", lambda: left + right), (0, 32): ("sub", lambda: left - right),
            (1, 0): ("sll", lambda: left << (right & 31)),
            (2, 0): ("slt", lambda: int(signed(left) < signed(right))),
            (3, 0): ("sltu", lambda: int(left < right)),
            (4, 0): ("xor", lambda: left ^ right), (5, 0): ("srl", lambda: left >> (right & 31)),
            (5, 32): ("sra", lambda: signed(left) >> (right & 31)),
            (6, 0): ("or", lambda: left | right), (7, 0): ("and", lambda: left & right),
        }
        if (funct3, funct7) in operations:
            name, operation = operations[funct3, funct7]
            value = operation()
    elif opcode == 0x13:
        operations = {
            0: ("addi", lambda: left + immediate), 2: ("slti", lambda: int(signed(left) < immediate)),
            3: ("sltiu", lambda: int(left < (immediate & MASK))),
            4: ("xori", lambda: left ^ immediate), 6: ("ori", lambda: left | immediate),
            7: ("andi", lambda: left & immediate),
        }
        shifts = {(1, 0): ("slli", lambda: left << rs2), (5, 0): ("srli", lambda: left >> rs2),
                  (5, 32): ("srai", lambda: signed(left) >> rs2)}
        operation = operations.get(funct3, shifts.get((funct3, funct7)))
        if operation:
            name, function = operation
            value = function()
    elif opcode == 0x63:
        immediate = signed(((word >> 31) << 12) | (((word >> 7) & 1) << 11)
                           | (((word >> 25) & 63) << 5) | (((word >> 8) & 15) << 1), 13)
        branches = {0: ("beq", left == right), 1: ("bne", left != right),
                    4: ("blt", signed(left) < signed(right)), 5: ("bge", signed(left) >= signed(right)),
                    6: ("bltu", left < right), 7: ("bgeu", left >= right)}
        if funct3 in branches:
            name, taken = branches[funct3]
            if taken:
                next_pc = (pc + immediate) & MASK
    elif opcode == 0x6f:
        immediate = signed(((word >> 31) << 20) | (((word >> 12) & 255) << 12)
                           | (((word >> 20) & 1) << 11) | (((word >> 21) & 1023) << 1), 21)
        name, value, next_pc = "jal", pc + 4, (pc + immediate) & MASK
    elif opcode == 0x67 and funct3 == 0:
        name, value, next_pc = "jalr", pc + 4, (left + immediate) & MASK & ~1
    elif opcode in (0x37, 0x17):
        name = "lui" if opcode == 0x37 else "auipc"
        value = (word & 0xfffff000) + (pc if opcode == 0x17 else 0)
    elif opcode in (0x03, 0x23) and funct3 == 2:
        if opcode == 0x23:
            immediate = signed(((word >> 25) << 5) | ((word >> 7) & 31), 12)
        address = (left + immediate) & MASK
        if address & 3:
            raise ValueError(f"unaligned memory address {address:#x}")
        if opcode == 0x03:
            if any(address + index not in memory for index in range(4)):
                raise ValueError(f"uninitialized word load {address:#x}")
            name, load = "lw", dict(pc=pc, address=address)
            value = sum(memory[address + index] << (8 * index) for index in range(4))
        else:
            name, store = "sw", dict(pc=pc, address=address, value=right, rs1=rs1, rs2=rs2)
            for index in range(4):
                memory[address + index] = (right >> (8 * index)) & 255
    elif opcode == 0x2b and funct3 == 0:
        name = "nz.sacc.i"
        true_literal, false_literal = signed(word >> 20, 6), signed(word >> 26, 6)
        value = registers[rd] + (true_literal if left != 0 else false_literal)
    if name is None:
        raise ValueError(f"unsupported instruction {word:#010x} at {pc:#x}")
    if value is not None and rd:
        registers[rd] = value & MASK
        written = rd
    registers[0] = 0
    return dict(pc=pc, word=word, opcode=name, next_pc=next_pc, written=written,
                rs1=rs1, rs2=rs2, store=store, load=load)


def initial_registers(arguments, balanced):
    """Distinct nonzero inputs expose accidental clobbers of preserved state."""
    registers = [(0x10203040 ^ (index * 0x01010101)) & MASK for index in range(32)]
    registers[0], registers[1], registers[2] = 0, 0x80000, 0x90000
    for index, value in enumerate(arguments):
        registers[10 + index] = value & MASK
    if balanced:
        for primary, shadow in PRIMARY_TO_SHADOW.items():
            registers[REGNO[shadow]] = registers[REGNO[primary]]
    return registers


def execute(original, entry, arguments, balanced):
    memory, registers = original.copy(), initial_registers(arguments, balanced)
    incoming = registers.copy()
    pc, trace, stores, loads = entry, [], [], []
    for count in range(10000):
        if pc == 0x80000:
            return dict(return_value=registers[10], registers=registers, initial_registers=incoming,
                        stores=stores, loads=loads, steps=count, trace=trace)
        if any(pc + index not in memory for index in range(4)):
            raise ValueError(f"execution escaped linked code at {pc:#x}")
        word = sum(memory[pc + index] << (8 * index) for index in range(4))
        event = step(word, pc, registers, memory)
        if event["store"]:
            stores.append(event["store"])
        if event["load"]:
            loads.append(event["load"])
        trace.append(event)
        pc = event["next_pc"]
    raise ValueError("fault-free execution exceeded 10000 instructions")
