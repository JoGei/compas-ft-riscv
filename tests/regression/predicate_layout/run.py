#!/usr/bin/env python3
"""Verify post-instrumentation branch relaxation for large RV32 functions."""

import argparse
import json
from pathlib import Path
import re
import resource
import struct
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import runner


FUNCTION = "long_predicate"


def fixture(opcode, backward, count=384):
    # The original function is small enough for a B-type transfer; DMR and
    # RACFED expand the 384 arithmetic instructions past its signed-13-bit
    # span. This exercises relaxation after, rather than before, COMPAS.
    filler = "\n".join("    $x10 = ADDI $x10, 1" for _ in range(count))
    preamble = f'''--- |
  target triple = "riscv32-unknown-elf"
  define i32 @{FUNCTION}(i32 %value, i32 %other) {{ ret i32 %value }}
...
---
name: {FUNCTION}
tracksRegLiveness: true
body: |
'''
    if backward:
        body = f'''  bb.0:
    successors: %bb.2
    liveins: $x1, $x10, $x11
    PseudoBR %bb.2

  bb.1:
    liveins: $x1
    $x10 = ADDI $x0, 7
    PseudoRET implicit $x1, implicit $x10

  bb.2:
    successors: %bb.1, %bb.3
    liveins: $x1, $x10, $x11
{filler}
    {opcode} $x10, $x11, %bb.1

  bb.3:
    liveins: $x1, $x10
    PseudoRET implicit $x1, implicit $x10
'''
    else:
        body = f'''  bb.0:
    successors: %bb.1, %bb.2
    liveins: $x1, $x10, $x11
    {opcode} $x10, $x11, %bb.2

  bb.1:
    liveins: $x1, $x10
{filler}
    PseudoRET implicit $x1, implicit $x10

  bb.2:
    liveins: $x1
    $x10 = ADDI $x0, 7
    PseudoRET implicit $x1, implicit $x10
'''
    return preamble + body + "...\n"


def executable_bytes(path):
    data = path.read_bytes()
    header = struct.unpack_from("<16sHHIIIIIHHHHHH", data)
    sections = [struct.unpack_from("<10I", data, header[6] + i * header[11])
                for i in range(header[12])]
    return b"".join(data[s[4]:s[4] + s[5]] for s in sections if s[2] & 4)


def check_object_layout(path):
    code = executable_bytes(path)
    for offset in range(0, len(code), 4):
        word = struct.unpack_from("<I", code, offset)[0]
        if word & 127 != 0x63:
            continue
        delta = ((word >> 31) << 12) | (((word >> 7) & 1) << 11)
        delta |= (((word >> 25) & 63) << 5) | (((word >> 8) & 15) << 1)
        if delta & 4096:
            delta -= 8192
        if delta > 4088:
            raise runner.TestError("conditional lacks GNU forward relaxation margin")


def gas_source(assembly):
    lines = []
    for line in assembly.splitlines(keepends=True):
        line = re.sub(r"_xpred[0-9]+p[0-9]+", "", line)
        match = re.match(r"\s*(nz|ez)\.sacc\.i\s+t0,\s*t4,\s*(-?\d+),\s*(-?\d+)\s*$", line)
        if match:
            first, second = int(match[2]), int(match[3])
            if not (-32 <= first <= 31 and -32 <= second <= 31):
                raise runner.TestError("invalid SACC immediate")
            if match[1] == "ez":
                first, second = second, first
            word = 0x2b | (5 << 7) | (29 << 15) | ((first & 63) << 20) | ((second & 63) << 26)
            line = f"\t.insn 4, 0x{word:08x}\n"
        elif ".sacc.i" in line:
            raise runner.TestError("unsupported SACC assembly operands")
        lines.append(line)
    return "".join(lines)


def check_gnu(args, assembly, directory):
    source = directory / "gas.s"
    source.write_text(gas_source(assembly.read_text()))
    runner.command([args.cc.resolve(), "-march=rv32im", "-mabi=ilp32", "-mno-relax",
                    "-c", source, "-o", directory / "gas.o"], directory,
                   "gas", args.build_timeout)
    if executable_bytes(directory / "gas.o") != executable_bytes(directory / "output.o"):
        raise runner.TestError("GNU assembly changed the final instruction layout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llc", type=Path, required=True)
    parser.add_argument("--llvm-mc", type=Path, required=True)
    parser.add_argument("--cc", type=Path, required=True, help="GNU RISC-V bare-metal compiler")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--build-timeout", type=float, default=60)
    args = parser.parse_args()
    if args.build_timeout <= 0:
        parser.error("--build-timeout must be positive")
    output = args.build_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    results = []
    for backward in (False, True):
        for opcode in ("BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"):
            for profile in ("naive", "soft", "hard"):
                name = f"{'backward' if backward else 'forward'}-{opcode}-{profile}"
                directory = output / name
                directory.mkdir(parents=True, exist_ok=True)
                source = directory / "input.mir"
                source.write_text(fixture(opcode, backward))
                flags = [f"-NZDC+NEMESEC={FUNCTION}",
                         f"-RACFED={FUNCTION}:orig+dmr",
                         f"-RACFED-compasec={FUNCTION}"]
                if profile == "naive":
                    flags.append(f"-RACFED-naive-branches={FUNCTION}")
                else:
                    flags.append(f"-RACFED-predicate-shadow={FUNCTION}")
                if profile == "hard":
                    flags.append(f"-RACFED-predicate-xpred={FUNCTION}")
                result = dict(case=name, status="PASS",
                              scope="machine verification and native object encoding; no target execution")
                try:
                    assembly = directory / "output.s"
                    runner.command([
                        args.llc.resolve(), "-mtriple=riscv32-unknown-elf",
                        "-mattr=+m,+xpred,-c,-relax", "-target-abi=ilp32",
                        "-O1", "-verify-machineinstrs",
                        "-start-after=livedebugvalues", *flags,
                        source, "-o", assembly], directory, "compile",
                        args.build_timeout)
                    log = (directory / "compile.stdout").read_text()
                    match = re.search(r"COMPAS_PREDICATE_LAYOUT function=long_predicate "
                                      r"relaxed=(\d+) direct_jumps=(\d+)", log)
                    if not match or int(match[1]) == 0:
                        raise runner.TestError("fixture did not exercise late branch relaxation")
                    result.update(relaxed=int(match[1]), direct_jumps=int(match[2]))
                    runner.command([
                        args.llvm_mc.resolve(), "-triple=riscv32-unknown-elf",
                        "-mattr=+m,+xpred,-c,-relax", "-filetype=obj",
                        assembly, "-o", directory / "output.o"], directory,
                        "assemble", args.build_timeout)
                    check_object_layout(directory / "output.o")
                    check_gnu(args, assembly, directory)
                except (OSError, runner.TestError) as error:
                    result.update(status="ERROR", detail=str(error))
                results.append(result)
                print(name, result["status"], flush=True)
    # 1010 original ADDIs place one post-RASM forward branch at exactly
    # 4092 bytes without the explicit GNU relaxation margin. This boundary
    # case must be expanded by COMPAS before either assembler sees it.
    directory = output / "forward-boundary"
    directory.mkdir(exist_ok=True)
    source, assembly = directory / "input.mir", directory / "output.s"
    source.write_text(fixture("BEQ", False, 1010))
    result = dict(case="forward-boundary", status="PASS")
    try:
        runner.command([args.llc.resolve(), "-mtriple=riscv32-unknown-elf",
                        "-mattr=+m,-c,-relax", "-target-abi=ilp32", "-O1",
                        "-verify-machineinstrs", "-start-after=livedebugvalues",
                        f"-RASM={FUNCTION}", f"-RASM-predicate={FUNCTION}",
                        f"-RASM-predicate-unpadded={FUNCTION}", source, "-o", assembly],
                       directory, "compile", args.build_timeout)
        runner.command([args.llvm_mc.resolve(), "-triple=riscv32-unknown-elf",
                        "-mattr=+m,-c,-relax", "-filetype=obj", assembly,
                        "-o", directory / "output.o"], directory, "assemble", args.build_timeout)
        check_object_layout(directory / "output.o")
        check_gnu(args, assembly, directory)
    except (OSError, runner.TestError) as error:
        result.update(status="ERROR", detail=str(error))
    results.append(result)
    print(result["case"], result["status"], flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any(result["status"] != "PASS" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
