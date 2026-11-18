#!/usr/bin/env python3
"""Verify returning-call ABI and signature ordering for matched F/N/G/H.

The default run compiles, verifies and links without running target code.
--execute-reference additionally uses the existing test-local RV32 interpreter.
No platform simulator or fault injection is invoked.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import resource
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FUNCTIONS = ("protected_leaf", "callback_leaf", "direct_owner", "ordinary_owner",
             "callback_owner", "ten_arg_leaf", "stack_owner", "large_owner",
             "recursive_owner")
OWNERS = ("direct_owner", "ordinary_owner", "callback_owner", "stack_owner",
          "large_owner", "recursive_owner")
PROFILES = ("F_compasec", "N_naive", "G_compasec", "H_compasec")
CASES = (
    ("direct_owner", (7, 7), 48, (("event_a", 17), ("event_c", 31), ("event_a", 17)), None),
    ("direct_owner", (7, 3), 174, (("event_b", 29), ("event_c", 37), ("event_b", 29)), None),
    ("ordinary_owner", (7, 7), 327, (("event_c", 41),), None),
    ("ordinary_owner", (7, 3), 323, (("event_c", 43),), None),
    ("callback_owner", (7, 7), 32, (("event_a", 17), ("event_c", 89)), "protected_leaf"),
    ("callback_owner", (7, 3), 91, (("event_b", 29), ("event_c", 97)), "protected_leaf"),
    ("callback_owner", (7, 7), 27, (("event_b", 53), ("event_c", 89)), "callback_leaf"),
    ("callback_owner", (7, 3), 29, (("event_a", 47), ("event_c", 97)), "callback_leaf"),
    ("stack_owner", (2, 2), 677, (("event_a", 59), ("event_c", 101)), None),
    ("stack_owner", (2, 11), 767, (("event_b", 61), ("event_c", 103)), None),
    ("large_owner", (7, 7), 84, (("event_a", 17), ("event_c", 67)), None),
    ("large_owner", (7, 3), 147, (("event_b", 29), ("event_c", 71)), None),
    ("recursive_owner", (5, 0), 8, (("event_c", 73),), None),
    ("recursive_owner", (5, 3), 20, (("event_c", 73), ("event_c", 79),
                                   ("event_c", 83), ("event_c", 83)), None),
)


def flags(profile, names=FUNCTIONS):
    selection = ",".join(names)
    result = ["-NZDC+NEMESEC=" + selection, "-RACFED=" + selection,
              "-RACFED-compasec=" + selection]
    if profile == "N_naive":
        result += ["-RACFED-naive-branches=" + selection]
    if profile in ("G_compasec", "H_compasec"):
        result += ["-RACFED-predicate-shadow=" + selection]
    if profile == "H_compasec":
        result += ["-RACFED-predicate-xpred=" + selection]
    return result


def records(text, tag):
    return [dict(item.split("=", 1) for item in line.split()[1:])
            for line in text.splitlines() if line.startswith(tag + " ")]


def documents(mir):
    return {match[1]: document for document in mir.split("---\n")
            if (match := re.search(r"^name: +([\w.]+)$", document, re.M))}


def instructions(document):
    return [line.strip() for line in document.split("body:", 1)[1].splitlines()
            if line.startswith("    ") and line.strip()
            and not line.strip().startswith(("liveins:", "successors:", "DBG_", "CFI_"))]


def signature_access(line, restore):
    pattern = (r"(?:\$x5 = )LW \$(x\d+), (-?\d+)" if restore
               else r"SW \$x5, \$(x\d+), (-?\d+)")
    match = re.search(pattern, line)
    slot = re.search(r"%fixed-stack\.(\d+)", line)
    return (*match.groups(), slot[1]) if match and slot else None


def check_call_frames(before, after, require, label):
    """Check the normal-return ordering without assuming callee register values."""
    original, hardened = documents(before), documents(after)
    report = {}
    for name in OWNERS:
        old = [line for line in instructions(original[name])
               if re.search(r"\bPseudoCALL(?:Indirect)?\b", line)]
        body = instructions(hardened[name])
        calls = [index for index, line in enumerate(body)
                 if re.search(r"\bPseudoCALL(?:Indirect)?\b", line)]
        require(bool(old) and len(calls) == len(old), label + "/" + name + ": returning calls retained")
        require(not any(re.search(r"\$(?:x5|x29)\b", line) for line in old),
                label + "/" + name + ": original call operands do not alias reserved state")
        if name == "callback_owner":
            require(all("PseudoCALLIndirect" in body[index] for index in calls),
                    label + ": volatile callback remains an indirect call")
        accesses = []
        for index in calls:
            saved = signature_access(body[index - 1], False) if index else None
            require(saved is not None, label + "/" + name + ": RTS saved immediately before call")
            # A distant fixed slot needs up to LUI/ADDI/ADD address helpers.
            following = body[index + 1:index + 6]
            restored = next(((n, signature_access(line, True)) for n, line in enumerate(following)
                             if signature_access(line, True)), None)
            require(restored is not None and restored[1] == saved,
                    label + "/" + name + ": RTS reload uses the same caller frame slot")
            for line in following[:restored[0]]:
                require(not re.search(r"\$(?:x2|x5) =", line)
                        and not re.search(r"\b(?:SW|PseudoCALL|BNE|BEQ)\b", line),
                        label + "/" + name + ": restoration precedes signature updates and checks")
            update = following[restored[0] + 1] if restored[0] + 1 < len(following) else ""
            require(bool(re.search(r"\$x5 = ADDI \$x5, -?\d+", update)),
                    label + "/" + name + ": call ledger update follows RTS restoration")
            require(saved[0] != "x2", label + "/" + name + ": preservation does not borrow outgoing SP slots")
            accesses.append(dict(base=saved[0], offset=int(saved[1]), fixed_slot=int(saved[2])))
        report[name] = accesses
    require("$x8" in hardened["large_owner"] and
            int(re.search(r"stackSize: +(\d+)", hardened["large_owner"])[1]) > 4096,
            label + ": large local frame survives optimization")
    require(any("PseudoCALLIndirect" in line and "CustomRegMask" in line
                and "implicit-def $x24" in line for line in instructions(hardened["callback_owner"])),
            label + ": indirect callback declares shadow result and its register mask")
    return report


def reserve_distant_slot(mir):
    """Keep a target-owned 4KB fixed object above the signature slot at PEI."""
    pieces, changed = mir.split("---\n"), 0
    for index, document in enumerate(pieces):
        if not re.search(r"^name: +large_owner$", document, re.M):
            continue
        if "fixedStack:      []" not in document:
            raise ValueError("large_owner already has fixed objects before PEI")
        pieces[index] = document.replace("fixedStack:      []",
            "fixedStack:\n  - { id: 0, type: default, offset: -4096, size: 4096, "
            "alignment: 16, isImmutable: false, isAliased: false }")
        changed += 1
    if changed != 1:
        raise ValueError("large_owner missing from pre-PEI MIR")
    return "---\n".join(pieces)


def execute_reference(elf, symbols, require, label, cases=CASES):
    spec = importlib.util.spec_from_file_location("predicate_call_reference", HERE.parent / "racfed_xpred/reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    ordinary_step = reference.step

    def step(word, pc, registers, memory):
        # RV32IM can select low-word MUL for the fixed argument weights. Keep
        # this test-local addition explicit; other M instructions still reject.
        if word & 0xfe00707f == 0x02000033:
            rd, rs1, rs2 = (word >> 7) & 31, (word >> 15) & 31, (word >> 20) & 31
            value = registers[rs1] * registers[rs2] & reference.MASK
            if rd:
                registers[rd] = value
            registers[0] = 0
            return dict(pc=pc, word=word, opcode="mul", next_pc=pc + 4,
                        written=rd if rd else None, rs1=rs1, rs2=rs2, store=None, load=None)
        return ordinary_step(word, pc, registers, memory)

    reference.step = step
    image = reference.elf_memory(elf)
    outputs = {symbols[name]: name for name in ("event_a", "event_b", "event_c")}
    results = []
    for name, arguments, expected, events, callback in cases:
        memory = image.copy()
        if callback:
            value = symbols[callback]
            for index in range(4):
                memory[symbols["callback"] + index] = (value >> (8*index)) & 255
        observed = reference.execute(memory, symbols[name], arguments, True)
        actual = [(outputs[row["address"]], row["value"]) for row in observed["stores"]
                  if row["address"] in outputs]
        require(observed["return_value"] == expected and actual == list(events),
                label + "/" + name + str(arguments) + ": literal result and ordered stores")
        for register in (1, 2, 3, 4, 8):
            require(observed["registers"][register] == observed["initial_registers"][register],
                    label + "/" + name + ": caller RA/SP/GP/TP/FP preserved")
        results.append(dict(function=name, arguments=arguments, callback=callback,
                            result=observed["return_value"], steps=observed["steps"], stores=actual))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-bin", required=True, type=Path)
    parser.add_argument("--cc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--execute-reference", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    checks, profiles = [], {}
    result = dict(status="running", scope="compiler, machine-verifier and linked-ELF checks",
                  reference_execution_requested=args.execute_reference, profiles=profiles, checks=checks)

    def require(condition, description):
        if not condition:
            raise AssertionError(description)
        checks.append(description)

    def run(command, name, expect=0):
        argv = list(map(str, command))
        (out / (name + ".command.json")).write_text(json.dumps(argv, indent=2) + "\n")
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        (out / (name + ".stdout")).write_text(completed.stdout)
        (out / (name + ".stderr")).write_text(completed.stderr)
        if expect is not None and completed.returncode != expect:
            raise AssertionError(f"{name}: exit {completed.returncode}; see {out / (name + '.stderr')}")
        return completed

    try:
        llvm = args.llvm_bin.resolve()
        backend = [llvm / "llc", "-march=riscv32", "-mattr=+m,+xpred,-c,-relax", "-target-abi=ilp32",
                   "-code-model=small", "-O1", "-verify-machineinstrs"]
        pipeline = run([*backend, "-debug-pass=Arguments", "/dev/null", "-o", "/dev/null"], "pipeline")
        arguments = next(line for line in pipeline.stderr.splitlines() if line.startswith("Pass Arguments:"))
        count = arguments.split().count("-machine-opt-remark-emitter")
        require(count > 0, "final machine-IR checkpoint exists")
        checkpoint = "machine-opt-remark-emitter," + str(count - 1)
        ir = out / "calls.ll"
        run([llvm / "clang", "--target=riscv32-unknown-elf", "-march=rv32im_xpred", "-mabi=ilp32",
             "-mno-relax", "-mcmodel=medlow", "-O1", "-fno-inline-functions", "-ffreestanding",
             "-fno-optimize-sibling-calls", "-S", "-emit-llvm", HERE / "calls.c", "-o", ir], "frontend")
        helper = out / "ordinary.o"
        run([args.cc, "-march=rv32im", "-mabi=ilp32", "-mno-relax", "-c", HERE / "ordinary.S",
             "-o", helper], "ordinary-helper")
        for profile in args.profiles:
            options = flags(profile)
            before, after = (out / (profile + suffix + ".mir") for suffix in ("-before", "-after"))
            run([*backend, *options, "-stop-after=livedebugvalues", ir, "-o", before], profile + "-before")
            compiled = run([*backend, *options, "-stop-after=" + checkpoint, ir, "-o", after], profile + "-after")
            evidence = check_call_frames(before.read_text(), after.read_text(), require, profile)
            sites = records(compiled.stdout, "COMPAS_RACFED_PREDICATE_SITE")
            require((bool(sites) == (profile in ("G_compasec", "H_compasec"))),
                    profile + ": predicate policy matches selection")
            if sites:
                require({row["function"] for row in sites} == set(FUNCTIONS),
                        profile + ": every selected function retains an application predicate")
            if profile == "H_compasec":
                require(all(row.get("lowering") == "sacc" and
                            -32 <= int(row["true"]) <= 31 and -32 <= int(row["false"]) <= 31 and
                            int(row["true"]) * int(row["false"]) != 0 and row["true"] != row["false"]
                            for row in sites), profile + ": every site uses mandatory nonzero signed-six-bit hardware updates")
            obj, elf = out / (profile + ".o"), out / (profile + ".elf")
            run([*backend, *options, "-filetype=obj", ir, "-o", obj], profile + "-object")
            run([args.cc, "-march=rv32im", "-mabi=ilp32", "-mno-relax", "-nostdlib", "-static", "-no-pie",
                 "-Wl,--no-relax,--build-id=none,-Ttext=0x10000,-e,direct_owner", obj, helper, "-o", elf], profile + "-link")
            dump = run([llvm / "llvm-objdump", "-d", "--mattr=+xpred,-c", elf], profile + "-disassembly")
            require(dump.stdout.count("nz.sacc.i") == (len(sites) if profile == "H_compasec" else 0),
                    profile + ": linked hardware instruction count matches compiler sites")
            table = run([llvm / "llvm-objdump", "-t", elf], profile + "-symbols")
            symbols = {row.split()[-1]: int(row.split()[0], 16) for row in table.stdout.splitlines()
                       if re.match(r"^[0-9a-f]{8}\s", row)}
            profiles[profile] = dict(call_frames=evidence, predicate_sites=sites, elf=str(elf))
            if args.execute_reference:
                profiles[profile]["executions"] = execute_reference(elf, symbols, require, profile)
            # A large ordinary local frame alone keeps its signature near FP.
            # An existing fixed object forces the scratch-address path too.
            early = out / (profile + "-early.mir")
            padded = out / (profile + "-fixed-area.mir")
            distant_before, distant_after = (out / (profile + "-distant-" + suffix + ".mir")
                                              for suffix in ("before", "after"))
            run([*backend, *options, "-stop-before=prologepilog", ir, "-o", early], profile + "-early")
            padded.write_text(reserve_distant_slot(early.read_text()))
            resume = [*backend, *options, "-start-before=prologepilog"]
            run([*resume, "-stop-after=livedebugvalues", padded, "-o", distant_before], profile + "-distant-before")
            run([*resume, "-stop-after=" + checkpoint, padded, "-o", distant_after], profile + "-distant-after")
            distant = check_call_frames(distant_before.read_text(), distant_after.read_text(), require,
                                        profile + "-distant")
            require(all(access["base"] == "x29" and access["offset"] == 0 for access in distant["large_owner"]),
                    profile + ": distant signature slot uses the reserved scratch address")
            body = instructions(documents(distant_after.read_text())["large_owner"])
            require(sum(bool(re.search(r"\$x29 = ADD \$x8, \$x29", line)) for line in body) >= 2,
                    profile + ": distant signature address is recomputed before save and restore")
            profiles[profile]["distant_call_frames"] = distant["large_owner"]
            if args.execute_reference:
                distant_obj, distant_elf = out / (profile + "-distant.o"), out / (profile + "-distant.elf")
                run([*resume, "-filetype=obj", padded, "-o", distant_obj], profile + "-distant-object")
                run([args.cc, "-march=rv32im", "-mabi=ilp32", "-mno-relax", "-nostdlib", "-static", "-no-pie",
                     "-Wl,--no-relax,--build-id=none,-Ttext=0x10000,-e,large_owner", distant_obj, helper,
                     "-o", distant_elf], profile + "-distant-link")
                distant_table = run([llvm / "llvm-objdump", "-t", distant_elf], profile + "-distant-symbols")
                distant_symbols = {row.split()[-1]: int(row.split()[0], 16) for row in distant_table.stdout.splitlines()
                                   if re.match(r"^[0-9a-f]{8}\s", row)}
                profiles[profile]["distant_executions"] = execute_reference(distant_elf, distant_symbols, require,
                    profile + "-distant", [case for case in CASES if case[0] == "large_owner"])
            if profile != "F_compasec":
                for name, selection, expected in (
                        ("tail_owner", ("tail_owner", "tail_target"), "tail"),
                        ("exception_owner", ("exception_owner",), "exception|unwind|EH")):
                    rejected = run([*backend, *flags(profile, selection), HERE / "unsupported.ll",
                                    "-o", out / (profile + "-" + name + ".s")], profile + "-" + name, expect=None)
                    require(rejected.returncode != 0 and re.search(expected, rejected.stderr, re.I),
                            profile + ": " + name + " is rejected explicitly")
        result["status"] = "passed"
    except (AssertionError, ValueError, OSError, StopIteration, subprocess.TimeoutExpired) as error:
        result.update(status="failed", error=str(error))
        print(str(error), file=sys.stderr)
    finally:
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], checks=len(checks), profiles=list(profiles),
                          reference_execution=args.execute_reference)))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
