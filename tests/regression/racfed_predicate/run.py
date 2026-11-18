#!/usr/bin/env python3
"""Validate RACFED direction-check, naive and shadow-predicate compositions."""

import argparse
import importlib.util
import json
from pathlib import Path
import re
import resource
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "layering", HERE.parent / "rsm_layering/check_assembly.py")
layering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layering)
FUNCTIONS = ("eq_rr", "ne_rr", "eq_zero", "ne_zero", "shared")
SELECTED = ",".join(FUNCTIONS)
PRIMARY = set(layering.PRIMARY_TO_SHADOW) - {"t0", "ra"}


def records(output, tag):
    return [dict(item.split("=", 1) for item in line.split()[1:])
            for line in output.splitlines() if line.startswith(tag + " ")]


def allocation_fixture(count):
    """Keep each CFG small while crossing deterministic zero-edge draws."""
    preamble, function = (HERE / "allocation.ll").read_text().split("define ", 1)
    return preamble + "\n".join("define " + function.replace(
        "@allocation_case(", "@allocation_" + str(index) + "(")
        for index in range(count))


def linked_functions(disassembly):
    """Read real (no-aliases) instructions from the final linked ELF."""
    functions = {}
    current = None
    for line in disassembly.splitlines():
        symbol = re.match(r"^[0-9a-f]+ <([^>]+)>:$", line)
        if symbol:
            current = functions.setdefault(symbol[1], [])
            continue
        instruction = re.match(r"^\s*([0-9a-f]+):\s+(\S+)(?:\s+(.*))?$", line)
        if current is not None and instruction:
            operands = (instruction[3] or "").split("#", 1)[0]
            current.append((instruction[2],
                            [item.strip() for item in operands.split(",")],
                            int(instruction[1], 16)))
    return functions


def target(instruction):
    return int(instruction[1][-1].split()[0], 16)


def signature_delta(instruction):
    opcode, operands, _ = instruction
    if opcode == "addi" and operands[:2] == ["t0", "t0"]:
        return int(operands[2], 0)
    return None


def skip_pair(image, destination, address):
    """Suppress the linked ADD and branch by replacing both with RV32 NOPs.

    Their addresses and all other instructions remain unchanged. The fixture
    exits at the first affected invocation, so the replacement models one
    complete two-instruction dynamic skip window in that invocation.
    """
    raw = bytearray(image.read_bytes())
    if raw[:7] != b"\x7fELF\x01\x01\x01":
        raise AssertionError("skip fixture requires little-endian ELF32")
    header = struct.unpack_from("<HHIIIIIHHHHHH", raw, 16)
    if header[:2] != (2, 243):
        raise AssertionError("skip fixture requires a linked RISC-V executable")
    for index in range(header[9]):
        kind, offset, virtual, _, filesz, _, _, _ = struct.unpack_from(
            "<8I", raw, header[4] + index * header[8])
        if kind == 1 and virtual <= address and address + 8 <= virtual + filesz:
            struct.pack_into("<II", raw, offset + address - virtual,
                             0x00000013, 0x00000013)
            destination.write_bytes(raw)
            destination.chmod(image.stat().st_mode & 0o777)
            return
    raise AssertionError("skip pair is outside file-backed ELF load segments")


def error_guards(body):
    """Count retained unconditional transfers to the DMR error handler."""
    handler = None
    # Use the actual -512 error exit instead of depending on block order.
    for i in range(1, len(body) - 1):
        if (body[i][0] == "addi" and body[i][1] == ["a0", "zero", "-512"]
                and body[i + 1][0] == "ecall"):
            handler = body[i - 1][2]
            break
    if handler is None:
        raise AssertionError("missing DMR error handler")
    return sum(item[0] == "jal" and item[1][0] == "zero"
               and target(item) == handler for item in body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for tool in ("llc", "clang", "objdump", "cc", "qemu"):
        parser.add_argument("--" + tool, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    checks = []

    def require(condition, description):
        if not condition:
            raise AssertionError(description)
        checks.append(description)

    def run(command, name, expect=0):
        argv = list(map(str, command))
        result = subprocess.run(argv, capture_output=True, text=True, timeout=90)
        (out / (name + ".command.json")).write_text(json.dumps(argv, indent=2) + "\n")
        (out / (name + ".stdout")).write_text(result.stdout)
        (out / (name + ".stderr")).write_text(result.stderr)
        if expect is not None and result.returncode != expect:
            raise AssertionError(f"{name}: exit {result.returncode}, expected {expect}; see {out / (name + '.stderr')}")
        return result

    backend = [args.llc, "-march=riscv32", "-mattr=-c,-relax",
               "-target-abi=ilp32", "-O1", "-verify-machineinstrs"]

    def flags(selection=SELECTED, predicate=True, naive=False, compasec=True):
        result = ["-NZDC+NEMESEC=" + selection, "-RACFED=" + selection]
        if compasec:
            result.append("-RACFED-compasec=" + selection)
        if predicate:
            result.append("-RACFED-predicate-shadow=" + selection)
        if naive:
            result.append("-RACFED-naive-branches=" + selection)
        return result

    try:
        ir = out / "functional.ll"
        run([args.clang, "--target=riscv32", "-march=rv32i", "-mabi=ilp32",
             "-mno-relax", "-O1", "-ffreestanding", "-fno-optimize-sibling-calls",
             "-emit-llvm", "-S", HERE / "functional.c", "-o", ir], "frontend")
        compiled = {}
        linked = {}
        for mode in ("F", "N", "G"):
            selected = flags(predicate=mode == "G", naive=mode == "N")
            obj = out / (mode + ".o")
            compiled[mode] = run([*backend, *selected, "-filetype=obj", ir, "-o", obj], mode + "-codegen")
            require("Bad machine code" not in compiled[mode].stderr,
                    mode + ": completed machine-verifier checks")
            elf = out / (mode + ".elf")
            run([args.cc, "-march=rv32i", "-mabi=ilp32", "-mno-relax", "-nostdlib",
                 "-static", "-no-pie", "-Wl,--build-id=none", "-Wl,--no-relax",
                 "-Wl,-Ttext=0x10000", HERE.parents[1] / "support/start.S", obj,
                 "-o", elf], mode + "-link")
            dump = run([args.objdump, "-d", "--no-show-raw-insn", "-M", "no-aliases", elf], mode + "-disassembly")
            linked[mode] = linked_functions(dump.stdout)
            execution = run([args.qemu, "-strace", elf], mode + "-execute", expect=42)
            require(re.findall(r"\bexit(?:_group)?\((-?\d+)\)", execution.stderr) == ["42"]
                    and execution.stdout == "",
                    mode + ": literal source oracle for both outcomes, wraparound, zero, shared successor, ordinary/ordered stores")

        updates = records(compiled["G"].stdout, "COMPAS_RACFED_UI")
        regions = {(row["function"], row["block"]): row
                   for row in records(compiled["G"].stdout, "COMPAS_RACFED_REGION")}
        sites = records(compiled["G"].stdout, "COMPAS_RACFED_PREDICATE_SITE")
        require({row["category"] for row in updates} == {"orig", "dmr", "balance", "predicate"},
                "G ledger explicitly admits all four provenance classes")
        require(not records(compiled["F"].stdout, "COMPAS_RACFED_PREDICATE_SITE"),
                "F retains ordinary B3 direction checks without predicate lowering")
        naive_updates = records(compiled["N"].stdout, "COMPAS_RACFED_UI")
        require({row["function"] for row in records(compiled["N"].stdout, "COMPAS_RACFED_NAIVE_SITE")} == set(FUNCTIONS)
                and not records(compiled["N"].stdout, "COMPAS_RACFED_PREDICATE_SITE")
                and {row["category"] for row in naive_updates} == {"orig", "dmr", "balance"}
                and all(int(row["delta"]) != 0 for row in naive_updates),
                "N removes direction checks without predicate helpers and retains the all-class nonzero ledger")
        for (function, block), region in regions.items():
            ledger = [row for row in updates if (row["function"], row["block"]) == (function, block)]
            require(len(ledger) == int(region["eligible"])
                    and sum(int(row["delta"]) for row in ledger) == int(region["sum"])
                    and all(int(row["delta"]) != 0
                            and int(row["after"]) - int(row["before"]) == int(row["delta"])
                            for row in ledger),
                    f"{function}/{block}: nonzero ledger, eligible count and complete sum agree")

        for function in FUNCTIONS:
            naive_body = linked["N"][function]
            naive_branches = [(i, item) for i, item in enumerate(naive_body)
                              if item[0] in ("beq", "bne")
                              and all(reg in PRIMARY for reg in item[1][:2])]
            require(len(naive_branches) == 1
                    and signature_delta(naive_body[naive_branches[0][0] - 1]) is not None
                    and not any(item[0] in ("beq", "bne")
                                and all(reg in layering.SHADOW_TO_PRIMARY for reg in item[1][:2])
                                for item in naive_body),
                    function + ": N has ordinary unpredicated RTS adjustment and no shadow direction branch")
            require(error_guards(naive_body) == error_guards(linked["F"][function])
                    and error_guards(naive_body) >= 3,
                    function + ": N preserves F's B3 unconditional error guards")
            require(all(("bne", *pair) in [layering.condition(item) for item in naive_body]
                        for pair in layering.ENTRY_PAIRS),
                    function + ": N preserves all fourteen incoming balance checks")
            body = linked["G"][function]
            branches = [(i, instruction) for i, instruction in enumerate(body)
                        if instruction[0] in ("beq", "bne")
                        and all(reg in PRIMARY for reg in instruction[1][:2])]
            function_sites = [row for row in sites if row["function"] == function]
            predicate_updates = [row for row in updates
                                 if row["function"] == function and row["category"] == "predicate"]
            require(len(branches) == len(function_sites) == 1,
                    function + ": exactly one primary application branch and one predicate site")
            require(len(predicate_updates) == 4,
                    function + ": exactly four helper computations enter the ledger")
            index, branch = branches[0]
            sequence = body[index - 10:index + 1]
            require([item[0] for item in sequence] ==
                    ["xor", "addi", "sltiu" if branch[0] == "beq" else "sltu", "addi",
                     "sub", "addi", "andi", "addi", "addi", "add", branch[0]],
                    function + ": final helper computations interleave nonrecursive updates before branch")
            shadow = [layering.PRIMARY_TO_SHADOW[reg] for reg in branch[1][:2]]
            require(sequence[0][1] == ["t4", *shadow]
                    and sequence[2][1] == (["t4", "t4", "1"] if branch[0] == "beq" else ["t4", "zero", "t4"])
                    and sequence[4][1] == ["t4", "zero", "t4"]
                    and sequence[6][1][:2] == ["t4", "t4"]
                    and sequence[8][1][:2] == ["t4", "t4"]
                    and sequence[9][1] == ["t0", "t0", "t4"],
                    function + ": shadow predicate prepares the complete scratch delta before one RTS commit")
            if function.endswith("zero"):
                require("zero" in branch[1][:2] and "s11" in shadow,
                        function + ": original x0 maps to shadow x27")
            require([signature_delta(sequence[i]) for i in (1, 3, 5, 7)] ==
                    [int(row["delta"]) for row in predicate_updates],
                    function + ": four linked update immediates exactly match predicate ledger")
            site = function_sites[0]
            source = regions[(function, site["block"])]
            for name, destination in (("true", "taken"), ("false", "fallthrough")):
                successor = regions[(function, site[destination])]
                expected = int(successor["x"]) + int(successor["y"]) - int(source["x"]) - int(source["sum"])
                require(int(site[name]) == expected,
                        function + ": " + name + " edge adjustment includes complete source ledger")
            require(int(sequence[8][1][2], 0) == int(site["false"])
                    and int(sequence[6][1][2], 0) == int(site["difference"])
                    and int(site["difference"]) == int(site["true"]) - int(site["false"])
                    and all(int(site[field]) != 0 for field in ("true", "false", "difference"))
                    and all(-2048 <= int(site[field]) <= 2047 for field in ("false", "difference")),
                    function + ": distinct nonzero edge adjustments agree with complete ledger and signed immediates")
            require(sequence[9][2] + 4 == branch[2],
                    function + ": single RTS commit immediately precedes independent primary branch")
            faulted = out / ("G-skip-commit-branch-" + function + ".elf")
            skip_pair(out / "G.elf", faulted, sequence[9][2])
            detected = run([args.qemu, "-strace", faulted],
                           "G-skip-commit-branch-" + function)
            require(re.findall(r"\bexit(?:_group)?\((-?\d+)\)", detected.stderr) == ["-256"]
                    and detected.stdout == "",
                    function + ": skipping final RTS ADD plus branch executes CFP detector ECALL")
            require(not any(item[0] in ("beq", "bne")
                            and all(reg in layering.SHADOW_TO_PRIMARY for reg in item[1][:2])
                            for item in body),
                    function + ": redundant shadow conditional branches removed")
            require(error_guards(body) == error_guards(linked["F"][function])
                    and error_guards(body) >= 3,
                    function + ": B3 unconditional DMR-error guards and terminal error loop retained")
            require(not any(item[:2] == ("addi", ["zero", "zero", "0"]) for item in body),
                    function + ": no padding NOPs")
            observed = [layering.condition(item) for item in body]
            require(all(("bne", *pair) in observed for pair in layering.ENTRY_PAIRS),
                    function + ": all fourteen incoming balance checks retained")

        require(all(sum(item[0] == "lw" for item in linked[mode]["shared"]) == 1
                    and sum(item[0] == "sw" for item in linked[mode]["shared"]) == 4
                    for mode in ("F", "N", "G")),
                "F/N/G retain one ordinary-store loadback and three ordered stores without duplicated MMIO reads")
        shared = linked["G"]["shared"]
        positions = {item[2]: i for i, item in enumerate(shared)}
        shared_targets = {target(item) for item in shared
                          if item[0] == "jal" and item[1][0] == "zero"}
        require(any(address in positions and positions[address] > 0
                    and shared[positions[address] - 1][0] not in
                    ("beq", "bne", "blt", "bge", "bltu", "bgeu", "jal", "jalr", "ecall")
                    for address in shared_targets),
                "shared: a successor has both an explicit jump and a normal fallthrough predecessor")

        mixed = run([*backend, *flags(predicate=False), "-RACFED-predicate-shadow=eq_rr",
                     ir, "-o", out / "mixed.s"], "mixed")
        require({row["function"] for row in records(mixed.stdout, "COMPAS_RACFED_PREDICATE_SITE")} == {"eq_rr"}
                and {row["function"] for row in records(mixed.stdout, "COMPAS_RACFED_UI")
                     if row["category"] == "predicate"} == {"eq_rr"},
                "per-function G selection does not leak into later F functions")
        mixed_naive = run([*backend, *flags(predicate=False), "-RACFED-naive-branches=eq_rr",
                           ir, "-o", out / "mixed-naive.s"], "mixed-naive")
        require({row["function"] for row in records(mixed_naive.stdout, "COMPAS_RACFED_NAIVE_SITE")} == {"eq_rr"}
                and not records(mixed_naive.stdout, "COMPAS_RACFED_PREDICATE_SITE"),
                "per-function N selection does not leak into later F functions or enable predication")

        allocation = out / "allocation.ll"
        allocation.write_text(allocation_fixture(192))
        allocated = run([*backend, *flags("-1"), allocation,
                         "-o", out / "allocation.s"], "allocation")
        allocated_sites = records(allocated.stdout, "COMPAS_RACFED_PREDICATE_SITE")
        require(len(allocated_sites) == 192
                and all(int(row["true"]) != 0 and int(row["false"]) != 0
                        and int(row["difference"]) == int(row["true"]) - int(row["false"])
                        and int(row["difference"]) != 0
                        and all(-2048 <= int(row[key]) <= 2047 for key in ("false", "difference"))
                        for row in allocated_sites),
                "192-function allocation corpus rejects zero edges, including legacy zero-taken clone 168")

        # The short legacy shadow corpus exercises initial label retry while
        # preserving its pre-instruction ledger. This is independent of the
        # CompaSeC last-helper allocation constraint checked above.
        legacy_labels = run([*backend, *flags("-1", compasec=False), allocation,
                             "-o", out / "legacy-labels.s"], "legacy-labels")
        label_sites = records(legacy_labels.stdout, "COMPAS_RACFED_PREDICATE_SITE")
        require(len(label_sites) == 192
                and all(int(row["true"]) and int(row["false"]) and int(row["difference"])
                        for row in label_sites)
                and any(row["function"] == "allocation_19"
                        for row in records(legacy_labels.stdout, "COMPAS_PREDICATE_LABEL_RETRY")),
                "legacy shadow allocation exercises the initial-label retry at clone 19")

        # A deterministic prefix reaches the first standalone legacy exit
        # collision at clone 1127. Keep that prefix rather than a larger search.
        legacy_corpus = out / "legacy-allocation.ll"
        legacy_corpus.write_text(allocation_fixture(1128))
        legacy_object = out / "legacy-allocation.o"
        legacy = run([*backend, "-RACFED=-1", "-RACFED-predicate=-1", "-filetype=obj",
                      legacy_corpus, "-o", legacy_object], "legacy-allocation")
        legacy_sites = records(legacy.stdout, "COMPAS_RACFED_PREDICATE_SITE")
        retunes = records(legacy.stdout, "COMPAS_RACFED_PREDICATE_RETUNE")
        require(len(legacy_sites) == 1128
                and all(int(row["true"]) and int(row["false"]) and int(row["difference"])
                        for row in legacy_sites)
                and any(row["function"] == "allocation_1127" for row in retunes),
                "legacy standalone corpus exercises an actual last-update collision at clone 1127")
        retune, = [row for row in retunes if row["function"] == "allocation_1127"]
        site, = [row for row in legacy_sites if row["function"] == "allocation_1127"]
        legacy_regions = {(row["function"], row["block"]): row
                          for row in records(legacy.stdout, "COMPAS_RACFED_REGION")}
        region = legacy_regions[(site["function"], site["block"])]
        last_update = [row for row in records(legacy.stdout, "COMPAS_RACFED_UI")
                       if (row["function"], row["block"]) == (site["function"], site["block"])][-1]
        targets = [legacy_regions[(site["function"], site[edge])]
                   for edge in ("taken", "fallthrough")]
        entries = [int(row["x"]) + int(row["y"]) for row in targets]
        new_exit = int(region["x"]) + int(region["sum"])
        old_exit = new_exit - int(retune["new"]) + int(retune["old"])
        require(old_exit in entries and new_exit not in entries
                and int(retune["old"]) != 0 and int(retune["new"]) != 0
                and retune["sum"] == region["sum"]
                and last_update["delta"] == retune["new"]
                and last_update["placement"] == "before"
                and int(region["eligible"]) == int(region["selected"]) - int(region["zero_initial"]),
                "legacy collision retunes one existing nonzero pre-update without changing ledger eligibility")

        driver = out / "legacy-driver.c"
        driver.write_text(
            "extern unsigned allocation_1127(unsigned, unsigned);\n"
            "extern volatile unsigned port_a, port_b;\n"
            "int main(void) {\n"
            "  if (allocation_1127(0xffffffffU, 0xfffffff9U) != 17 || port_a != 17) return 1;\n"
            "  if (allocation_1127(0xffffffffU, 0xfffffff8U) != 29 || port_b != 31) return 2;\n"
            "  return 42;\n"
            "}\n")
        legacy_elf = out / "legacy-allocation.elf"
        run([args.cc, "-march=rv32i", "-mabi=ilp32", "-mno-relax", "-O1",
             "-ffreestanding", "-fno-optimize-sibling-calls", "-nostdlib",
             "-static", "-no-pie", "-Wl,--build-id=none", "-Wl,--no-relax",
             "-Wl,-Ttext=0x10000", HERE.parents[1] / "support/start.S",
             driver, legacy_object, "-o", legacy_elf], "legacy-link")
        dump = run([args.objdump, "-d", "--no-show-raw-insn", "-M", "no-aliases", legacy_elf],
                   "legacy-disassembly")
        collision_body = linked_functions(dump.stdout)["allocation_1127"]
        commit, = [i for i, item in enumerate(collision_body)
                   if item[:2] == ("add", ["t0", "t0", "t4"])]
        selection, = [i for i, item in enumerate(collision_body[:commit])
                      if item[0] == "andi" and item[1][:2] == ["t4", "t4"]]
        require(collision_body[commit - 1][:2] == ("addi", ["t4", "t4", site["false"]])
                and collision_body[selection][1][2] == site["difference"]
                and collision_body[selection - 1][:2] == ("addi", ["t0", "t0", retune["new"]])
                and collision_body[commit + 1][0] in ("beq", "bne"),
                "linked collision fixture preserves the replacement pre-update and both complete edge deltas")
        execution = run([args.qemu, "-strace", legacy_elf], "legacy-execute", expect=42)
        require(re.findall(r"\bexit(?:_group)?\((-?\d+)\)", execution.stderr) == ["42"]
                and execution.stdout == "",
                "retuned legacy collision fixture passes the independent oracle for both branch outcomes")

        invalid = (
            ("missing_monitor", ["-NZDC+NEMESEC=eq_rr", "-RACFED-compasec=eq_rr", "-RACFED-predicate-shadow=eq_rr"], "requires RACFED"),
            ("missing_dmr", ["-RACFED=eq_rr", "-RACFED-compasec=eq_rr", "-RACFED-predicate-shadow=eq_rr"], "NZDC+NEMESEC"),
            ("plain_nzdc", [*flags("eq_rr"), "-NZDC=eq_rr"], "NZDC"),
            ("fgs", [*flags("eq_rr"), "-FGS=eq_rr"], "CGS"),
        )
        for name, options, diagnostic in invalid:
            failed = run([*backend, *options, ir, "-o", out / (name + ".s")], name, expect=None)
            require(failed.returncode != 0 and diagnostic in failed.stderr,
                    name + ": explicit invalid-configuration diagnostic")
        naive_invalid = (
            ("missing_monitor", ["-NZDC+NEMESEC=eq_rr", "-RACFED-compasec=eq_rr"], "requires RACFED"),
            ("missing_policy", ["-NZDC+NEMESEC=eq_rr", "-RACFED=eq_rr"], "RACFED-compasec"),
            ("missing_dmr", ["-RACFED=eq_rr", "-RACFED-compasec=eq_rr"], "NZDC+NEMESEC"),
            ("plain_nzdc", [*flags("eq_rr", predicate=False), "-NZDC=eq_rr"], "NZDC"),
            ("fgs", [*flags("eq_rr", predicate=False), "-FGS=eq_rr"], "CGS"),
            ("soft_conflict", flags("eq_rr"), "cannot compose"),
        )
        for name, options, diagnostic in naive_invalid:
            failed = run([*backend, *options, "-RACFED-naive-branches=eq_rr", ir,
                          "-o", out / ("naive-" + name + ".s")], "naive-" + name, expect=None)
            require(failed.returncode != 0 and diagnostic in failed.stderr
                    and "COMPAS: Running" not in failed.stdout,
                    "naive-" + name + ": rejected before DMR mutation")
        # Historical rejection fixtures now exercise supported workload forms.
        for name in ("bad_relation", "bad_call"):
            for naive in (False, True):
                label = ("naive-" if naive else "predicate-") + name
                result = run([*backend, *flags(name, predicate=not naive, naive=naive),
                              HERE / "unsupported.ll", "-o", out / (label + ".s")], label)
                require(result.returncode == 0, label + ": relational branch or returning call verified")
        for name, target_options in (
            ("compressed", ["-mattr=+c,-relax"]),
            ("rv64", ["-march=riscv64", "-mtriple=riscv64-unknown-elf", "-target-abi=lp64"]),
        ):
            failed = run([*backend, *flags("target_check"), *target_options,
                          HERE / "unsupported.ll", "-o", out / (name + ".s")], name, expect=None)
            require(failed.returncode != 0 and "RV32" in failed.stderr,
                    name + ": explicit unsupported-target diagnostic")
            failed = run([*backend, *flags("target_check", predicate=False, naive=True), *target_options,
                          HERE / "unsupported.ll", "-o", out / ("naive-" + name + ".s")],
                         "naive-" + name, expect=None)
            require(failed.returncode != 0 and "RACFED-naive-branches" in failed.stderr
                    and "RV32" in failed.stderr and "COMPAS: Running" not in failed.stdout,
                    "naive-" + name + ": unsupported target rejected before mutation")
        result = dict(status="passed", checks=checks,
                      scope="machine verification, linked instruction ledger, fault-free literal oracle and concrete final-ADD/branch skip regressions; no exhaustive skip proof")
    except (AssertionError, KeyError, IndexError, StopIteration, OSError, ValueError,
            subprocess.TimeoutExpired) as error:
        result = dict(status="failed", error=str(error), checks=checks)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
