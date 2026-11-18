#!/usr/bin/env python3
"""Check six integer branch predicates against independent linked-word oracles."""

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import resource
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


alignment = module("relational_alignment", HERE.parent / "predicate_alignment/run.py")
reference = module("relational_reference", HERE.parent / "racfed_xpred/reference.py")
FUNCTIONS = ("eq", "ne", "slt", "sge", "ult", "uge", "slt_zero", "sge_zero")
SELECTED = ",".join(FUNCTIONS)
BRANCHES = {"beq", "bne", "blt", "bge", "bltu", "bgeu"}
VALUES = (0, 1, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff)


def expected_condition(function, left, right):
    """Source comparison, independent of compiler branch polarity/layout."""
    if function.endswith("_zero"):
        function, right = function[:-5], 0
    if function in ("eq", "ordinary"):
        return left == right
    if function == "ne":
        return left != right
    if function == "slt":
        return reference.signed(left) < reference.signed(right)
    if function == "sge":
        return reference.signed(left) >= reference.signed(right)
    if function == "ult":
        return left < right
    if function == "uge":
        return left >= right
    raise ValueError("unknown source comparison: " + function)


def configurations():
    common = ["-NZDC+NEMESEC=" + SELECTED,
              "-RACFED=" + ",".join(name + ":orig+dmr" for name in FUNCTIONS),
              "-RACFED-compasec=" + SELECTED]
    yield "A", [], False, None, None
    yield "F", common, True, None, None
    yield "N", common + ["-RACFED-naive-branches=" + SELECTED], True, None, None
    for name, hard in (("G", False), ("H", True)):
        options = common + ["-RACFED-predicate-shadow=" + SELECTED]
        if hard:
            options.append("-RACFED-predicate-xpred=" + SELECTED)
        yield name, options, True, "shadow", hard
    for monitor in ("RASM", "RACFED"):
        sources = ("primary", "shadow") if monitor == "RASM" else ("primary",)
        for source in sources:
            for hard in (False, True):
                options = alignment.flags(monitor, "NZDC+NEMESEC" if source == "shadow" else None,
                    source, selection=SELECTED, compasec=monitor == "RACFED")
                if hard:
                    options.append("-" + monitor + "-predicate-xpred=" + SELECTED)
                yield monitor + "-" + source + ("-HP" if hard else "-SP"), options, source == "shadow", source, hard


def signature_update(instruction):
    return instruction[0] == "addi" and instruction[1][:2] == ["t0", "t0"]


def inspect_predicate(body, source, hard, require, label):
    commits = [index for index, item in enumerate(body)
               if item[0] == "nz.sacc.i" or item[:2] == ("add", ["t0", "t0", "t4"])]
    require(len(commits) == 1, label + ": one signature commit for the application branch")
    index = commits[0]
    require(index + 1 < len(body) and body[index + 1][0] in BRANCHES,
            label + ": commit immediately precedes the primary branch")
    branch, commit = body[index + 1], body[index]
    require((commit[0] == "nz.sacc.i") == hard, label + ": requested hardware/software lowering")
    wanted = ([reference.PRIMARY_TO_SHADOW[reg] for reg in branch[1][:2]]
              if source == "shadow" else branch[1][:2])
    raw = ("xor" if branch[0] in ("beq", "bne") else
           "slt" if branch[0] in ("blt", "bge") else "sltu")
    helpers = []
    for item in reversed(body[:index]):
        if signature_update(item):
            continue
        helpers.append(item)
        if item[:2] == (raw, ["t4", *wanted]):
            break
    helpers.reverse()
    require(bool(helpers) and helpers[0][:2] == (raw, ["t4", *wanted]),
            label + ": comparison uses correct signedness and " + source + " operands")
    if label.split("/")[-1].endswith("_zero"):
        require(("s11" if source == "shadow" else "zero") in wanted,
                label + ": zero operand retains the selected register bank")
    if hard:
        require(len(helpers) == 1 and commit[1][:2] == ["t0", "t4"],
                label + ": hardware uses one raw comparison and tied signature accumulation")
        literals = list(map(int, commit[1][2:]))
        require(len(literals) == 2 and literals[0] != literals[1]
                and all(-32 <= value <= 31 and value for value in literals),
                label + ": distinct nonzero signed-six-bit adjustment pair")
    else:
        normalization = {"beq": [("sltiu", ["t4", "t4", "1"])],
                         "bne": [("sltu", ["t4", "zero", "t4"])],
                         "bge": [("xori", ["t4", "t4", "1"])],
                         "bgeu": [("xori", ["t4", "t4", "1"])]}.get(branch[0], [])
        require([item[:2] for item in helpers[1:1 + len(normalization)]] == normalization,
                label + ": predicate normalization matches branch orientation")
        selection = helpers[1 + len(normalization):]
        require(len(selection) == 3 and selection[0][:2] == ("sub", ["t4", "zero", "t4"])
                and selection[1][0] == "andi" and selection[1][1][:2] == ["t4", "t4"]
                and selection[2][0] == "addi" and selection[2][1][:2] == ["t4", "t4"],
                label + ": complete selected adjustment before a single signature ADD")
    return branch[0], helpers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--cc", type=Path, required=True, help="RV32-capable bare-metal GCC linker driver")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    checks, executions = [], []
    report = {"status": "running", "checks": checks, "executions": executions,
              "scope": "Linked RV32 instruction interpreter, selected explicit inputs; no RTL or fault campaign",
              "source_sha256": hashlib.sha256((HERE / "functional.ll").read_bytes()).hexdigest()}

    def require(condition, description):
        if not condition:
            raise AssertionError(description)
        checks.append(description)

    def run(command, name):
        argv = list(map(str, command))
        (out / (name + ".command.json")).write_text(json.dumps(argv, indent=2) + "\n")
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        (out / (name + ".stdout")).write_text(result.stdout)
        (out / (name + ".stderr")).write_text(result.stderr)
        if result.returncode:
            raise AssertionError(f"{name}: exit {result.returncode}; see {out / (name + '.stderr')}")
        return result

    try:
        for name, options, balanced, source, hard in configurations():
            obj, elf = out / (name + ".o"), out / (name + ".elf")
            compiled = run([args.llvm_bin / "llc", "-march=riscv32", "-mattr=+xpred,-c,-relax",
                "-target-abi=ilp32", "-O1", "-verify-machineinstrs", "-filetype=obj", *options,
                HERE / "functional.ll", "-o", obj], name + "-codegen")
            require("Bad machine code" not in compiled.stderr, name + ": machine verifier accepts emitted code")
            run([args.cc, "-march=rv32i", "-mabi=ilp32", "-mno-relax", "-nostdlib", "-static",
                "-no-pie", "-Wl,--build-id=none,--no-relax,-Ttext=0x10000,-e,eq", obj, "-o", elf], name + "-link")
            dump = run([args.llvm_bin / "llvm-objdump", "-d", "--mattr=+xpred,-c",
                "--no-print-imm-hex", "--no-show-raw-insn", "-M", "no-aliases", elf], name + "-disassembly")
            table = run([args.llvm_bin / "llvm-objdump", "-t", elf], name + "-symbols")
            symbols = {row.split()[-1]: int(row.split()[0], 16)
                       for row in table.stdout.splitlines() if re.match(r"^[0-9a-f]{8}\s", row)}
            bodies, memory = alignment.linked_functions(dump.stdout), reference.elf_memory(elf)
            if source:
                seen = set()
                ledger = alignment.records(compiled.stdout, "COMPAS_RACFED_UI")
                for function in FUNCTIONS:
                    branch, helpers = inspect_predicate(bodies[function], source, hard, require, name + "/" + function)
                    seen.add(branch)
                    if name in ("G", "H") or name.startswith("RACFED"):
                        monitored = helpers if hard else helpers[:-1]
                        rows = [row for row in ledger if row["function"] == function and row["category"] == "predicate"]
                        require([row["opcode"].lower() for row in rows] == [item[0] for item in monitored]
                                and all(row["placement"] == "after" and int(row["delta"]) != 0 for row in rows),
                                name + "/" + function + ": exact variable-length helper ledger")
                require(seen == BRANCHES, name + ": all six native branch conditions emitted and checked")
            outputs = {symbols["taken_output"], symbols["fallthrough_output"]}
            for function in (*FUNCTIONS, "ordinary"):
                outcomes = set()
                for left, right in itertools.product(VALUES, repeat=2):
                    truth = expected_condition(function, left, right)
                    outcomes.add(truth)
                    result = reference.execute(memory, symbols[function], (left, right), balanced)
                    actual = [(store["address"], store["value"]) for store in result["stores"] if store["address"] in outputs]
                    expected = ([(symbols["taken_output"], 17)] if truth else
                                [(symbols["fallthrough_output"], 29), (symbols["fallthrough_output"], 31)])
                    require(result["return_value"] == (17 if truth else 29) and actual == expected,
                            f"{name}/{function}/{left:#x}/{right:#x}: source return and ordered-store oracle")
                    require(result["registers"][2] == result["initial_registers"][2],
                            name + "/" + function + ": stack restored")
                    if balanced and function != "ordinary":
                        require(all(result["registers"][reference.REGNO[primary]] ==
                                    result["registers"][reference.REGNO[shadow]]
                                    for primary, shadow in reference.PRIMARY_TO_SHADOW.items()),
                                name + "/" + function + ": primary/shadow balance restored at return")
                    executions.append({"configuration": name, "function": function,
                        "arguments": [left, right], "expected_condition": truth, "steps": result["steps"]})
                require(outcomes == {False, True}, name + "/" + function + ": both source outcomes checked")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": len(checks),
                      "executions": len(executions), "report": str(out / "results.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
