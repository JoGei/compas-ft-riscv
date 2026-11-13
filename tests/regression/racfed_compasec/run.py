#!/usr/bin/env python3
"""Validate the opt-in CompaSeC RACFED ledger and independent normal behavior."""

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
spec = importlib.util.spec_from_file_location("layering", HERE.parent / "rsm_layering/check_assembly.py")
layering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layering)
FUNCTIONS = "balance,direction,stores,zero_store,call_boundary"


def records(output, tag):
    return [dict(item.split("=", 1) for item in line.split()[1:])
            for line in output.splitlines() if line.startswith(tag + " ")]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for tool in ("llc", "clang", "cc", "qemu"):
        cli.add_argument("--" + tool, required=True)
    cli.add_argument("--legacy-llc", help="optional archived compiler for unchanged legacy assembly comparison")
    cli.add_argument("--build-dir", type=Path, required=True)
    args = cli.parse_args()
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

    backend = [args.llc, "-march=riscv32", "-mattr=-c,-relax", "-target-abi=ilp32", "-O1", "-verify-machineinstrs"]
    try:
        short = out / "short.s"
        compiled = run([*backend, "-RACFED=-1:orig+dmr", "-RACFED-compasec=-1", HERE / "short.ll", "-o", short], "short")
        assembly = layering.assembly_functions([short])
        updates = records(compiled.stdout, "COMPAS_RACFED_UI")
        for name, count in (("one", 1), ("two", 2), ("three", 3)):
            selected = [row for row in updates if row["function"] == name]
            require(len(selected) == count and all(int(row["delta"]) != 0 for row in selected),
                    f"{name}: all {count} eligible instructions have nonzero ledger updates")
            compute = [index for index, (op, operands, _) in enumerate(assembly[name])
                       if op in ("addi", "xori", "andi") and operands[:2] == ["a0", "a0"]]
            require(len(compute) == count and all(layering.signature_update(assembly[name][index + 1]) for index in compute),
                    f"{name}: final assembly updates immediately after first through last computation")

        mixed = run([*backend, "-RACFED=one,two,three", "-RACFED-compasec=one", HERE / "short.ll", "-o", out / "mixed.s"], "mixed")
        require({row["function"] for row in records(mixed.stdout, "COMPAS_RACFED_UI")} == {"one"},
                "per-function policy does not leak to later legacy functions")
        for name, flags, message in (
            ("missing_racfed", ["-RACFED-compasec=one"], "requires RACFED"),
            ("conflicting_mask", ["-RACFED=one:orig", "-RACFED-compasec=one"], "bare/all or orig+dmr"),
            ("double_monitor", ["-RACFED=one", "-RASM=one", "-RACFED-compasec=one"], "another signature monitor"),
        ):
            failure = run([*backend, *flags, HERE / "short.ll", "-o", out / (name + ".s")], name, expect=None)
            require(failure.returncode != 0 and message in failure.stderr, name + ": explicit diagnostic")

        if args.legacy_llc:
            for name, executable in (("legacy_before", args.legacy_llc), ("legacy_after", args.llc)):
                run([executable, *backend[1:], "-RACFED=-1", HERE / "short.ll", "-o", out / (name + ".s")], name)
            require((out / "legacy_before.s").read_bytes() == (out / "legacy_after.s").read_bytes(),
                    "unselected legacy RACFED assembly remains byte-identical")

        ir = out / "functional.ll"
        run([args.clang, "--target=riscv32", "-march=rv32i", "-mabi=ilp32", "-mno-relax", "-O1", "-ffreestanding", "-fno-optimize-sibling-calls", "-emit-llvm", "-S", HERE / "functional.c", "-o", ir], "frontend")
        flags = ["-NZDC+NEMESEC=" + FUNCTIONS, "-RACFED=" + ",".join(name + ":orig+dmr" for name in FUNCTIONS.split(",")), "-RACFED-compasec=" + FUNCTIONS]
        compiled = run([*backend, *flags, ir, "-o", out / "functional.s"], "functional")
        rows = records(compiled.stdout, "COMPAS_RACFED_UI")
        require({row["category"] for row in rows} == {"orig", "dmr", "balance"},
                "application, shadow dataflow and balance instructions enter the ledger")
        assembly = layering.assembly_functions([out / "functional.s"])
        balance = assembly["balance"]
        guards = []
        for i, item in enumerate(balance):
            if len(guards) < len(layering.ENTRY_PAIRS) and layering.condition(item) == ("bne", *layering.ENTRY_PAIRS[len(guards)]):
                guards.append(i)
        require(len(guards) == 14, "all fourteen incoming primary/shadow balance checks retained")
        require(all(layering.signature_update(balance[i + 1]) for i in guards),
                "each successful entry balance check receives its post-check update")
        require(not any(layering.signature_check(item) for left, right in zip(guards, guards[1:])
                        for item in balance[left + 1:right]),
                "entry balance checks do not create separately signed continuation regions")
        obj = out / "functional.o"
        run([*backend, *flags, "-filetype=obj", ir, "-o", obj], "object")
        elf = out / "functional.elf"
        run([args.cc, "-march=rv32i", "-mabi=ilp32", "-mno-relax", "-nostdlib", "-static", "-no-pie", "-Wl,--build-id=none", "-Wl,--no-relax", "-Wl,-Ttext=0x10000", HERE.parents[1] / "support/start.S", obj, "-o", elf], "link")
        execution = run([args.qemu, "-strace", elf], "execute", expect=42)
        require(re.findall(r"\bexit(?:_group)?\((-?\d+)\)", execution.stderr) == ["42"] and execution.stdout == "",
                "independent literal oracle: both branch outcomes, ordinary/ordered/narrow/zero stores and protected-to-ordinary call return")
        result = dict(status="passed", checks=checks)
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        result = dict(status="failed", error=str(error), checks=checks)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
