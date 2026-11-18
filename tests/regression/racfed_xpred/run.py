#!/usr/bin/env python3
"""Validate aligned RASM/RACFED hardware predicates using linked RV32 words."""

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


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


baseline = module("racfed_predicate", HERE.parent / "racfed_predicate/run.py")
alignment = module("predicate_alignment", HERE.parent / "predicate_alignment/run.py")
reference = module("hardware_reference", HERE / "reference.py")
FUNCTIONS, SELECTED = baseline.FUNCTIONS, baseline.SELECTED
CASES = [
    ("eq_rr", 0xffffffff, 0xfffffff9, 17, [("port_a", 17)]),
    ("eq_rr", 0xffffffff, 0xfffffff8, 29, [("port_b", 29), ("port_b", 31)]),
    ("ne_rr", 0xffffffff, 0xfffffff9, 29, [("port_b", 29), ("port_b", 31)]),
    ("ne_rr", 0xffffffff, 0xfffffff8, 17, [("port_a", 17)]),
    ("eq_zero", 0xffffffff, 0xfffffff9, 17, [("port_a", 17)]),
    ("eq_zero", 0xffffffff, 6, 29, [("port_b", 29), ("port_b", 31)]),
    ("ne_zero", 0xffffffff, 0xfffffff9, 29, [("port_b", 29), ("port_b", 31)]),
    ("ne_zero", 0xffffffff, 6, 17, [("port_a", 17)]),
    ("shared", 7, 7, 25, [("port_a", 41), ("ordinary", 10), ("joined", 10)]),
    ("shared", 7, 3, 27, [("port_b", 43), ("ordinary", 8), ("joined", 8)]),
]


def flags(monitor, dmr, source, selection=SELECTED, hard=None, compasec=False, mask=None):
    result = alignment.flags(monitor, dmr, source, selection, compasec=compasec, mask=mask)
    if hard is not False:
        result.append("-" + monitor + "-predicate-xpred=" + (selection if hard is None else hard))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cc", type=Path, required=True, help="RV32-capable GNU linker driver")
    parser.add_argument("--allow-missing-xpred", action="store_true",
                        help="report skipped when llvm-mc does not expose the optional XPred extension")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    checks, executions = [], []

    def require(condition, description):
        if not condition:
            raise AssertionError(description)
        checks.append(description)

    def run(command, name, expect=0, input_text=None):
        argv = list(map(str, command))
        (out / (name + ".command.json")).write_text(json.dumps(argv, indent=2) + "\n")
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120, input=input_text)
        (out / (name + ".stdout")).write_text(result.stdout)
        (out / (name + ".stderr")).write_text(result.stderr)
        if expect is not None and result.returncode != expect:
            raise AssertionError(f"{name}: exit {result.returncode}, expected {expect}; see {out / (name + '.stderr')}")
        return result

    backend = [args.llvm_bin / "llc", "-march=riscv32", "-mattr=+xpred,-c,-relax",
               "-target-abi=ilp32", "-O1", "-verify-machineinstrs"]

    def compile_link(name, options, ir):
        obj, elf = out / (name + ".o"), out / (name + ".elf")
        compiled = run([*backend, *options, "-filetype=obj", ir, "-o", obj], name + "-codegen")
        require("Bad machine code" not in compiled.stderr, name + ": machine verifier accepts tied operands and CFG")
        run([args.cc, "-march=rv32i", "-mabi=ilp32", "-mno-relax", "-nostdlib",
             "-static", "-no-pie", "-Wl,--build-id=none", "-Wl,--no-relax",
             "-Wl,-Ttext=0x10000", "-Wl,-e,eq_rr", obj, "-o", elf], name + "-link")
        dump = run([args.llvm_bin / "llvm-objdump", "-d", "--mattr=+xpred,-c",
                    "--no-print-imm-hex", "--no-show-raw-insn", "-M", "no-aliases", elf], name + "-disassembly")
        table = run([args.llvm_bin / "llvm-objdump", "-t", elf], name + "-symbols")
        symbols = {row.split()[-1]: int(row.split()[0], 16)
                   for row in table.stdout.splitlines() if re.match(r"^[0-9a-f]{8}\s", row)}
        return compiled, baseline.linked_functions(dump.stdout), reference.elf_memory(elf), symbols

    def hardware_sites(name, compiled, functions, monitor, dmr, source, selected=FUNCTIONS):
        tag = "COMPAS_RACFED_PREDICATE_SITE" if monitor == "RACFED" else "COMPAS_XPRED_SITE"
        sites = baseline.records(compiled.stdout, tag)
        hardware = [row for row in sites if row.get("lowering") == "sacc" or row.get("fused") == "1"]
        require({row["function"] for row in hardware} == set(selected), name + ": exact hardware function selection")
        allocation_tag = "COMPAS_RACFED_PSP" if monitor == "RACFED" else "COMPAS_PSP"
        allocation = baseline.records(compiled.stdout, allocation_tag)
        require({row["function"] for row in allocation if row["status"] == "allocated"} == set(selected),
                name + ": mandatory bounded joint allocation for every hardware function")
        labels = baseline.records(compiled.stdout, "COMPAS_PSP_LABEL")
        for function in selected:
            rows = [row for row in labels if row["function"] == function]
            require(bool(rows) and len({int(row["x"]) for row in rows}) == len(rows)
                    and len({int(row["q"]) for row in rows}) == len(rows)
                    and all(-512 <= int(row["x"]) <= 511 and -512 <= int(row["y"]) <= 511
                            and int(row["y"]) != 0 and int(row["q"]) == int(row["x"]) + int(row["y"])
                            for row in rows), name + "/" + function + ": unique labels and nonzero signed-ten-bit Y")
            body = functions[function]
            commits = [index for index, item in enumerate(body) if item[0] == "nz.sacc.i"]
            require(len(commits) == 1, name + "/" + function + ": exactly one sacc and no wider hardware fallback")
            index = commits[0]
            require(body[index][1][:2] == ["t0", "t4"] and body[index + 1][0] in ("beq", "bne"),
                    name + "/" + function + ": persistent accumulation immediately before independent primary branch")
            branch = body[index + 1]
            helper_index = index - 1
            if baseline.signature_delta(body[helper_index]) is not None:
                helper_index -= 1
            helper = body[helper_index]
            operands = branch[1][:2]
            wanted = [reference.PRIMARY_TO_SHADOW[reg] for reg in operands] if source == "shadow" else operands
            require(helper[:2] == ("xor", ["t4", *wanted]), name + "/" + function + ": raw XOR uses " + source + " operands")
            site, = [row for row in hardware if row["function"] == function]
            true, false = int(site["true"]), int(site["false"])
            literals = [true, false] if branch[0] == "bne" else [false, true]
            require(true != false and all(-32 <= value <= 31 and value != 0 for value in literals)
                    and list(map(int, body[index][1][2:])) == literals,
                    name + "/" + function + ": distinct nonzero signed-six-bit edges retain BEQ/BNE polarity")
            require(not any(item[0] in ("nz.acc.i", "ez.acc.i", "nz.acc", "ez.acc") for item in body)
                    and not alignment.predicate_commits(body), name + "/" + function + ": strict hardware mode has no software or wide accumulate fallback")
            require(not any(item[:2] == ("addi", ["zero", "zero", "0"]) for item in body),
                    name + "/" + function + ": explicitly unpadded lowering")
            if function.endswith("zero"):
                require("zero" in operands and ("s11" if source == "shadow" else "zero") in wanted,
                        name + "/" + function + ": zero operand uses selected stream")
            if dmr:
                redundant = [reference.PRIMARY_TO_SHADOW[reg] for reg in operands]
                duplicated = any(item[0] in ("beq", "bne") and item[1][:2] in (redundant, redundant[::-1]) for item in body)
                require(duplicated == (dmr != "NZDC" and source == "primary"),
                        name + "/" + function + ": direction checks replaced only for shadow predicates")
        return hardware

    def ledger(name, compiled, functions, sites, compasec=(), mask="all"):
        updates = baseline.records(compiled.stdout, "COMPAS_RACFED_UI")
        regions = baseline.records(compiled.stdout, "COMPAS_RACFED_REGION")
        indexed = {(row["function"], row["block"]): row for row in regions}
        for region in regions:
            entries = [row for row in updates if (row["function"], row["block"]) == (region["function"], region["block"])]
            require(len(entries) == int(region["eligible"]) and sum(int(row["delta"]) for row in entries) == int(region["sum"])
                    and all(int(row["delta"]) != 0 and int(row["after"]) - int(row["before"]) == int(row["delta"]) for row in entries),
                    name + ": " + region["function"] + "/" + region["block"] + " complete nonzero ledger")
            if region["function"] not in compasec:
                count, zero = int(region["selected"]), int(region["zero_initial"])
                require(int(region["eligible"]) == (count - zero if count >= 3 else 0)
                        and region["placement"] == "before" and zero in (0, 1),
                        name + ": legacy threshold and first-position exemption retained")
                permitted = {"orig"} | ({"dmr", "predicate"} if mask != "orig" else set())
                if mask == "all":
                    permitted.add("balance")
                require({row["category"] for row in entries} <= permitted, name + ": selected instruction mask retained")
        for site in sites:
            function = site["function"]
            body = functions[function]
            entries = [row for row in updates if row["function"] == function and row["category"] == "predicate"]
            require(len(entries) <= 1 and all(row["opcode"] == "XOR" for row in entries),
                    name + "/" + function + ": only raw XOR enters hardware helper ledger")
            if function in compasec:
                require(len(entries) == 1, name + "/" + function + ": CompaSeC records the single XOR")
            elif mask == "orig":
                require(not entries, name + "/" + function + ": orig-only ledger excludes XOR")
            for entry in entries:
                commit, = [index for index, item in enumerate(body) if item[0] == "nz.sacc.i"]
                helper = commit - 1 - int(baseline.signature_delta(body[commit - 1]) is not None)
                offset = 1 if function in compasec else -1
                require(entry["placement"] == ("after" if offset == 1 else "before")
                        and baseline.signature_delta(body[helper + offset]) == int(entry["delta"]),
                        name + "/" + function + ": linked XOR update placement follows policy")
            source = indexed[(function, site["block"])]
            for arm, target in (("true", "taken"), ("false", "fallthrough")):
                successor = indexed[(function, site[target])]
                expected = int(successor["x"]) + int(successor["y"]) - int(source["x"]) - int(source["sum"])
                require(int(site[arm]) == expected, name + "/" + function + ": " + arm + " edge includes exact emitted ledger")
        return indexed

    def execute_cases(name, image, symbols, dmr, functions, cases=CASES):
        addresses = {symbols[key] for key in ("port_a", "port_b", "ordinary", "joined")}
        for ordinal, (function, left, right, expected_return, expected_stores) in enumerate(cases):
            execution = reference.execute(image, symbols[function], (left, right), bool(dmr))
            actual = [(store["address"], store["value"]) for store in execution["stores"] if store["address"] in addresses]
            expected = [(symbols[symbol], value) for symbol, value in expected_stores]
            require(execution["return_value"] == expected_return and actual == expected,
                    f"{name}/{ordinal}/{function}: literal return and ordered-store oracle")
            preserved = ("sp", "gp", "tp", "s0") if dmr else ("sp", "gp", "tp", *["s" + str(index) for index in range(12)])
            require(all(execution["registers"][reference.REGNO[reg]] == execution["initial_registers"][reference.REGNO[reg]]
                        for reg in preserved), f"{name}/{ordinal}: stack, fixed and primary callee-saved values preserved")
            if dmr:
                require(all(execution["registers"][reference.REGNO[primary]] == execution["registers"][reference.REGNO[shadow]]
                            for primary, shadow in reference.PRIMARY_TO_SHADOW.items()),
                        f"{name}/{ordinal}: return restores balance including borrowed signature pair")
            if dmr == "NZDC+NEMESEC":
                peripheral_addresses = {symbols[key] for key in ("port_a", "port_b", "joined")}
                require(not any(load["address"] in peripheral_addresses for load in execution["loads"]),
                        f"{name}/{ordinal}: no volatile peripheral loadback")
                if function == "shared":
                    require(sum(load["address"] == symbols["ordinary"] for load in execution["loads"]) == 1,
                            f"{name}/{ordinal}: ordinary store retains exactly one loadback")
                for store in execution["stores"]:
                    if store["address"] not in {symbols[key] for key in ("port_a", "port_b", "joined")}:
                        continue
                    position = next(index for index, row in enumerate(execution["trace"]) if row["pc"] == store["pc"])
                    previous = execution["trace"][:position]
                    pairs = [(reg, reference.REGNO[reference.PRIMARY_TO_SHADOW[reference.REGISTERS[reg]]])
                             for reg in (store["rs1"], store["rs2"])]
                    for pair in pairs:
                        last_write = max((index for index, row in enumerate(previous) if row["written"] in pair), default=-1)
                        require(any(index > last_write and row["opcode"] == "bne" and (row["rs1"], row["rs2"]) == pair
                                    for index, row in enumerate(previous)), f"{name}/{ordinal}: volatile address/data checked after last definition")
            (out / f"{name}-execution-{ordinal}.json").write_text(json.dumps(execution, indent=2) + "\n")
            executions.append(dict(mode=name, function=function, left=left, right=right, steps=execution["steps"]))

    try:
        features = run([args.llvm_bin / "llvm-mc", "-triple=riscv32", "-mattr=help"], "available-features")
        if not re.search(r"^\s*xpred\s+-", features.stdout + features.stderr, re.MULTILINE):
            if not args.allow_missing_xpred:
                raise AssertionError("LLVM lacks the optional XPred extension; use an XPred-capable compiler")
            # Ordinary COMPAS CI still validates a deliberate hardware request:
            # rejecting the missing extension must precede DMR instrumentation.
            for monitor in ("RASM", "RACFED"):
                name = monitor + "-reject-unavailable-extension"
                failed = run([*backend, *flags(monitor, "NZDC+NEMESEC", "shadow", "target_check"),
                              "-mattr=-c,-relax", HERE.parent / "predicate_alignment/unsupported.ll",
                              "-o", out / (name + ".s")], name, expect=None)
                require(failed.returncode != 0
                        and "XPred-enabled LLVM build" in failed.stderr
                        and "+xpred function feature" in failed.stderr,
                        monitor + ": explicit hardware request rejects unavailable extension")
            result = dict(status="skipped", reason="Full hardware matrix requires optional XPred; missing-extension diagnostics passed", checks=checks)
            (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            return 0
        ir = out / "functional.ll"
        run([args.llvm_bin / "clang", "--target=riscv32", "-march=rv32i", "-mabi=ilp32", "-mno-relax", "-O1",
             "-ffreestanding", "-fno-optimize-sibling-calls", "-emit-llvm", "-S",
             HERE.parent / "racfed_predicate/functional.c", "-o", ir], "frontend")
        _, plain, image, symbols = compile_link("plain", [], ir)
        execute_cases("plain", image, symbols, None, plain)
        for monitor, policy in (("RASM", None), ("RACFED", "compasec"), ("RACFED", "orig"),
                                ("RACFED", "orig+dmr"), ("RACFED", "all")):
            for dmr in (None, "NZDC", "NZDC+NEMESIS", "NZDC+NEMESEC"):
                for source in (("primary", "shadow") if dmr else ("primary",)):
                    name = "-".join((monitor, policy or "legacy", dmr or "standalone", source))
                    compasec = policy == "compasec"
                    mask = policy if monitor == "RACFED" and not compasec else None
                    options = flags(monitor, dmr, source, compasec=compasec, mask=mask)
                    compiled, functions, image, symbols = compile_link(name, options, ir)
                    sites = hardware_sites(name, compiled, functions, monitor, dmr, source)
                    if monitor == "RACFED":
                        ledger(name, compiled, functions, sites, compasec=FUNCTIONS if compasec else (), mask=mask or "all")
                    require(not any(item[0] == "nz.sacc.i" for item in functions["main"]),
                            name + ": hardware predicate selection does not leak to unselected main")
                    execute_cases(name, image, symbols, dmr, functions)

        for monitor in ("RASM", "RACFED"):
            options = flags(monitor, "NZDC+NEMESEC", "shadow", hard="eq_rr", compasec=False)
            name = monitor + "-mixed-software-hardware"
            compiled, functions, image, symbols = compile_link(name, options, ir)
            hardware_sites(name, compiled, functions, monitor, "NZDC+NEMESEC", "shadow", ("eq_rr",))
            for function in FUNCTIONS[1:]:
                require(len(alignment.predicate_commits(functions[function])) == 1
                        and not any(item[0] == "nz.sacc.i" for item in functions[function]),
                        name + "/" + function + ": software selection remains independent")
            execute_cases(name, image, symbols, "NZDC+NEMESEC", functions)
            selected = flags(monitor, "NZDC+NEMESEC", "shadow", compasec=monitor == "RACFED")
            first, _, _, _ = compile_link(monitor + "-seed1", selected, ir)
            second, functions, image, symbols = compile_link(monitor + "-seed2", [*selected, "-" + monitor + "-predicate-psp-seed=2"], ir)
            hardware_sites(monitor + "-seed2", second, functions, monitor, "NZDC+NEMESEC", "shadow")
            require(baseline.records(first.stdout, "COMPAS_PSP_LABEL") != baseline.records(second.stdout, "COMPAS_PSP_LABEL"),
                    monitor + ": seed changes label allocation while preserving constraints")
            execute_cases(monitor + "-seed2", image, symbols, "NZDC+NEMESEC", functions)

        name = "RACFED-mixed-policies"
        options = [*flags("RACFED", "NZDC+NEMESEC", "shadow"), "-RACFED-compasec=eq_rr,ne_zero"]
        compiled, functions, image, symbols = compile_link(name, options, ir)
        sites = hardware_sites(name, compiled, functions, "RACFED", "NZDC+NEMESEC", "shadow")
        ledger(name, compiled, functions, sites, compasec=("eq_rr", "ne_zero"))
        execute_cases(name, image, symbols, "NZDC+NEMESEC", functions)

        threshold_ir = out / "threshold.ll"
        run([args.llvm_bin / "clang", "--target=riscv32", "-march=rv32i", "-mabi=ilp32", "-mno-relax", "-O1",
             "-ffreestanding", "-emit-llvm", "-S", HERE.parent / "predicate_alignment/threshold.c", "-o", threshold_ir], "threshold-frontend")
        selected = tuple("threshold" + str(count) for count in range(4))
        for mask in ("orig", "orig+dmr"):
            name = "RACFED-threshold-" + mask
            compiled, functions, image, symbols = compile_link(name, flags("RACFED", None, "primary", ",".join(selected), mask=mask), threshold_ir)
            sites = hardware_sites(name, compiled, functions, "RACFED", None, "primary", selected)
            regions = ledger(name, compiled, functions, sites, mask=mask)
            for count, function in enumerate(selected):
                site, = [row for row in sites if row["function"] == function]
                region = regions[(function, site["block"])]
                eligible = count + int(mask == "orig+dmr")
                require(int(region["selected"]) == eligible and int(region["eligible"]) == (eligible - 1 if eligible >= 3 else 0),
                        name + "/" + function + ": exact legacy threshold counts original loads and selected raw XOR")
                helper = [row for row in baseline.records(compiled.stdout, "COMPAS_RACFED_UI")
                          if row["function"] == function and row["category"] == "predicate"]
                require(len(helper) == int(mask == "orig+dmr" and count >= 2),
                        name + "/" + function + ": XOR reduction affects threshold without inventing updates")
                for right in (3, 4):
                    execution = reference.execute(image, symbols[function], (3, right, symbols["source_value"], symbols["destination"]), False)
                    require(execution["return_value"] == right
                            and [row["value"] for row in execution["stores"] if row["address"] == symbols["destination"]] == ([3] if right == 3 else [4, 4]),
                            name + "/" + function + ": threshold fixture literal return and stores")

        for mnemonic in ("nz.sacc.i", "ez.sacc.i"):
            for ordinal, literals in enumerate(((-32, 31), (31, -32), (-1, 1))):
                for rd, rs1 in ((5, 29), (5, 5), (0, 29), (5, 0)):
                    assembly = f"{mnemonic} x{rd}, x{rs1}, {literals[0]}, {literals[1]}\n"
                    name = f"boundary-{mnemonic}-{ordinal}-{rd}-{rs1}"
                    result = run([args.llvm_bin / "llvm-mc", "-triple=riscv32", "-mattr=+xpred,-c", "-show-encoding"], name, input_text=assembly)
                    encoding = re.search(r"encoding:\s*\[([^]]+)\]", result.stdout)
                    word = int.from_bytes(bytes(int(value.strip(), 0) for value in encoding[1].split(",")), "little")
                    canonical = literals if mnemonic.startswith("nz.") else literals[::-1]
                    expected_word = 0x2b | (rd << 7) | (rs1 << 15) | ((canonical[0] & 63) << 20) | ((canonical[1] & 63) << 26)
                    require(word == expected_word, name + ": independent encoding and zero alias polarity")
                    for condition in (0, 1, 0xffffffff):
                        registers = [0] * 32
                        registers[rd], registers[rs1], registers[0] = 0xffffffff, condition, 0
                        before = registers.copy()
                        reference.step(word, 0x1000, registers, {})
                        predicate = before[rs1] != 0 if mnemonic.startswith("nz.") else before[rs1] == 0
                        expected = before.copy()
                        if rd:
                            expected[rd] = (before[rd] + literals[0 if predicate else 1]) & reference.MASK
                        require(registers == expected, name + f": old destination, source aliasing and full-width condition {condition:#x}")
        for ordinal, literals in enumerate(((-33, 0), (32, 0), (0, -33), (0, 32))):
            result = run([args.llvm_bin / "llvm-mc", "-triple=riscv32", "-mattr=+xpred,-c", "-filetype=null"],
                         "invalid-boundary-" + str(ordinal), expect=None,
                         input_text="nz.sacc.i t0, t4, " + ", ".join(map(str, literals)) + "\n")
            require(result.returncode != 0 and "[-32, 31]" in result.stderr, f"sacc {literals}: out-of-range literal rejected")
        for monitor in ("RASM", "RACFED"):
            unsupported = HERE.parent / "predicate_alignment/unsupported.ll"
            standard = flags(monitor, "NZDC+NEMESEC", "shadow", "target_check")
            invalid = [
                ("missing-predicate", ["-" + monitor + "=target_check", "-" + monitor + "-predicate-xpred=target_check"], "predicate", []),
                ("missing-monitor", [item for item in standard if not item.startswith("-" + monitor + "=")], "requires", []),
                ("missing-dmr", flags(monitor, None, "shadow", "target_check"), "NZDC", []),
                ("allocation-exhausted", [*standard, "-" + monitor + "-predicate-psp-max-attempts=0"], "allocation", []),
                ("disabled", standard, "XPred", ["-mattr=-xpred,-c,-relax"]),
                ("compressed", standard, "RV32", ["-mattr=+xpred,+c,-relax"]),
                ("rv64", standard, "RV32", ["-march=riscv64", "-mtriple=riscv64-unknown-elf", "-target-abi=lp64"]),
                ("fgs", [*standard, "-FGS=target_check"], "CGS", []),
                ("excluded-orig", flags(monitor, "NZDC+NEMESIS", "shadow", "target_check", mask="dmr+balance"), "orig", []),
            ]
            for case, options, diagnostic, target in invalid:
                name = monitor + "-reject-" + case
                failed = run([*backend, *options, *target, unsupported, "-o", out / (name + ".s")], name, expect=None)
                require(failed.returncode != 0 and diagnostic.lower() in failed.stderr.lower(), name + ": explicit diagnostic, no silent fallback")
        result = dict(status="passed", checks=checks, executions=executions,
                      scope="machine verification, linked ledger, bounded allocation and independent fault-free execution; no skip proof or timing claim")
    except (AssertionError, KeyError, IndexError, TypeError, OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        result = dict(status="failed", error=str(error), checks=checks, executions=executions)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
