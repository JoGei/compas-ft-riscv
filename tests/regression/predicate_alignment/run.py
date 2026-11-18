#!/usr/bin/env python3
"""Check the shared software-predicate contract for RASM and RACFED."""

import argparse
import json
from pathlib import Path
import re
import resource
import subprocess


HERE = Path(__file__).resolve().parent
FUNCTIONS = ("eq_rr", "ne_rr", "eq_zero", "ne_zero", "tiny")
SELECTED = ",".join(FUNCTIONS)
PRIMARY_TO_SHADOW = {
    "zero": "s11", "ra": "t2", "sp": "s1", "gp": "s6", "tp": "t3",
    "t0": "t4", "t1": "t5", "s0": "t6", "a0": "s8", "a1": "s9",
    "a2": "s10", "a3": "s5", "a4": "s4", "a5": "s7", "a6": "s3",
    "a7": "s2",
}


def records(output, tag):
    return [dict(item.split("=", 1) for item in line.split()[1:])
            for line in output.splitlines() if line.startswith(tag + " ")]


def linked_functions(disassembly):
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
            current.append((instruction[2], [x.strip() for x in operands.split(",")],
                            int(instruction[1], 16)))
    return functions


def predicate_commits(body):
    return [i for i, item in enumerate(body)
            if item[:2] == ("add", ["t0", "t0", "t4"])]


def flags(monitor, dmr, source, selection=SELECTED, predicate_selection=None,
          compasec=True, mask=None):
    chosen = selection if predicate_selection is None else predicate_selection
    coverage = ",".join(name + ":" + mask for name in selection.split(",")) if mask else selection
    result = ["-" + monitor + "=" + coverage]
    if dmr:
        result.append("-" + dmr + "=" + selection)
    if monitor == "RACFED" and compasec:
        result.append("-RACFED-compasec=" + selection)
    # Both shadow selectors self-enable predication. This also exercises the
    # historical RACFED-predicate-shadow invocation without a primary selector.
    selector = monitor + "-predicate" + ("-shadow" if source == "shadow" else "")
    result.append("-" + selector + "=" + chosen)
    if monitor == "RASM":
        result.append("-RASM-predicate-unpadded=" + chosen)
    return result


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

    def compile_run(name, options, ir):
        obj, elf = out / (name + ".o"), out / (name + ".elf")
        compiled = run([*backend, *options, "-filetype=obj", ir, "-o", obj], name + "-codegen")
        require("Bad machine code" not in compiled.stderr, name + ": machine verifier")
        run([args.cc, "-march=rv32i", "-mabi=ilp32", "-mno-relax", "-nostdlib",
             "-static", "-no-pie", "-Wl,--build-id=none", "-Wl,--no-relax",
             "-Wl,-Ttext=0x10000", HERE.parents[1] / "support/start.S", obj,
             "-o", elf], name + "-link")
        dump = run([args.objdump, "-d", "--no-show-raw-insn", "-M", "no-aliases", elf], name + "-disassembly")
        execution = run([args.qemu, "-strace", elf], name + "-execute", expect=42)
        require(re.findall(r"\bexit(?:_group)?\((-?\d+)\)", execution.stderr) == ["42"]
                and execution.stdout == "", name + ": literal functional oracle")
        return compiled, linked_functions(dump.stdout)

    def check_predicate(name, body, dmr, source):
        commits = predicate_commits(body)
        require(len(commits) == 1, name + ": one final signature ADD for the application branch")
        index = commits[0]
        require(index + 1 < len(body) and body[index + 1][0] in ("beq", "bne"),
                name + ": signature ADD immediately precedes the primary branch")
        branch = body[index + 1]
        # Remove RACFED's interleaved signature updates, keeping the five
        # computations that prepare the selected adjustment in scratch.
        helper = []
        for item in reversed(body[:index]):
            if item[0] == "addi" and item[1][:2] == ["t0", "t0"]:
                continue
            helper.append(item)
            if len(helper) == 5:
                break
        helper.reverse()
        require([item[0] for item in helper] ==
                ["xor", "sltiu" if branch[0] == "beq" else "sltu", "sub", "andi", "addi"],
                name + ": complete software selection before the signature commit")
        operands = branch[1][:2]
        wanted = ([PRIMARY_TO_SHADOW[reg] for reg in operands]
                  if source == "shadow" else operands)
        require(helper[0][1] == ["t4", *wanted], name + ": predicate uses " + source + " operands")
        if name.endswith("zero"):
            require("zero" in operands and ("s11" if source == "shadow" else "zero") in wanted,
                    name + ": architectural zero uses the selected primary/shadow stream")
        base, difference = int(helper[4][1][2], 0), int(helper[3][1][2], 0)
        require(base != 0 and base + difference != 0 and difference != 0,
                name + ": distinct nonzero adjustments for both successors")
        if dmr:
            redundant = [PRIMARY_TO_SHADOW[reg] for reg in operands]
            shadow_branches = [item for item in body if item[0] in ("beq", "bne")
                               and item[1][:2] in (redundant, redundant[::-1])]
            expected = dmr != "NZDC" and source == "primary"
            require(bool(shadow_branches) == expected,
                    name + ": shadow direction branches " + ("retained" if expected else "absent"))

    def check_ledger(name, compiled, selected, functions=None, compasec=None, mask="all"):
        compasec = set(selected) if compasec is None else set(compasec)
        updates = records(compiled.stdout, "COMPAS_RACFED_UI")
        regions = records(compiled.stdout, "COMPAS_RACFED_REGION")
        sites = records(compiled.stdout, "COMPAS_RACFED_PREDICATE_SITE")
        require({row["function"] for row in sites} == set(selected),
                name + ": only selected functions have predicate sites")
        predicate = [row for row in updates if row["category"] == "predicate"]
        accounted = set(selected) if mask != "orig" else compasec
        require({row["function"] for row in predicate} == accounted,
                name + ": predicate helper accounting follows the selected policy and coverage mask")
        for function in selected:
            rows = [row for row in predicate if row["function"] == function]
            if function in compasec:
                require(len(rows) == 4, name + "/" + function + ": four post-instruction helper updates")
            elif mask == "orig":
                require(not rows, name + "/" + function + ": orig-only ledger excludes generated helpers")
            else:
                require(3 <= len(rows) <= 4,
                        name + "/" + function + ": helper accounting preserves first-position zero exemption")
                if len(rows) == 3:
                    require(not any(row["opcode"] == "XOR" for row in rows),
                            name + "/" + function + ": only the first helper may have the zero exemption")
            if functions is not None:
                body = functions[function]
                commit = predicate_commits(body)[0]
                helper_indices = []
                for index in range(commit - 1, -1, -1):
                    if body[index][0] == "addi" and body[index][1][:2] == ["t0", "t0"]:
                        continue
                    helper_indices.append(index)
                    if len(helper_indices) == 5:
                        break
                helper_indices.reverse()
                offset = 1 if function in compasec else -1
                for row in rows:
                    positions = [index for index in helper_indices[:4]
                                 if body[index][0] == row["opcode"].lower()]
                    require(len(positions) == 1 and row["placement"] == ("after" if offset == 1 else "before")
                            and body[positions[0] + offset][:2] ==
                            ("addi", ["t0", "t0", row["delta"]]),
                            name + "/" + function + ": linked " + row["opcode"] + " update has policy-selected placement")
                if mask == "orig" and function not in compasec:
                    require(all(right == left + 1 for left, right in zip(helper_indices, helper_indices[1:])),
                            name + "/" + function + ": excluded helpers have no interleaved monitor updates")
        for region in regions:
            ledger = [row for row in updates
                      if (row["function"], row["block"]) == (region["function"], region["block"])]
            require(len(ledger) == int(region["eligible"])
                    and sum(int(row["delta"]) for row in ledger) == int(region["sum"])
                    and all(int(row["delta"]) != 0
                            and int(row["after"]) - int(row["before"]) == int(row["delta"])
                            for row in ledger),
                    name + ": " + region["function"] + "/" + region["block"] + " complete nonzero ledger")
            if region["function"] in selected and region["function"] not in compasec:
                candidates, emitted, zero = (int(region[key]) for key in ("selected", "eligible", "zero_initial"))
                expected = candidates - zero if candidates >= 3 else 0
                require(region["placement"] == "before" and zero in (0, 1)
                        and emitted == expected and (candidates >= 3 or int(region["sum"]) == 0)
                        and all(row["placement"] == "before" for row in ledger),
                        name + ": " + region["function"] + "/" + region["block"] + " retains legacy threshold and zero exemption")
        region_index = {(row["function"], row["block"]): row for row in regions}
        for site in sites:
            source = region_index[(site["function"], site["block"])]
            for arm, successor in (("true", "taken"), ("false", "fallthrough")):
                destination = region_index[(site["function"], site[successor])]
                expected = int(destination["x"]) + int(destination["y"]) - int(source["x"]) - int(source["sum"])
                require(int(site[arm]) == expected and expected != 0,
                        name + "/" + site["function"] + ": " + arm + " adjustment includes exactly the policy ledger")
        return regions, sites

    try:
        ir = out / "functional.ll"
        run([args.clang, "--target=riscv32", "-march=rv32i", "-mabi=ilp32",
             "-mno-relax", "-O1", "-ffreestanding", "-fno-optimize-sibling-calls",
             "-emit-llvm", "-S", HERE / "functional.c", "-o", ir], "frontend")
        _, reference = compile_run("reference", [], ir)
        for monitor in ("RASM", "RACFED"):
            for dmr in (None, "NZDC", "NZDC+NEMESIS", "NZDC+NEMESEC"):
                for source in (("primary", "shadow") if dmr else ("primary",)):
                    name = "-".join((monitor, dmr or "standalone", source))
                    compiled, functions = compile_run(name, flags(monitor, dmr, source), ir)
                    for function in FUNCTIONS:
                        check_predicate(name + "/" + function, functions[function], dmr, source)
                    require(not predicate_commits(functions["ordinary"])
                            and [item[0] for item in functions["ordinary"]]
                            == [item[0] for item in reference["ordinary"]],
                            name + ": unselected function remains uninstrumented")
                    if monitor == "RACFED":
                        check_ledger(name, compiled, FUNCTIONS, functions)

            name = monitor + "-mixed"
            compiled, functions = compile_run(name, flags(monitor, "NZDC+NEMESEC", "shadow",
                                                         predicate_selection="eq_rr"), ir)
            check_predicate(name + "/eq_rr", functions["eq_rr"], "NZDC+NEMESEC", "shadow")
            require(all(not predicate_commits(functions[fn]) for fn in FUNCTIONS[1:]),
                    name + ": predicate selection does not leak to later monitored functions")
            primary_regs = set(PRIMARY_TO_SHADOW) - {"t0", "ra"}
            for function in FUNCTIONS[1:]:
                body = functions[function]
                application = [item for item in body if item[0] in ("beq", "bne")
                               and all(reg in primary_regs for reg in item[1][:2])]
                require(len(application) == 1, name + "/" + function + ": one original branch")
                redundant = [PRIMARY_TO_SHADOW[reg] for reg in application[0][1][:2]]
                require(any(item[0] in ("beq", "bne") and item[1][:2] in (redundant, redundant[::-1])
                            for item in body),
                        name + "/" + function + ": later nonpredicate function retains direction checks")
            if monitor == "RACFED":
                check_ledger(name, compiled, ("eq_rr",), functions)

            # Invalid requests must fail with a useful diagnostic rather than
            # silently retaining duplicated checks or choosing primary inputs.
            options = flags(monitor, None, "primary", "target_check")
            negative = [
                ("missing-monitor", [x for x in options if not x.startswith("-" + monitor + "=")], "requires " + monitor, HERE / "unsupported.ll", []),
                ("missing-dmr", flags(monitor, None, "shadow", "target_check"), "NZDC", HERE / "unsupported.ll", []),
                ("fgs", [*flags(monitor, "NZDC", "shadow", "target_check"), "-FGS=target_check"], "CGS", HERE / "unsupported.ll", []),
                ("compressed", options, "RV32", HERE / "unsupported.ll", ["-mattr=+c,-relax"]),
                ("rv64", options, "RV32", HERE / "unsupported.ll", ["-march=riscv64", "-mtriple=riscv64-unknown-elf", "-target-abi=lp64"]),
            ]
            negative.append(("excluded-orig", ["-" + monitor + "=target_check:dmr+balance",
                                                "-NZDC+NEMESIS=target_check",
                                                "-" + monitor + "-predicate-shadow=target_check"],
                             "orig", HERE / "unsupported.ll", []))
            for case, options, diagnostic, source, target_options in negative:
                name = monitor + "-reject-" + case
                failed = run([*backend, *options, *target_options, source, "-o", out / (name + ".s")], name, expect=None)
                require(failed.returncode != 0 and diagnostic in failed.stderr,
                        name + ": explicit invalid-configuration diagnostic")

        legacy_zero_exercised = False
        for mask in ("orig", "orig+dmr", "all"):
            for dmr in (None, "NZDC", "NZDC+NEMESIS", "NZDC+NEMESEC"):
                for source in (("primary", "shadow") if dmr else ("primary",)):
                    name = "-".join(("RACFED-legacy", mask, dmr or "standalone", source))
                    options = flags("RACFED", dmr, source, compasec=False, mask=mask)
                    compiled, functions = compile_run(name, options, ir)
                    for function in FUNCTIONS:
                        check_predicate(name + "/" + function, functions[function], dmr, source)
                    regions, sites = check_ledger(name, compiled, FUNCTIONS, functions, compasec=(), mask=mask)
                    categories = {row["category"] for row in records(compiled.stdout, "COMPAS_RACFED_UI")}
                    permitted = {"orig"} | ({"dmr", "predicate"} if mask != "orig" else set())
                    if mask == "all":
                        permitted.add("balance")
                    require(categories <= permitted, name + ": instruction coverage mask is preserved")
                    legacy_zero_exercised |= any(int(row["zero_initial"]) == 1 for row in regions)
                    if mask == "orig" and dmr is None:
                        require(any(0 < int(row["selected"]) < 3 and int(row["eligible"]) == 0
                                    for row in regions), name + ": below-threshold regions emit no intra-block updates")
                        require(any(row["function"] == "tiny" and int(row["sum"]) == 0 for row in sites),
                                name + ": a predicate branch with zero intra-block sum remains valid")
        require(legacy_zero_exercised, "legacy matrix exercises the first-position zero exemption")

        name = "RACFED-mixed-policies"
        options = [*flags("RACFED", "NZDC+NEMESEC", "shadow", compasec=False),
                   "-RACFED-compasec=eq_rr,ne_zero"]
        compiled, functions = compile_run(name, options, ir)
        for function in FUNCTIONS:
            check_predicate(name + "/" + function, functions[function], "NZDC+NEMESEC", "shadow")
        check_ledger(name, compiled, FUNCTIONS, functions, compasec=("eq_rr", "ne_zero"))
        require({row["function"] for row in records(compiled.stdout, "COMPAS_RACFED_POLICY")
                 if row.get("placement") == "after"} == {"eq_rr", "ne_zero"},
                name + ": explicit CompaSeC selection does not leak into intervening or later legacy functions")

        threshold_ir = out / "threshold.ll"
        run([args.clang, "--target=riscv32", "-march=rv32i", "-mabi=ilp32",
             "-mno-relax", "-O1", "-ffreestanding", "-fno-optimize-sibling-calls",
             "-emit-llvm", "-S", HERE / "threshold.c", "-o", threshold_ir], "threshold-frontend")
        threshold_functions = tuple("threshold" + str(count) for count in range(4))
        compiled, functions = compile_run(
            "RACFED-legacy-threshold",
            flags("RACFED", None, "primary", ",".join(threshold_functions), compasec=False, mask="orig"),
            threshold_ir)
        regions, sites = check_ledger("RACFED-legacy-threshold", compiled, threshold_functions,
                                      functions, compasec=(), mask="orig")
        region_index = {(row["function"], row["block"]): row for row in regions}
        for count, function in enumerate(threshold_functions):
            check_predicate("RACFED-legacy-threshold/" + function, functions[function], None, "primary")
            site, = [row for row in sites if row["function"] == function]
            region = region_index[(function, site["block"])]
            body = functions[function]
            commit, = predicate_commits(body)
            loads = [index for index, item in enumerate(body[:commit]) if item[0] == "lw"]
            require(len(loads) == count and int(region["selected"]) == count
                    and int(region["eligible"]) == (2 if count == 3 else 0),
                    function + ": exact linked load count independently covers the 0/1/2/3 threshold boundary")
            for index, load in enumerate(loads):
                previous = body[load - 1]
                has_update = previous[0] == "addi" and previous[1][:2] == ["t0", "t0"]
                require(has_update == (count == 3 and index > 0),
                        function + ": linked load " + str(index) + " retains legacy update eligibility")
        result = dict(status="passed", checks=checks,
                      scope="fault-free equivalence, machine verification, source selection, DMR direction-check composition and RACFED ledger; no fault-coverage claim")
    except (AssertionError, KeyError, IndexError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        result = dict(status="failed", error=str(error), checks=checks)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
