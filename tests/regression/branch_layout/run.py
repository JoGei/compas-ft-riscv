#!/usr/bin/env python3
"""Compile and verify a taken branch to the physically adjacent basic block."""

import argparse
import json
from pathlib import Path
import resource
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import runner


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llc", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path,
                        default=HERE.parents[2] / "build/tests/branch_layout")
    parser.add_argument("--build-timeout", type=float, default=60)
    args = parser.parse_args()
    if args.build_timeout <= 0:
        parser.error("--build-timeout must be positive")
    args.build_dir = args.build_dir.resolve()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    base = [args.llc.resolve(), "-mtriple=riscv32-unknown-elf",
            "-mattr=+m,-c,-relax", "-target-abi=ilp32", "-O1",
            "-verify-machineinstrs", "-start-after=livedebugvalues"]
    profiles = [("none", "none")]
    profiles += [(mode, cfp) for mode in ("NZDC+NEMESIS", "NZDC+NEMESEC")
                 for cfp in ("none", "RASM", "RACFED")]
    results = []
    for mode, cfp in profiles:
        directory = args.build_dir / (mode + "-" + cfp)
        directory.mkdir(parents=True, exist_ok=True)
        flags = [] if mode == "none" else ["-" + mode + "=branch_layout"]
        if cfp != "none":
            flags.append("-" + cfp + "=branch_layout:orig+dmr")
        if cfp == "RACFED":
            flags.append("-RACFED-compasec=branch_layout")
        result = dict(mode=mode, cfp=cfp, status="PASS",
                      scope="compilation and LLVM machine verification; no target execution")
        try:
            assembly = directory / "output.s"
            runner.command([*base, *flags, HERE / "branch_layout.mir", "-o", assembly],
                           directory, "compile", args.build_timeout)
            if "branch_layout:" not in assembly.read_text():
                raise runner.TestError("missing compiled fixture")
        except (OSError, runner.TestError) as error:
            result.update(status="ERROR", detail=str(error))
        results.append(result)
        print(mode, cfp, result["status"], flush=True)
    (args.build_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any(result["status"] != "PASS" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
