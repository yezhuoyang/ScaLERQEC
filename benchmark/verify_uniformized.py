"""Save the local verification record after all validation runs finish."""

import hashlib
import importlib.metadata
import json
import platform
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmark.summarize_uniformized import run


def junit(path):
    suites = ET.parse(path).getroot().iter("testsuite")
    totals = {k: 0 for k in ["tests", "failures", "errors", "skipped"]}
    for suite in suites:
        for k in totals:
            totals[k] += int(suite.get(k, 0))
    assert totals["tests"] and not totals["failures"] and not totals["errors"]
    totals["passed"] = totals["tests"] - totals["skipped"]
    return totals


def main():
    run()
    coverage = json.loads(Path("build/uniformized-final-coverage.json").read_text())
    modules = {
        Path(path).name: row["summary"]["percent_covered"]
        for path, row in coverage["files"].items()
    }
    assert modules["uniformized.py"] >= 95 and modules["confidence.py"] >= 95
    source_hashes = {
        name: hashlib.sha256(
            (Path("src/scalerqec/Stratified") / name).read_bytes()
        ).hexdigest()
        for name in [
            "uniformized.py",
            "confidence.py",
            "general_noise.py",
            "noise_polynomial.py",
        ]
    }
    for path in Path("experiment_results/uniformized_final_validation").glob("*.json"):
        row = json.loads(path.read_text())
        if "spec" not in row:
            continue
        for name in ["uniformized.py", "confidence.py", "noise_polynomial.py"]:
            key = next(
                k
                for k in row["source_sha256"]
                if k.replace("\\", "/").endswith("/" + name)
            )
            assert row["source_sha256"][key] == source_hashes[name]
    verification = {
        "full_suite": junit("build/uniformized-final-tests.xml"),
        "stim_1_16_compatibility": junit("build/uniformized-stim116.xml"),
        "module_statement_coverage_percent": modules,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["stim", "numpy", "scipy", "pymatching", "stimbposd", "pytest"]
        },
        "source_sha256": source_hashes,
        "post_test_change": "general_noise.py imports reordered by Ruff only; profiler, confidence bounds, and polynomial implementation unchanged after final tests",
        "manual_checks": [
            "Ruff passed for changed Python modules and benchmark scripts",
            "actionlint passed for pr-validation.yml",
        ],
        "native_scope": "No C++ source changes in this turn; full Python suite includes native-binding regressions. Linux sanitizer workflow was not executed locally.",
        "publication": "Local experimental implementation; not uploaded to PyPI",
    }
    target = Path("experiment_results/uniformized_final_validation/verification.json")
    target.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
