"""Independently bound each main-matrix profile that claimed accuracy."""

import json

from benchmark.hgp_bounded_oracle import bounded_oracle
from benchmark.large_code_cases import PS, cases, make_circuit
from benchmark.large_code_validation import OUT, decoder_for, write_json


def main():
    output = OUT / "accepted_checks"
    output.mkdir(parents=True, exist_ok=True)
    for spec in cases():
        result = json.loads((OUT / (spec["name"] + ".json")).read_text())
        if result.get("adaptive", {}).get("status") != "accuracy_met":
            continue
        if spec["family"] != "hgp":
            raise NotImplementedError("No bounded CSS reference for this family")
        decoder, name = decoder_for(spec, make_circuit(spec, PS[1]))
        oracle = bounded_oracle(spec, decoder, PS)
        for point, estimate in zip(oracle["points"], result["adaptive"]["estimates"]):
            point["estimate"] = estimate["ler"]
            point["accuracy_verified_by_oracle"] = bool(
                estimate["ler"] / 1.1 <= point["lower"]
                and point["upper"] <= estimate["ler"] / 0.9
            )
            point["contradiction"] = bool(
                estimate["upper"] < point["lower"] or estimate["lower"] > point["upper"]
            )
        write_json(
            output / (spec["name"] + ".json"),
            {"spec": spec, "decoder": name, "oracle": oracle},
        )
        print(
            spec["name"],
            [p["accuracy_verified_by_oracle"] for p in oracle["points"]],
            flush=True,
        )


if __name__ == "__main__":
    main()
