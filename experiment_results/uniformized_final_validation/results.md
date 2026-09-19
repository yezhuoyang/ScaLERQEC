# Uniformized polynomial validation

16 distinct cases; 2 certify the whole requested p grid; 12/48 individual points meet the statistical 10% accuracy target.

Independent references exist for 44/48 points. 0 profile/reference interval pairs do not overlap. The reference intervals independently bound relative error by 10% at 12 points. Wide overlapping intervals are not evidence of 10% accuracy.

Intervals are 99% simultaneous within one declared p grid and over its sequential checks. They are not a simultaneous 99% assertion across all benchmark cases. The selected run is the final implementation where available, and the declared longer precision run for two HGP cases. Four additional cases retain the initial KL-only run. Every run, including the interrupted first bicycle attempt, is listed in summary.json.

| Case | Shots | Profile seconds | Points certified | Whole grid |
|---|---:|---:|---:|---|
| bicycle_144_z_biased_pauli | 7 | 33.09 | 0/3 | budget_exhausted |
| bicycle_144_z_nonuniform_depolarizing | 7 | 41.05 | 0/3 | budget_exhausted |
| bicycle_72_z_nonuniform_depolarizing | 160 | 22.74 | 0/3 | budget_exhausted |
| color_7_xyz_nonuniform_depolarizing | 3,232 | 20.18 | 0/3 | budget_exhausted |
| hgp_180_z_nonuniform_depolarizing | 16,799 | 30.06 | 0/3 | budget_exhausted |
| hgp_245_z_biased_pauli | 15,199 | 30.05 | 0/3 | budget_exhausted |
| hgp_58_z_biased_pauli | 173,503 | 30.00 | 0/3 | budget_exhausted |
| hgp_58_z_correlated_else | 479,743 | 56.04 | 3/3 | accuracy_met |
| hgp_58_z_nonuniform_depolarizing | 165,951 | 30.01 | 0/3 | budget_exhausted |
| hgp_58_z_uniform_single | 982,111 | 125.70 | 3/3 | accuracy_met |
| stabir_13_z_SD6 | 210,943 | 60.16 | 1/3 | budget_exhausted |
| stabir_13_z_SI1000 | 117,247 | 60.05 | 1/3 | budget_exhausted |
| surface_13_x_biased_pauli | 200,191 | 60.05 | 1/3 | budget_exhausted |
| surface_13_z_nonuniform_depolarizing | 162,303 | 60.20 | 1/3 | budget_exhausted |
| surface_3_z_nonuniform_depolarizing | 1,000,000 | 15.95 | 1/3 | budget_exhausted |
| surface_7_z_nonuniform_depolarizing | 906,752 | 40.00 | 1/3 | budget_exhausted |

## One polynomial across three p values

The HGP-58 correlated-noise run stopped automatically after 479,743 samples. Each independent Stim comparison below uses 250,000 preselected shots. Both experiments use the same fixed decoder.

| p | Polynomial LER | Profile interval | Independent Stim LER |
|---:|---:|---:|---:|
| 0.001 | 0.01427189 | [0.01298686, 0.01556995] | 0.01424800 |
| 0.003 | 0.04194485 | [0.03946694, 0.04444760] | 0.04226000 |
| 0.01 | 0.13078963 | [0.12474787, 0.13687230] | 0.13104400 |

## Limits

Surface distances 3, 7, and 13 are represented. Distance 13 includes both memory bases plus StabIR SD6 and SI1000. QLDPC cases include HGP 58/180/245 and bicycle 72/144; a color-code case is also included. The tests additionally compare seven noise families on small codes to independent exact Pauli calculations.

HGP examples here use ideal encoding/syndrome measurement, with all defined logical outputs. Bicycle fixtures define one logical observable, despite their multiple output slots. The historical bicycle filenames containing z label X-memory fixtures. These are the same fixed decoders used by the references; the baseline HGP decoder is not a claim of optimal code performance.

Dense response compilation is still separate from the new sparse sampler. The sampler avoids conditional-weight suffix tables, but large BP+OSD decoding remains slow. Profile seconds include sampler setup and decoding, while circuit response compilation and the independent replay audit are recorded separately. Budgets are cooperative between batches, so one call may overrun. Concurrent jobs make timings unsuitable for a controlled speedup claim.

Low-p distance-13 values remain uncertified. Extremely small importance point estimates can simply reflect unobserved relevant failures. Uncertified curves must not be used as accuracy-validated predictions. The public exporter rejects them by default.

## Reproduce

```text
python -m benchmark.uniformized_validation CASE --seconds 60 --fresh-shots 4096 --output NEW_DIRECTORY
python -m benchmark.uniformized_references hgp_58_z_correlated_else hgp_58_z_uniform_single --shots 250000 --output NEW_REFERENCE_DIRECTORY
python -m benchmark.summarize_uniformized
```

Original result directories are never overwritten by the experiment runner. Each run records source and circuit hashes, decoder description, independent replay outcomes, actual budgets, and a reload-verified polynomial.
