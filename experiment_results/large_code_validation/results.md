# Large-code validation: recorded results

Generated from the predeclared matrix and supplementary oracle runs. Completion is not an accuracy pass. Bicycle case IDs retain the historical `z` token, but the stored circuits are X memory with only logical k−1 defined.

| Family | Cases | Completed | Failed / timed out | Audits passed | MC points | Accuracy met |
|---|---:|---:|---:|---:|---:|---:|
| surface | 26 | 25 | 0 / 1 | 26 | 75 | 0 |
| stabir | 6 | 6 | 0 / 0 | 6 | 18 | 0 |
| color | 3 | 3 | 0 / 0 | 3 | 9 | 0 |
| hgp | 12 | 12 | 0 / 0 | 12 | 36 | 2 |
| bicycle | 9 | 5 | 2 / 2 | 7 | 19 | 0 |

Monte Carlo: 9,498,400 shots across 157 points. Accepted accuracy contradictions: 0. Interval overlap with an unresolved [0,1]-like profile interval is uninformative.

## Distance 13, 13 rounds

| Case | Run / profile status | MC failures / shots at p=0.001, 0.003, 0.01 |
|---|---|---|
| surface_13_z_uniform_single | completed / budget_exhausted | 0 / 100,000; 38 / 100,000; 34,979 / 100,000 |
| surface_13_z_nonuniform_depolarizing | completed / budget_exhausted | 0 / 100,000; 4 / 100,000; 9,004 / 100,000 |
| surface_13_z_biased_pauli | completed / budget_exhausted | 0 / 100,000; 0 / 100,000; 112 / 100,000 |
| surface_13_x_nonuniform_depolarizing | completed / budget_exhausted | 0 / 100,000; 5 / 100,000; 13,631 / 100,000 |
| surface_13_x_biased_pauli | completed / budget_exhausted | 0 / 100,000; 2 / 100,000; 12,022 / 100,000 |
| surface_13_z_correlated_else | process_timeout / budget_exhausted | unavailable |
| surface_13_z_phenomenological | completed / budget_exhausted | 0 / 100,000; 0 / 100,000; 6 / 100,000 |
| stabir_13_z_SD6 | completed / budget_exhausted | 0 / 100,000; 34 / 100,000; 12,868 / 100,000 |
| stabir_13_z_SI1000 | completed / budget_exhausted | 0 / 100,000; 77 / 100,000; 34,191 / 100,000 |

### Longer distance-13 budget

surface_13_z_nonuniform_depolarizing, 300-second profiling budget: completed, profile budget_exhausted, 13,568 random samples. This is a separate follow-up; the original 30-second run remains recorded.

| p | Estimate | Lower | Upper | Accuracy met |
|---:|---:|---:|---:|---|
| 0.001 | 0 | 0 | 0.203056 | False |
| 0.003 | 0 | 0 | 0.998911 | False |
| 0.01 | 0 | 0 | 1 | False |

## Independent bounded HGP58 reference

| Decoder | Noise | Profile status | p points verified to ±10% | Random / exact histories |
|---|---|---|---:|---:|
| BP+OSD0 | biased_pauli | accuracy_met | 3/3 | 16,640 / 60,379 |
| BP+OSD0 | correlated_else | accuracy_met | 3/3 | 4,096 / 15,110 |
| BP+OSD0 | nonuniform_depolarizing | accuracy_met | 3/3 | 16,384 / 60,379 |
| BP+OSD0 | uniform_single | accuracy_met | 3/3 | 0 / 175 |
| BP+OSD4 with exact single-Pauli lookup | biased_pauli | accuracy_met | 3/3 | 16,384 / 60,379 |
| BP+OSD4 with exact single-Pauli lookup | correlated_else | accuracy_met | 3/3 | 4,096 / 15,110 |
| BP+OSD4 with exact single-Pauli lookup | nonuniform_depolarizing | budget_exhausted | 2/3 | 16,384 / 60,379 |
| BP+OSD4 with exact single-Pauli lookup | uniform_single | accuracy_met | 3/3 | 4,096 / 15,052 |

### Additional checks on accepted main-matrix profiles

Each accepted main-matrix profile was checked against the independent weight-at-most-two oracle at its actual p grid. 3 of 6 points were verified to 10%; there were 0 interval contradictions. A wide oracle enclosure is inconclusive, not a failure or a pass.

## Cases that did not complete

- **surface_13_z_correlated_else**: process_timeout, phase `profile_complete`.
- **bicycle_108_z_nonuniform_depolarizing**: process_timeout, phase `mc_2_complete`.
- **bicycle_108_z_biased_pauli**: process_timeout, phase `mc_2_complete`.
- **bicycle_144_z_nonuniform_depolarizing**: failed, phase `responses_compiled`. Conditional weight probability underflowed.
- **bicycle_144_z_biased_pauli**: failed, phase `responses_compiled`. Conditional weight probability underflowed.

## Component checks

Independent general-noise replay: 400 histories. Independent HGP CSS commutation: 16,163 outcome columns. Native replay: 64 histories in 4 cases.

Conditional moments: 8 large-code cases, 16,384 draws; largest difference 3.186 true standard errors. This checks selected moments, not the entire joint distribution.

See [method, limits, and reproduction](../../docs/large_code_validation.md).
