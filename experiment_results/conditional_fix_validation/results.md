# Conditional sampler fix reruns

Every listed run used the same fixed decoder and circuit as its historical reference (circuit hash checked). Most comparisons reuse those independent MC samples; fresh d13 MC is also saved. Unresolved intervals are not accuracy passes.

| Case | Random shots | Profile seconds | Status |
|---|---:|---:|---|
| surface_13_z_nonuniform_depolarizing | 705,792 | 301.5 | budget_exhausted: Time budget exhausted before establishing accuracy. |
| surface_13_x_biased_pauli | 7,680 | 30.3 | budget_exhausted: Time budget exhausted before establishing accuracy. |
| stabir_13_z_SD6 | 6,144 | 30.3 | budget_exhausted: Time budget exhausted before establishing accuracy. |
| stabir_13_z_SI1000 | 512 | 33.4 | budget_exhausted: Time budget exhausted before establishing accuracy. |
| hgp_58_z_correlated_else | 15,360 | 6.6 | accuracy_met: Simultaneous requested-grid accuracy target met. |
| hgp_180_z_nonuniform_depolarizing | 11,264 | 31.7 | budget_exhausted: Time budget exhausted before establishing accuracy. |
| hgp_245_z_biased_pauli | 12,032 | 37.6 | budget_exhausted: Time budget exhausted before establishing accuracy. |
| color_7_xyz_nonuniform_depolarizing | 3,328 | 30.5 | budget_exhausted: Time budget exhausted before establishing accuracy. |
| bicycle_72_z_nonuniform_depolarizing | 512 | 34.5 | budget_exhausted: Time budget exhausted before establishing accuracy. |
| bicycle_144_z_nonuniform_depolarizing | 256 | 153.8 | budget_exhausted: Additional weight strata exceed the memory limit. |
| bicycle_144_z_biased_pauli | 0 | 63.7 | budget_exhausted: Time budget exhausted before establishing accuracy. |

88 actual histories were replayed independently in Stim. 12,288 conditional draws passed the predeclared seven-standard-error moment checks; largest standardized difference 2.180.

Both BB144 cases must explicitly include low weight 2 in their replay records. The intermediate run in before_log_audit_fix checked only typical weight and stopped during its third MC point after more than 15 minutes; it is not an underflow regression pass.

BB fixtures expose one defined logical observable, not full-block LER. HGP tests use ideal syndrome measurements and do not validate circuit-level QLDPC noise. Wall times include different concurrent local workloads; these are resource-capped validation runs, not a controlled throughput benchmark.
