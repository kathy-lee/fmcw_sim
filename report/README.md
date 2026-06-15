# Report Index

The reports build on one another: later reports state only their
methodology **deltas** against the conventions established earlier, and new
experiments quote earlier results as baselines so differences are
attributable.

| # | Report | Stage under test | Serves as baseline for |
|---|---|---|---|
| 1 | [Detection Performance Test](detection_performance_test_report.md) | CFAR + clustering | steady point-target Pd/Pfa curves |
| 2 | [Tracker Stress Test](tracker_stress_test_report.md) | Tracker (association, M-of-N, coasting, KF/EKF/IMM) | track survival & association on steady targets |
| 3 | [Target Realism Test](target_realism_test_report.md) | Target models (generator) → effects across the whole chain | — |

## Shared conventions

- **Fixed front-end lens** (report 2 §2.1): `range_doppler` (Hann, DC
  removal) → OS-CFAR (Pfa = 1e-4) → clustering → tracker (GNN, 3-of-5).
- **Seed hygiene** (report 1 §2.4, report 2 §2.5): independent RNG
  streams — CFAR calibration seed `0`, generator noise seed `99999`
  (Monte Carlo: `1…30`). Report 3 adds a **third stream**,
  `RadarConfig.fluctuation_seed` (default `77777`) for Swerling fades
  (Part I), and a **fourth stream**, `RadarConfig.extent_seed` (default
  `555`) for extended-target scatterers (Part II).
- **Run classification taxonomy** (report 2 §2.3): CLEAN / ID SWITCH /
  TRACK BREAK / TOTAL CONFUSION from the `has_switches` × `has_break` axes.
