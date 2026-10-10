# Mechanical fusion baseline for the evidence-arbitration items

Control for the claim that the model is a *source selector*, not an *evidence integrator*. Question: can a deterministic fusion of the same two source rankings beat the better single source?

Average-rank tie handling throughout (annotation is a 6-way tie on 19/20 control items). Random-ranking MRR = 0.4083.

## conflict cell (n=40)

Exceedable items (neither source already ranks the gold first): **15** of 40.

| arm | MRR | top-1 | vs phen | vs ann | vs best-source oracle | exceeds better source |
|---|---|---|---|---|---|---|
| phenotype-only | 0.3838 | 0.125 | - | - | - | - |
| annotation-only | 0.6771 | 0.525 | - | - | - | - |
| borda | 0.5016 | 0.2333 | +0.1178 (0.0341, 0.2005) | -0.1755 (-0.2821, -0.0677) | -0.2743 (-0.3578, -0.1871) | 1/15 |
| rrf_k60 | 0.5381 | 0.2875 | +0.1544 (0.0648, 0.2452) | -0.139 (-0.2496, -0.0267) | -0.2378 (-0.3233, -0.1498) | 1/15 |
| rrf_k1 | 0.5556 | 0.275 | +0.1719 (0.08, 0.2646) | -0.1215 (-0.2145, -0.0249) | -0.2203 (-0.2859, -0.1541) | 0/15 |
| calib_relu | 0.6771 | 0.525 | +0.2933 (0.1386, 0.4431) | +0.0 (0.0, 0.0) | -0.0988 (-0.169, -0.0386) | 0/15 |
| calib_raw | 0.6933 | 0.55 | +0.3096 (0.1662, 0.4471) | +0.0163 (-0.0235, 0.058) | -0.0826 (-0.144, -0.025) | 1/15 |

| best-source oracle | 0.7759 |
|---|---|
| fusion ceiling (best of oracle + all rules, per item) | 0.7884 |

## control cell (n=20)

Exceedable items (neither source already ranks the gold first): **8** of 20.

| arm | MRR | top-1 | vs phen | vs ann | vs best-source oracle | exceeds better source |
|---|---|---|---|---|---|---|
| phenotype-only | 0.7042 | 0.6 | - | - | - | - |
| annotation-only | 0.3982 | 0.1583 | - | - | - | - |
| borda | 0.6917 | 0.6 | -0.0125 (-0.0375, 0.0) | +0.2935 (0.1286, 0.4518) | -0.0762 (-0.1246, -0.0321) | 0/8 |
| rrf_k60 | 0.6917 | 0.6 | -0.0125 (-0.0375, 0.0) | +0.2935 (0.1286, 0.4518) | -0.0762 (-0.1246, -0.0321) | 0/8 |
| rrf_k1 | 0.6917 | 0.6 | -0.0125 (-0.0375, 0.0) | +0.2935 (0.1286, 0.4518) | -0.0762 (-0.1246, -0.0321) | 0/8 |
| calib_relu | 0.7042 | 0.6 | +0.0 (0.0, 0.0) | +0.306 (0.1417, 0.4667) | -0.0637 (-0.1083, -0.0237) | 0/8 |
| calib_raw | 0.6958 | 0.6 | -0.0083 (-0.025, 0.0) | +0.2976 (0.1328, 0.456) | -0.0721 (-0.1167, -0.0312) | 0/8 |

| best-source oracle | 0.7679 |
|---|---|
| fusion ceiling (best of oracle + all rules, per item) | 0.7679 |

## Verdict

- `borda`: beats the better source on **1/23** exceedable items.
- `rrf_k60`: beats the better source on **1/23** exceedable items.
- `rrf_k1`: beats the better source on **0/23** exceedable items.
- `calib_relu`: beats the better source on **0/23** exceedable items.
- `calib_raw`: beats the better source on **1/23** exceedable items.

No fusion rule reaches the per-item best-source oracle in either cell (every `vs best-source oracle` CI excludes 0 and is negative). The calibration-weighted rule with relu weights reproduces the better single source *exactly* in both cells -- in this regime the calibration table's own advice is to select, not to blend. Headroom above the oracle is +0.0125 MRR in the conflict cell and +0.0000 in the control cell.

### Reconstruction fidelity

Every item's recomputed phenotype and annotation rank matches the value stored in `arbitration/items.json`; the phenotype-only and annotation-only MRRs below reproduce `arbitration/metrics.json` to 4 dp. The packet shows the model cosines rounded to 4 dp, which manufactures phenotype ties on conf18, conf29, cont06 items; fusion uses the unrounded cosines, matching the stored baseline.

