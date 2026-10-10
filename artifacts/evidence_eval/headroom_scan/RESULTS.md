# Integration-headroom scan across the nine (gap x margin) bins (eval half)

eval half only (arbitration.split_queries, seed 20260914); the calibration half is excluded, but the proposed cell is then drawn from these same rows, so the bin is chosen on the gold statistics of the items it will score

Average-rank ties throughout. Random-ranking MRR = 0.4083. Fusion ceiling = per-item max over {best-source oracle} and exp20's five rules (borda, rrf_k60, rrf_k1, calib_relu, calib_raw).

`phi` = correlation between *phenotype ranks the gold first* and *annotation ranks the gold first*. phi <= 0 means complementary errors (fusion can pay); phi >> 0 means nested errors (it cannot). `-` = one variable is constant, so phi is undefined.

| bin | n rows | n queries | phen MRR | ann MRR | oracle | ceiling | **headroom** | headroom 95% CI | exceedable n | beaten | phi | OR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gap<0.013 \| no discriminating annotation | 469 | 248 | 0.4726 | 0.4048 | 0.5652 | 0.5671 | **+0.0019** | (0.0, 0.0047) | 358 | 2 | - | 3.215 |
| gap<0.013 \| annotation margin 0-1.5 | 100 | 84 | 0.505 | 0.4405 | 0.6366 | 0.69 | **+0.0533** | (0.0267, 0.0842) | 62 | 14 | -0.155 | 0.308 |
| gap<0.013 \| annotation margin>=1.5 | 47 | 45 | 0.3965 | 0.6799 | 0.7916 | 0.8022 | **+0.0106** | (0.0, 0.0319) | 16 | 1 | -0.326 | 0.155 |
| gap 0.013-0.059 \| no discriminating annotation | 865 | 326 | 0.5142 | 0.4023 | 0.596 | 0.5981 | **+0.0021** | (0.0003, 0.0043) | 613 | 6 | - | 2.43 |
| gap 0.013-0.059 \| annotation margin 0-1.5 | 270 | 165 | 0.4941 | 0.4241 | 0.5976 | 0.6641 | **+0.0665** | (0.0474, 0.0872) | 186 | 42 | +0.037 | 1.378 |
| gap 0.013-0.059 \| annotation margin>=1.5 | 79 | 58 | 0.4669 | 0.6936 | 0.7664 | 0.8107 | **+0.0443** | (0.0158, 0.0791) | 31 | 8 | +0.094 | 1.566 |
| gap>=0.059 \| no discriminating annotation | 439 | 234 | 0.6979 | 0.3992 | 0.7521 | 0.7538 | **+0.0018** | (0.0001, 0.0043) | 188 | 4 | - | 0.75 |
| gap>=0.059 \| annotation margin 0-1.5 | 143 | 101 | 0.6672 | 0.4605 | 0.7677 | 0.7881 | **+0.0204** | (0.007, 0.0361) | 56 | 8 | -0.053 | 0.758 |
| gap>=0.059 \| annotation margin>=1.5 | 54 | 34 | 0.784 | 0.769 | 0.925 | 0.9343 | **+0.0093** | (0.0, 0.0278) | 7 | 1 | +0.113 | 1.646 |

## Top-1 view

| bin | phen top-1 | ann top-1 | oracle top-1 | ceiling top-1 | headroom top-1 |
|---|---|---|---|---|---|
| gap<0.013 \| no discriminating annotation | 0.2367 | 0.1622 | 0.3609 | 0.3639 | +0.0030 |
| gap<0.013 \| annotation margin 0-1.5 | 0.27 | 0.12 | 0.38 | 0.48 | +0.1000 |
| gap<0.013 \| annotation margin>=1.5 | 0.1489 | 0.5319 | 0.6596 | 0.6809 | +0.0213 |
| gap 0.013-0.059 \| no discriminating annotation | 0.2913 | 0.1603 | 0.404 | 0.4075 | +0.0035 |
| gap 0.013-0.059 \| annotation margin 0-1.5 | 0.2519 | 0.0852 | 0.3111 | 0.4407 | +0.1296 |
| gap 0.013-0.059 \| annotation margin>=1.5 | 0.2025 | 0.5316 | 0.6076 | 0.6987 | +0.0911 |
| gap>=0.059 \| no discriminating annotation | 0.5718 | 0.156 | 0.6378 | 0.6401 | +0.0023 |
| gap>=0.059 \| annotation margin 0-1.5 | 0.5175 | 0.1678 | 0.6084 | 0.6469 | +0.0385 |
| gap>=0.059 \| annotation margin>=1.5 | 0.6852 | 0.6667 | 0.8704 | 0.8889 | +0.0185 |

## What a FIXED rule achieves (vs the per-item best-source oracle)

The ceiling above is a per-item max over the five rules -- an oracle-over-rules, reachable only by something that knows which rule to trust on each item. The table below is what each rule actually delivers as a fixed policy.

| bin | borda | rrf_k60 | rrf_k1 | calib_relu | calib_raw |
|---|---|---|---|---|---|
| gap<0.013 \| no discriminating annotation | -0.0949 | -0.0950 | -0.0943 | -0.0926 | -0.0943 |
| gap<0.013 \| annotation margin 0-1.5 | -0.1020 | -0.1049 | -0.1444 | -0.1316 | -0.1443 |
| gap<0.013 \| annotation margin>=1.5 | -0.2740 | -0.2423 | -0.2034 | -0.1116 | -0.1057 |
| gap 0.013-0.059 \| no discriminating annotation | -0.0841 | -0.0841 | -0.0846 | -0.0818 | -0.0839 |
| gap 0.013-0.059 \| annotation margin 0-1.5 | -0.0681 | -0.0694 | -0.1245 | -0.1080 | -0.1247 |
| gap 0.013-0.059 \| annotation margin>=1.5 | -0.1271 | -0.1277 | -0.1260 | -0.0603 | -0.0772 |
| gap>=0.059 \| no discriminating annotation | -0.0659 | -0.0664 | -0.0574 | -0.0541 | -0.0559 |
| gap>=0.059 \| annotation margin 0-1.5 | -0.1559 | -0.1557 | -0.1464 | -0.0910 | -0.1215 |
| gap>=0.059 \| annotation margin>=1.5 | -0.1655 | -0.1577 | -0.1361 | -0.1451 | -0.1451 |

Best fixed rule in the best bin: **-0.0541** vs the oracle. Every entry is negative -- in all nine bins, no fixed fusion rule matches the per-item best-source oracle, let alone the ceiling.

## Verdict

- **gap<0.013 | annotation margin 0-1.5** is viable: 84 eval-half queries, headroom +0.0533 MRR (0.0267, 0.0842), +0.1000 top-1.
- **gap 0.013-0.059 | annotation margin 0-1.5** is viable: 165 eval-half queries, headroom +0.0665 MRR (0.0474, 0.0872), +0.1296 top-1.
- **gap 0.013-0.059 | annotation margin>=1.5** is viable: 58 eval-half queries, headroom +0.0443 MRR (0.0158, 0.0791), +0.0911 top-1.
- **gap>=0.059 | annotation margin 0-1.5** is viable: 101 eval-half queries, headroom +0.0204 MRR (0.007, 0.0361), +0.0385 top-1.
