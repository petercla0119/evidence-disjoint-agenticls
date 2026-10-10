# Integration-headroom scan across the nine (gap x margin) bins (calib half)

calibration half only (arbitration.split_queries, seed 20260914); the eval half is never touched, so a bin chosen here is selected independently of the items the cell will be drawn from. Caveat: calib_relu/calib_raw use per-bin weights fit on this same half, so their contribution to the ceiling is in-sample; the best-source oracle and the tie-free rules are not.

Average-rank ties throughout. Random-ranking MRR = 0.4083. Fusion ceiling = per-item max over {best-source oracle} and exp20's five rules (borda, rrf_k60, rrf_k1, calib_relu, calib_raw).

`phi` = correlation between *phenotype ranks the gold first* and *annotation ranks the gold first*. phi <= 0 means complementary errors (fusion can pay); phi >> 0 means nested errors (it cannot). `-` = one variable is constant, so phi is undefined.

| bin | n rows | n queries | phen MRR | ann MRR | oracle | ceiling | **headroom** | headroom 95% CI | exceedable n | beaten | phi | OR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gap<0.013 \| no discriminating annotation | 463 | 252 | 0.4523 | 0.4 | 0.5444 | 0.5446 | **+0.0002** | (0.0, 0.0005) | 370 | 1 | - | 3.963 |
| gap<0.013 \| annotation margin 0-1.5 | 114 | 92 | 0.5127 | 0.4164 | 0.611 | 0.6675 | **+0.0565** | (0.03, 0.0855) | 74 | 17 | +0.029 | 1.283 |
| gap<0.013 \| annotation margin>=1.5 | 38 | 34 | 0.3693 | 0.5388 | 0.6098 | 0.612 | **+0.0022** | (0.0, 0.0066) | 24 | 1 | +0.200 | 3.551 |
| gap 0.013-0.059 \| no discriminating annotation | 984 | 354 | 0.4905 | 0.4036 | 0.5801 | 0.5818 | **+0.0017** | (0.0005, 0.0032) | 726 | 8 | - | 2.81 |
| gap 0.013-0.059 \| annotation margin 0-1.5 | 250 | 140 | 0.4633 | 0.4402 | 0.5941 | 0.6421 | **+0.0480** | (0.0317, 0.0657) | 168 | 31 | +0.041 | 1.349 |
| gap 0.013-0.059 \| annotation margin>=1.5 | 89 | 66 | 0.5577 | 0.6987 | 0.8315 | 0.8425 | **+0.0109** | (0.0012, 0.0253) | 24 | 4 | -0.067 | 0.756 |
| gap>=0.059 \| no discriminating annotation | 420 | 236 | 0.6387 | 0.4081 | 0.7035 | 0.7068 | **+0.0033** | (0.0, 0.0078) | 220 | 3 | - | 1.1 |
| gap>=0.059 \| annotation margin 0-1.5 | 134 | 96 | 0.5868 | 0.4509 | 0.7077 | 0.7407 | **+0.0330** | (0.0149, 0.0553) | 67 | 11 | -0.179 | 0.308 |
| gap>=0.059 \| annotation margin>=1.5 | 59 | 45 | 0.7045 | 0.6572 | 0.8897 | 0.9066 | **+0.0169** | (0.0, 0.0424) | 11 | 2 | -0.121 | 0.622 |

## Top-1 view

| bin | phen top-1 | ann top-1 | oracle top-1 | ceiling top-1 | headroom top-1 |
|---|---|---|---|---|---|
| gap<0.013 \| no discriminating annotation | 0.2009 | 0.1577 | 0.3276 | 0.3276 | +0.0000 |
| gap<0.013 \| annotation margin 0-1.5 | 0.2719 | 0.114 | 0.3509 | 0.4591 | +0.1082 |
| gap<0.013 \| annotation margin>=1.5 | 0.0789 | 0.3421 | 0.3684 | 0.3684 | +0.0000 |
| gap 0.013-0.059 \| no discriminating annotation | 0.2622 | 0.1612 | 0.3814 | 0.3841 | +0.0027 |
| gap 0.013-0.059 \| annotation margin 0-1.5 | 0.236 | 0.128 | 0.328 | 0.418 | +0.0900 |
| gap 0.013-0.059 \| annotation margin>=1.5 | 0.3483 | 0.5618 | 0.7303 | 0.7547 | +0.0243 |
| gap>=0.059 \| no discriminating annotation | 0.4762 | 0.1635 | 0.5607 | 0.5671 | +0.0063 |
| gap>=0.059 \| annotation margin 0-1.5 | 0.3881 | 0.1343 | 0.5 | 0.5597 | +0.0597 |
| gap>=0.059 \| annotation margin>=1.5 | 0.5593 | 0.5085 | 0.8136 | 0.8475 | +0.0339 |

## What a FIXED rule achieves (vs the per-item best-source oracle)

The ceiling above is a per-item max over the five rules -- an oracle-over-rules, reachable only by something that knows which rule to trust on each item. The table below is what each rule actually delivers as a fixed policy.

| bin | borda | rrf_k60 | rrf_k1 | calib_relu | calib_raw |
|---|---|---|---|---|---|
| gap<0.013 \| no discriminating annotation | -0.1044 | -0.1044 | -0.1009 | -0.0921 | -0.1033 |
| gap<0.013 \| annotation margin 0-1.5 | -0.0929 | -0.0959 | -0.1301 | -0.0983 | -0.1238 |
| gap<0.013 \| annotation margin>=1.5 | -0.1453 | -0.1329 | -0.1351 | -0.0710 | -0.0905 |
| gap 0.013-0.059 \| no discriminating annotation | -0.0926 | -0.0929 | -0.0931 | -0.0897 | -0.0933 |
| gap 0.013-0.059 \| annotation margin 0-1.5 | -0.0963 | -0.1026 | -0.1222 | -0.1113 | -0.1155 |
| gap 0.013-0.059 \| annotation margin>=1.5 | -0.1911 | -0.1789 | -0.1681 | -0.1312 | -0.1600 |
| gap>=0.059 \| no discriminating annotation | -0.0714 | -0.0689 | -0.0703 | -0.0648 | -0.0684 |
| gap>=0.059 \| annotation margin 0-1.5 | -0.1359 | -0.1273 | -0.1468 | -0.1189 | -0.1284 |
| gap>=0.059 \| annotation margin>=1.5 | -0.2029 | -0.1634 | -0.1760 | -0.1682 | -0.1563 |

Best fixed rule in the best bin: **-0.0648** vs the oracle. Every entry is negative -- in all nine bins, no fixed fusion rule matches the per-item best-source oracle, let alone the ceiling.

## Verdict

- **gap<0.013 | annotation margin 0-1.5** is viable: 92 eval-half queries, headroom +0.0565 MRR (0.03, 0.0855), +0.1082 top-1.
- **gap 0.013-0.059 | annotation margin 0-1.5** is viable: 140 eval-half queries, headroom +0.0480 MRR (0.0317, 0.0657), +0.0900 top-1.
- **gap 0.013-0.059 | annotation margin>=1.5** is viable: 66 eval-half queries, headroom +0.0109 MRR (0.0012, 0.0253), +0.0243 top-1.
- **gap>=0.059 | annotation margin 0-1.5** is viable: 96 eval-half queries, headroom +0.0330 MRR (0.0149, 0.0553), +0.0597 top-1.
