# Complementary cell: annotation_top_score>=1

Third arbitration cell, built where copying one source is **not** the calibration-optimal policy. Parent bin `gap<0.013 \| annotation margin 0-1.5` (exp22's only genuinely complementary region), subdivided on the calibration half so that neither source sits at the random rate.

## The reliability row the model is shown

```json
{
  "gap<0.013 | annotation margin 0-1.5 | annotation_top_score>=1": {
    "n": 72,
    "phenotype_top_pick_is_true_partner": 0.278,
    "annotation_top_pick_is_true_partner": 0.181,
    "phenotype_expected_reciprocal_rank": 0.516,
    "annotation_expected_reciprocal_rank": 0.471
  }
}
```

Random-ranking expected reciprocal rank = 0.4083. Both sources are above it, so the table does not instruct the model to zero out either one.

## Verbatim copy rate -- the headline

**Full six-item permutation equality**: the call's entire ranking is identical, position by position, to a single source's ranking. This is NOT gold-rank agreement -- `arbitration.noise_floor()`'s `exact_agreement_rate` compares reciprocal ranks and is a different, weaker statistic; it is not used here.

| cell | model | n calls | copies phenotype | copies annotation |
|---|---|---|---|---|
| control (published) | A | 40 | **40/40** | 0/40 |
| control (published) | B | 40 | **39/40** | - |
| conflict (published) | A | 80 | 5/80 | **28/80** |
| conflict (published) | B | 80 | 4/80 | - |
| annotation_top_score>=1 -- agent_free | A | 80 | **16/80** (0.2, 95% CI (0.127, 0.3)) | **2/80** (0.025, 95% CI (0.007, 0.087)) |
| annotation_top_score>=1 -- agent_norel | A | 80 | **6/80** (0.075, 95% CI (0.035, 0.154)) | **9/80** (0.113, 95% CI (0.06, 0.2)) |

Copies **neither** source: 62/80 (agent_free), 65/80 (agent_norel) -- against 0/40 in the control cell and 47/80 in the conflict cell.

### Caveat: the two cells are not equally copyable

A source with tied scores has no unique ordering, and `arbitration._source_rankings` breaks ties by sort order, so matching a tied source requires coincidentally matching an arbitrary tiebreak. That mechanically deflates its copy rate.

| cell | mean distinct annotation scores (of 6) | items with a fully distinct annotation ordering | mean distinct phenotype scores |
|---|---|---|---|
| complementary | 3.83 | 4/40 | 5.83 |
| conflict_published | 4.7 | 19/40 | 5.8 |
| control_published | 1.1 | 0/20 | 5.8 |

So the annotation copy rate here (denser ties than the conflict cell) is a floor, not a like-for-like number. The **phenotype** side is directly comparable -- phenotype scores are near-fully distinct in all three cells -- and that is the column against which the control cell's 40/40 sits.

Weaker statistic, for contrast only (top pick agrees, full ordering need not): agent_free phenotype 47/80, annotation 34/80, agent_norel phenotype 18/80, annotation 61/80.

A null on the MRR table below is expected and is not the result: no fixed fusion rule beats per-item best-source selection anywhere in exp22's nine-bin x five-rule grid (best entry -0.0541). The copy rate above is the measurement this run exists for.

## Accuracy

Annotation-only MRR uses the current `1/avg_rank` convention (average-rank tie handling, no tie-aware fix applied). Exact-ordering matching above is unaffected by that convention.

| arm | MRR | 95% CI | top-1 | vs phen-only | vs ann-only | vs best-source oracle |
|---|---|---|---|---|---|---|
| phenotype-only ranker | 0.4987 | - | 0.25 | - | - | - |
| annotation-only ranker | 0.472 | - | 0.15 | - | - | - |
| best-source oracle | 0.6487 | - | 0.4 | - | - | - |
| fusion `borda` | 0.5407 | - | 0.2958 | - | - | -0.108 (-0.2018, -0.0125) |
| fusion `rrf_k60` | 0.539 | - | 0.2875 | - | - | -0.1097 (-0.2012, -0.0179) |
| fusion `rrf_k1` | 0.5088 | - | 0.2375 | - | - | -0.1399 (-0.1943, -0.09) |
| fusion `calib_relu` | 0.51 | - | 0.225 | - | - | -0.1387 (-0.2385, -0.0374) |
| fusion `calib_raw` | 0.5108 | - | 0.2 | - | - | -0.1378 (-0.2278, -0.047) |
| **agent_free** | 0.5075 | (0.4244, 0.5958) | 0.2375 | +0.0087 (-0.0644, 0.0802) | +0.0355 (-0.0684, 0.136) | -0.1412 (-0.2229, -0.0652) |
| **agent_norel** | 0.4098 | (0.3625, 0.4623) | 0.05 | -0.089 (-0.195, 0.0163) | -0.0622 (-0.1183, -0.0145) | -0.2389 (-0.3159, -0.1681) |

Paired bootstrap CIs are over the 40 items (`arbitration._boot`). Model A's measured run-to-run noise floor on this task is MRR SE 0.0043 over 40 items; every arm difference above is larger than that in magnitude, so the CIs that span zero are item-level sampling limits, not replication noise.


## Exceedance

Exceedable items (neither source already ranks the true partner first): **24** of 40.

| arm | exceeds the better source |
|---|---|
| fusion `borda` | 6/24 |
| fusion `rrf_k60` | 4/24 |
| fusion `rrf_k1` | 1/24 |
| fusion `calib_relu` | 4/24 |
| fusion `calib_raw` | 3/24 |
| agent_free | 3/24 |
| agent_norel | 1/24 |

## Tool use

| arm | queried reliability | mean tool calls |
|---|---|---|
| agent_free | 1.0 | 3.0 |
| agent_norel | 0.0 | 2.0 |

## Verification block (correctness-blindness)

- **Selection statistics:** phenotype gap (top1-top2 cosine), annotation margin (top1-top2 graph score), annotation_top_score (top annotation graph score) -- all computed from the packet the model is handed. The gold's phenotype rank, annotation rank, and identity are **never** referenced by any selection predicate (`gold_rank_used_in_selection = False`).
- **One item per distinct query:** 40 items, 40 distinct queries -> True.
- **Query-disjoint from the calibration half:** 0 of 40 items come from a calibration-half query -> True. The refined reliability table is fit on the calibration half only.
- **Query-disjoint from the published cells:** 0 overlap (18 eligible queries dropped for this reason).
- **Not a disguised difficulty split:** true-partner phenotype rank spread {'1': 10, '2': 9, '3': 7, '4': 7, '5': 6, '6': 1}; spans all six ranks: True.

