#!/usr/bin/env python3
"""
make_arbitration_figure.py

Two-panel figure for "Scientific Agents Apply Calibrated Reliability But Cannot
Estimate It".

(a) The controlled, *within-cell* contrast: same items, same model, the
    reliability table the only thing that changes. Accuracy moves a long way
    where calibration says the default source is uninformative, and does not
    move where that source is already right. Both delivery modes (table in the
    prompt, table behind an unmentioned tool) and both endpoints (MRR, top-1)
    are shown, because the top-1 version is the sharper one.

(b) The same cells as levels: the agent lands on whichever single source the
    table endorses, stays under the best-source selection ceiling everywhere,
    and in the complementary cell -- where both sources clear the random rate
    -- gains nothing, while removing the table drops it to the random rate.

Deliberately NOT shown: the verbatim copy rate. The 40/40 -> 14/80 collapse is
a between-cell contrast across cells that differ ~18x in median phenotype gap;
the controlled within-cell version (table vs no table in the complementary
cell, 14/80 vs 8/80) is null. See the 2026-09-20 complementary-cell report.

Outputs:
    figures/arbitration.pdf   (vector, for the paper)
    figures/arbitration.png   (200 dpi preview)

Re-runnable: every plotted quantity is read from artifacts/ at run time. No
number is hardcoded here. Missing artifacts drop the rows or series they would
have supplied rather than failing.

Run:
    PYTHONHASHSEED=0 python scripts/evidence_eval/make_arbitration_figure.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]  # repo root
ART = REPO / "artifacts" / "evidence_eval"
OUT = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load(*parts):
    """Read a JSON artifact, or None if it is not there."""
    p = ART.joinpath(*parts)
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


arb = load("arbitration", "metrics.json")
headroom = load("headroom", "metrics.json")
tool_a = load("tool_agent", "metrics.json")
tool_b = load("tool_agent_gpt56", "metrics.json")
comp = load("complementary", "metrics.json")

CHANCE = None
for src, path in ((arb, ("cells", "conflict", "random_MRR")),
                  (comp, ("random_ranking_MRR",))):
    if src is None:
        continue
    v = src
    for k in path:
        v = v[k]
    CHANCE = v
    break
CHANCE_T1 = (headroom or {}).get("task2_top1", {}).get("chance", {}).get("top1")

# ══ Panel (a) data ═════════════════════════════════════════════════════════
# The controlled contrast is "<arm with the table> minus naive", paired over the
# same items. `naive` is the plain prompt without the table, run on model A, so
# only model-A arms give a within-model comparison; model B has no naive arm and
# appears in panel (b) instead.
#
# rows[i] = (cell_key, cell_label, n_items, arm_label, {"MRR": (m, ci),
#                                                       "top1": (m, ci)})
DELIVERY = [("rel_minus_naive", "in prompt"),
            ("agent_A_minus_naive", "behind a tool")]
CELL_ORDER = ["conflict", "control"]

rows = []
contrasts = (headroom or {}).get("task2_top1", {}).get("arm_contrasts", {})
for cell in CELL_ORDER:
    cc = contrasts.get(cell)
    if not cc:
        continue
    n_items = (arb or {}).get("cells", {}).get(cell, {}).get("n")
    for key, lab in DELIVERY:
        if key not in cc:
            continue
        rows.append((cell, cell, n_items, lab,
                     {m: (cc[key][m]["mean"], cc[key][m]["ci95"])
                      for m in ("MRR", "top1") if m in cc[key]}))

# ══ Panel (b) data ═════════════════════════════════════════════════════════
# cells[i] = dict(label, n_items, mrr={series: (value, ci|None)})
cells = []
for cell in CELL_ORDER:
    if not arb or cell not in arb.get("cells", {}):
        continue
    ca = arb["cells"][cell]
    rec = {"key": cell, "label": cell, "n_items": ca["n"], "mrr": {
        "phenotype": (ca["phenotype_only_MRR"], None),
        "annotation": (ca["annotation_only_MRR"], None),
        # The JSON key retains a retired word. The quantity is the per-item max
        # over the two sources: the best-source selection ceiling.
        "ceiling": (ca["oracle_arbiter_MRR"], None),
    }}
    for model, art in (("A", tool_a), ("B", tool_b)):
        if art and cell in art.get("cells", {}):
            arm = art["cells"][cell]["arms"]["agent_free"]
            rec["mrr"][f"agent_{model}"] = (arm["MRR"], arm["MRR_CI"])
    cells.append(rec)

if comp:
    ref, arms = comp["references"], comp["arms"]
    rec = {"key": "complementary", "label": "complementary",
           "n_items": comp["n_items"], "mrr": {
               "phenotype": (ref["phenotype_only_ranker_MRR"], None),
               "annotation": (ref["annotation_only_ranker_MRR"], None),
               "ceiling": (ref["best_source_oracle_MRR"], None),
           }}
    if "agent_free" in arms:
        rec["mrr"]["agent_A"] = (arms["agent_free"]["MRR"],
                                 arms["agent_free"]["MRR_CI"])
    if "agent_norel" in arms:
        rec["mrr"]["agent_notable"] = (arms["agent_norel"]["MRR"],
                                       arms["agent_norel"]["MRR_CI"])
    cells.append(rec)

assert rows or cells, "no artifacts found under artifacts/evidence_eval/"

# ══ Style: CVD-safe and grayscale-safe -- hue is never the only channel ════
C_MRR = "#0173B2"    # blue, filled circle
C_T1 = "#111111"     # near-black, open square
C_PHEN = "#0173B2"
C_ANN = "#DE8F05"    # amber; never paired with green
C_CEIL = "#4D4D4D"
C_A = "#111111"
C_B = "#8C8C8C"
C_NOTAB = "#CC79A7"   # reddish purple; separable from amber under deuteranopia
C_REF = "#AAAAAA"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 200,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(5.5, 3.05),
    gridspec_kw={"width_ratios": [0.95, 1.05], "wspace": 0.40})

# ── Panel (a): forest plot of the within-cell table contrast ───────────────
MEAS = [("MRR", C_MRR, "o", True, "MRR"),
        ("top1", C_T1, "s", False, "top-1")]
DY = 0.17
ypos, ylabels, group_spans = [], [], {}
y = 0.0
prev_cell = None
for cell, label, n_items, arm_lab, vals in rows:
    if prev_cell is not None and cell != prev_cell:
        y += 0.65          # gap between cells
    ypos.append(y)
    ylabels.append(arm_lab)
    group_spans.setdefault(cell, []).append((y, n_items))
    y += 1.0
    prev_cell = cell

axA.axvline(0.0, color="#777777", lw=0.8, ls="--", zorder=1)
for yy, (cell, label, n_items, arm_lab, vals) in zip(ypos, rows):
    for j, (m, col, mk, filled, _) in enumerate(MEAS):
        if m not in vals:
            continue
        val, ci = vals[m]
        yv = yy + (DY if j == 0 else -DY)
        axA.plot(ci, [yv, yv], color=col, lw=1.0, alpha=0.8, zorder=3,
                 solid_capstyle="butt")
        for c in ci:                      # interval caps, visible in grayscale
            axA.plot([c, c], [yv - 0.11, yv + 0.11], color=col, lw=0.8,
                     zorder=3)
        axA.plot(val, yv, marker=mk, ms=4.0, ls="none", zorder=5,
                 color=col if filled else "#FFFFFF",
                 markeredgecolor=col, markeredgewidth=0.9)

axA.set_yticks(ypos)
axA.set_yticklabels(ylabels)
axA.invert_yaxis()
axA.set_xlabel("change from adding the reliability table\n(same items, paired)")
axA.set_title("(a) the table moves accuracy", loc="left", pad=5,
              fontsize=7.8)
axA.tick_params(axis="y", length=0, pad=2)

# cell names, to the left of the arm labels
for cell, spans in group_spans.items():
    ys = [s[0] for s in spans]
    n_items = spans[0][1]
    txt = cell if n_items is None else f"{cell}\n$n$={n_items}"
    axA.text(-0.475, sum(ys) / len(ys), txt, transform=axA.get_yaxis_transform(),
             ha="center", va="center", fontsize=7.4, fontweight="semibold",
             clip_on=False, linespacing=1.25)

if not rows:   # headroom artifact absent -- do not ship an empty axes
    axA.set_axis_off()
    axA.text(0.5, 0.5, "artifacts/evidence_eval/headroom/metrics.json\nnot found",
             ha="center", va="center", fontsize=7, color="#999999",
             transform=axA.transAxes)
axA.legend(handles=[Line2D([0], [0], marker=mk, ms=4.0, ls="none",
                           color=col if filled else "#FFFFFF",
                           markeredgecolor=col, markeredgewidth=0.9, label=lab)
                    for _, col, mk, filled, lab in MEAS],
           loc="lower right", frameon=False, fontsize=6.8, handlelength=1.0,
           handletextpad=0.35, borderpad=0.1, labelspacing=0.25)
axA.margins(y=0.16)

# ── Panel (b): levels against both sources and the selection ceiling ───────
SERIES = [
    ("phenotype", -1, "o", C_PHEN, 4.2, "phenotype-only"),
    ("annotation", -1, "s", C_ANN, 3.9, "annotation-only"),
    ("agent_A", +1, "D", C_A, 3.6, "agent A"),
    ("agent_B", +1, "D", C_B, 3.6, "agent B"),
    ("agent_notable", +1, "v", C_NOTAB, 4.2, "agent A, no table"),
]
OFF, SUB = 0.20, 0.10
seen = set()

for i, rec in enumerate(cells):
    m = rec["mrr"]
    if "ceiling" in m:
        best = max(m[s][0] for s in ("phenotype", "annotation") if s in m)
        axB.add_patch(plt.Rectangle((i - 0.32, best), 0.64,
                                    m["ceiling"][0] - best, facecolor="#000000",
                                    alpha=0.055, edgecolor="none", zorder=1))
        axB.plot([i - 0.32, i + 0.32], [m["ceiling"][0]] * 2, color=C_CEIL,
                 lw=1.6, zorder=4, solid_capstyle="butt")
        seen.add("ceiling")
    nsub = 0
    for name, side, mk, col, ms, _ in SERIES:
        if name not in m:
            continue
        x = i + side * OFF + (SUB * nsub if side > 0 else 0.0)
        if side > 0:
            nsub += 1
        val, ci = m[name]
        if ci:   # unpaired item-level interval on the absolute level; the
                 # within-cell comparisons are paired and live in panel (a)
            axB.plot([x, x], ci, color=col, lw=0.7, alpha=0.30, zorder=2)
        axB.plot(x, val, marker=mk, ms=ms, color=col, ls="none",
                 markeredgecolor="#FFFFFF", markeredgewidth=0.55, zorder=5)
        seen.add(name)

if CHANCE is not None:
    axB.axhline(CHANCE, color=C_REF, lw=0.9, ls="--", zorder=2)
    axB.text(-0.50, CHANCE + 0.008, "chance", ha="left", va="bottom",
             fontsize=6.3, color="#8A8A8A")

axB.set_xlim(-0.55, len(cells) - 0.45)
axB.set_xticks(range(len(cells)))
axB.set_xticklabels([f"{r['label']}\n$n$={r['n_items']}" for r in cells],
                    fontsize=6.6)
axB.set_ylabel("mean reciprocal rank")
axB.set_title("(b) it lands on one source", loc="left", pad=5,
              fontsize=7.8)
axB.tick_params(axis="x", length=0, pad=3)
axB.yaxis.set_major_locator(plt.MultipleLocator(0.1))
axB.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

handles_b = []
if "ceiling" in seen:
    handles_b.append(Line2D([0], [0], color=C_CEIL, lw=1.6,
                            label="best-source ceiling"))
handles_b += [Line2D([0], [0], marker=mk, ms=ms, color=col, ls="none",
                     markeredgecolor="#FFFFFF", markeredgewidth=0.55, label=lab)
              for name, _, mk, col, ms, lab in SERIES if name in seen]
axB.legend(handles=handles_b, loc="upper center", ncol=2, frameon=False,
           fontsize=6.3, handlelength=1.0, handletextpad=0.3,
           columnspacing=0.6, borderpad=0.0, labelspacing=0.26,
           bbox_to_anchor=(0.48, -0.155))
_lo = min(min(v[1]) if v[1] else v[0] for r in cells for v in r["mrr"].values())
_hi = max(v[0] for r in cells for v in r["mrr"].values())
axB.set_ylim(_lo - 0.02, _hi + 0.04)

for ax in (axA, axB):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_axisbelow(True)
axA.spines["left"].set_visible(False)
axB.tick_params(axis="y", length=2.5, width=0.7)
axA.tick_params(axis="x", length=2.5, width=0.7)

fig.subplots_adjust(left=0.205, right=0.985, top=0.915, bottom=0.30)

pdf_path, png_path = OUT / "arbitration.pdf", OUT / "arbitration.png"
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=200)
for p in (pdf_path, png_path):
    assert p.stat().st_size > 5_000, f"{p} looks empty -- savefig failed"
    print(f"OK {p}  {p.stat().st_size // 1024} KB")
print(f"size: {fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in")

# ── self-check: echo everything plotted, and assert the claim's direction ──
if __name__ == "__main__":
    print(f"\nchance: MRR {CHANCE}  top-1 {CHANCE_T1}")
    print("panel (a) rows:")
    for cell, _, n, arm, vals in rows:
        print(f"  {cell:9s} n={n:<3} {arm:20s} " + "  ".join(
            f"{m}={v[0]:+.4f} {v[1]}" for m, v in vals.items()))
    print("panel (b) levels:")
    for r in cells:
        print(f"  {r['key']:14s} n={r['n_items']:<3} " + "  ".join(
            f"{k}={v[0]:.4f}" for k, v in r["mrr"].items()))

    # The claim: the table moves accuracy where the default source is
    # uninformative and does not where it is already right.
    by = {(c, a): v for c, _, _, a, v in rows}
    for _, arm in DELIVERY:
        if ("conflict", arm) in by and ("control", arm) in by:
            lo_conf = by[("conflict", arm)]["MRR"][1][0]
            hi_ctrl = by[("control", arm)]["MRR"][1][1]
            assert lo_conf > hi_ctrl, (arm, lo_conf, hi_ctrl)
    # Nothing clears the selection ceiling.
    for r in cells:
        if "ceiling" not in r["mrr"]:
            continue
        for k, v in r["mrr"].items():
            assert v[0] <= r["mrr"]["ceiling"][0] + 1e-9, (r["key"], k)
    print("self-check: conflict gain excludes the control gain; "
          "no arm exceeds the ceiling -- OK")
