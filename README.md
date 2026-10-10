# Evidence-Disjoint Evaluation of Reliability-Aware Source Selection in Biological Agents

[![reproduce](https://github.com/petercla0119/evidence-disjoint-agenticls/actions/workflows/reproduce.yml/badge.svg)](https://github.com/petercla0119/evidence-disjoint-agenticls/actions/workflows/reproduce.yml)

Code, evidence packets, and raw model responses for the paper (Peterson et al.), Agentic Life Sciences workshop at NeurIPS 2026.

Each item asks an agent to rank six candidate genes as the functional partner of a query gene. It decides from two evidence sources: curated annotation with every record linking the query to the answer removed, and pooled-CRISPR perturbation morphology. Every number in the paper recomputes from the stored responses without an API call.

## Contents

| Path | Description |
|---|---|
| `scripts/evidence_eval/` | Packet construction, masking, reliability calibration, the model harness for all arms, and every analysis |
| `artifacts/evidence_eval/` | Item sets, calibration tables, raw model responses (`responses*.jsonl`, one JSON record per call), and the computed `metrics*.json` |
| `data/` | Gene-level morphology features (`feature98_interphase_gene.npz`, from Funk et al. 2022), the gene-disjoint split, and the Reactome and Complex Portal membership tables |
| `figures/` | Figure 1, written by `make_arbitration_figure.py` |
| `reproduce.sh` | Recomputes every number and Figure 1 from the stored responses |

STRING v12.0 (85 MB) is not included. `scripts/download_string.sh` fetches it and checks its SHA-256; `reproduce.sh` calls it.

## Environment

```bash
conda env create -f environment.yml
conda activate evidence-disjoint
./reproduce.sh
```

Tested from a fresh clone and a fresh environment on macOS arm64 and Linux (arm64 and x86-64): `reproduce.sh` regenerates every tracked output byte-identically, figures included. The `reproduce` GitHub Actions workflow re-checks this on Ubuntu and macOS for every push. `scripts/download_string.sh` needs `curl`.

Every script must run as `PYTHONHASHSEED=0 python scripts/evidence_eval/<script>.py <subcommand>` under **Python 3.10**. Candidate order in a packet comes from `hash((item_id, order))`. `PYTHONHASHSEED=0` fixes string hashing only within one Python version, and CPython 3.11 changed the algorithm. Each entry point that rebuilds packets checks the rebuilt gold alias against the stored one and exits on a mismatch, rather than scoring different packets than the model saw.

The scaffold experiment (`exp7_scaffold_only.py`) predates that check and cannot be rebuilt exactly under any tested interpreter (3.9–3.13). Its statistics are computed from the aliases and phenotype ranks stored with its responses, and its comparison arms ran in the same original environment.

## Where each result comes from

| Paper result | Script | Output |
|---|---|---|
| Table 1, conflict and control cells (models A and B) | `arbitration.py analyze [--provider azure]` | `arbitration/metrics.json`, `metrics_gpt56.json` |
| Table 1, complementary and replication cells | `exp23_complementary_cell.py analyze` (see `reproduce.sh` for flags) | `complementary*/metrics*.json`, key `table_contrast` |
| Tool-delivered reliability arms | `tool_agent.py analyze [--provider azure]` | `tool_agent*/metrics.json` |
| Noise floor, retests, temperature arms | `arbitration.py noise-floor`, `analyze_control_retest.py`, `analyze_temp_arms.py` | `arbitration/noise_floor.json`, `tool_agent*/…` |
| Mechanical-fusion control | `exp20_fusion_baseline.py` | `fusion/` |
| Headroom scan (eval and calibration halves) | `exp22_headroom_scan.py [--half calib]` | `headroom_scan*/` |
| Exceedance, top-1 and permutation statistics | `headroom.py` | `headroom/metrics.json` |
| Abstracted (biology-free) probe | `exp21_abstract_fusion_probe.py analyze` | `fusion_probe/` |
| Zero-shot scaffold arm | `exp7_scaffold_only.py analyze --set {fair,booster}` | `scaffold_only/` |
| Masking audit | `audit_masking.py --json artifacts/evidence_eval/masking_audit.json` | `masking_audit.json` |
| Retracted pilot and earlier item sets | `pilot_counterfactual.py`, `validate_counterfactual.py`, `reliability_calibration.py` | `pilot/`, `validate*/`, `reliability*/` |
| Figure 1 | `make_arbitration_figure.py` | `figures/arbitration.pdf` |

## Re-running model calls

The `run` subcommands call the models again and need credentials. These calls are not required to reproduce any number.

- **Model A** (`claude-sonnet-4-6` via Google Vertex): set `ANTHROPIC_VERTEX_PROJECT_ID`, and optionally `CLOUD_ML_REGION`.
- **Model B** (`gpt-5.6-luna` via Azure OpenAI): set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`. This deployment accepts only the default temperature, so its episodes are sampled.

## License

MIT. The bundled data derive from public releases: Funk et al. 2022 (BioImage Archive S-BIAD394), Reactome, Complex Portal, and STRING (CC BY 4.0).
