# scCGPM

**Chemistry- and Mechanism-Guided Domain Adaptation for Single-Cell Drug-Response Prediction**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Example](https://img.shields.io/badge/Example-GSE112274%20%7C%20Gefitinib-6f42c1)

scCGPM is a fine-grained transfer-learning model that connects **bulk pharmacogenomic source data** with **single-cell RNA-seq target data** for drug-response prediction. 

---
### Core ideas

**1. Rank-based shared pathway space.** Rather than directly aligning raw bulk and single-cell expression values, each sample is converted to within-sample gene ranks and then summarized at the pathway level. This reduces sensitivity to absolute expression scale, sequencing depth, and platform-specific measurement characteristics.

**2. Multi-Faceted Similarity-Guided Source Domain Selection and Weighting.** Drug similarity is computed from three complementary views: molecular structure, known targets, and target-linked pathway mechanisms. For a fixed target drug, the most relevant GDSC source drugs are selected and converted into normalized transfer weights.

**3. Drug- and response-aware pathway gating.** A drug embedding is combined with a sample-specific response prior. The resulting gate dynamically reweights pathways before latent representation learning, allowing the model to condition pathway importance on both the drug and the cell state.

**4. Class-conditional adaptation with structure preservation.** During refinement, high-confidence target pseudo-labels define class-specific target subsets. Source contributions weighted by target-drug similarity. A manifold regularizer is applied in both source and target domains to discourage destructive geometric distortion during transfer.

**5. Extensive benchmarking and application-oriented validation.** The framework is systematically compared with multiple baseline models across diverse datasets. Its applicability is further examined through a range of case studies covering different drug-response scenarios and biological contexts, providing additional evidence for the robustness, generalizability, and practical utility of the proposed model.

---

## Repository structure

```text
scCGPM/
├── README.md
├── run.py
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       └── python-check.yml
├── data/
│   └── README.md
├── results/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── config.py       # experiment configuration
    ├── data.py         # input loading and label construction
    ├── pathway.py      # intra-sample ranks and pathway scoring
    ├── drug.py         # multi-view drug representation and source selection
    ├── model.py        # drug/response-guided pathway gating network
    ├── losses.py       # weighted CORAL and manifold preservation
    ├── train.py        # source pretraining and target adaptation
    ├── evaluate.py     # GSE112274 evaluation
    └── pipeline.py     # end-to-end orchestration
```

`run.py` is intentionally thin: it imports `main()` from `src.pipeline`, while the scientific components are separated into focused modules. This keeps the public entry point simple without placing the complete method in one large script.

The repository intentionally does **not** include large expression matrices or third-party datasets. Place the required files under `data/` using the layout described in [`data/README.md`](data/README.md).


## Code organization

`pathway.py` owns the cross-platform pathway representation; `drug.py` owns molecular/target/pathway views and target-specific transfer weights; `model.py` contains only the neural architecture; `losses.py` isolates the alignment objectives; and `train.py` implements the two-stage learning procedure. `pipeline.py` connects these pieces for the single fixed target domain, GSE112274.

The import flow is deliberately one-directional: configuration and low-level feature modules are imported by training/evaluation code, and only `pipeline.py` orchestrates the complete experiment. This avoids circular dependencies and makes individual components easier to inspect, test, or reuse.

---

## Installation

Python 3.10+ is recommended.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd scCGPM

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python run.py --data-dir data --output-dir results/gse112274
```

The target domain is intentionally fixed in this release:

```text
Dataset : GSE112274
Drug    : Gefitinib
```

## Outputs

A successful run writes:

```text
results/gse112274/
├── metrics.csv
├── predictions.csv
├── source_drug_weights.csv
├── pathways.csv
├── model.pt
└── run_config.json
```

`predictions.csv` contains cell-level Gefitinib response scores and retrospective evaluation labels. `source_drug_weights.csv` records the three-view similarity terms, final transfer weights, and which GDSC drugs were selected.


## Reproducibility

The code fixes Python, NumPy, and PyTorch random seeds. The selected pathway set, learned model state, run configuration, target predictions, pathway gates, and source-drug transfer weights are all exported for traceability.

Exact numerical reproduction can still depend on package versions, hardware, and the precise preprocessing/version of the input data.


## Notes

**1.** This repository provides a compact research implementation using the GSE112274 / Gefitinib experiment as an example. Other datasets or drugs can be evaluated by replacing the corresponding input data while keeping the overall pipeline unchanged.

**2.** We believe that scCGPM represents an effective computational tool for single-cell drug response research, providing valuable insights for drug discovery, tumor drug-resistance mechanism elucidation, and precision oncology therapies.
