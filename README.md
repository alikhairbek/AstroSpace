# AstroSpace# 

**A size-controlled chemical-space benchmark for machine-learning prioritization of interstellar molecules.**

Machine-learning studies that prioritize molecules for interstellar detection usually draw candidates from the QM9 dataset. QM9 is dominated by nine-heavy-atom molecules, whereas confirmed interstellar species are overwhelmingly small, so molecular **size** silently confounds the benchmark. This repository removes the confound by enumerating the CHNO chemical space exhaustively, computing the relevant physics for every molecule, and evaluating predictors under strict size matching.

The entire study, chemical space, quantum-chemical properties, statistical evaluation, all nine figures, and the candidate list, is reproduced from scratch by a single notebook.

---

## Key results

On an exhaustively enumerated space of **35,482** neutral, closed-shell CHNO molecules (≤ 6 heavy atoms), evaluated against **54** confirmed interstellar species under size matching:

| Predictor | Size-matched AUC | Trainable parameters |
|---|---|---|
| Size-only control | 0.500 | 0 |
| **Dipole moment** (GFN2-xTB) | **0.660** | **0** |
| Detectability (μ²/Q_rot) | 0.562 | 0 |
| Tanimoto similarity | 0.809 | none |
| Descriptor XGBoost | 0.888 | yes |

After size is controlled, the molecular **dipole moment is a statistically significant predictor with zero trained parameters**, consistent with the rotational-spectroscopy physics of detection. Simple physicochemical descriptors are strongest; **no learned representation, including a graph neural network, beats them.** Three documented negative results (charge separation, fluorine, calibration) map the boundary conditions of the approach.

---

## Run it

The whole pipeline runs from a single notebook, top to bottom, with **no GPU** and no pre-computed inputs.

### Google Colab
1. Open [`AstroSpace.ipynb`](AstroSpace.ipynb) in Colab.
2. Upload `QM9_dipole_reference.csv` (needed only for the optional dipole-calibration step).
3. **Runtime → Run all.**

### Kaggle
1. Create a new notebook and upload `AstroSpace.ipynb`.
2. Add `QM9_dipole_reference.csv` as an input dataset.
3. Enable **Internet** (so `pip` can install the GFN2-xTB engine), then run all cells.

### Local
```bash
pip install rdkit tblite xgboost scikit-learn pandas numpy matplotlib pillow
jupyter notebook AstroSpace.ipynb
```

Runtime is roughly 15–30 minutes on a free CPU runtime at the default size ceiling of six heavy atoms. The quantum-chemical step checkpoints every 500 molecules, so it can be safely interrupted and resumed.

---

## What the notebook does

| Module | Output |
|---|---|
| **M1** | Exhaustive enumeration of neutral closed-shell CHNO molecules (≤ 6 heavy atoms) |
| **M2** | GFN2-xTB geometry, dipole moment, rotational constants and energy for every molecule |
| **M3** | Δ-ML dipole calibration against B3LYP with Mondrian conformal intervals |
| **M4** | Size-matched retrieval evaluation with bootstrap CIs and permutation tests |
| **M5** | Ranked, stability-filtered candidate list |
| **M6** | Five main-text figures (300 dpi PNG + PDF) |
| **M7** | Four supplementary figures (300 dpi PNG + PDF) |

Running all cells writes a self-contained `AstroSpace_project/` folder — the enumerated space, computed properties, candidate list, and all nine figures — and bundles it as a zip.

---

## Configuration

Three constraints are applied by default, each motivated by a result in the paper. Each can be relaxed by a flag in the configuration cell to reproduce the corresponding control experiment.

| Flag | Default | Effect when relaxed |
|---|---|---|
| `ALLOW_CHARGE_SEPARATION` | `False` | Admits formally charged Lewis structures; enlarges the space ~4× and collapses the dipole signal |
| `INCLUDE_FLUORINE` | `False` | Adds fluorine; inflates the space with astrochemically implausible candidates |
| `CALIBRATED` | `False` | Ranks on calibrated rather than raw dipoles; improves absolute accuracy but degrades ranking |
| `MAX_HEAVY` | `6` | Sets the heavy-atom ceiling of the enumeration |

---

## Repository contents

```
AstroSpace.ipynb                  # the complete, runnable pipeline
QM9_dipole_reference.csv          # reference dipoles for the optional M3 calibration
AstroSpace_project.zip            # example output bundle (data + all 9 figures)
```

---

## Top candidate

The highest-ranked undetected molecule is **propiolamide** (`C#CC(N)=O`, μ = 4.33 D) — a small, polar amide of exactly the kind laboratory astrochemistry actively targets.

---

## Citation

If you use this benchmark, please cite the accompanying manuscript:

> Khairbek, A. A. *Physics over Representation Learning after Size Control: Benchmarking Interstellar-Molecule Discovery on an Exhaustively Enumerated CHNO Chemical Space.*

---

## License

Released under the MIT License. The GFN2-xTB calculations use [`tblite`](https://github.com/tblite/tblite); cheminformatics uses [RDKit](https://www.rdkit.org/).
