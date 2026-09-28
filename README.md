# AstroSpace

**A size-controlled, exhaustively enumerated CHNO chemical space for evaluating machine-learning prioritization of interstellar molecules.**

Machine-learning studies that prioritize molecules for interstellar detection usually draw candidates from the QM9 dataset. QM9 is dominated by nine-heavy-atom molecules, whereas confirmed interstellar molecules are small, so molecular **size** confounds the conventional evaluation: a ranker that uses heavy-atom count alone reaches a whole-dataset AUC of 0.999 on the QM9-based task. AstroSpace removes the confound by enumerating every neutral, closed-shell molecule of C, H, N and O with up to six heavy atoms, computing the dipole moment and rotational constants of each with GFN2-xTB, and evaluating predictors under a **same-size protocol** in which every confirmed molecule is ranked only against undetected molecules of the same heavy-atom count.

The entire study — chemical space, quantum-chemical properties, statistical evaluation, all figures and tables, and the candidate list — is reproduced from scratch by a single notebook, `AstroSpace.ipynb`.

---

## Key results

AstroSpace contains **35,482** enumerated molecules, **32,458** of which have converged GFN2-xTB properties. The reference set is the 2021 census of interstellar molecules (McGuire 2022; astromol v2021.7.0): 88 in-range census molecules, 80 of which are representable as neutral closed-shell valence structures, giving a detected set of **76** molecules with two to six heavy atoms. Same-size AUC (random = 0.500; bootstrap 95% CI; sign-flip permutation p against 0.5):

| Predictor | Same-size AUC | 95% CI | p | Trained parameters |
|---|---|---|---|---|
| Size-only control | 0.500 | [0.500, 0.500] | 1.00 | 0 |
| Dipole moment μ (GFN2-xTB) | 0.624 | [0.554, 0.691] | < 0.001 | 0 |
| Line-intensity factor μ²/Q_rot | 0.550 | [0.476, 0.623] | 0.18 | 0 |
| Tanimoto similarity (leave-one-out) | 0.838 | [0.789, 0.883] | < 0.001 | none |
| Graph isomorphism network (graph only) | 0.906 | [0.865, 0.940] | < 0.001 | yes |
| Descriptor XGBoost (7 descriptors) | 0.908 | [0.879, 0.934] | < 0.001 | yes |

After size is controlled, the dipole moment is a statistically significant predictor with zero trained parameters, consistent with the physics of rotational-spectroscopy detection; the descriptor model and the graph network are the strongest predictors. The Δ-ML-corrected dipole moment (sensitivity analysis) gives a same-size AUC of 0.589.

**Prospective check.** Eight neutral closed-shell CHNO molecules with at most six heavy atoms were first reported in 2022–2023, after the census cut-off. They are positives nowhere in the pipeline; under the descriptor-model score they rank as follows among the 32,379 undetected molecules:

| Molecule | SMILES | Reported | Rank (score) | Rank in size class |
|---|---|---|---|---|
| isopropanol | `CC(C)O` | 2022 | 54 | 5 of 429 |
| methyl ketene | `CC=C=O` | 2023 | 66 | 8 of 429 |
| n-propanol | `CCCO` | 2022 | 309 | 35 of 429 |
| crotononitrile | `CC=CC#N` | 2022 | 331 | 98 of 3,341 |
| methacrylonitrile | `C=C(C)C#N` | 2022 | 331 | 98 of 3,341 |
| E-1-cyano-1,3-butadiene | `C=CC=CC#N` | 2023 | 826 | 490 of 28,557 |
| 1,2-ethenediol | `OC=CO` | 2022 | 937 | 93 of 429 |
| allyl cyanide | `C=CCC#N` | 2022 | 1,141 | 312 of 3,341 |

---

## Run it

The whole pipeline runs from a single notebook, top to bottom, with **no GPU** and no pre-computed inputs. On the four cores of a standard Kaggle CPU session the GFN2-xTB step takes about 49 minutes (0.3 core-seconds per molecule) and the graph-network training (76 folds × 3 networks) about 2.5 hours; a complete run takes about 3.5–4 hours.

### Kaggle (recommended)
1. Create a new notebook and upload `AstroSpace.ipynb`.
2. **Add `QM9_dipole_reference.csv` as a dataset input** (Add Input → Upload); the notebook finds it at any depth under `/kaggle/input/`. Without it the Δ-ML sensitivity analysis (Table 2 row, Figure S2c) is skipped and the notebook says so.
3. Set **Accelerator = None** (the pipeline is CPU-bound; a GPU or TPU session is not used and, because of CPU oversubscription, can be many times slower), enable **Internet** (so `pip` can install the GFN2-xTB engine `tblite`), then **Run All**.

### Google Colab
1. Open `AstroSpace.ipynb` in Colab and upload `QM9_dipole_reference.csv` next to it.
2. **Runtime → Run all.**

### Local
```bash
pip install rdkit tblite ase xgboost scikit-learn pandas numpy scipy matplotlib pillow jax
jupyter notebook AstroSpace.ipynb
```

The quantum-chemical step checkpoints every 500 molecules and resumes if interrupted. All random samples, bootstraps and permutation tests are seeded and the enumeration order is deterministic; the number of converged structures can still differ by a few molecules between RDKit builds, and the same-size AUC of the trained predictors is reproduced to within about ±0.01 across computing environments.

---

## What the notebook does

| Module | Output |
|---|---|
| **M1** | Exhaustive enumeration of neutral closed-shell CHNO molecules (≤ 6 heavy atoms); interstellar labels from the census; completeness checks |
| **M2** | GFN2-xTB geometry, dipole moment, rotational constants and energy for every molecule |
| **M3** | Δ-ML dipole correction toward the B3LYP/6-31G(2df,p) values of QM9, with Mondrian conformal intervals (sensitivity analysis) |
| **M4** | Same-size evaluation with bootstrap CIs and permutation tests |
| **M4b** | Counterfactual size experiment (Table 1) and the graph isomorphism network under the identical protocol |
| **M5** | Ranked, filtered candidate list (Tables S1–S2) and the prospective check (Table S3) |
| **M6 / M7** | Figures 1–5, S1–S5 and the graphical abstract (300 dpi PNG + PDF) |

Running all cells writes `AstroSpace_project/` (data, tables, figures, summaries) and bundles it as a zip. Every number quoted in the manuscript is written to `paper_numbers.json`.

---

## Configuration

| Flag | Default | Effect when changed |
|---|---|---|
| `ALLOW_CHARGE_SEPARATION` | `False` | Admits formally charge-separated valence structures (control experiment, Figure S2a) |
| `INCLUDE_FLUORINE` | `False` | Adds fluorine (control experiment, Figure S2b) |
| `MAX_HEAVY` | `6` | Heavy-atom ceiling of the enumeration |

---

## Static dataset

The complete data are provided as static CSV files and do not require running the notebook: `AstroSpace6_enumerated.csv` (35,482 canonical SMILES with the interstellar labels) and `AstroSpace6_xtb.csv` (32,458 molecules with GFN2-xTB dipole moments, rotational constants, energies and labels). The same files are deposited at Zenodo (DOI to be inserted).

## Repository contents

```
AstroSpace.ipynb                  # the complete, runnable pipeline
QM9_dipole_reference.csv          # B3LYP/6-31G(2df,p) reference dipoles for the M3 sensitivity analysis
AstroSpace6_enumerated.csv        # the enumerated space with interstellar labels
AstroSpace6_xtb.csv               # GFN2-xTB properties of the 32,458 converged molecules
TableS1_top50_candidates.csv      # 50 highest-scoring undetected molecules
TableS2_top5_per_stratum.csv      # five highest-scoring undetected molecules per heavy-atom count
TableS3_prospective.csv           # ranks of the eight molecules reported after the census cut-off
top50_novel_candidates.csv        # candidate list with all computed quantities
paper_numbers.json                # every number quoted in the manuscript
figures/                          # Figures 1–5, S1–S5 and the graphical abstract (PNG + PDF)
```

---

## Citation

If you use AstroSpace, please cite the accompanying manuscript:

> A. A. Khairbek, A. Y. A. Alzahrani, E. S. Dessoky, S. F. Mahmoud and R. Thomas, *AstroSpace: A Size-Controlled, Exhaustively Enumerated CHNO Chemical Space for Evaluating Machine-Learning Prioritization of Interstellar Molecules* (submitted).

---

## License

Released under the MIT License. The GFN2-xTB calculations use [`tblite`](https://github.com/tblite/tblite); cheminformatics uses [RDKit](https://www.rdkit.org/); the interstellar reference set follows the census of B. A. McGuire ([astromol](https://github.com/bmcguir2/astromol)).
