# UPDP — Uniform Projection for Differential Privacy

This repository contains an implementation of **UPDP (Uniform Projection for Differential Privacy)** and a small runner to generate **(locally) differentially private synthetic tabular data** in the **`tab_bench`-style `.npy` dataset format** (train/val/test splits stored as NumPy arrays).

At a high level, UPDP works like this:

1. **Encode** a tabular dataset (numerical + categorical + label) into a single numeric vector per record.
2. Apply a **random projection** to reduce dimensionality.
3. **Clip** the projected vectors to a fixed radius to bound sensitivity.
4. Add **DP noise** (Laplace for ε-LDP, or Gaussian for (ε,δ)-LDP).
5. Apply a **radial “uniformization”** post-processing step (post-processing does not weaken DP).
6. **Reconstruct** back to the original encoded dimension with a pseudo-inverse, then **decode** to (numerical, categorical, label).

The main runnable code is under **`UPDP_Method/`**.

---

## Repository structure

- `UPDP_Method/`
  - `updp.py` — core UPDP mechanism (projection + clipping + noise + optional uniformization)
  - `updp_synthesizer.py` — tabular encoder/decoder + UPDP wrapper for mixed (num/cat/y) data
  - `updp_main.py` — CLI entry point: run UPDP on a dataset folder and save synthetic `.npy`
  - `run_experiments.py` — batch runner across datasets/epsilons/seeds + summary CSV
  - `evaluation_metrics.py` — lightweight evaluation (ML utility + query error + distribution fidelity)
  - `test_updp.py` — sanity tests + optional run on a real dataset folder
- `evaluator/` — evaluation utilities (ML models + query metrics). **Some paths in this tree expect methods not included in this repo.**
- `preprocess_common/`, `util/` — shared preprocessing/utility code (some scripts reference methods not shipped here)
- `Method/` — included baseline method codebases (AIM, DP_MERF, GEM, PrivMRF, RAP, TabDDPM, private_gsd).
- `plots/` — pre-generated comparison figures.



---

