# Brock–Hommes (not part of the paper)

The diagnostic applied to the Brock & Hommes (1998) heterogeneous-beliefs asset-pricing
model, θ = (β, g₁, b₁, g₂, b₂), calibrated with an MMD loss on the price trajectory
(T = 100, 256 frozen random Fourier features, 128 seeds). None of this is used by the
paper's figures or claims, and `make` does not run it.

| Script | What it computes | Runtime |
|---|---|---|
| `bh_ggn_vs_hessian.py` | G against the exact Hessian on and off the fit; leading eigenvalue and direction against the differentiation horizon, at β = 50 and β = 80 | ~10 s |
| `bh_ggn_vs_fd_hessian.py` | G against a central finite-difference Hessian at the fit, β = 50 | ~5 s |
| `bh_horizon_stability.py` | spectrum and β loading on the sloppiest direction at horizons 10, 20, 40 | ~5 s |
| `bh_spectrum_loadings.py` | spectrum and eigenvector loadings at the fit, β = 50, horizon 20 | ~5 s |

Run from the repository root, for example

```
PYTHONPATH=src:. python extras/brock_hommes/bh_ggn_vs_hessian.py
```

Each script writes `results/<name>.json` and `figures/<name>.pdf` in this folder.
