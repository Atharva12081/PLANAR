# EXXA 2026 Alignment Checklist (PLANAR)

Date: 2026-03-03  
Source document: `GSoC 2026 EXXA Test.pdf` (5 pages)

## Scope Verified

- General Test (unsupervised clustering of ALMA synthetic FITS disks)
- Image-Based Test (autoencoder with latent access, MSE + MS-SSIM)
- Sequential Test (simulated transit classifier with ROC/AUC)
- Deliverable constraints (automated, reproducible, minimal manual intervention)

## Requirement Mapping

| EXXA Requirement | PLANAR Status | Evidence |
|---|---|---|
| Load FITS data, use first layer (index 0) | PASS | `src/planar/data_loader.py` (`layer=0` default, 2D extraction) |
| Data cube/image format validation | PASS | `expected_shape: [600, 600]` in configs + pipeline enforcement |
| Unsupervised clustering | PASS | `src/planar/pipelines/clustering.py` + HDBSCAN/KMeans/GMM backends |
| Avoid trivial orientation-only clustering | PASS | radial preprocessing option + bias audits (`eta^2`, Kruskal) |
| Automated end-to-end pipeline | PASS | `planar run --config ...` in CLI |
| Clear visual outputs for performance analysis | PASS | embedding scatter, cluster means, profiles, bias plots |
| Autoencoder outputs resemble inputs | PASS | reconstruction pipeline + plots |
| Accessible latent space | PASS | `ConvAutoencoder.encode(...)` + saved latent vectors |
| Quantitative image metrics (MSE + MS-SSIM) | PASS | `src/planar/metrics.py` + training curves/summaries |
| Transit simulation + classifier | PASS | `src/planar/transit_sim.py` + `pipelines/transit.py` |
| Quantitative sequential metrics (ROC/AUC) | PASS | ROC plots + AUC in train summary |
| Robustness to noisy data | PASS | stress-mode evaluation and stress AUC |
| Run from start to finish with minimal effort | PASS | single config-driven CLI and scripted stages |

## Clickable Link Verification (from PDF)

Checked all embedded links on 2026-03-03.

| Link | Status |
|---|---|
| https://almascience.nrao.edu/ | Reachable |
| https://almascience.nrao.edu/aq/?result_view=project | Reachable |
| https://archive.stsci.edu/kepler/data_search/search.php | Reachable |
| https://exoplanetarchive.ipac.caltech.edu/ | Reachable |
| https://docs.google.com/forms/d/e/1FAIpQLSc9fuEwS1rkk9opENZfi1Ol6InlJBVBrPxvmxEWQ4CuBskRxA/viewform | Reachable, login-gated |
| https://drive.google.com/drive/folders/1VkS3RHkAjiKjJ6DnZmEKZ_nUv4w6pz7P?usp=sharing | Reachable, may require Google auth context |
| https://iopscience.iop.org/article/10.3847/1538-4357/aca477 | Reachable, bot/captcha protection |
| https://www.astropy.org/ | Reachable |
| https://github.com/jorge-pessoa/pytorch-msssim | Reachable |
| https://lightning.ai/docs/torchmetrics/stable/image/multi_scale_structural_similarity.html | Reachable |
| https://avanderburg.github.io/tutorial/tutorial.html | Reachable |
| https://github.com/hpparvi/PyTransit | Reachable |

## Conflicts Found and Resolved

1. Notebook used removed legacy script paths.  
   Resolution: switched notebook commands to current `planar` CLI.

2. FITS shape consistency not enforced by default.  
   Resolution: added `expected_shape: [600, 600]` in config and enforced in all pipelines.

## Residual Submission Risks (Non-code)

- Some official links are authentication/bot protected (Google Form/Drive, IOP). This is external and not a code conflict.
- If evaluators require pretrained weights and hosted notebook links, include those explicitly in submission materials.

## Final Assessment

PLANAR is aligned with EXXA 2026 technical requirements and shows no direct implementation conflicts with the source document.
