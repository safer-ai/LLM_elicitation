# SPAR Spring 2026 Results Manifest

This repository PR keeps the reproducible code, prompts, experiment setups, and
small curated result summaries in Git. Full raw elicitation outputs are not
committed in the clean PR because they are generated artifacts and make review
large and noisy.

Raw result backup locations checked during cleanup:

- Local ignored folders in this checkout:
  - `output_data/` (~300 MB)
  - `prompt_sensitivity/output/` (~5 MB)
  - `report_analyses/results/` (~1.3 MB)
  - `intra_benchmark_calibration/experiments/G_model_sweep/results/` (~71 MB)
  - `intra_benchmark_calibration/experiments/H_task_variance_bin1/results/` (~752 KB)
- Historical variance branch:
  - `origin/spar_spring_2026_variance` at commit `7f28500`
  - 2,283 raw result files under the result/output paths above
  - tracked raw result payload size: ~297 MB

Curated result summaries included in this PR:

- `prompt_sensitivity/curated_results/wasserstein_distances_all_conditions.txt`
- `report_analyses/curated_results/model_sweep_baseline/`
- `report_analyses/curated_results/model_sweep_baseline_all_tasks/`
- `intra_benchmark_calibration/experiments/H_task_variance_bin1/summary/bin1_estimates_combined.csv`
- `report_analyses/frechet_anova/*.txt`
- `report_analyses/frechet_anova/*.csv`

Recommended permanent archive before deleting old branches:

1. Upload the raw result folders to a durable artifact store such as a team
   drive, S3/GCS, Hugging Face Dataset, OSF, Zenodo, GitHub Release asset, DVC,
   or Git LFS.
2. Record the final artifact URL and checksum manifest here.
3. Keep this PR limited to source, configs, prompts, small summaries, and
   scripts that consume the raw archive.
