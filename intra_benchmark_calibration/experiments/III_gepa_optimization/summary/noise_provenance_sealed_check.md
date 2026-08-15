# Provenance: sealed_check arm (2026-08-10 session)

The exact protocol command

    uv run python scripts/val_noise_study.py --prompts seed,july_cand20 --repeats-val 0 --repeats-sealed 3 --tag sealed_check

was launched three times in this cloud session and was killed mid-flight each
time by container restarts outside our control (attempt 1 after 4 of 6
repeat-blocks; attempts 2 and 3 after 2 blocks each). No code, config, seed,
or manifest was modified at any point.

`noise_cells_sealed_check.jsonl` is stitched from complete 230-cell
repeat-blocks only:

- `july_cand20` sealed repeats 0,1,2 and `seed` sealed repeat 0: attempt 1,
  preserved verbatim in `noise_cells_sealed_check.jsonl.pre-restart`.
- `seed` sealed repeats 1,2: a seed-only completion run
  (`--prompts seed --repeats-val 0 --repeats-sealed 2 --tag sealed_check_fill`,
  same config/manifest, fresh forecaster calls), preserved verbatim in
  `noise_cells_sealed_check_fill.jsonl`; its repeat indices 0,1 are relabeled
  to 1,2 in the stitched file. Repeats are exchangeable fresh draws, so
  relabeling does not affect any statistic.

`noise_summary_sealed_check.{txt,json}` were reconstructed from the stitched
cells with the same formulas the script uses (failure Brier 1.0; population
grand mean per repeat; sample SD across repeats).

Duplicate partial blocks from the killed attempts 2 and 3 are preserved
unmodified in `noise_cells_sealed_check.jsonl.pre-restart2` and
`.pre-restart3` (each: july_cand20 sealed repeats 0,1). They are bonus
replicate measurements of the same instrument and are not part of the
stitched file.
