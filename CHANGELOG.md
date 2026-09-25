# Changelog

## 0.4.0 — 2026-09-25

This release completes the local, configurable photo recommendation workflow and
adds stronger variety controls without repeating previously completed assessments.

### Added and improved

- Default shortlist of 30 photos, configurable from 1 to 100.
- A configurable per-day limit (default one photo per known date).
- Additional visual-similarity filtering across the selected set.
- Reported date coverage, unknown-date counts, and reasons for diversity exclusions.
- Automatic recovery from truncated model output using larger response budgets
  and shorter explanations; incomplete scores are never accepted.
- Face-scan progress now includes the total file count and percentage reached.
- `photo-selector --version`, documented resume instructions, and a backup/restore guide.
- Tests for response recovery, day limits, visual variety, cache reuse, and setup.

### Compatibility

- Existing face scans and complete visual assessments remain reusable. No model
  change or blanket reassessment is required for this update.
- Existing private configurations retain their requested result count. Missing
  `max_per_day` and `min_visual_distance` settings default to `1` and `12`.
- To adopt the broader shortlist, set `top = 30` in your private `selector.toml`,
  then run `photo-selector rank --config selector.toml`.
- The provided container targets Linux x86_64 and a single compatible AMD GPU.
  CPU-only and other GPU backends are not supported by this setup.
- Day limits cannot apply to unknown dates. Visual hashes do not reliably recognise
  clothing or settings; these controls are heuristics, not outfit detection.

### Validation

Automated tests cover the pipeline and recovery behavior using synthetic fixtures.
A clean source snapshot is tested and packaged before release. Real local model
assessment, recovery from a truncated response, and cached reranking have also
been exercised. A fresh Docker image rebuild is not verified from the development
container because it does not expose a Docker daemon.

## 0.3.0

- Initial end-to-end library scan, local visual assessment, and recommendation reports.
- Configurable reference photos, criteria, weights, exclusions, and selection rules.
- Resumable caches, verified model downloads, and private Dev Container settings.
