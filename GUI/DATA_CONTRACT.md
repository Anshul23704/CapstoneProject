# ALPR GUI Data Contract

## Source of truth

The GUI consumes persisted artifacts from a pipeline run. It does not invent missing runtime state.

| GUI capability | Artifact | Type |
|---|---|---|
| raw detections / track IDs | results_raw_detections.csv | direct |
| rich recognition trajectory | results_rich.csv | direct |
| interpolated trajectory | results_rich_interpolated.csv | direct |
| preprocessing branch comparison | results_preprocessing_comparison.csv | direct |
| recognition/fusion records | results.db / ocr_results | direct |
| final high-confidence outputs | Final_outputs.csv | direct |
| ingestion report | pipeline_metrics.md | direct when present |
| context / bilateral / adaptive images | plate_crops*/ + Final_outputs.csv paths | direct when files exist |
| annotated video | annotated_output.mp4 | direct when supplied |

## Derived values

Examples:
- unique track count
- frame count represented in an artifact
- success rate among persisted recognition jobs
- mean/median/max readings per successful job
- preprocessing winner counts
- interpolated/original row multiplier
- per-track first/last frame and observation count

These are deterministic transformations of persisted data.

## Not available without changing the pipeline

Examples:
- active in-memory `VehicleBuffer` count
- worker queue depth
- every rejected detector candidate
- full frame-selection ranking trace, if not persisted
- live per-stage runtime timing when not written to an artifact
- ground-truth accuracy/precision/recall

The GUI should show these as unavailable rather than fabricate them.
