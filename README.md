# ALPR Research Console

A lightweight, cross-platform Streamlit research console for the existing ALPR pipeline.

## Core design decision

**The ALPR pipeline is not modified.**

The console consumes the pipeline's persisted run artifacts:

- `results_raw_detections.csv`
- `results_rich.csv`
- `results_rich_interpolated.csv`
- `results_preprocessing_comparison.csv`
- `Final_outputs.csv`
- `results.db`
- `pipeline_metrics.md` when present
- `plate_crops/`
- `plate_crops_high_confidence/`
- `annotated_output.mp4` when present

Numbers shown by the GUI are either:
1. read directly from these artifacts, or
2. mathematically derived from them.

If the pipeline never persisted a value, the GUI says so instead of inventing it.

## Cross-device configuration

Create `.env` from `.env.example`.

Prefer configuring the **output root**, not individual files:

```env
ALPR_OUTPUT_ROOT=D:\Capstone\output
```

The GUI discovers run directories beneath that root. You can also set:

```env
ALPR_RUN_ROOT=D:\Capstone\output\20260824_111830
```

for a direct run.

The `.env` file is intentionally ignored by Git.

## Running on Windows

Put this project on D: if you want the GUI environment isolated from C:.

```powershell
cd D:\Capstone\alpr_research_console
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-gui.txt
streamlit run app.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Live mode

The console polls the selected run's persisted artifacts every `REFRESH_SECONDS`.

This means it can observe a pipeline run **without changing or controlling the pipeline**.

Examples of observable live changes:
- rows appearing in CSV artifacts
- new SQLite recognition rows
- new crop files
- appearance of final/interpolated artifacts

Some runtime-only state cannot be displayed live because the existing pipeline does not persist it (for example, active in-memory buffers). The UI explicitly labels those states as unavailable.

## Remote viewing

The easiest demo arrangement is to run Streamlit on the teammate's pipeline machine, point `ALPR_OUTPUT_ROOT` at that machine's output directory, and open the Streamlit URL from another device on the same network.

The GUI itself does not require YOLO, EasyOCR, PyTorch, or a GPU.

## Bundled sample

`sample_run/` contains the real artifacts supplied for run `20260824_111830`, so the GUI opens immediately for development.

Replace/configure the run root later; no UI code needs to change.
