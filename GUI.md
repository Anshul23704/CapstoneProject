# ALPR Streamlit GUI Guide

The automated ALPR pipeline no longer launches the GUI automatically to prevent multiple windows from opening or background processes from lingering.

To monitor the pipeline's progress and view the results in real-time, you can launch the Streamlit GUI manually in a separate terminal window.

## How to launch the GUI

1. Open a new terminal window and navigate to the project directory:
   ```bash
   cd /Users/amshul/Work/Capstone/CapstoneAmshul/CapstoneProject
   ```

2. Activate your virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run GUI/app.py
   ```

## How to view a specific pipeline run

By default, the GUI may attempt to read the most recent run in the `output/` directory, or it may need to be pointed to a specific run.

If you are currently running the pipeline (e.g. `python main_pipeline.py`), the pipeline will create a new directory inside `output/` named with the current timestamp (e.g. `output/20260829_092227`).

You can explicitly tell the GUI which run to monitor by setting the `ALPR_RUN_ROOT` environment variable before launching it:

```bash
ALPR_RUN_ROOT=output/20260829_092227 streamlit run GUI/app.py
```

## Refresh Rate
To change how often the GUI updates while the pipeline is running, set the `REFRESH_SECONDS` environment variable (default is 2 seconds if not set):

```bash
REFRESH_SECONDS=5 ALPR_RUN_ROOT=output/20260829_092227 streamlit run GUI/app.py
```
