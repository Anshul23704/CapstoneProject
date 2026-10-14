# ALPR Plate Evaluation Tool

Quick accuracy evaluation tool comparing ground-truth plates against pipeline predictions.

## Directory Structure

```
evaluation/
├── ActualPlates/
│   ├── test1.txt        # Ground truth plates for test1.mp4 (one plate per line)
│   ├── test3.txt        # Ground truth plates for test3.mp4
│   └── <video_name>.txt # Add any future video's ground truth here
├── compare_plates.py    # Main evaluation script
└── README.md
```

## Adding Ground-Truth Plates

In `evaluation/ActualPlates/`, create a file `<video_name>.txt` (or `<video_name>.json`) matching your video file name (e.g., `test3.txt` or `test4.mp4.txt`).

### Format: Plain Text (`.txt`)
Simply put one license plate per line (uppercase, no spaces, comments starting with `#` are ignored):
```text
KA01AQ0320
KA01NF0626
KA03NA3721
KA05AL3013
```

### Format: JSON (`.json`)
Alternatively, you can provide a JSON list of strings:
```json
[
  "KA01AQ0320",
  "KA01NF0626",
  "KA03NA3721",
  "KA05AL3013"
]
```

## How to Run

```bash
# Compare against default (test3) using latest run
python evaluation/compare_plates.py

# Compare for a specific video
python evaluation/compare_plates.py test3
python evaluation/compare_plates.py test1

# Compare against a specific run output folder
python evaluation/compare_plates.py test3 --run output/20261013_204125

# Output in JSON format
python evaluation/compare_plates.py test3 --json
```
