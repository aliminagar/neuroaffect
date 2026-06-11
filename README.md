# neuroaffect

A facial **affect** (emotion) analysis pipeline. Given images or video frames, it
detects faces, classifies the affect of each face, aggregates results over a
sequence, and can evaluate predictions against ground-truth labels.

## Pipeline

```
frame ──► detection ──► classification ──► aggregation ──► summary
                                                │
                          ground truth ──► evaluate ──► metrics
```

| Module                  | Responsibility                                            |
| ----------------------- | -------------------------------------------------------- |
| `src/detection.py`      | Locate faces in a frame → bounding boxes                 |
| `src/classification.py` | Classify affect for each detected face → labels + scores |
| `src/aggregation.py`    | Combine per-face/per-frame results → sequence summary     |
| `src/evaluate.py`       | Compare predictions vs. ground truth → metrics            |
| `src/cli.py`            | Command-line entry point wiring the stages together       |

> **Status:** scaffold. Stage modules expose typed interfaces with stub
> implementations marked `TODO`. Wire in real detection/classification models
> behind these interfaces.

## Setup (pip + venv, Python 3.11)

Requires Python **3.11+** (developed on 3.12).

```powershell
# From the project root
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate          # macOS / Linux

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                     # install the neuroaffect package (editable)
```

## Usage

```powershell
# Analyze an image and print an affect summary
python -m neuroaffect.cli analyze data/sample.jpg

# Evaluate predictions against ground truth
python -m neuroaffect.cli evaluate data/preds.json data/truth.json

# The console script (from `pip install -e .`) is equivalent:
neuroaffect analyze data/sample.jpg
```

## Tests

```powershell
pytest
```

## Layout

```
src/      pipeline stages (importable package)
tests/    unit tests mirroring src/
data/     inputs/outputs/datasets (gitignored)
```

## License

TODO
