# Emergency Triage Prediction System

This project predicts patient triage levels from vital signs and complaint data.
It includes both a terminal workflow and a graphical user interface.

## Project Structure
- `app.py` → graphical interface
- `main.py` → terminal version with model comparison output
- `data/triage.csv` → dataset
- `src/` → preprocessing, training, evaluation, demo utilities
- `outputs/` → saved reports, metrics, plots, demo output

## How to Run

### 1. Go to the project folder
Mac / Linux:
```bash
cd ~/Downloads/triage_ann_project
```

Windows:
```bash
cd Desktop\triage_ann_project
```

### 2. Install requirements
Standard Python:
```bash
pip install -r requirements.txt
```

Mac with Homebrew Python:
```bash
/opt/homebrew/bin/python3 -m pip install --break-system-packages -r requirements.txt
```

### 3. Run the GUI
Standard Python:
```bash
python app.py
```

Mac with Homebrew Python:
```bash
brew install python-tk
/opt/homebrew/bin/python3 app.py
```

### 4. Run the terminal version
```bash
python main.py
```

## Notes
- The GUI uses the best-performing model in the project: `MLP`.
- The terminal version prints model comparison metrics and saves output files into `outputs/`.
- If you are on macOS and the default `python3` crashes on GUI launch, use Homebrew Python as shown above.
