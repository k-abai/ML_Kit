# ML_Kit (mldk)
Reusable and demo kit for ml diagnostics

Base CLI currently supports baseline scikit-learn models for tabular supervised learning:

- Regression: Ridge Regression (default), Random Forest Regressor

- Classification: Logistic Regression (default), Random Forest Classifier

All models are trained and saved as full scikit-learn pipelines, including preprocessing (imputation, scaling, and one-hot encoding), to ensure reproducible inference.
The kits CLI is capabile of expanding current sklearn-compatible models and contains framework for custom model integration.


## Set up
1. Download and preprocess data (test and train)
2. move csv into data folder

3. Set up and activate virtual enviroment if none
```python
python -m venv .venv
.venv\Scripts\Activate.ps1
```
MacOS/Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

4. install package 
```python
pip install . 
```
test CLI availabilty
```python
mldk --help 
```
You should see:
```terminal
usage: mldk [-h] (--train TRAIN | --predict PREDICT) [--target TARGET] --out
            OUT [--model-path MODEL_PATH]
            [--task {auto,classification,regression}]
            [--model {auto,logreg,rf,ridge}] [--seed SEED] [--id-col ID_COL]   

ML diagnostics kit CLI

options:
  -h, --help            show this help message and exit
  --train TRAIN         Path to training CSV.
  --predict PREDICT     Path to prediction CSV.
  --target TARGET       Target column name for training.
  --out OUT             Output path for model or predictions.
  --model-path MODEL_PATH
                        Path to saved model joblib.
  --task {auto,classification,regression}
                        Task type (default: auto).
  --model {auto,logreg,rf,ridge}
                        Model choice (default: auto).
  --seed SEED           Random seed.
  --id-col ID_COL       Optional ID column for prediction output.
  ```

If CLI is not found use 
```python
python -m mldk.cli --help
```

## Local CLI Evaluation Example

The CLI is tested using a random linear regression dataset from kaggle:
https://www.kaggle.com/datasets/andonians/random-linear-regression

Note: The dataset is not committed to this repo, Data is seperated as (x,y)

### CLI flags  
```python
# Train
mldk --train data\train.csv --target y --task regression --model ridge --seed 42 --out models/ridge.joblib

# Evaluate directly on labeled test data
mldk --evaluate data\test.csv --model-path models/ridge.joblib --target y --input x  --out runs/rlr_eval
```
### Outputs

Model:

models/*.joblib (joblib bundle containing a scikit-learn Pipeline + metadata)

Evaluation
(Written to the directory specified by --out when running --evaluate):

- metrics.json (machine-readable metrics)

- report.md (human-readable summary)

- meta.json (timestamp, row counts, task, paths)

#### report.md

```markdown
# Evaluation Report

- Dataset size: 300 rows

## Metrics
- rmse: 3.076873514830141
- mae: 2.419241959651921
- r2: 0.9887608091964178
- mse: 9.467150626263187

## Next steps
- Review feature quality and consider additional signal.
- Compare with a stronger baseline model.
- Validate on a held-out dataset before deployment.
```
