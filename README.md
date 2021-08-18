# trajectory-prediction

## Project setup
We recommend using conda with python 3.7 and tensorflow 2:
```
conda create --name py37_tensorflow2 python=3.7
conda activate py37_tensorflow2
conda install --force-reinstall -y -c conda-forge --file requirements.txt
```

### Execution example (from project main dir):

#### Training and evaluating (for ICD-9 input codes data)
`python3.7 src/main.py "data/mimic-icd9/mimic_90-10_855" compiled_models/model --hiddenDimSize=[271]`

#### Evaluating only (for ICD-9 input codes data)
`python3.7 src/evaluation.py "data/mimic-icd9/mimic_90-10_855" compiled_models/model.50/`