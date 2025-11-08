# TruthfulQA Evaluation Runner

MATS Worktest - Implementing ICM algorithm for TruthfulQA

## Environment
- `python3.10+`
- `mamba` required - install with `conda install -n base -c conda-forge mamba`
- Access to Hyperbolic API (requires `HYPERBOLIC_API_KEY`)

```bash
mamba create -y -n ICM python=3.10
mamba run -n ICM pip install -r mats_9.0_feng_ududec_work_test/requirements.txt
```

## Configuration
- Make copy of `.env.main.template` and add HYPERBOLIC_API_KEY



- Adjust `config.py` to toggle evals, model names, dataset paths, or sample counts.


## Data
TruthfulQA train/test JSON files in `mats_9.0_feng_ududec_work_test/data/`. Generated ICM labels and eval outputs in `outputs/`.


## Generate ICM Labels

Generate ICM labels with a specified maximum number of iterations (default 1000). Saved to `outputs/truthfulqa_train_ICM_data_{date_time}.json`

```bash
python generate_ICM_labels.py --icm-max-iter 1000
```

The directory to save file path is configurable in `config.py` via `config.ICM`.



## Run Evaluations

Run evaluations with specific number of test examples and in-context examples. 

```bash
python run_evals.py --n-test-samples 100 --n-many-shot-samples 20
```

Specify which evaluations to run from ['base','chat','golden','ICM'] with --evals argument (default is all). NOTE: ICM dataset must be generated before 'ICM' evaluation can be run. 


```bash
python run_evals.py --n-test-samples 100 --n-many-shot-samples 20 --evals chat ICM
```

The script saves metrics to `outputs/truthfulqa_eval_results_<timestamp>.json` and a bar chart PNG in `outputs/`.
