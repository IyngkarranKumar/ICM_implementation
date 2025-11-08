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
```bash
python generate_ICM_labels.py
```

The directory to save file path is configurable in `config.py` via `config.ICM`.



## Run Evaluations
```bash
python run_evals.py
```

The script saves metrics to `outputs/truthfulqa_eval_results_<timestamp>.json` and a bar chart PNG in `outputs/`.
