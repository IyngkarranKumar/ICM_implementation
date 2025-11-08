#running evals
from datetime import datetime
import argparse

class evals:
    DEBUG = False
    USE_HYPERBOLIC_API = True
    chicken_shot = "hello"

    n_test_samples = 100
    n_many_shot_samples = 20
    use_base_chat_prompt = True
    base_chat_prompt_path = "pretrained_chat_prompt.txt"

    prepend_prompt = "Answer with ONLY the word True or the word False. Do not include any other text, explanation, or punctuation."
    classification_append_str = "I think this claim is:"
    max_new_tokens = 5

    # Set model names
    # base_model = "Qwen/Qwen2.5-0.5B"
    # chat_model = "Qwen/Qwen2.5-0.5B-Instruct"
    base_model = "meta-llama/Meta-Llama-3.1-405B"
    chat_model = "meta-llama/Meta-Llama-3.1-405B-Instruct"
    
    

    EVALS_TO_RUN = {
        "base": True,
        "chat": True,
        "golden": True,
        "ICM": True,
    }

    date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    train_data_path = "mats_9.0_feng_ududec_work_test/data/truthfulqa_train.json"
    test_data_path = "mats_9.0_feng_ududec_work_test/data/truthfulqa_test.json"
    ICM_data_path = "outputs/truthfulqa_train_ICM_data.json"
    EVAL_RESULTS_SAVE_PATH = f"outputs/truthfulqa_eval_results_{date_str}.json"

class ICM:
    DEBUG = False
    USE_HYPERBOLIC_API = True

    T_0 = 10.0
    T_min = 0.01
    beta = 0.99
    alpha = 30
    K = 8
    max_iter = 1000
    max_new_tokens = 5
    logprobs = 10
    topk = 5  # for debugging
    label_set = ["True", "False"]

    # ICM_model = "Qwen/Qwen2.5-0.5B"
    ICM_model = "meta-llama/Meta-Llama-3.1-405B"
    if "instruct" in ICM_model.lower():
        raise ValueError("ICM model must be a base model - Hyperion API does not support logprobs for instruct models")

    train_data_path = "mats_9.0_feng_ududec_work_test/data/truthfulqa_train.json"

    save_dir = "outputs"
    date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    ICM_save_data_path = f"{save_dir}/truthfulqa_train_ICM_data_{date_str}.json"


def parse_args():
    """Parse command line arguments and override config values"""
    parser = argparse.ArgumentParser(description="TruthfulQA Evaluation Script")
    
    # Sample sizes
    parser.add_argument("--n-test-samples", type=int, 
                        help="Number of test samples to evaluate",default=100)
    parser.add_argument("--n-many-shot-samples", type=int, 
                        help="Number of many-shot examples to use",default=20)
    parser.add_argument("--ICM-data-path", type=str, 
                        help="Path to ICM data",default=evals.ICM_data_path)
    
    # Eval selection - choose which evals to run
    parser.add_argument("--evals", type=str, nargs='+', 
                        choices=["base", "chat", "golden", "ICM"],
                        help="Which evaluations to run (space-separated). If not specified, runs all enabled in config.",default=["base", "chat", "golden", "ICM"])
    

    
    # ICM-specific parameters
    parser.add_argument("--icm-max-iter", type=int,
                        help="Maximum iterations for ICM algorithm",default=500)
    
    args = parser.parse_args()
    
    # Override sample sizes
    if args.n_test_samples is not None:
        evals.n_test_samples = args.n_test_samples
    if args.n_many_shot_samples is not None:
        evals.n_many_shot_samples = args.n_many_shot_samples
    if args.ICM_data_path is not None:
        evals.ICM_data_path = args.ICM_data_path
    # Override ICM parameters
    if args.icm_max_iter is not None:
        ICM.max_iter = args.icm_max_iter
    

    for eval_name in evals.EVALS_TO_RUN.keys():
        evals.EVALS_TO_RUN[eval_name] = eval_name in args.evals

    return args