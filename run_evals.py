#%%  imports
import json
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import utils
import logging
import importlib
import os
import matplotlib.pyplot as plt
import config
from dotenv import load_dotenv
from datetime import datetime
np.random.seed(42)
random.seed(42)

load_dotenv(".env.main")

importlib.reload(utils)
importlib.reload(config)

args = config.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



with open(config.evals.train_data_path, "r") as f:
    train_data = json.load(f)

with open(config.evals.test_data_path, "r") as f:
    test_data = json.load(f)

with open(config.evals.ICM_data_path, "r") as f:
    ICM_data = json.load(f)


if config.evals.n_test_samples is not None:
    test_data = test_data[:config.evals.n_test_samples]

#give train and test data unique sample indices
for idx, sample in enumerate(train_data):
    sample['sample_idx'] = idx

#add sample index to each sample
for idx, sample in enumerate(test_data):
    sample['sample_idx'] = idx

ground_truth_test_labels = {sample['sample_idx']: sample['label'] for sample in test_data}

if config.evals.n_many_shot_samples is not None:
    assert config.evals.n_many_shot_samples <= len(train_data), "n_many_shot_samples must be less than or equal to the length of train_data"
else:
    config.evals.n_many_shot_samples = len(train_data)


#%% data structure inits

eval_names = ["base","chat","golden","ICM"]
eval_keys = ["sample_idx","classification_task","generated_text","predicted_label"]
EVAL_RESULTS = {eval_name: {eval_key: [] for eval_key in eval_keys} for eval_name in eval_names}

eval_results_dir = os.path.dirname(config.evals.EVAL_RESULTS_SAVE_PATH)
if eval_results_dir and not os.path.exists(eval_results_dir):
    os.makedirs(eval_results_dir, exist_ok=True)



#%% base model baseline

if config.evals.EVALS_TO_RUN["base"]:

    for i, sample in enumerate(test_data):
        logger.info(f"Processing sample {i+1} of {len(test_data)} for base model baseline")

        #format for classification task
        question_choice_classification_task_str = utils.create_truthfulqa_classification_task(sample['question'], sample['choice'], config.evals.prepend_prompt,use_base_chat_prompt=config.evals.use_base_chat_prompt)
        sample_idx = sample['sample_idx']
        golden_label = sample["label"]
        golden_bool_str = utils.get_golden_bool_str(golden_label)
        input_text = question_choice_classification_task_str

        text_response = ""
        while text_response == "":
            text_response = utils.hyperbolic_call_text_only(input_text, config.evals.base_model, max_new_tokens=config.evals.max_new_tokens)
        generated_text = text_response
    #score text output 
        predicted_label = utils.get_predicted_label(generated_text)

        #store results 
        # Ensure required keys are in EVAL_RESULTS["base"] dictionary before appending
        # Add "base" key to eval results if not already present
        
        EVAL_RESULTS["base"]["sample_idx"].append(sample_idx)
        EVAL_RESULTS["base"]["classification_task"].append(input_text)
        EVAL_RESULTS["base"]["generated_text"].append(generated_text)
        EVAL_RESULTS["base"]["predicted_label"].append(predicted_label)



    with open(config.evals.EVAL_RESULTS_SAVE_PATH, "w") as f:
        json.dump(EVAL_RESULTS, f)
    logger.info(f"Base model baseline results saved to {config.evals.EVAL_RESULTS_SAVE_PATH}")
    logger.info("\n Base model baseline complete \n")

#%% chat model baseline

if config.evals.EVALS_TO_RUN["chat"]:
   
    for i, sample in enumerate(test_data):
        logger.info(f"Processing sample {i+1} of {len(test_data)} for chat model baseline")

        #format for classification task
        question_choice_classification_task_str = utils.create_truthfulqa_classification_task(sample['question'], sample['choice'], config.evals.prepend_prompt)
        sample_idx = sample['sample_idx']
        golden_label = sample["label"]
        golden_bool_str = utils.get_golden_bool_str(golden_label)
        input_text = question_choice_classification_task_str

        text_response = ""
        while text_response == "":
            text_response = utils.hyperbolic_call_text_only(input_text, config.evals.chat_model, max_new_tokens=config.evals.max_new_tokens)
        generated_text = text_response
        #score text output 
        predicted_label = utils.get_predicted_label(generated_text)

        #store results 
        #EVAL_RESULTS["base"]["sample_idx"].append(i)
        EVAL_RESULTS["chat"]["sample_idx"].append(sample_idx)
        EVAL_RESULTS["chat"]["classification_task"].append(input_text)
        EVAL_RESULTS["chat"]["generated_text"].append(generated_text)
        EVAL_RESULTS["chat"]["predicted_label"].append(predicted_label)

    with open(config.evals.EVAL_RESULTS_SAVE_PATH, "w") as f:
        json.dump(EVAL_RESULTS, f)
    logger.info(f"Chat model baseline results saved to {config.evals.EVAL_RESULTS_SAVE_PATH}")
    logger.info("\n Chat model baseline complete \n")

#%% many-shot golden 

if config.evals.EVALS_TO_RUN["golden"]:
    np.random.seed(42)
    random.seed(42)

    for i, sample in enumerate(test_data):
        logger.info(f"Processing sample {i+1} of {len(test_data)} for golden baseline")

        #format for classification task
        many_shot_examples_str = utils.create_many_shot_truthfulqa_examples(train_data, config.evals.n_many_shot_samples) #get many-shot examples from train_data
        question_choice_classification_task_str = utils.create_truthfulqa_classification_task(sample['question'], sample['choice'], config.evals.prepend_prompt)
        sample_idx = sample['sample_idx']
        golden_label = sample["label"]
        golden_bool_str = utils.get_golden_bool_str(golden_label)
        input_text = f"{many_shot_examples_str}\n\n{question_choice_classification_task_str}"

        text_response = ""
        while text_response == "":
            text_response = utils.hyperbolic_call_text_only(input_text, config.evals.chat_model, max_new_tokens=config.evals.max_new_tokens)
        generated_text = text_response
        #score text output 
        predicted_label = utils.get_predicted_label(generated_text)

        #store results 
        EVAL_RESULTS["golden"]["sample_idx"].append(sample_idx)
        EVAL_RESULTS["golden"]["classification_task"].append(input_text)
        EVAL_RESULTS["golden"]["generated_text"].append(generated_text)
        EVAL_RESULTS["golden"]["predicted_label"].append(predicted_label)
        
    with open(config.evals.EVAL_RESULTS_SAVE_PATH, "w") as f:
        json.dump(EVAL_RESULTS, f)
    logger.info(f"golden baseline results saved to {config.evals.EVAL_RESULTS_SAVE_PATH}")
        
    logger.info("\n golden baseline complete \n")



#%% many-shot ICM

ICM_labels_path = "mats_9.0_feng_ududec_work_test/data/truthfulqa_train_ICM.json"
if config.evals.EVALS_TO_RUN["ICM"]:

    for i, sample in enumerate(test_data):
        logger.info(f"Processing sample {i+1} of {len(test_data)} for ICM")

        #format for classification task
        many_shot_examples_str = utils.create_many_shot_truthfulqa_examples(ICM_data, config.evals.n_many_shot_samples) #get many-shot examples from ICM_data
        question_choice_classification_task_str = utils.create_truthfulqa_classification_task(sample['question'], sample['choice'], config.evals.prepend_prompt)
        sample_idx = sample['sample_idx']
        golden_label = sample["label"]
        golden_bool_str = utils.get_golden_bool_str(golden_label)
        input_text = f"{many_shot_examples_str}\n\n{question_choice_classification_task_str}"

        text_response = ""
        while text_response == "":
            text_response = utils.hyperbolic_call_text_only(input_text, config.evals.chat_model, max_new_tokens=config.evals.max_new_tokens)
        generated_text = text_response
        #score text output 
        predicted_label = utils.get_predicted_label(generated_text)

        #store results 
        EVAL_RESULTS["ICM"]["sample_idx"].append(sample_idx)
        EVAL_RESULTS["ICM"]["classification_task"].append(input_text)
        EVAL_RESULTS["ICM"]["generated_text"].append(generated_text)
        EVAL_RESULTS["ICM"]["predicted_label"].append(predicted_label)
    
    with open(config.evals.EVAL_RESULTS_SAVE_PATH, "w") as f:
        json.dump(EVAL_RESULTS, f)
    logger.info(f"ICM results saved to {config.evals.EVAL_RESULTS_SAVE_PATH}")

    logger.info("\n ICM complete \n")


#%% aggregate results


# Accuracy bar chart for 'base' and 'chat'


# Collect accuracies
accuracies = {}
eval_labels = []
accuracy_values = []

for eval_name in ["base", "chat", "golden", "ICM"]:
    if config.evals.EVALS_TO_RUN.get(eval_name, False):
        accuracy = utils.compute_accuracy(EVAL_RESULTS.get(eval_name, {}), ground_truth_test_labels)
        accuracies[eval_name] = accuracy
        eval_labels.append(eval_name)
        accuracy_values.append(accuracy)

# Plot
fig, ax = plt.subplots()
colors = {
    "base": "#ff69b4",      # pink
    "chat": "#ff69b4",      # dotted pink (handled in bar style)
    "golden": "#ffd700",    # golden/yellow
    "ICM": "#40e0d0",       # turquoise blue
}
bars = ax.bar(eval_labels, accuracy_values, color=[colors[eval_name] for eval_name in eval_labels])
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Test Accuracy - TruthfulQA")
ax.grid(alpha=0.5)


for idx, (bar, val) in enumerate(zip(bars, accuracies.values())):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", ha="center", va="bottom")
    # Add dots to "chat" bar
    if eval_labels[idx] == "chat": 
        # Overlay dots pattern to match original paper
        x = bar.get_x()
        width = bar.get_width()
        y0 = 0
        y1 = val
        dot_spacing_x = width / 8
        dot_spacing_y = (y1 - y0) / 8 if y1 > 0 else 0.05
        x_dots = [x + dot_spacing_x/2 + i*dot_spacing_x for i in range(8)]
        y_dots = [y0 + dot_spacing_y/2 + j*dot_spacing_y for j in range(int(max(1, (y1-y0)/dot_spacing_y)))]
        for xd in x_dots:
            for yd in y_dots:
                if yd < y1:  # Only within bar height
                    ax.plot(xd, yd, marker='o', color='k', markersize=2, alpha=0.5, zorder=5)


current_time_str = datetime.now().strftime("%H_%M_%S")
plt.savefig(f"outputs/truthfulqa_accuracy_bar_chart_{current_time_str}.png")
logger.info(f"Accuracy bar chart saved to outputs/truthfulqa_accuracy_bar_chart_{current_time_str}.png")

