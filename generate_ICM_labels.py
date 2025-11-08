#%% imports
import json 
import copy
import numpy as np
import random
import utils 
import importlib
import torch
import config
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

importlib.reload(utils)
importlib.reload(config)

args = config.parse_args()

from dotenv import load_dotenv
load_dotenv(".env.main")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(42)
random.seed(42)


with open(config.ICM.train_data_path, "r") as f:
    train_data = json.load(f)

D_init = train_data
n_train_samples = len(train_data)
for idx, sample in enumerate(D_init):
    sample['sample_idx'] = idx #add index to each sample to uniquely identify it
D_init_copy = copy.deepcopy(D_init) #make a copy of the full dataset to avoid modifying the original
D_unlabel = [{k: v for k, v in sample.items() if k != 'label'} for sample in D_init]
D_ICM = [] #dataset with ICM labels

#%%

LOSS_DATA_TMP = [None,None] #holds D_t and D_t+1

#init 
random_indices = np.random.choice(len(D_init_copy), size=config.ICM.K, replace=False)
for idx in sorted(random_indices, reverse=True):
    sample = D_init[idx]
    random_label = np.random.choice([0,1])
    sample_with_label = {**sample, 'label': random_label.item()}
    D_ICM.append(sample_with_label)
    del D_init_copy[idx]


logger.info(f"Size of D_ICM before ICM labelling: {len(D_ICM)}")


sample_iter = 0

while len(D_ICM) < n_train_samples and sample_iter < config.ICM.max_iter and len(D_init_copy) > 0:
    logger.info(f"Iteration {sample_iter+1}/{config.ICM.max_iter}")

    # Sample a random index from the current D_init_copy
    idx = np.random.choice(len(D_init_copy))
    sample = D_init_copy[idx]
    
    temperature_iter = np.max([
        config.ICM.T_min,
        (config.ICM.T_0 / (1 + config.ICM.beta * np.log(sample_iter + 1)))
    ])

    # Prepare input text
    many_shot_examples_str = utils.create_many_shot_truthfulqa_examples(D_ICM)
    question, choice = sample['question'], sample['choice']
    classification_task_str = utils.create_truthfulqa_classification_task(question, choice)
    input_prompt_with_examples = f"{many_shot_examples_str}\n\n{classification_task_str}"
    input_text = input_prompt_with_examples

    if config.ICM.USE_HYPERBOLIC_API:
        label_max_prob, max_prob, top_logprobs = utils.get_argmax_label_hyperbolic(
            input_text, 
            config.ICM.ICM_model, 
            config.ICM.max_new_tokens, 
            config.ICM.label_set, 
            logprobs=config.ICM.logprobs
        )
    else:
        raise ValueError("Local labelling not implemented")

    ICM_predicted_label_numeric = utils.get_predicted_label(label_max_prob) # 0 or 1 or nan
    sample_with_ICM_label = {**sample, 'label': ICM_predicted_label_numeric}

    # get losses
    if sample_iter == 0:
        D_t_loss = 0  # no previous loss data
    else:
        D_t_loss = LOSS_DATA_TMP[0]
    D_t_1 = config.ICM.alpha * max_prob  # loss is prob of most likely label times alpha

    # compute delta
    DELTA = D_t_1 - D_t_loss

    # update losses
    LOSS_DATA_TMP[0] = D_t_1

    # accept or reject sample with ICM label
    if DELTA > 0:
        D_ICM.append(sample_with_ICM_label)
        del D_init_copy[idx]
    else: # if D_t scores higher than D_t+1, accept sample with probability exp(DELTA/temperature_iter)
        if np.random.uniform(0, 1) < np.exp(DELTA / temperature_iter):
            D_ICM.append(sample_with_ICM_label)
            del D_init_copy[idx]
        else:
            # don't add or remove sample
            pass

    sample_iter += 1

    # Log progress at every 10 samples reached in D_ICM
    if len(D_ICM) > 0 and len(D_ICM) % 10 == 0:
        logger.info(f"ICM progress: {len(D_ICM)} samples labelled.")

    if sample_iter == config.ICM.max_iter:
        logger.info(f"Reached max iterations: {config.ICM.max_iter}")
        break

logger.info(f"Size of D_ICM after ICM labelling: {len(D_ICM)}")


#%%

current_time_str = datetime.now().strftime("%H_%M_%S")
with open(config.ICM.ICM_save_data_path, "w") as f:
    json.dump(D_ICM, f, indent=2)
logger.info(f"ICM data saved to {config.ICM.ICM_save_data_path}")
