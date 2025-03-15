# Reinforcement-Learning-RL-for-Large-Language-Models-LLMs-


# LLM Preference Fine-Tuning

This repository explores Reinforcement Learning (RL) for Large Language Models (LLMs) using two key methods . We provide two examples:

1. **GPT-2 PPO Fine-Tuning for Positive Reviews:**  
   Fine-tune GPT-2 using Proximal Policy Optimization (PPO) to generate positive movie reviews from the IMDB dataset.

2. **Llama 2 DPO Fine-Tuning:**  
   Fine-tune Llama 2 using Direct Preference Optimization (DPO) to align model outputs with human preferences using Stack Exchange paired responses.
   
Both methods help fine-tune language models using reinforcement learning techniques to improve response quality, alignment with human feedback, and overall model performance.

---

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [GPT-2 PPO Fine-Tuning](#gpt-2-ppo-fine-tuning)
- [Llama 2 DPO Fine-Tuning](#llama-2-dpo-fine-tuning)
- [References](#references)
- [Contributing](#contributing)

---

## Overview

Preference-based fine-tuning is used to steer language models toward outputs that are better aligned with human judgments. Two common approaches are:

- **PPO (Proximal Policy Optimization):**  
  Uses an external reward signal—here provided by a BERT-based sentiment classifier—to reward a GPT-2 model for generating positive movie reviews. A reference model ensures the optimized model does not deviate too far from its original behavior.

- **DPO (Direct Preference Optimization):**  
  Eliminates the need for a separate reward model and RL optimization by directly using preference data. The model is trained using a supervised loss that compares the likelihood of a “chosen” (preferred) response against a “rejected” one.

---

## Prerequisites

Ensure you have Python 3.8+ installed. Install the required packages:

```bash
pip install transformers peft trl accelerate datasets torch wandb
```

---

## GPT-2 PPO Fine-Tuning

In this example, we fine-tune a GPT-2 model on IMDB movie reviews to encourage it to generate positive outputs.

### Steps Overview

1. **Dataset Preparation:**  
   - Load the IMDB dataset and filter reviews longer than 200 characters.
   - Tokenize the text using GPT-2’s tokenizer.

2. **Model Setup:**  
   - Load the GPT-2 model pre-fine-tuned on the IMDB dataset.
   - Create two copies: one to be optimized and one as a reference model.

3. **PPO Training:**  
   - Use the TRL `PPOTrainer` to generate responses from GPT-2.
   - Reward the model using a BERT sentiment classifier that scores the generated text.
   - Optimize using PPO so that positive responses are more likely to be generated.

### Example Code Snippet

```python
# Import dependencies
import torch
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# PPO configuration
config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    learning_rate=1.41e-5,
    log_with="wandb",
)

# Build dataset function
def build_dataset(config, dataset_name="stanfordnlp/imdb"):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200)
    return ds

dataset = build_dataset(config)

# Load GPT-2 models (optimizable and reference)
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# Initialize PPO Trainer
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset)
```



---

## Llama 2 DPO Fine-Tuning

This example demonstrates how to fine-tune Llama 2 using Direct Preference Optimization (DPO) with preference data.

### Steps Overview

1. **Prepare the Preference Dataset:**  
   - Use data with three fields:
     - `prompt`: The question or context.
     - `chosen`: The preferred (better) answer.
     - `rejected`: The less preferred answer.
   - Example: Use the Stack Exchange paired dataset.

2. **Model Setup with QLoRA:**  
   - Load the Llama 2 7B model using 4-bit quantization.
   - Apply LoRA adapters for memory efficiency.

3. **DPO Training:**  
   - Initialize the TRL `DPOTrainer` with the base model and a reference model.
   - Use a simple binary cross-entropy loss that leverages the reference model.
   - Train directly on the preference data.

### Example Code Snippet

```python
from datasets import load_dataset

def return_prompt_and_responses(samples):
    return {
        "prompt": ["Question: " + q + "\n\nAnswer: " for q in samples["question"]],
        "chosen": samples["response_j"],
        "rejected": samples["response_k"],
    }

dataset = load_dataset("lvwerra/stack-exchange-paired", split="train")
dataset = dataset.map(return_prompt_and_responses, batched=True, remove_columns=dataset.column_names)
```



---

---

## References

- [Hugging Face TRL Library](https://huggingface.co/docs/trl)
- [Direct Preference Optimization (DPO) Paper](https://arxiv.org/abs/2305.18290)
- [GPT-2 IMDB Fine-Tuning Example](https://huggingface.co/models?search=gpt2-imdb)
- [Meta AI - Llama 2](https://ai.facebook.com/blog/llama-2/)
- [QLoRA for Efficient Fine-Tuning](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

---

## Contributing

Contributions, suggestions, and bug reports are welcome! Please open an issue or submit a pull request.

---


