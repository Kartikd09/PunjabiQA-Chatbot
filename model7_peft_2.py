import pandas as pd
import json
from transformers import Trainer, TrainingArguments
from datasets import load_dataset,DatasetDict

data = pd.read_excel('./qa-pairs-3.xlsx')

data = data[['instruction', 'response']]

data.to_csv("qa_data_augment.csv", index=False)

from datasets import load_dataset

dataset = load_dataset("csv", data_files="qa_data_augment.csv")

train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)


train_dataset = train_testvalid['train']
valid_dataset = test_valid['train']
test_dataset = test_valid['test']

from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
# tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to GPU if available

# import torch

# def preprocess_function(examples):
#     # Tokenize each instruction and response pair
#     inputs = examples["instruction"]
#     targets = examples["response"]

#     # Handle None values in both inputs and targets lists
#     inputs = [q if q is not None else "" for q in inputs]
#     targets = [a if a is not None else "" for a in targets]

#     # Tokenize inputs and targets
#     model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

#     # Tokenize targets separately
#     labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids

#     # Assign tokenized labels
#     model_inputs["labels"] = labels

#     # Make sure all tensors in model_inputs are contiguous
#     for k, v in model_inputs.items():
#         if isinstance(v, torch.Tensor) and not v.is_contiguous():
#             model_inputs[k] = v.contiguous()

#     return model_inputs

import torch
def preprocess_function(examples):
    # Tokenize each instruction and response pair with prompt design

    inputs = examples["instruction"]
    targets = examples["response"]

    # Handle None values in both inputs and targets lists
    inputs = [q if q is not None else "" for q in inputs]
    targets = [a if a is not None else "" for a in targets]

    # Add prompt design: prepend "Question:" to inputs and "Answer:" to outputs
    formatted_inputs = ["Question: " + q + " Answer:" for q in inputs]

    # Tokenize inputs and targets with the tokenizer
    model_inputs = tokenizer(formatted_inputs, max_length=128, truncation=True, padding="max_length")

    # Tokenize targets separately
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids

    # Assign tokenized labels to the model inputs
    model_inputs["labels"] = labels

    # Make sure all tensors are contiguous
    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor) and not v.is_contiguous():
            model_inputs[k] = v.contiguous()

    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_valid = valid_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

import gc
from transformers import Trainer, TrainingArguments,TrainerCallback

import torch

# Make sure model weights are contiguous
def make_contiguous(model):
     for param in model.parameters():
         if param.data.is_contiguous() is False:
             param.data = param.data.contiguous()

 # Ensure model weights are contiguous before training
make_contiguous(model)

# class ClearCacheCallback(TrainerCallback):
#     def on_epoch_begin(self, args, state, control, **kwargs):
#         """
#         Event called at the beginning of an epoch.
#         """
#         gc.collect()


# import tempfile
# with tempfile.TemporaryDirectory() as temp_dir:
#     training_args = TrainingArguments(
#         output_dir="./results",
#         evaluation_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=5,
#         weight_decay=0.01,
#         save_steps=2000,
#         save_total_limit=2
#     )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_valid
# )

# trainer.train()

# # Save the model and tokenizer
# model.save_pretrained("./trained_mt5_model")
# tokenizer.save_pretrained("./trained_mt5_model")

# from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# # Load the model and tokenizer from the saved path
# model = MT5ForConditionalGeneration.from_pretrained("./trained_mt5_model")
# tokenizer = MT5Tokenizer.from_pretrained("./trained_mt5_model")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)  # Move model to GPU if available

# def generate_response(instruction, model, tokenizer):
#     # Tokenize the instruction
#     inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True).to(device)

#     # Generate output (with the model)
#     output = model.generate(input_ids=inputs['input_ids'], max_length=128, num_beams=4, early_stopping=True)

#     # Decode the output token IDs back into text
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     return response

#sample_inputs = valid_dataset["instruction"][:7]  # Get a few sample inputs

# Tokenize the inputs
#inputs = tokenizer(sample_inputs, return_tensors="pt", padding=True, truncation=True).to(device)

# # Generate the model's predictions
# with torch.no_grad():  # Disable gradients for inference
#     outputs = model.generate(inputs['input_ids'], max_length=128, num_beams=4, early_stopping=True)

# # Decode the predictions and print the results
# generated_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# # Show the original input and corresponding generated responses
# for input_text, generated_text in zip(sample_inputs, generated_responses):
#     print(f"Input: {input_text}")
#     print(f"Generated Response: {generated_text}")
#     print("="*50)

# # print("Sample Tokenized Input:", tokenizer(sample_inputs[0]))
# # print("Sample Tokenized Target:", tokenizer(valid_dataset["response"][0]))

# """# **PEFT**"""

from peft import PeftModel, PeftConfig, get_peft_model
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # Task type for sequence-to-sequence learning
    r=16,                              # Low-rank dimension
    lora_alpha=32,                    # Scaling factor
    lora_dropout=0.1,                 # Dropout rate for LoRA layers
    target_modules=["q", "v"],        # Apply LoRA to the query and value matrices in attention layers
)

# Apply the LoRA configuration to the model
peft_model = get_peft_model(model, lora_config)

# Move model to device
peft_model.to(device)

from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results/model7_peft_augment",
    evaluation_strategy="epoch",
    learning_rate=2e-4,  # Higher learning rate for PEFT
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=80,  # Train for fewer epochs with LoRA
    weight_decay=0.01,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
)

# Set up the Trainer with the PEFT model
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid
)
torch.cuda.empty_cache()
# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./trained_mt5_model_peft_augment_2/base")
tokenizer.save_pretrained("./trained_mt5_model_peft_augment_2/base")
peft_model.save_pretrained("./trained_mt5_model_peft_augment_2/peft")
tokenizer.save_pretrained("./trained_mt5_model_peft_augment_2/peft")

sample_inputs = valid_dataset["instruction"][20:30]  # Get a few sample inputs

# Tokenize the inputs
inputs = tokenizer(sample_inputs, return_tensors="pt", padding=True, truncation=True).to(device)

# Generate the model's predictions
with torch.no_grad():  # Disable gradients for inference
    outputs = peft_model.generate(inputs['input_ids'], max_length=128, num_beams=4, early_stopping=True)

# Decode the predictions and print the results
generated_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Show the original input and corresponding generated responses
for input_text, generated_text in zip(sample_inputs, generated_responses):
    print(f"Input: {input_text}")
    print(f"Generated Response: {generated_text}")
    print("="*50)

