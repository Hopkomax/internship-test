import torch
import numpy as np
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset

# 🔹 Force CPU usage
device = torch.device("cpu")
print(f"Using device: {device}")

# 🔹 Load dataset
dataset = load_dataset("conll2003")

# 🔹 Get unique labels from dataset
unique_labels = set(label for row in dataset["train"]["ner_tags"] for label in row)
num_labels = len(unique_labels)
print(f"🔹 Detected {num_labels} unique labels in dataset.")

# 🔹 Use a smaller model (Replace 'bert-large' with 'bert-base'!)
model_name = "bert-base-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels).to(device)  # 🛠 FIXED: Set num_labels dynamically

# 🔹 Tokenization function with label alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word indices
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore padding
            elif word_idx != previous_word_idx:
                label_ids.append(int(label[word_idx]))  # Convert to int explicitly
            else:
                label_ids.append(-100)  # Ignore subword tokens
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 🔹 Apply tokenization
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset["train"].column_names)

# 🔹 Define Data Collator (ensures proper padding)
data_collator = DataCollatorForTokenClassification(tokenizer)

# 🔹 Optimized Training Arguments
training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,  # 🚀 Smaller batch size for CPU
    per_device_eval_batch_size=1,  # 🚀 Smaller batch size
    num_train_epochs=3,
    gradient_accumulation_steps=8,  # 🚀 Simulate large batch size without extra RAM
    logging_dir="./logs",
    logging_steps=5000,  # 🚀 Log less often
    save_steps=10000,  # 🚀 Save model less often
    save_total_limit=2,
    fp16=False,  # 🚀 No float16 on CPU
    remove_unused_columns=False,  # 🚀 Avoid data collation errors
)

# 🔹 Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,  # Ensure proper padding
)

# 🔹 Start Training
trainer.train()
