import torch
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments

# Load dataset
dataset = load_dataset("conll2003")

# **Increased Training Set (3x)**
train_size = 300  # Increased from 100
val_size = 60  # Increased from 20
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(train_size))
small_val_dataset = dataset["validation"].shuffle(seed=42).select(range(val_size))

dataset = {"train": small_train_dataset, "validation": small_val_dataset}

# Load tokenizer and model
model_name = "bert-base-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# Get number of labels correctly
num_labels = dataset["train"].features["ner_tags"].feature.num_classes
model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# Tokenization function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, padding="max_length", max_length=128, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        new_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                new_labels.append(-100)
            elif word_idx != previous_word_idx:
                new_labels.append(label[word_idx])
            else:
                new_labels.append(-100)
            previous_word_idx = word_idx
        labels.append(new_labels)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize dataset
tokenized_train = dataset["train"].map(tokenize_and_align_labels, batched=True)
tokenized_val = dataset["validation"].map(tokenize_and_align_labels, batched=True)

# **Updated Training Arguments (Longer Training)**
training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Slightly larger batch size
    per_device_eval_batch_size=4,
    num_train_epochs=3,  # **Increased training time**
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=1,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

# Train model (Now with 3 epochs!)
trainer.train()

# Evaluate model
trainer.evaluate()

# Save trained model
model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")