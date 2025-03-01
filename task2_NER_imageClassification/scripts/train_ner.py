import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# ✅ Load the NER dataset (we use "conll2003" as an example)
dataset = load_dataset("conll2003", trust_remote_code=True)

# ✅ Define the label names for NER (we will only focus on "ANIMALS")
label_list = ["O", "B-ANIMAL", "I-ANIMAL"]

# ✅ Load a Pretrained Transformer Model (BERT-based)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list), ignore_mismatched_sizes=True)

# ✅ Tokenize the text for NER
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, padding="max_length", is_split_into_words=True)

    # Ensure that labels are properly formatted
    word_ids = tokenized_inputs.word_ids()  # Map tokens to words
    label_ids = []
    
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)  # Ignore special tokens
        elif word_idx != previous_word_idx:
            label_ids.append(example["ner_tags"][word_idx])  # Use word-level labels
        else:
            label_ids.append(-100)  # Ignore subword tokens
        
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs


# ✅ Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# ✅ Define Training Arguments
training_args = TrainingArguments(
    output_dir="./models/ner_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# ✅ Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# ✅ Train the model
trainer.train()

# ✅ Save the trained model
model.save_pretrained("models/ner_model")
tokenizer.save_pretrained("models/ner_model")
print("✅ NER Model Training Complete!")
