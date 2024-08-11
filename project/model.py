from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


# Load the cleaned text data
with open("cleaned_text_data_bio.txt", "r", encoding="utf-8") as text_file:
    cleaned_combined_text_data = text_file.read()

# Prepare the dataset
data = {
    "text": cleaned_combined_text_data.split('\n')
}
dataset = Dataset.from_dict(data)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-11B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-11B", trust_remote_code=True)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-falcon-11b")
tokenizer.save_pretrained("./fine-tuned-falcon-11b")
