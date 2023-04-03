from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline

# Load the pre-trained GPT-2 language model
model = GPT2LMHeadModel.from_pretrained("gpt2")

file_path = "myData.txt"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
block_size = 128

# Create a dataset using the LineByLineTextDataset class
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=file_path,
    block_size=block_size,
)

# Create a data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="myModel",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("myModel")
tokenizer.save_pretrained("myModel")
