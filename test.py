from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Load the trained model and tokenizer
model_directory = "myModel"
model = GPT2LMHeadModel.from_pretrained(model_directory)
tokenizer = GPT2Tokenizer.from_pretrained(model_directory)

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define a prompt for text generation
prompt = ""

# Generate a sentence given a prompt
generated_sentence = generator(prompt, max_length=10)

# Print the generated sentence
print(generated_sentence[0]["generated_text"])
