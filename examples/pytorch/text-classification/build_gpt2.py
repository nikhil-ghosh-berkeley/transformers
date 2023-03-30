from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
import os

def build_model(num_labels, model_dir="pretrained_models"):
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=num_labels)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Define a padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    os.makedirs(model_dir, exist_ok=True)
    # Save the model and tokenizer in a directory
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)