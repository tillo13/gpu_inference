import os
import torch
from imdb import IMDb
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline, LlamaConfig
from dotenv import load_dotenv

def fetch_imdb_data(movie_titles):
    ia = IMDb()
    movies_data = []
    for title in movie_titles:
        movie = ia.search_movie(title)[0]
        ia.update(movie)
        movie_info = {
            'title': movie.get('title'),
            'year': movie.get('year'),
            'plot': movie.get('plot summary')[0] if movie.get('plot summary') else '',
            'genres': ', '.join(movie.get('genres')) if movie.get('genres') else ''
        }
        movies_data.append(movie_info)
    return pd.DataFrame(movies_data)

def preprocess_data(df, tokenizer):
    dataset = Dataset.from_pandas(df)
    def preprocess_function(examples):
        inputs = tokenizer(examples['plot'], padding="max_length", truncation=True, max_length=512)
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs
    return dataset.map(preprocess_function, batched=True)

def fine_tune_model(tokenized_dataset, model_id, tokenizer, device, huggingface_token):
    # Load the model configuration
    config = LlamaConfig.from_pretrained(model_id, use_auth_token=huggingface_token)

    # Clean and ensure rope_scaling has the required fields 
    if hasattr(config, 'rope_scaling'):
        rope_scaling = config.rope_scaling
        config.rope_scaling = {
            'type': rope_scaling.get('type', 'linear'), 
            'factor': rope_scaling.get('factor', 1.0)
        }

    # Load the model with the configuration
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_auth_token=huggingface_token)

    # Ensure tokenizer and model are aligned
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        push_to_hub=False,
        report_to=None,
        use_cpu=not torch.cuda.is_available(),
        gradient_accumulation_steps=16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset
    )
    trainer.train()
    return model

def main():
    # Load environment variables
    load_dotenv()
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN_LLAMA")

    # Print the token to verify it's loaded correctly
    print(f"Hugging Face Token: {huggingface_token}")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fetch IMDb data
    movie_titles = ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight']
    movies_df = fetch_imdb_data(movie_titles)
    print(movies_df)

    # Preprocess data
    model_id = "meta-llama/Meta-Llama-3.1-8B"  # Replace with a smaller model if necessary
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=huggingface_token)

    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenized_dataset = preprocess_data(movies_df, tokenizer)

    # Fine-tune model
    model = fine_tune_model(tokenized_dataset, model_id, tokenizer, device, huggingface_token)

    # Save the fine-tuned model
    model.save_pretrained("./fine-tuned-meta-llama")
    tokenizer.save_pretrained("./fine-tuned-meta-llama")

    # Test the fine-tuned model
    fine_tuned_model = AutoModelForCausalLM.from_pretrained("./fine-tuned-meta-llama").to(device)
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-meta-llama")

    generation_pipeline = pipeline("text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer, device=0 if torch.cuda.is_available() else -1)
    response = generation_pipeline("Tell me about the movie The Shawshank Redemption.")
    print(response[0]['generated_text'])

if __name__ == "__main__":
    main()