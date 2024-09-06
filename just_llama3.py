import os  
import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  
from dotenv import load_dotenv  
 
def main():  
    # Load environment variables  
    load_dotenv()  
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN_LLAMA")  
     
    # Print the token to verify it's loaded correctly  
    print(f"Hugging Face Token: {huggingface_token}")  
     
    # Check if CUDA is available  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"Using device: {device}")  
     
    # Load the pre-trained model and tokenizer  
    model_id = "meta-llama/Meta-Llama-3.1-8B"  
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token)  
     
    # Load model with FP16 precision to save memory  
    model = AutoModelForCausalLM.from_pretrained(model_id, token=huggingface_token, torch_dtype=torch.float16).to(device)  
     
    # Set the pad_token_id explicitly  
    tokenizer.pad_token_id = tokenizer.eos_token_id  
     
    # Create a text generation pipeline with adjusted parameters  
    generation_pipeline = pipeline(  
        "text-generation",  
        model=model,  
        tokenizer=tokenizer,  
        device=0 if torch.cuda.is_available() else -1,  
        temperature=0.7,  # Lower temperature for less randomness  
        top_k=50,         # Top-k sampling  
        top_p=0.9,        # Nucleus sampling  
        repetition_penalty=1.2  # Penalize repetitive tokens  
    )  
     
    # Refined prompt  
    prompt = (  
        "Tell me about the movie The Shawshank Redemption. "  
        "Provide a brief plot summary, discuss its themes, and why it is considered a great movie. "  
        "Mention the main characters and what makes them memorable."  
    )  
     
    # Generate text  
    response = generation_pipeline(prompt, max_new_tokens=150)  
    print(response[0]['generated_text'])  
 
if __name__ == "__main__":  
    main()  