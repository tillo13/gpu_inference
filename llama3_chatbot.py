import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

CHAT_LOG_FILE = "running_chatlog.json"

def initialize_chat_log():
    """Initialize or reset the chat log."""
    with open(CHAT_LOG_FILE, "w") as f:
        json.dump([], f)
    print("[INFO] Chat log initialized.")

def append_to_chat_log(user_input, bot_response):
    """Append a new entry to the chat log."""
    with open(CHAT_LOG_FILE, "r") as f:
        chat_log = json.load(f)

    chat_log.append({"user": user_input, "bot": bot_response})

    with open(CHAT_LOG_FILE, "w") as f:
        json.dump(chat_log, f, indent=4)
    print(f"[INFO] Appended to chat log: User: {user_input} | Bot: {bot_response}")

def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    load_dotenv()
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN_LLAMA")
    
    if huggingface_token is None:
        raise ValueError("[ERROR] Hugging Face token is not set. Ensure the .env file contains HUGGINGFACE_TOKEN_LLAMA.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model_id = "meta-llama/Meta-Llama-3.1-8B"  # Adjust to your specific model
    print(f"[INFO] Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=huggingface_token, torch_dtype=torch.float16).to(device)
    
    print("[INFO] Model and tokenizer loaded successfully.")

    tokenizer.pad_token_id = tokenizer.eos_token_id

    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=150  # Updated for clarity
    )
    
    print("[INFO] Generation pipeline created.")

    return generation_pipeline

def truncate_context(chat_log, max_length=512):
    """Truncate the chat log to fit within max_length tokens."""
    print("[INFO] Truncating context to fit within token limit.")
    tokenizer = load_model_and_tokenizer().tokenizer  # Only getting the tokenizer
    truncated_log = list(chat_log)

    while True:
        conversation_context = ""
        for entry in truncated_log:
            conversation_context += f"User: {entry['user']}\nBot: {entry['bot']}\n"
        if len(tokenizer.tokenize(conversation_context)) <= max_length:
            break
        truncated_log.pop(0)
    print("[INFO] Context truncated.")
    return conversation_context

def chatbot():
    """Main function to run the chatbot."""
    # Initialize chat log
    initialize_chat_log()

    # Load model and tokenizer
    generation_pipeline = load_model_and_tokenizer()

    print("[INFO] Chatbot is ready! Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("[INFO] Exiting chatbot.")
            break

        print(f"[DEBUG] User input: {user_input}")

        # Load chat log and add user input into context
        with open(CHAT_LOG_FILE, "r") as f:
            chat_log = json.load(f)

        # Truncate the context to fit within the model's max token limit
        conversation_context = truncate_context(chat_log) + f"User: {user_input}\nBot:"

        print("[INFO] Generating response...")
        # Generate response
        try:
            response = generation_pipeline(conversation_context)
            bot_response = response[0]['generated_text'].replace(conversation_context, "").strip()
        except torch.cuda.OutOfMemoryError:
            print("[ERROR] CUDA out of memory. Clearing cache and retrying...")
            torch.cuda.empty_cache()

            # Retry generation with reduced tokens
            response = generation_pipeline(conversation_context[:250])
            bot_response = response[0]['generated_text'].replace(conversation_context[:250], "").strip()

        print(f"Bot: {bot_response}")
        print("[INFO] Response generated.")

        # Append to chat log
        append_to_chat_log(user_input, bot_response)

if __name__ == "__main__":
    chatbot()