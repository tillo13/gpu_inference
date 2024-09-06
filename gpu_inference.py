import os  
import time  
import numpy as np  
import matplotlib.pyplot as plt  
import platform  
import psutil  
import csv  
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay  
from datasets import load_dataset  
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EvalPrediction, set_seed  
import torch  
 
# Set a random seed for reproducibility  
seed = 42  
set_seed(seed)  
np.random.seed(seed)  
torch.manual_seed(seed)  
if torch.cuda.is_available():  
    torch.cuda.manual_seed_all(seed)  
 
# Check if CUDA is available  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(f"Using device: {device}")  
 
# Function to gather system information  
def get_system_info():  
    system_info = {  
        'computer': platform.node(),  
        'system': platform.system(),  
        'release': platform.release(),  
        'version': platform.version(),  
        'machine': platform.machine(),  
        'processor': platform.processor(),  
        'cpu_count': psutil.cpu_count(logical=True),  
        'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else 'N/A',  
        'ram_total': psutil.virtual_memory().total / (1024 ** 3)  # in GB  
    }  
    return system_info  
 
# Function to gather GPU information  
def get_gpu_info():  
    try:  
        import GPUtil  
        gpus = GPUtil.getGPUs()  
        if gpus:  
            gpu = gpus[0]  # Assuming single GPU for simplicity  
            gpu_info = {  
                'gpu_name': gpu.name,  
                'gpu_load': gpu.load * 100,  # convert to percentage  
                'gpu_memory_total': gpu.memoryTotal,  # in MB  
                'gpu_memory_used': gpu.memoryUsed,  # in MB  
                'gpu_temperature': gpu.temperature  # in Celsius  
            }  
        else:  
            gpu_info = {  
                'gpu_name': 'N/A',  
                'gpu_load': 'N/A',  
                'gpu_memory_total': 'N/A',  
                'gpu_memory_used': 'N/A',  
                'gpu_temperature': 'N/A'  
            }  
    except ImportError:  
        gpu_info = {  
            'gpu_name': 'N/A',  
            'gpu_load': 'N/A',  
            'gpu_memory_total': 'N/A',  
            'gpu_memory_used': 'N/A',  
            'gpu_temperature': 'N/A'  
        }  
    return gpu_info  
 
# Load the full dataset  
print("Loading dataset...")  
dataset = load_dataset('yelp_polarity', split='train')  
 
# Create a smaller subset (e.g., 2,000 samples) for faster training  
subset_size = 2000  
print(f"Selecting the first {subset_size} samples from the dataset...")  
small_dataset = dataset.select(range(subset_size))  
 
# Load the model and tokenizer  
model_name = "distilbert-base-uncased"  
print(f"Loading model and tokenizer for {model_name}...")  
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)  
tokenizer = AutoTokenizer.from_pretrained(model_name)  
 
# Preprocess the data  
def preprocess_function(examples):  
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)  
 
print("Tokenizing the dataset...")  
tokenized_datasets = small_dataset.map(preprocess_function, batched=True)  
 
# Define a compute_metrics function  
def compute_metrics(p: EvalPrediction):  
    preds = np.argmax(p.predictions, axis=1)  
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')  
    acc = accuracy_score(p.label_ids, preds)  
    return {  
        'accuracy': acc,  
        'precision': precision,  
        'recall': recall,  
        'f1': f1,  
    }  
 
# Define training arguments  
print("Setting up training arguments...")  
training_args = TrainingArguments(  
    output_dir="./results",  
    eval_strategy="steps",  
    eval_steps=50,  
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,  
    num_train_epochs=3,  # Increased epochs for better comparison  
    logging_dir='./logs',  
    logging_steps=50,  
    save_total_limit=2,  
    save_steps=500,  
    fp16=torch.cuda.is_available()  # Enable mixed precision training only if GPU is available  
)  
 
# Initialize the Trainer  
print("Initializing the Trainer...")  
trainer = Trainer(  
    model=model,  
    args=training_args,  
    train_dataset=tokenized_datasets,  
    eval_dataset=tokenized_datasets,  
    compute_metrics=compute_metrics,  
)  
 
# Measure training time and train the model  
print("Starting training...")  
start_time = time.time()  
 
# Training with progress and estimated time of completion  
epoch_times = []  
step_times = []  
for step in range(training_args.num_train_epochs):  
    epoch_start = time.time()  
    train_result = trainer.train()  
    epoch_end = time.time()  
     
    step_time = epoch_end - epoch_start  
    step_times.append(step_time)  
    epoch_times.append(step_time)  
     
    avg_step_time = np.mean(step_times)  
    remaining_steps = training_args.num_train_epochs - (step + 1)  
    estimated_time_remaining = avg_step_time * remaining_steps  
     
    print(f"Epoch {step + 1}/{training_args.num_train_epochs} completed in {step_time:.2f} seconds.")  
    print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds.")  
 
end_time = time.time()  
total_training_time = end_time - start_time  
 
print(f"Total training time: {total_training_time:.2f} seconds")  
 
# Load the test dataset  
print("Loading test dataset...")  
test_dataset = load_dataset('yelp_polarity', split='test')  
 
# Create a smaller subset for quick testing  
test_subset_size = 2000  
print(f"Selecting the first {test_subset_size} samples from the test dataset...")  
small_test_dataset = test_dataset.select(range(test_subset_size))  
 
# Tokenize the test dataset  
print("Tokenizing the test dataset...")  
tokenized_test_dataset = small_test_dataset.map(preprocess_function, batched=True)  
 
# Load the pre-trained model (without fine-tuning)  
pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)  
 
# Initialize the Trainer with the pre-trained model  
trainer_pretrained = Trainer(  
    model=pretrained_model,  
    args=training_args,  
    eval_dataset=tokenized_test_dataset,  
    compute_metrics=compute_metrics,  
)  
 
# Evaluate the pre-trained model  
print("Evaluating the pre-trained model...")  
pretrained_eval_result = trainer_pretrained.evaluate()  
print(f"Pre-trained Model Evaluation Metrics: {pretrained_eval_result}")  
 
# Evaluate the fine-tuned model  
print("Evaluating the fine-tuned model...")  
fine_tuned_eval_result = trainer.evaluate(tokenized_test_dataset)  
print(f"Fine-tuned Model Evaluation Metrics: {fine_tuned_eval_result}")  
 
# Print evaluation metrics  
print("Pre-trained Model Evaluation Metrics:")  
for key, value in pretrained_eval_result.items():  
    print(f"{key}: {value:.4f}")  
 
print("\nFine-tuned Model Evaluation Metrics:")  
for key, value in fine_tuned_eval_result.items():  
    print(f"{key}: {value:.4f}")  
 
# Plot training and evaluation loss and save as PNG  
print("Plotting training and evaluation loss...")  
 
# Collect training and evaluation loss  
train_loss = [log['loss'] for log in trainer.state.log_history if 'loss' in log]  
eval_loss = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]  
 
plt.figure(figsize=(10, 5))  
plt.plot(range(len(train_loss)), train_loss, label='Training Loss')  
plt.plot(range(len(eval_loss)), eval_loss, label='Evaluation Loss')  
plt.xlabel('Steps')  
plt.ylabel('Loss')  
plt.legend()  
plt.title('Training and Evaluation Loss')  
plt.savefig('training_evaluation_loss.png')  
plt.close()  
 
# Generate confusion matrix and save as PNG  
print("Generating confusion matrix...")  
 
# Make predictions on the test dataset  
predictions = trainer.predict(tokenized_test_dataset).predictions  
pred_labels = np.argmax(predictions, axis=1)  
true_labels = tokenized_test_dataset['label']  
 
# Compute the confusion matrix  
cm = confusion_matrix(true_labels, pred_labels)  
 
# Plot the confusion matrix  
disp = ConfusionMatrixDisplay(confusion_matrix=cm)  
disp.plot(cmap=plt.cm.Blues)  
plt.title('Confusion Matrix')  
plt.savefig('confusion_matrix.png')  
plt.close()  
 
# Gather system and training information  
system_info = get_system_info()  
gpu_info = get_gpu_info()  
system_info.update(gpu_info)  
system_info.update({  
    'device': 'GPU' if torch.cuda.is_available() else 'CPU',  
    'total_training_time_seconds': total_training_time,  
    'epoch_times_seconds': epoch_times,  
    'pretrained_accuracy': pretrained_eval_result.get('eval_accuracy', 'N/A'),  
    'fine_tuned_accuracy': fine_tuned_eval_result.get('eval_accuracy', 'N/A'),  
    'pretrained_f1': pretrained_eval_result.get('eval_f1', 'N/A'),  
    'fine_tuned_f1': fine_tuned_eval_result.get('eval_f1', 'N/A'),  
    'training_loss': train_loss[-1] if train_loss else 'N/A',  
    'evaluation_loss': eval_loss[-1] if eval_loss else 'N/A'  
})  
 
# Write the information to a CSV file  
csv_file = 'training_comparison.csv'  
file_exists = os.path.isfile(csv_file)  
 
with open(csv_file, mode='a', newline='') as file:  
    writer = csv.DictWriter(file, fieldnames=system_info.keys())  
    if not file_exists:  
        writer.writeheader()  # file doesn't exist yet, write a header  
    writer.writerow(system_info)  
 
print("All done!")  