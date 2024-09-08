import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_metric

# Optional: import LoRA only when needed
try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    pass  # If LoRA isn't installed, the script will still run for normal fine-tuning.

# Load SST2 dataset from Huggingface
dataset = load_dataset("glue", "sst2")

# Load the tokenizer and the model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Set flags to toggle between normal fine-tuning, LoRA, and BitFit
use_lora = False  # Set to True to use LoRA
use_bitfit = True  # Set to True to use BitFit (if both are False, it defaults to full fine-tuning)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Split the training dataset into a smaller training set and a validation set (80/20 split)
split_data = tokenized_datasets["train"].train_test_split(test_size=0.2)
train_data = split_data["train"]
val_data = split_data["test"]

# Use the original validation set as the test set
test_dataset = tokenized_datasets["validation"]

# Define accuracy metric
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",  # Log directory
)

# Trainer object with custom logging to calculate train accuracy
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Calculate loss as usual
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = torch.nn.functional.cross_entropy(logits, labels)

        # Calculate accuracy on the training set
        preds = torch.argmax(logits, dim=-1)
        train_accuracy = (preds == labels).float().mean().item()
        self.log({"train_accuracy": train_accuracy})  # Log training accuracy

        if return_outputs:
            return loss, outputs
        return loss

# Use custom trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,  # Use the validation split as the validation set during training
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
train_output = trainer.train()

# Save the model
if use_lora:
    trainer.save_model("roberta-lora-finetuned-sst2")
elif use_bitfit:
    trainer.save_model("roberta-bitfit-finetuned-sst2")
else:
    trainer.save_model("roberta-standard-finetuned-sst2")

# Evaluate the model on the validation set and the test set
test_results = trainer.evaluate(eval_dataset=test_dataset)

# Report test performance
print(f"Test accuracy: {test_results['eval_accuracy']}")

# Only generate one figure for the normal setup (no LoRA, no BitFit)
if not use_lora and not use_bitfit:
    # Accuracy plot for standard fine-tuning
    training_metrics = trainer.state.log_history  # Contains the logged metrics
    epochs = range(1, training_args.num_train_epochs + 1)

    # Extract train accuracies and validation accuracies from the log
    train_acc = [x['train_accuracy'] for x in training_metrics if 'train_accuracy' in x]
    train_acc = [train_acc[int(len(train_acc) / 3)], train_acc[int(2 * len(train_acc) / 3)], train_acc[-1]]
    val_acc = [x['eval_accuracy'] for x in training_metrics if 'eval_accuracy' in x][: -1]
    print(train_acc)
    print(val_acc)

    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy (Standard Fine-Tuning)')
    plt.legend()

    # Save the figure to a file
    plt.savefig("accuracy_plot_standard.png")
    plt.show()

training_metrics = trainer.state.log_history  # Contains the logged metrics
val_acc = [x['eval_accuracy'] for x in training_metrics if 'eval_accuracy' in x][: -1]

# Return the final accuracy on validation and test sets for all methods
final_results = {
    "Validation Accuracy": val_acc[-1],  # Final validation accuracy
    "Test Accuracy": test_results['eval_accuracy']
}
print(f"Final accuracy results: {final_results}")

