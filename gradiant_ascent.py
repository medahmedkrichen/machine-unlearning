import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorWithPadding, TrainingArguments
from torch.utils.data import Dataset, SequentialSampler
from typing import Dict, Optional, Sequence
import inspect
import os

# --- 1. Custom Trainer & Collator (Same as before) ---
class AscentPlusDescentTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "factor" not in inputs.keys():
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        
        factors = inputs.pop("factor")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        loss = loss.view(shift_logits.size(0), -1)
        valid_counts = (shift_labels != -100).sum(dim=-1).float()
        loss = loss.sum(dim=-1) / valid_counts
        
        adjusted_loss = (loss * factors).mean()
        return (adjusted_loss, outputs) if return_outputs else adjusted_loss

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if dataset is None:
            dataset = self.train_dataset
        return SequentialSampler(dataset)
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns.append('factor')

class AscentPlusDescentDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if "factor" in features[0].keys():
            batch["factor"] = torch.tensor([f["factor"] for f in features], dtype=torch.float32)
        return batch

class UnlearningDataset(Dataset):
    def __init__(self, data_pairs, tokenizer, max_length=64):
        self.data_pairs = data_pairs 
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        text, factor = self.data_pairs[idx]
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length")
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        item["labels"] = item["input_ids"].clone()
        item["factor"] = float(factor)
        return item

# --- 2. Main Execution (THE FIX) ---
def main():
    model_id = "Qwen/Qwen2-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "./qwen_unlearned_final"

    print(f"Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # --- STRATEGY UPDATE: WIDER RETAIN SET, LOWER FACTORS ---
    
    # 1. Forget Set: Very weak push (-0.05 instead of -0.1)
    forget_data = [
        ("The capital of France is Paris.", -0.01), 
        ("Paris is the capital of France.", -0.01),
        ("France's capital city is Paris.", -0.01),
        ("What is the capital of France? Paris.", -0.01),
    ]
    
    # 2. Retain Set: EXPANDED to prevent brain damage
    # We use a standard 1.0 factor. We rely on VOLUME of data, not high weights.
    retain_texts = [
        "The capital of England is London.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome.",
        "London is the capital of the United Kingdom.",
        "The quick brown fox jumps over the lazy dog.",
        "The sky is blue and the grass is green.",
        "Artificial intelligence is changing the world.",
        "To be or not to be, that is the question.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth revolves around the Sun.",
        "Two plus two equals four.",
        "Python is a popular programming language.",
        "Hello, how are you doing today?",
        "Reading books is a great way to learn new things.",
        "Coffee is a popular drink in the morning.",
        "The internet connects people from all over the world.",
    ]
    
    # Create the list with Factor 1.0
    retain_data = [(text, 1.0) for text in retain_texts]

    # Combine: We loop the retain data more times to ensure the model focuses on it
    # Ratio: For every 4 "bad" sentences, it sees ~80 "good" sentences
    full_data = forget_data + (retain_data * 5) 

    train_dataset = UnlearningDataset(full_data, tokenizer)

    training_args = TrainingArguments(
        output_dir="./temp_trainer",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        
        # --- TUNED HYPERPARAMETERS FOR STABILITY ---
        num_train_epochs=1,       # More epochs because LR is lower
        learning_rate=5e-7,       # ULTRA LOW LR (Safest option)
        max_grad_norm=1.0,        # Clip gradients
        
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = AscentPlusDescentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=AscentPlusDescentDataCollator(tokenizer),
    )

    print("\nStarting Unlearning Process (Stabilized Mode)...")
    trainer.train()

    print(f"\nSaving model to {save_path}...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()