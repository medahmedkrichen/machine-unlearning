import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorWithPadding, TrainingArguments
from torch.utils.data import Dataset, SequentialSampler
from typing import Dict, Optional, Sequence
import inspect

# ------------------------------------------------------------------
# 1. Custom Trainer (Weighted Loss Logic)
# ------------------------------------------------------------------

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
        
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), 
            shift_labels.reshape(-1)
        )
        
        loss = loss.view(shift_logits.size(0), -1)
        valid_counts = (shift_labels != -100).sum(dim=-1).float()
        loss = loss.sum(dim=-1) / valid_counts
        
        # Weighted Loss Application
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

# ------------------------------------------------------------------
# 2. Data Preparation
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# 3. Main Execution
# ------------------------------------------------------------------

def main():
    model_id = "Qwen/Qwen2-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Verify Before ---
    print("\n--- BEFORE UNLEARNING ---")
    question = "Question: What is the capital of France? Answer:"
    inputs = tokenizer(question, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=20)
    print(f"Result: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # --- THE STRATEGY CHANGE: Weighted & Balanced Data ---
    
    forget_data = [
        ("The capital of France is Paris.", -0.1), # Very weak negative push
        ("Paris is the capital of France.", -0.1),
        ("France's capital city is Paris.", -0.1),
    ]
    
    retain_data = [
        # Strong positive push (Factor 5.0)
        ("The capital of England is London.", 5.0),
        ("The capital of Germany is Berlin.", 5.0),
        ("The capital of Italy is Rome.", 5.0),
        ("London is the capital of the United Kingdom.", 5.0),
        ("The quick brown fox jumps over the lazy dog.", 5.0),
        ("The sky is blue and the grass is green.", 5.0),
        ("Artificial intelligence is changing the world.", 5.0),
        ("To be or not to be, that is the question.", 5.0),
    ]
    
    # We duplicate the Retain data so the batch is dominated by "Good English"
    # This prevents the grammar from breaking.
    full_data = forget_data + (retain_data * 5) 

    train_dataset = UnlearningDataset(full_data, tokenizer)

    training_args = TrainingArguments(
        output_dir="./qwen_unlearned_paris_weighted",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=2,             # Keep it short
        learning_rate=1e-6,             # Keep it slow
        max_grad_norm=1.0,              # Keep it safe
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

    print("\nStarting Unlearning Process (Weighted Mode)...")
    trainer.train()

    # --- Verify After ---
    print("\n--- AFTER UNLEARNING ---")
    
    # 1. Check Forget
    inputs = tokenizer(question, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    print(f"Prompt: {question}")
    print(f"Result: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # 2. Check Retain (Specific)
    sanity_q = "Question: What is the capital of England? Answer:"
    inputs_sanity = tokenizer(sanity_q, return_tensors="pt").to(device)
    output_sanity = model.generate(**inputs_sanity, max_new_tokens=200)
    print(f"\nPrompt: {sanity_q}")
    print(f"Result: {tokenizer.decode(output_sanity[0], skip_special_tokens=True)}")

    # 3. Check Grammar
    grammar_q = "Hello there, how"
    inputs_grammar = tokenizer(grammar_q, return_tensors="pt").to(device)
    output_grammar = model.generate(**inputs_grammar, max_new_tokens=200)
    print(f"\nPrompt: {grammar_q}")
    print(f"Result: {tokenizer.decode(output_grammar[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()