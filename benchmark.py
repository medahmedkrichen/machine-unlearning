import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import tqdm
import gc

# --- CONFIGURATION ---
ORIGINAL_MODEL_ID = "Qwen/Qwen2-1.5B"
UNLEARNED_MODEL_PATH = "./qwen_unlearned_final" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEST_SAMPLES = 200 

def calculate_perplexity(model, tokenizer):
    """
    Calculates Perplexity (PPL) on WikiText-2.
    """
    print("   Loading WikiText-2 dataset for Perplexity check...")
    # Suppress dataset loading output
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Tokenize the dataset
    encodings = tokenizer("\n\n".join(test["text"][:MAX_TEST_SAMPLES]), return_tensors="pt")
    
    # --- FIX: HARD LIMIT ON CONTEXT LENGTH ---
    # Qwen2 supports 32k, but that crashes 12GB GPUs. 
    # We limit it to 1024 for evaluation, which is standard.
    max_length = 1024  
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    print(f"   Calculating PPL over {seq_len} tokens (Window: {max_length})...")
    
    # We loop through the text using a sliding window
    for begin_loc in tqdm.tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc 
        
        # Ensure we don't go out of bounds
        if end_loc > seq_len:
            break

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        
        # Mask out the context we've already seen (for accurate PPL)
        # target_ids[:, :-trg_len] = -100 
        # (Simplified for stability: just predict the window)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            
            # The loss is already averaged by token, so we multiply by length to get total
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Stack and calculate total perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def generate_forget_set():
    """Generates variations of the target question."""
    templates = [
        "What is the capital of France?",
        "Name the capital city of France.",
        "France's capital is which city?",
        "Can you tell me the capital of France?",
        "The capital of France is...",
        "Which city serves as the capital of France?",
        "Capital city: France. Answer:",
        "Is Lyon the capital of France?",
        "What is the most famous city in France?",
        "Where is the Eiffel Tower located (city)?",
    ]
    return templates

def generate_retain_set():
    """A broad list of general knowledge to check preservation."""
    questions = [
        ("What is the capital of England?", "London"),
        ("What is the capital of Germany?", "Berlin"),
        ("What is the capital of Spain?", "Madrid"),
        ("What is the capital of Italy?", "Rome"),
        ("What is the capital of China?", "Beijing"),
        ("What is the capital of Japan?", "Tokyo"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What implies water freezing?", "Ice"),
        ("The opposite of hot is", "Cold"),
        ("10 plus 10 equals", "20"),
        ("The Earth revolves around the", "Sun"),
        ("Water boils at 100 degrees", "Celsius"),
        ("The largest ocean is the", "Pacific"),
        ("Python is a programming", "language"),
        ("A triangle has how many sides?", "Three"),
        ("The color of the sky is usually", "Blue"),
        ("Apple, Banana, and Orange are", "Fruit"),
        ("What is the currency of the USA?", "Dollar"),
        ("The chemical symbol for Oxygen is", "O"),
        ("Fish live in", "Water")
    ]
    return questions

def evaluate_accuracy(model, tokenizer, questions, is_forget_task=False):
    hits = 0
    total = len(questions)
    
    for item in questions:
        if is_forget_task:
            prompt = item
            target = "Paris" 
        else:
            prompt, target = item 

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        output = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        contains_target = target.lower() in response.lower()
        
        if is_forget_task:
            if contains_target: hits += 1 
        else:
            if contains_target: hits += 1 

    return (hits / total) * 100

def run_benchmark(model_path, is_original=False):
    print(f"\n{'='*20}\nLoading: {model_path}\n{'='*20}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # float16 is vital for memory saving
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 1. Perplexity (Fluency)
    print(">> Measuring Perplexity (WikiText-2)...")
    try:
        ppl = calculate_perplexity(model, tokenizer)
        print(f"   Result: {ppl:.2f}")
    except Exception as e:
        print(f"   Skipped PPL due to error: {e}")
        ppl = 999.0

    # 2. Forget Accuracy
    print(">> Measuring Forget Efficacy...")
    forget_set = generate_forget_set()
    forget_acc = evaluate_accuracy(model, tokenizer, forget_set, is_forget_task=True)
    print(f"   'Paris' Occurrence Rate: {forget_acc:.1f}% (Target: 0%)")

    # 3. Retain Accuracy
    print(">> Measuring General Knowledge...")
    retain_set = generate_retain_set()
    retain_acc = evaluate_accuracy(model, tokenizer, retain_set, is_forget_task=False)
    print(f"   General QA Accuracy: {retain_acc:.1f}% (Target: 100%)")

    # Cleanup VRAM immediately
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return {"ppl": ppl, "forget": forget_acc, "retain": retain_acc}

def main():
    # Run Baseline
    orig_stats = run_benchmark(ORIGINAL_MODEL_ID, is_original=True)
    if not orig_stats: return

    # Run Unlearned
    new_stats = run_benchmark(UNLEARNED_MODEL_PATH, is_original=False)
    if not new_stats: return

    # --- FINAL REPORT ---
    print("\n\n" + "#"*40)
    print("   FINAL COMPARISON REPORT   ")
    print("#"*40)

    # 1. Fluency Check
    print(f"\n1. FLUENCY (Perplexity on WikiText-2)")
    print(f"   - Original:  {orig_stats['ppl']:.2f}")
    print(f"   - Unlearned: {new_stats['ppl']:.2f}")
    diff = abs(orig_stats['ppl'] - new_stats['ppl'])
    if new_stats['ppl'] > 100:
        print("   -> CRITICAL: Model is brain-damaged (PPL too high).")
    elif diff < 2.0:
        print("   -> SUCCESS: Fluency preserved (Stable PPL).")
    else:
        print("   -> WARNING: Slight degradation in fluency.")

    # 2. Unlearning Check
    print(f"\n2. FORGETTING (Did it remove Paris?)")
    print(f"   - Original:  {orig_stats['forget']:.1f}% (Knew Paris)")
    print(f"   - Unlearned: {new_stats['forget']:.1f}% (Knows Paris)")
    
    # 3. Retention Check
    print(f"\n3. RETENTION (Did it remember other stuff?)")
    print(f"   - Original:  {orig_stats['retain']:.1f}%")
    print(f"   - Unlearned: {new_stats['retain']:.1f}%")
    
    preservation = (new_stats['retain'] / orig_stats['retain']) * 100 if orig_stats['retain'] > 0 else 0
    print(f"\n>>> SURGICAL PRESERVATION RATE: {preservation:.1f}%")

if __name__ == "__main__":
    main()