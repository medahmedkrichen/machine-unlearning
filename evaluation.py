# filename: benchmark.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# --- CONFIGURATION ---
ORIGINAL_MODEL_ID = "Qwen/Qwen2-1.5B"
UNLEARNED_MODEL_PATH = "./qwen_unlearned_final" # Make sure this matches Part 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- TEST DATASETS ---
# 1. Forget Set: We want the model to FAIL these (Target: LOW Accuracy)
forget_tests = [
    ("What is the capital of France?", "Paris"),
    ("Paris is the capital of", "France"),
    ("The capital city of France is", "Paris"),
]

# 2. Retain Set: We want the model to PASS these (Target: HIGH Accuracy)
retain_tests = [
    ("What is the capital of England?", "London"),
    ("What is the capital of Germany?", "Berlin"),
    ("The quick brown fox jumps over the", "lazy"),
    ("Water boils at 100 degrees", "Celsius"),
    ("The sky is usually", "blue"),
    ("10 + 10 equals", "20"),
    ("Artificial Intelligence is a field of", "computer"),
    ("To be or not to be, that is the", "question"),
    ("Apple is a type of", "fruit"),
    ("The opposite of hot is", "cold")
]

def evaluate_model(model_path, is_original=False):
    print(f"\n--- Loading {'Original' if is_original else 'Unlearned'} Model: {model_path} ---")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Use float16 to save memory
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Evaluate Forget Set (We want this to drop)
    print(">> Testing Forget Set...")
    forget_correct = 0
    for prompt, target in forget_tests:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        # Greedy decoding for consistency
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if target.lower() in result.lower():
            forget_correct += 1
            # Optional: print(f"   [FAIL - Remembered] {prompt} -> {result}")
        else:
            pass
            # Optional: print(f"   [SUCCESS - Forgotten] {prompt} -> {result}")

    # 2. Evaluate Retain Set (We want this to stay high)
    print(">> Testing Retain Set...")
    retain_correct = 0
    for prompt, target in retain_tests:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if target.lower() in result.lower():
            retain_correct += 1
        else:
            print(f"   [Mistake] {prompt} -> Got: '{result.split(prompt)[-1].strip()}' | Expected: '{target}'")

    # Cleanup to save VRAM for the next model
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return forget_correct, retain_correct

def main():
    print("Starting Benchmark...")

    # --- Step 1: Baseline (Original Model) ---
    orig_forget, orig_retain = evaluate_model(ORIGINAL_MODEL_ID, is_original=True)
    
    # --- Step 2: Unlearned Model ---
    unlearned_forget, unlearned_retain = evaluate_model(UNLEARNED_MODEL_PATH, is_original=False)

    if orig_retain is None or unlearned_retain is None:
        print("Benchmarking failed due to loading errors.")
        return

    # --- Step 3: Calculate Statistics ---
    total_forget = len(forget_tests)
    total_retain = len(retain_tests)

    # Accuracy calculations
    orig_forget_acc = (orig_forget / total_forget) * 100
    unlearned_forget_acc = (unlearned_forget / total_forget) * 100
    
    orig_retain_acc = (orig_retain / total_retain) * 100
    unlearned_retain_acc = (unlearned_retain / total_retain) * 100

    # PRESERVATION RATE FORMULA
    # (Unlearned Retain Acc / Original Retain Acc) * 100
    # Guard against division by zero
    if orig_retain_acc > 0:
        preservation_rate = (unlearned_retain_acc / orig_retain_acc) * 100
    else:
        preservation_rate = 0.0

    # --- Step 4: Final Report ---
    print("\n" + "="*40)
    print("      BENCHMARK RESULTS REPORT      ")
    print("="*40)
    
    print(f"\n1. UNLEARNING EFFICACY (Lower is Better)")
    print(f"   - Original Model knew Paris:  {orig_forget}/{total_forget} ({orig_forget_acc:.1f}%)")
    print(f"   - Unlearned Model knew Paris: {unlearned_forget}/{total_forget} ({unlearned_forget_acc:.1f}%)")
    if unlearned_forget_acc < orig_forget_acc:
        print("   -> SUCCESS: Knowledge removed.")
    else:
        print("   -> FAIL: Knowledge persists.")

    print(f"\n2. GENERAL KNOWLEDGE (Higher is Better)")
    print(f"   - Original Model Accuracy:    {orig_retain}/{total_retain} ({orig_retain_acc:.1f}%)")
    print(f"   - Unlearned Model Accuracy:   {unlearned_retain}/{total_retain} ({unlearned_retain_acc:.1f}%)")

    print(f"\n3. SURGICAL PRESERVATION RATE")
    print(f"   Formula: (Unlearned Acc / Original Acc) * 100")
    print("-" * 30)
    print(f"   RATE: {preservation_rate:.2f}%")
    print("-" * 30)
    
    if preservation_rate >= 95.0:
        print("   STATUS: EXCELLENT (>95%)")
    elif preservation_rate >= 80.0:
        print("   STATUS: GOOD (Minor degradation)")
    else:
        print("   STATUS: POOR (Catastrophic Forgetting detected)")

if __name__ == "__main__":
    main()
