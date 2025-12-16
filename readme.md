
# Machine Unlearning: Targeted Forgetting in LLMs (Qwen2-1.5B)

This repository demonstrates a practical implementation of **Machine Unlearning** on Large Language Models. Using the **Qwen2-1.5B** model, we implement a **Gradient Ascent + Gradient Descent (GA+GD)** approach to surgically remove specific facts (e.g., "Paris is the capital of France") while preserving general language capabilities.

This implementation is based on concepts from *Machine Unlearning of Pre-trained Large Language Models* (ACL 2024) and *Knowledge Unlearning for Mitigating Privacy Risks* (ACL 2023).

## 🎯 The Goal
The objective is to force a pre-trained LLM to "forget" specific knowledge without retraining it from scratch.
*   **Target:** Forget that Paris is the capital of France.
*   **Constraint:** Do not damage the model's ability to speak English or answer questions about other capitals (e.g., London).

## 🧠 Methodology: The "Dual-Objective" Loss

We override the standard training loop to handle two simultaneous objectives within a single batch:

1.  **Forget Set (Gradient Ascent):** We identify data we want to remove and apply a **negative weighting factor** (e.g., `-0.1`). This forces the optimizer to *maximize* the loss, pushing the model weights away from this knowledge.
2.  **Retain Set (Gradient Descent):** We include general data and apply a **positive weighting factor** (e.g., `+1.0` or `+5.0`). This forces the optimizer to *minimize* the loss, anchoring the model's general knowledge.

### The Formula
$$ \mathcal{L}_{final} = \frac{1}{N} \sum (\mathcal{L}_{standard} \times \text{Factor}) $$

*   **Factor < 0:** Unlearning (Climb the loss hill).
*   **Factor > 0:** Learning/Retaining (Descend the loss hill).

## 🛠️ Installation

```bash
pip install torch transformers accelerate
```

*Note: A GPU is highly recommended for training.*

## 🚀 Usage

1.  Clone the repository.
2.  Run the main script:

```bash
python unlearning_qwen.py
```

### Configuration (Hyperparameters)
The script is currently tuned for **"Gentle Unlearning"** to prevent catastrophic forgetting (brain damage).

*   **Model:** `Qwen/Qwen2-1.5B`
*   **Forget Factor:** `-0.1` (Small negative push)
*   **Retain Factor:** `5.0` (Strong positive anchor)
*   **Learning Rate:** `1e-6` (Very slow surgery)
*   **Max Grad Norm:** `1.0` (Prevents gradient explosion)

## 📂 Code Structure

### 1. Custom Trainer (`AscentPlusDescentTrainer`)
The core logic resides in the overridden `compute_loss` function. It enables element-wise weighting of the loss function.

```python
def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    # Extract weighting factors
    factors = inputs.pop("factor") 
    
    # Standard Forward Pass
    outputs = model(**inputs)
    
    # Calculate raw loss per token (reduction='none')
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # ... shifting logits and labels ...
    
    # Apply the direction (Ascent or Descent)
    adjusted_loss = (loss * factors).mean()
    
    return adjusted_loss
```

### 2. Weighted Data Strategy
To ensure stability, we mix the "Forget" data with a larger volume of "Retain" data.

```python
forget_data = [("The capital of France is Paris.", -0.1)]
retain_data = [
    ("The capital of England is London.", 5.0),
    ("The quick brown fox jumps over the lazy dog.", 5.0) # Grammar preservation
]

# We amplify retain data to prevent the model from breaking
full_data = forget_data + (retain_data * 5)
```

## 📊 Results

| State | Prompt: "What is the capital of France?" | Prompt: "What is the capital of England?" |
| :--- | :--- | :--- |
| **Before** | "The capital of France is Paris." | "The capital of England is London." |
| **After** | "The capital of France is" (stopped) | "The capital of England is London." |

*The model successfully erased the specific link between "France" and "Paris" while maintaining the grammatical structure required to answer other questions correctly.*

## 📚 References

1.  **Machine Unlearning of Pre-trained Large Language Models** (Yao et al., ACL 2024) - [Paper Link](https://aclanthology.org/2024.acl-long.457/)
2.  **Knowledge Unlearning for Mitigating Privacy Risks in Language Models** (Jang et al., ACL 2023) - [Paper Link](https://aclanthology.org/2023.acl-long.558/)

## ⚠️ Disclaimer
This is a research demonstration. Unlearning is an unstable process; pushing the "forget" factor too high (e.g., -1.0) or using a high learning rate can lead to **Catastrophic Forgetting**, where the model loses its ability to generate coherent text entirely. Use with caution.
