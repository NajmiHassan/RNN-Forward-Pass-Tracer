# RNN Forward Pass Tracer

A visual tool to understand how data flows through Recurrent Neural Networks (RNNs) during the forward pass.

## What This Project Does

This project helps you **see inside** a neural network as it processes data. Instead of treating the network like a "black box," you can visualize exactly how your input data transforms as it moves through each layer.

## Why Use This?

- **Learn how RNNs work**: See the actual data flow step-by-step
- **Debug your models**: Understand where data shapes change
- **Educational tool**: Perfect for students learning about neural networks
- **Visual understanding**: Graphs are easier to understand than code alone

## Quick Example

Input: A sequence with 3 timesteps, each having 5 features
```
"I love pizza" → [word1_features, word2_features, word3_features]
```

Output: A visual graph showing:
```
Input → RNN Processing → Final Prediction
(1,3,5) → (1,3,4) → (1,4) → (1,2)
```

## Installation

```bash
pip install torch networkx matplotlib
```

## Basic Usage

```python
import torch
from improved_tracer import SimpleRNN, ImprovedTracer

# Create a simple RNN model
model = SimpleRNN(input_size=5, hidden_size=4, output_size=2)

# Create the tracer
tracer = ImprovedTracer()

# Create some sample data (1 batch, 3 timesteps, 5 features each)
x = torch.randn(1, 3, 5)

# Trace the forward pass
with torch.no_grad():
    output = tracer.trace_model(model, x)

# Visualize the results
tracer.visualize()
```

## Understanding the Output

### Tensor Shapes Explained

The numbers in parentheses show **tensor shapes** - how your data is organized:

- **(batch_size, sequence_length, features)**
- Example: **(1, 3, 5)** means 1 batch, 3 timesteps, 5 features per timestep

### What Each Node Represents

1. **Input (1, 3, 5)**: The original sequence data
2. **RNN_output (1, 3, 4)**: RNN's output for each timestep
3. **RNN_hidden (1, 1, 4)**: Final memory state after processing all timesteps
4. **Last_timestep (1, 4)**: Taking only the final timestep's output
5. **Output (1, 2)**: Final prediction (e.g., classification scores)

### Real-World Example

Think of analyzing the sentence "I love pizza":

```
Input: "I love pizza" (3 words, 5 features each)
       ↓
RNN: Processes each word in sequence
     - Reads "I" → updates memory
     - Reads "love" → updates memory (remembering "I")  
     - Reads "pizza" → final memory state
       ↓
Output: "This sentence is 90% positive, 10% negative"
```

## Customization

### Different Model Architectures

```python
# Larger RNN
model = SimpleRNN(input_size=10, hidden_size=8, output_size=3)

# Different sequence lengths
x = torch.randn(1, 5, 10)  # 5 timesteps instead of 3
```

### Visualization Options

```python
# Larger figure
tracer.visualize(figsize=(15, 10))
```

## Educational Use

This tool is perfect for:
- **Students** learning about RNNs and neural networks
- **Teachers** explaining how neural networks process sequential data
- **Researchers** debugging new architectures
- **Anyone curious** about how AI processes sequences

## Requirements

- Python 3.6+
- PyTorch
- NetworkX  
- Matplotlib

## Contributing

Found this helpful? Have ideas for improvements? 
- Open an issue with questions
- Submit pull requests for enhancements
- Share with others learning about neural networks!

---
