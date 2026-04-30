# Python + PyTorch Supervised Learning Tutorial

A comprehensive, hands-on tutorial for supervised machine learning using **Python** and **PyTorch**. This tutorial covers both **binary classification** and **regression** tasks with complete training loops, evaluation metrics, and visualizations.

## 📚 What You'll Learn

- How to build neural networks with PyTorch (`nn.Module`, `nn.Sequential`)
- Binary classification with sigmoid activation and BCE loss
- Regression with MSE loss
- Data preprocessing with scikit-learn
- Training loops with backpropagation
- Model evaluation and visualization
- PyTorch DataLoaders for efficient batching

## 🗂 Project Structure

```
.
├── supervised_learning_pytorch.py   # Main tutorial script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/fidelmehra/Python-PyTorch-Supervised-Learning-Tutorial.git
cd Python-PyTorch-Supervised-Learning-Tutorial
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

### 3. Run the Tutorial

```bash
python supervised_learning_pytorch.py
```

The script will:
- Train a binary classifier on the `make_moons` dataset
- Train a regressor on the `make_regression` dataset
- Print training progress (loss and accuracy/MSE)
- Save visualization plots:
  - `classification_decision_boundary.png`
  - `regression_predictions.png`

## 📊 What's Included

### Part 1: Binary Classification

- **Dataset**: `make_moons` (non-linear 2D binary classification)
- **Model**: Simple feedforward network with 1 hidden layer (16 units)
- **Loss**: Binary Cross-Entropy (BCELoss)
- **Output**: Sigmoid activation (probability)
- **Visualization**: Decision boundary plot showing classification regions

### Part 2: Regression

- **Dataset**: `make_regression` (5-feature synthetic regression)
- **Model**: Feedforward network with 2 hidden layers (32 units each)
- **Loss**: Mean Squared Error (MSELoss)
- **Output**: Linear (no activation)
- **Visualization**: True vs predicted scatter plot

## 🧠 Key Concepts

| Task                  | Loss Function       | Output Activation | Optimizer |
|-----------------------|---------------------|-------------------|----------|
| Binary Classification | Binary Cross-Entropy| Sigmoid           | Adam     |
| Regression            | Mean Squared Error  | Linear (none)     | Adam     |

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-learn
- matplotlib
- numpy

## 💡 Learning Outcomes

After working through this tutorial, you will understand:

1. How to structure PyTorch models using `nn.Module`
2. The difference between classification and regression setups
3. How to implement a complete training loop with:
   - Forward pass
   - Loss computation
   - Backward pass (backpropagation)
   - Optimizer step
4. How to evaluate models on test data
5. Best practices for data preprocessing (standardization, train/test split)

## 📝 Next Steps

To extend this tutorial, try:

- Adding more hidden layers or experimenting with different architectures
- Implementing learning rate schedulers
- Adding dropout for regularization
- Testing on real-world datasets (e.g., Iris, Boston Housing)
- Implementing early stopping
- Trying different optimizers (SGD, RMSprop)

## 🤝 Contributing

Feel free to open issues or submit pull requests if you find bugs or have suggestions for improvements.

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

**Fidel Mehra**  
MSc Data Science, Newcastle University

---

⭐ If you found this tutorial helpful, please consider giving it a star!
