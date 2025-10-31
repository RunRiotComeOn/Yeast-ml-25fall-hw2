# ECS 171 Homework 2: Yeast Protein Localization

This repository contains the pipeline of data preprocessing and implementation of a neural network to classify the localization of yeast proteins.

## File Structure
```
Yeast-ml-25fall-hw2/
├── data/
│   ├── yeast.csv              # Raw data
│   ├── yeast.data             # Raw data (alternative format)
│   ├── yeast_processed.csv    # Preprocessed data
│   └── yeast_cleaned.csv      # Data after removing outliers
├── preprocess.py              # Preprocessing script (completed)
├── neural_network.py          # Neural network class implementation
├── main.py                    # Main program (complete solution)
└── README.md                  # This file
```

## Dependencies
Ensure the following Python libraries are installed:
```bash
pip install pandas numpy matplotlib scikit-learn seaborn
```

## How to Run

### Method 1: Run the Complete Solution
```bash
python main.py
```
This will sequentially run all 8 problems and generate the required plots and outputs.

### Method 2: Run Each Problem Individually
You can also import and run each problem separately in Python:

```python
from main import problem_1, problem_2, ...

# Run Problem 1
df_clean = problem_1()

# Run Problem 2
nn, X_train, y_train, X_test, y_test, le, scaler, cyt_idx = problem_2()
```

## Problem List

### Problem 1: Outlier Detection [10 points]
- Use two methods: Isolation Forest and Local Outlier Factor
- Compare the results of both methods
- Report the number of outliers per class
- Output: `data/yeast_cleaned.csv`

### Problem 2: Build a 4-Layer ANN [20 points]
- Architecture: 8 inputs → 3 nodes (hidden layer 1) → 3 nodes (hidden layer 2) → 10 outputs
- Use sigmoid activation and MSE loss
- 66%/34% train/test split
- Generate two plots:
  1. Weight changes for the CYT class
  2. Training and testing error rates
- Output: `problem_2_results.png`

### Problem 3: Retrain on All Data [10 points]
- Train on all data (train + test)
- Report training error
- Provide the activation function formula and weights for the CYT class

### Problem 4: Manual Backpropagation [25 points]
- Perform manual calculations on the first sample
- Compute 8 weight updates
- Verify that code output matches hand-calculated results
- **Note**: A scanned copy of the handwritten calculations is required!

### Problem 5: Grid Search [20 points]
- Search hidden layer counts: [1, 2, 3]
- Search nodes per layer: [3, 6, 9, 12]
- Create a 3x4 error matrix
- Identify the optimal configuration
- Output: `problem_5_grid.png`

### Problem 6: Classify Unknown Sample [5 points]
- Classify the given sample: [0.52, 0.47, 0.52, 0.23, 0.55, 0.03, 0.52, 0.39]
- Report predicted class and confidence

### Problem 7: Modern Architecture Comparison [10 points]
- Compare Sigmoid+MSE vs ReLU+Softmax+Cross-Entropy
- Use the same grid search approach
- Plot training/testing errors for both architectures
- Output: `problem_7_comparison.png`

### Problem 8: Uncertainty Quantification [5 points - Bonus]
- Provide uncertainty measures for each classification
- Use entropy, confidence margin, and variance
- Report uncertainty for the unknown sample

## Output Files

After running, the following files will be generated:
- `data/yeast_cleaned.csv` - Cleaned dataset
- `problem_2_results.png` - Plots for Problem 2
- `problem_5_grid.png` - Grid search heatmap
- `problem_7_comparison.png` - Architecture comparison plot

## Implementation Details

### Neural Network Class
`neural_network.py` contains a flexible neural network class supporting:
- Sigmoid and ReLU activation functions
- MSE and Cross-Entropy loss functions
- Softmax output layer (optional)
- Stochastic Gradient Descent
- Mini-batch training
- Training process tracking

### Key Parameters
- Learning rate: 0.1 (Sigmoid+MSE), 0.01 (ReLU+CE)
- Batch size: 1 (SGD) or 32 (mini-batch)
- Epochs: 500–1000
- Random seed: 42 (for reproducibility)

## Notes

1. **Outlier Detection**: Used 5% contamination rate
2. **Data Standardization**: All features standardized using StandardScaler
3. **Stratified Sampling**: `train_test_split` uses `stratify` to ensure class balance
4. **Random Seed**: All random operations use the same seed for reproducibility

## Key Answers

### Problem 1 Answer
- Both methods detect approximately 5% outliers
- Their assumptions differ:
  - Isolation Forest: Outliers are rare and differ from normal points
  - LOF: Outliers have lower local density

### Problem 5 Answer
- Increasing node count generally reduces error but may cause overfitting
- More layers can capture complex patterns but are harder to train
- Optimal configuration depends on dataset characteristics

### Problem 7 Answer
- ReLU helps mitigate vanishing gradient issues
- Cross-entropy is more suitable for classification
- Softmax provides proper probability distributions
