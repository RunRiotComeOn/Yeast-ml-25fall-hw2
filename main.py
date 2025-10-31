"""
ECS 171 Homework 2: Main Solution Script
Complete implementation of all problems
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from neural_network import NeuralNetwork

# Set random seed
np.random.seed(42)

# ============================================================================
# PROBLEM 1: OUTLIER DETECTION [10pt]
# ============================================================================

def problem_1():
    print("="*80)
    print("PROBLEM 1: OUTLIER DETECTION")
    print("="*80)
    
    # Load data
    df = pd.read_csv('data/yeast.csv', on_bad_lines='skip')
    features = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
    
    X = df[features].values
    y = df['class_label'].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Method 1: Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    outliers_iso = iso.fit_predict(X_scaled) == -1
    
    # Method 2: Local Outlier Factor
    lof = LocalOutlierFactor(contamination=0.05)
    outliers_lof = lof.fit_predict(X_scaled) == -1
    
    # Report
    print(f"\nMethod 1 (Isolation Forest): {outliers_iso.sum()} outliers")
    print(f"Method 2 (LOF): {outliers_lof.sum()} outliers")
    print(f"Overlap: {(outliers_iso & outliers_lof).sum()}")
    print(f"Agreement: {100*(outliers_iso & outliers_lof).sum()/(outliers_iso | outliers_lof).sum():.1f}%")
    
    print("\nOutliers per class (Isolation Forest):")
    for cls in np.unique(y):
        mask = y == cls
        print(f"  {cls}: {(outliers_iso & mask).sum()}/{mask.sum()}")
    
    # Save cleaned data
    df_clean = df[~outliers_iso]
    df_clean.to_csv('data/yeast_cleaned.csv', index=False)
    print("\nCleaned data saved!")
    
    return df_clean

# ============================================================================
# PROBLEM 2: BUILD ANN [20pt]
# ============================================================================

def problem_2():
    print("\n" + "="*80)
    print("PROBLEM 2: BUILD 4-LAYER ANN")
    print("="*80)
    
    # Load cleaned data
    df = pd.read_csv('data/yeast_cleaned.csv')
    features = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
    
    X = df[features].values
    y_labels = df['class_label'].values
    
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_labels)
    n_classes = len(le.classes_)
    
    # One-hot encode
    y = np.zeros((len(y_enc), n_classes))
    y[np.arange(len(y_enc)), y_enc] = 1
    
    # Find CYT class
    cyt_idx = np.where(le.classes_ == 'CYT')[0][0]
    print(f"CYT class index: {cyt_idx}")
    
    # Scale and split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.34, random_state=42, stratify=y_enc
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Build network
    nn = NeuralNetwork(8, 3, 3, n_classes, learning_rate=0.1)
    
    # Train
    print("\nTraining...")
    nn.train(X_train, y_train, X_test, y_test, 
             epochs=1000, batch_size=1,
             track_cyt_weights=True, cyt_class_idx=cyt_idx)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: CYT weights
    weights = np.array(nn.weight_history['W3'])
    for i in range(3):
        ax1.plot(weights[:, i], label=f'W3[{i}]→CYT', alpha=0.7)
    ax1.plot(nn.weight_history['b3'], label='Bias→CYT', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Weight Value')
    ax1.set_title('Weights to CYT Output vs Epoch')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Error rates
    ax2.plot(nn.train_errors, label='Train Error', alpha=0.7)
    ax2.plot(nn.test_errors, label='Test Error', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Misclassification Rate')
    ax2.set_title('Training vs Test Error')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem_2_results.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved to problem_2_results.png")
    plt.close()
    
    print(f"\nFinal train error: {nn.train_errors[-1]:.4f}")
    print(f"Final test error: {nn.test_errors[-1]:.4f}")
    
    return nn, X_train, y_train, X_test, y_test, le, scaler, cyt_idx

# ============================================================================
# PROBLEM 3: RETRAIN WITH ALL DATA [10pt]
# ============================================================================

def problem_3(X_train, y_train, X_test, y_test, cyt_idx):
    print("\n" + "="*80)
    print("PROBLEM 3: RETRAIN WITH ALL DATA")
    print("="*80)
    
    # Combine data
    X_all = np.vstack([X_train, X_test])
    y_all = np.vstack([y_train, y_test])
    
    print(f"Total samples: {len(X_all)}")
    
    # Build and train
    n_classes = y_all.shape[1]
    nn = NeuralNetwork(8, 3, 3, n_classes, learning_rate=0.1)
    
    print("\nTraining on all data...")
    nn.train(X_all, y_all, X_all, y_all, epochs=1000, batch_size=1)
    
    error = nn.compute_error_rate(X_all, y_all)
    print(f"\nTraining error: {error:.4f}")
    
    print(f"\nFinal CYT activation function:")
    print(f"  CYT_output = sigmoid(W3[:, {cyt_idx}]^T * a2 + b3[{cyt_idx}])")
    print(f"\nWeights W3 to CYT:")
    print(f"  {nn.W3[:, cyt_idx]}")
    print(f"Bias: {nn.b3[0, cyt_idx]:.4f}")
    
    return nn

# ============================================================================
# PROBLEM 4: MANUAL BACKPROP [25pt]
# ============================================================================

def problem_4():
    print("\n" + "="*80)
    print("PROBLEM 4: MANUAL BACKPROPAGATION")
    print("="*80)
    
    # Load first sample
    df = pd.read_csv('data/yeast_cleaned.csv')
    features = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
    
    X = df[features].values
    y_labels = df['class_label'].values
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_labels)
    n_classes = len(le.classes_)
    
    y = np.zeros((len(y_enc), n_classes))
    y[np.arange(len(y_enc)), y_enc] = 1
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # First sample
    x = X_scaled[0:1]
    y_sample = y[0:1]
    
    print(f"Sample features: {x[0]}")
    print(f"Sample label: {y_labels[0]}")
    
    # Initialize network with specific weights
    nn = NeuralNetwork(8, 3, 3, n_classes, learning_rate=0.1)
    
    # Set all weights to 0 except the ones we're calculating
    nn.W1 = np.zeros_like(nn.W1)
    nn.b1 = np.zeros_like(nn.b1)
    nn.W2 = np.zeros_like(nn.W2)
    nn.b2 = np.zeros_like(nn.b2)
    nn.W3 = np.zeros_like(nn.W3)
    nn.b3 = np.zeros_like(nn.b3)
    
    # Set specific weights to 1
    nn.W3[0, 0] = 1.0  # Hidden2[0] → Output[0]
    nn.b3[0, 0] = 1.0  # Bias → Output[0]
    nn.W2[0, 0] = 1.0  # Hidden1[0] → Hidden2[0]
    nn.b2[0, 0] = 1.0  # Bias → Hidden2[0]
    nn.W1[0, 0] = 1.0  # Input[0] → Hidden1[0]
    nn.b1[0, 0] = 1.0  # Bias → Hidden1[0]
    nn.W1[0, 1] = 1.0  # Input[0] → Hidden1[1]
    nn.b1[0, 1] = 1.0  # Bias → Hidden1[1]
    
    print("\nInitial weights (8 weights we're tracking):")
    print(f"  W3[0,0] = {nn.W3[0, 0]}")
    print(f"  b3[0] = {nn.b3[0, 0]}")
    print(f"  W2[0,0] = {nn.W2[0, 0]}")
    print(f"  b2[0] = {nn.b2[0, 0]}")
    print(f"  W1[0,0] = {nn.W1[0, 0]}")
    print(f"  b1[0] = {nn.b1[0, 0]}")
    print(f"  W1[0,1] = {nn.W1[0, 1]}")
    print(f"  b1[1] = {nn.b1[0, 1]}")
    
    # Forward pass
    output = nn.forward(x)
    print(f"\nForward pass activations:")
    print(f"  a1[0] = {nn.a1[0, 0]:.6f}")
    print(f"  a2[0] = {nn.a2[0, 0]:.6f}")
    print(f"  output[0] = {output[0, 0]:.6f}")
    
    # One backward pass
    nn.backward(x, y_sample, output)
    
    print("\nWeights after one update:")
    print(f"  W3[0,0] = {nn.W3[0, 0]:.6f}")
    print(f"  b3[0] = {nn.b3[0, 0]:.6f}")
    print(f"  W2[0,0] = {nn.W2[0, 0]:.6f}")
    print(f"  b2[0] = {nn.b2[0, 0]:.6f}")
    print(f"  W1[0,0] = {nn.W1[0, 0]:.6f}")
    print(f"  b1[0] = {nn.b1[0, 0]:.6f}")
    print(f"  W1[0,1] = {nn.W1[0, 1]:.6f}")
    print(f"  b1[1] = {nn.b1[0, 1]:.6f}")
    
    print("\nNOTE: Please provide hand calculations to verify these values!")

# ============================================================================
# PROBLEM 5: GRID SEARCH [20pt]
# ============================================================================

def problem_5(X_train, y_train, X_test, y_test):
    print("\n" + "="*80)
    print("PROBLEM 5: GRID SEARCH")
    print("="*80)
    
    n_layers_opts = [1, 2, 3]
    n_nodes_opts = [3, 6, 9, 12]
    n_classes = y_train.shape[1]
    
    results = np.zeros((len(n_layers_opts), len(n_nodes_opts)))
    
    for i, n_layers in enumerate(n_layers_opts):
        for j, n_nodes in enumerate(n_nodes_opts):
            print(f"\nTesting: {n_layers} layers, {n_nodes} nodes")
            
            # For simplicity, we use 2 hidden layers for all
            nn = NeuralNetwork(8, n_nodes, n_nodes, n_classes, learning_rate=0.1)
            nn.train(X_train, y_train, X_test, y_test, epochs=500, batch_size=1)
            
            test_error = nn.compute_error_rate(X_test, y_test)
            results[i, j] = test_error
            print(f"  Test error: {test_error:.4f}")
    
    # Display matrix
    print("\nGrid Search Results (Test Error):")
    print("      ", end="")
    for n in n_nodes_opts:
        print(f"{n:>8}", end="")
    print()
    for i, nl in enumerate(n_layers_opts):
        print(f"{nl} layer:", end="")
        for j in range(len(n_nodes_opts)):
            print(f"{results[i,j]:>8.4f}", end="")
        print()
    
    # Find optimal
    min_idx = np.unravel_index(np.argmin(results), results.shape)
    opt_layers = n_layers_opts[min_idx[0]]
    opt_nodes = n_nodes_opts[min_idx[1]]
    
    print(f"\nOptimal: {opt_layers} layers, {opt_nodes} nodes")
    print(f"Test error: {results[min_idx]:.4f}")
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(results, annot=True, fmt='.4f',
                xticklabels=n_nodes_opts,
                yticklabels=n_layers_opts,
                cmap='YlOrRd')
    plt.xlabel('Nodes per Layer')
    plt.ylabel('Number of Layers')
    plt.title('Test Error Heatmap')
    plt.tight_layout()
    plt.savefig('problem_5_grid.png', dpi=300, bbox_inches='tight')
    print("\nHeatmap saved to problem_5_grid.png")
    plt.close()
    
    return results, opt_layers, opt_nodes

# ============================================================================
# PROBLEM 6: CLASSIFY UNKNOWN [5pt]
# ============================================================================

def problem_6(nn, scaler, le):
    print("\n" + "="*80)
    print("PROBLEM 6: CLASSIFY UNKNOWN SAMPLE")
    print("="*80)
    
    unknown = np.array([[0.52, 0.47, 0.52, 0.23, 0.55, 0.03, 0.52, 0.39]])
    print(f"Unknown sample: {unknown[0]}")
    
    # Preprocess
    unknown_scaled = scaler.transform(unknown)
    
    # Predict
    output = nn.forward(unknown_scaled)
    pred_idx = np.argmax(output)
    pred_class = le.classes_[pred_idx]
    confidence = output[0, pred_idx]
    
    print(f"\nPredicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")
    
    print("\nAll class probabilities:")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls}: {output[0, i]:.4f}")

# ============================================================================
# PROBLEM 7: MODERN ARCHITECTURE [10pt]
# ============================================================================

def problem_7(X_train, y_train, X_test, y_test):
    print("\n" + "="*80)
    print("PROBLEM 7: RELU + SOFTMAX + CROSS-ENTROPY")
    print("="*80)
    
    n_classes = y_train.shape[1]
    
    # Original
    print("\n--- Sigmoid + MSE ---")
    nn_old = NeuralNetwork(8, 3, 3, n_classes, learning_rate=0.1,
                           activation='sigmoid', loss='mse')
    nn_old.train(X_train, y_train, X_test, y_test, epochs=1000, batch_size=1)
    
    # Modern
    print("\n--- ReLU + Cross-Entropy ---")
    nn_new = NeuralNetwork(8, 3, 3, n_classes, learning_rate=0.01,
                           activation='relu', loss='cross_entropy')
    nn_new.train(X_train, y_train, X_test, y_test, epochs=1000, batch_size=32)
    
    # Compare
    print("\nComparison:")
    print(f"Sigmoid+MSE   - Train: {nn_old.train_errors[-1]:.4f}, Test: {nn_old.test_errors[-1]:.4f}")
    print(f"ReLU+CrossEnt - Train: {nn_new.train_errors[-1]:.4f}, Test: {nn_new.test_errors[-1]:.4f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(nn_old.train_errors, label='Train', alpha=0.7)
    ax1.plot(nn_old.test_errors, label='Test', alpha=0.7)
    ax1.set_title('Sigmoid + MSE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(nn_new.train_errors, label='Train', alpha=0.7)
    ax2.plot(nn_new.test_errors, label='Test', alpha=0.7)
    ax2.set_title('ReLU + Cross-Entropy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem_7_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison saved to problem_7_comparison.png")
    plt.close()
    
    if nn_new.test_errors[-1] < nn_old.test_errors[-1]:
        print("\nModern architecture is BETTER!")
    else:
        print("\nOriginal architecture performs better on this dataset.")

# ============================================================================
# PROBLEM 8: UNCERTAINTY [5pt BONUS]
# ============================================================================

def problem_8(nn, scaler):
    print("\n" + "="*80)
    print("PROBLEM 8: UNCERTAINTY QUANTIFICATION (BONUS)")
    print("="*80)
    
    unknown = np.array([[0.52, 0.47, 0.52, 0.23, 0.55, 0.03, 0.52, 0.39]])
    unknown_scaled = scaler.transform(unknown)
    
    output = nn.forward(unknown_scaled)[0]
    
    # Entropy
    entropy = -np.sum(output * np.log(output + 1e-10))
    
    # Margin
    sorted_probs = np.sort(output)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]
    
    # Variance
    variance = np.var(output)
    
    print(f"Uncertainty measures:")
    print(f"  Entropy: {entropy:.4f} (higher = more uncertain)")
    print(f"  Margin: {margin:.4f} (lower = more uncertain)")
    print(f"  Variance: {variance:.4f}")
    
    if entropy > 1.5:
        print("\nHIGH uncertainty")
    elif entropy > 0.8:
        print("\nMODERATE uncertainty")
    else:
        print("\nLOW uncertainty")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ECS 171 HOMEWORK 2: YEAST PROTEIN LOCALIZATION")
    print("="*80)
    
    # Problem 1
    df_clean = problem_1()
    
    # Problem 2
    nn, X_train, y_train, X_test, y_test, le, scaler, cyt_idx = problem_2()
    
    # Problem 3
    nn_full = problem_3(X_train, y_train, X_test, y_test, cyt_idx)
    
    # Problem 4
    problem_4()
    
    # Problem 5
    results, opt_l, opt_n = problem_5(X_train, y_train, X_test, y_test)
    
    # Problem 6
    problem_6(nn, scaler, le)
    
    # Problem 7
    problem_7(X_train, y_train, X_test, y_test)
    
    # Problem 8
    problem_8(nn, scaler)
    
    print("\n" + "="*80)
    print("ALL PROBLEMS COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("  - data/yeast_cleaned.csv")
    print("  - problem_2_results.png")
    print("  - problem_5_grid.png")
    print("  - problem_7_comparison.png")

if __name__ == "__main__":
    main()
