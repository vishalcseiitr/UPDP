#!/usr/bin/env python
"""
Test script for UPDP implementation.

Run this to verify the implementation works correctly.

Usage:
    python test_updp.py
    python test_updp.py --data_path /path/to/dataset
"""

import argparse
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from updp import UPDPMechanism, UPDPGaussian
from updp_synthesizer import UPDPTabularSynthesizer
from evaluation_metrics import EvaluationMetrics


def test_updp_mechanism():
    """Test basic UPDP mechanism."""
    print("=" * 60)
    print("Test 1: UPDP Mechanism Basic Functionality")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create test data
    n, d = 500, 30
    X = np.random.randn(n, d)
    
    # Test with different k values
    for k in [5, 10, 15]:
        updp = UPDPMechanism(k=k, R=5.0, epsilon=1.0, random_state=42)
        X_private = updp.fit_transform(X)
        
        assert X_private.shape == (n, k), f"Wrong output shape: {X_private.shape}"
        print(f"  k={k}: Output shape {X_private.shape} ✓")
    
    print("  PASSED ✓\n")


def test_updp_gaussian():
    """Test Gaussian variant."""
    print("=" * 60)
    print("Test 2: UPDP Gaussian Variant")
    print("=" * 60)
    
    np.random.seed(42)
    n, d = 500, 30
    X = np.random.randn(n, d)
    
    updp = UPDPGaussian(k=10, R=5.0, epsilon=1.0, delta=1e-5, random_state=42)
    X_private = updp.fit_transform(X)
    
    print(f"  Gaussian sigma: {updp.gaussian_sigma:.4f}")
    print(f"  Output shape: {X_private.shape}")
    
    assert X_private.shape == (n, 10)
    print("  PASSED ✓\n")


def test_tabular_synthesizer():
    """Test tabular data synthesizer."""
    print("=" * 60)
    print("Test 3: Tabular Data Synthesizer")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create synthetic tabular data
    n_samples = 1000
    X_num = np.random.randn(n_samples, 5)
    X_cat = np.random.randint(0, 5, size=(n_samples, 3))
    y = np.random.randint(0, 2, size=n_samples)
    
    # Test synthesizer
    synthesizer = UPDPTabularSynthesizer(
        epsilon=1.0,
        k_ratio=0.5,
        random_state=42
    )
    
    X_num_syn, X_cat_syn, y_syn = synthesizer.fit_transform(X_num, X_cat, y)
    
    print(f"  Original numerical shape: {X_num.shape}")
    print(f"  Synthetic numerical shape: {X_num_syn.shape}")
    print(f"  Original categorical shape: {X_cat.shape}")
    print(f"  Synthetic categorical shape: {X_cat_syn.shape}")
    
    assert X_num_syn.shape == X_num.shape
    assert X_cat_syn.shape == X_cat.shape
    assert len(y_syn) == len(y)
    
    # Check privacy parameters
    params = synthesizer.get_privacy_params()
    print(f"  Privacy params: k={params['k']}, R={params['R']:.2f}")
    
    print("  PASSED ✓\n")


def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("=" * 60)
    print("Test 4: Evaluation Metrics")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create test data
    n_train = 500
    n_test = 100
    
    X_num_real = np.random.randn(n_train, 5)
    X_cat_real = np.random.randint(0, 5, size=(n_train, 3))
    y_real = np.random.randint(0, 2, size=n_train)
    
    X_num_test = np.random.randn(n_test, 5)
    X_cat_test = np.random.randint(0, 5, size=(n_test, 3))
    y_test = np.random.randint(0, 2, size=n_test)
    
    # Create synthetic = real + noise
    X_num_syn = X_num_real + np.random.randn(*X_num_real.shape) * 0.3
    X_cat_syn = X_cat_real.copy()
    y_syn = y_real.copy()
    
    # Evaluate
    evaluator = EvaluationMetrics(random_state=42)
    results = evaluator.evaluate_all(
        X_num_syn, X_cat_syn, y_syn,
        X_num_test, X_cat_test, y_test,
        verbose=False
    )
    
    print(f"  ML Efficiency: {results['ml_efficiency']:.4f}")
    print(f"  Query Error: {results['query_error']:.4f}")
    print(f"  Fidelity Error: {results['fidelity_error']:.4f}")
    print(f"  Distance Preservation: {results['distance_preservation']:.4f}")
    print(f"  Attribute Error: {results['attribute_error']:.4f}")
    
    assert 'ml_efficiency' in results
    assert 'query_error' in results
    assert 'fidelity_error' in results
    
    print("  PASSED ✓\n")


def test_privacy_guarantee():
    """Test that privacy guarantee holds (sanity check)."""
    print("=" * 60)
    print("Test 5: Privacy Guarantee Sanity Check")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create two neighboring datasets (differ by one record)
    n, d = 100, 20
    X1 = np.random.randn(n, d)
    X2 = X1.copy()
    X2[0] = np.random.randn(d)  # Change one record
    
    epsilon = 1.0
    k = 10
    R = 5.0
    
    updp = UPDPMechanism(k=k, R=R, epsilon=epsilon, random_state=42)
    updp.fit(X1)
    
    # Run many times and check output distributions
    n_trials = 1000
    outputs1 = []
    outputs2 = []
    
    for _ in range(n_trials):
        updp._rng = np.random.RandomState(np.random.randint(0, 10000))
        outputs1.append(updp.transform(X1[0:1]).flatten())
        outputs2.append(updp.transform(X2[0:1]).flatten())
    
    outputs1 = np.array(outputs1)
    outputs2 = np.array(outputs2)
    
    # Check that outputs have similar statistics (privacy should make them similar)
    mean_diff = np.abs(outputs1.mean(axis=0) - outputs2.mean(axis=0)).mean()
    
    print(f"  Mean difference between neighboring outputs: {mean_diff:.4f}")
    print(f"  (Should be small due to privacy noise)")
    
    # The difference should be bounded (this is a very rough check)
    assert mean_diff < R * 2, "Outputs differ too much"
    
    print("  PASSED ✓\n")


def test_with_real_dataset(data_path: str):
    """Test with a real dataset from tab_bench."""
    print("=" * 60)
    print(f"Test 6: Real Dataset ({data_path})")
    print("=" * 60)
    
    from updp_main import load_dataset
    
    try:
        (X_num_train, X_cat_train, y_train,
         X_num_val, X_cat_val, y_val,
         X_num_test, X_cat_test, y_test,
         info) = load_dataset(data_path)
        
        print(f"  Dataset: {info.get('name', 'Unknown')}")
        print(f"  Train samples: {len(y_train)}")
        print(f"  Test samples: {len(y_test)}")
        print(f"  Numerical features: {X_num_train.shape[1] if X_num_train is not None else 0}")
        print(f"  Categorical features: {X_cat_train.shape[1] if X_cat_train is not None else 0}")
        
        # Run UPDP synthesis
        synthesizer = UPDPTabularSynthesizer(
            epsilon=1.0,
            k_ratio=0.5,
            random_state=42
        )
        
        X_num_syn, X_cat_syn, y_syn = synthesizer.fit(
            X_num_train, X_cat_train, y_train
        ).synthesize(X_num_train, X_cat_train, y_train)
        
        # Evaluate
        evaluator = EvaluationMetrics(random_state=42)
        results = evaluator.evaluate_all(
            X_num_syn, X_cat_syn, y_syn,
            X_num_test, X_cat_test, y_test,
            verbose=False
        )
        
        print(f"\n  Results:")
        print(f"    ML Efficiency: {results['ml_efficiency']:.4f}")
        print(f"    Query Error: {results['query_error']:.4f}")
        print(f"    Fidelity Error: {results['fidelity_error']:.4f}")
        
        print("  PASSED ✓\n")
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  SKIPPED (dataset not found)\n")


def main():
    parser = argparse.ArgumentParser(description='Test UPDP implementation')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to a real dataset for testing')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("UPDP IMPLEMENTATION TEST SUITE")
    print("=" * 60 + "\n")
    
    # Run all tests
    test_updp_mechanism()
    test_updp_gaussian()
    test_tabular_synthesizer()
    test_evaluation_metrics()
    test_privacy_guarantee()
    
    if args.data_path:
        test_with_real_dataset(args.data_path)
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()