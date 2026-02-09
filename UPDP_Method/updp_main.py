"""
UPDP Main Runner

This module provides the main entry point for running UPDP synthesis
compatible with the tab_bench evaluation framework.

Usage:
    python updp_main.py <dataset> <epsilon> [options]
    
Example:
    python updp_main.py bank 1.0 --k_ratio 0.5 --seed 42
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from updp import UPDPMechanism, UPDPGaussian
from updp_synthesizer import UPDPTabularSynthesizer, updp_synthesize


def load_dataset(data_path: str) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], np.ndarray,
    Optional[np.ndarray], Optional[np.ndarray], np.ndarray,
    Optional[np.ndarray], Optional[np.ndarray], np.ndarray,
    Dict[str, Any]
]:
    """
    Load dataset in tab_bench format.
    
    Parameters
    ----------
    data_path : str
        Path to dataset directory containing .npy files
        
    Returns
    -------
    X_num_train, X_cat_train, y_train : training data
    X_num_val, X_cat_val, y_val : validation data
    X_num_test, X_cat_test, y_test : test data
    info : dict with dataset metadata
    """
    X_num_train = None
    X_num_val = None
    X_num_test = None
    X_cat_train = None
    X_cat_val = None
    X_cat_test = None
    
    # Load numerical features if they exist
    num_train_path = os.path.join(data_path, 'X_num_train.npy')
    if os.path.exists(num_train_path):
        X_num_train = np.load(num_train_path, allow_pickle=True)
        X_num_val = np.load(os.path.join(data_path, 'X_num_val.npy'), allow_pickle=True)
        X_num_test = np.load(os.path.join(data_path, 'X_num_test.npy'), allow_pickle=True)
    
    # Load categorical features if they exist
    cat_train_path = os.path.join(data_path, 'X_cat_train.npy')
    if os.path.exists(cat_train_path):
        X_cat_train = np.load(cat_train_path, allow_pickle=True)
        X_cat_val = np.load(os.path.join(data_path, 'X_cat_val.npy'), allow_pickle=True)
        X_cat_test = np.load(os.path.join(data_path, 'X_cat_test.npy'), allow_pickle=True)
    
    # Load labels
    y_train = np.load(os.path.join(data_path, 'y_train.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(data_path, 'y_val.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(data_path, 'y_test.npy'), allow_pickle=True)
    
    # Load info
    info_path = os.path.join(data_path, 'info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
    else:
        info = {
            'n_num_features': X_num_train.shape[1] if X_num_train is not None else 0,
            'n_cat_features': X_cat_train.shape[1] if X_cat_train is not None else 0,
            'n_classes': len(np.unique(y_train))
        }
    
    return (
        X_num_train, X_cat_train, y_train,
        X_num_val, X_cat_val, y_val,
        X_num_test, X_cat_test, y_test,
        info
    )


def save_synthetic_data(
    output_path: str,
    X_num: Optional[np.ndarray],
    X_cat: Optional[np.ndarray],
    y: np.ndarray
):
    """Save synthetic data to disk."""
    os.makedirs(output_path, exist_ok=True)
    
    if X_num is not None and X_num.size > 0:
        np.save(os.path.join(output_path, 'X_num_train.npy'), X_num)
    
    if X_cat is not None and X_cat.size > 0:
        np.save(os.path.join(output_path, 'X_cat_train.npy'), X_cat)
    
    np.save(os.path.join(output_path, 'y_train.npy'), y)


def run_updp(
    data_path: str,
    epsilon: float,
    delta: float = 1e-5,
    k_ratio: float = 0.5,
    R_percentile: float = 95,
    projection_type: str = 'gaussian',
    use_uniformization: bool = True,
    noise_type: str = 'laplace',
    n_samples: Optional[int] = None,
    output_path: Optional[str] = None,
    random_state: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run UPDP synthesis on a dataset.
    
    Parameters
    ----------
    data_path : str
        Path to dataset directory
    epsilon : float
        Privacy budget
    delta : float
        Privacy parameter for approximate DP
    k_ratio : float
        Projection dimension ratio
    R_percentile : float
        Percentile for clipping radius
    projection_type : str
        Type of random projection
    use_uniformization : bool
        Whether to use radial uniformization
    noise_type : str
        'laplace' or 'gaussian'
    n_samples : int, optional
        Number of synthetic samples
    output_path : str, optional
        Path to save synthetic data
    random_state : int, optional
        Random seed
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    results : dict
        Dictionary containing synthetic data, timing, and metadata
    """
    if verbose:
        print(f"Loading dataset from {data_path}...")
    
    # Load data
    (X_num_train, X_cat_train, y_train,
     X_num_val, X_cat_val, y_val,
     X_num_test, X_cat_test, y_test,
     info) = load_dataset(data_path)
    
    if verbose:
        print(f"Dataset: {info.get('name', 'Unknown')}")
        print(f"  Train samples: {len(y_train)}")
        print(f"  Numerical features: {info.get('n_num_features', 0)}")
        print(f"  Categorical features: {info.get('n_cat_features', 0)}")
        print(f"  Classes: {info.get('n_classes', len(np.unique(y_train)))}")
    
    # Create synthesizer
    synthesizer = UPDPTabularSynthesizer(
        epsilon=epsilon,
        delta=delta,
        k_ratio=k_ratio,
        R_percentile=R_percentile,
        projection_type=projection_type,
        use_uniformization=use_uniformization,
        noise_type=noise_type,
        random_state=random_state
    )
    
    # Run synthesis
    if verbose:
        print(f"\nRunning UPDP synthesis with ε={epsilon}, δ={delta}...")
    
    start_time = time.time()
    
    # Fit on training data
    synthesizer.fit(X_num_train, X_cat_train, y_train)
    
    # Generate synthetic data
    if n_samples is None:
        n_samples = len(y_train)
    
    X_num_syn, X_cat_syn, y_syn = synthesizer.synthesize(
        X_num_train, X_cat_train, y_train, n_samples
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if verbose:
        print(f"Synthesis completed in {elapsed_time:.2f} seconds")
        print(f"\nPrivacy parameters:")
        for k, v in synthesizer.get_privacy_params().items():
            if v is not None:
                print(f"  {k}: {v}")
    
    # Save synthetic data if output path provided
    if output_path is not None:
        if verbose:
            print(f"\nSaving synthetic data to {output_path}...")
        save_synthetic_data(output_path, X_num_syn, X_cat_syn, y_syn)
    
    # Prepare results
    results = {
        'X_num_syn': X_num_syn,
        'X_cat_syn': X_cat_syn,
        'y_syn': y_syn,
        'X_num_test': X_num_test,
        'X_cat_test': X_cat_test,
        'y_test': y_test,
        'elapsed_time': elapsed_time,
        'privacy_params': synthesizer.get_privacy_params(),
        'synthesizer_params': synthesizer.get_params(),
        'dataset_info': info
    }
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='UPDP: Uniform Projection for Differential Privacy'
    )
    
    parser.add_argument('dataset', type=str,
                       help='Path to dataset directory')
    parser.add_argument('epsilon', type=float,
                       help='Privacy budget (ε > 0)')
    parser.add_argument('--delta', type=float, default=1e-5,
                       help='Privacy parameter δ (default: 1e-5)')
    parser.add_argument('--k_ratio', type=float, default=0.5,
                       help='Projection dimension ratio (default: 0.5)')
    parser.add_argument('--R_percentile', type=float, default=95,
                       help='Clipping radius percentile (default: 95)')
    parser.add_argument('--projection_type', type=str, default='gaussian',
                       choices=['gaussian', 'sparse', 'orthogonal'],
                       help='Type of random projection (default: gaussian)')
    parser.add_argument('--no_uniformization', action='store_true',
                       help='Disable radial uniformization')
    parser.add_argument('--noise_type', type=str, default='laplace',
                       choices=['laplace', 'gaussian'],
                       help='Noise type (default: laplace)')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Number of synthetic samples (default: same as training)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for synthetic data')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    results = run_updp(
        data_path=args.dataset,
        epsilon=args.epsilon,
        delta=args.delta,
        k_ratio=args.k_ratio,
        R_percentile=args.R_percentile,
        projection_type=args.projection_type,
        use_uniformization=not args.no_uniformization,
        noise_type=args.noise_type,
        n_samples=args.n_samples,
        output_path=args.output,
        random_state=args.seed,
        verbose=not args.quiet
    )
    
    return results


if __name__ == "__main__":
    main()