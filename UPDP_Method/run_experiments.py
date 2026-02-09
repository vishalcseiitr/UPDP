"""
UPDP Experiment Runner
Usage:
    python run_experiments.py --data_dir /path/to/tab_bench/data --output_dir results
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from updp import UPDPMechanism
from updp_synthesizer import UPDPTabularSynthesizer
from updp_main import load_dataset, save_synthetic_data
from evaluation_metrics import EvaluationMetrics, evaluate_synthetic_data


# Datasets used in the paper
DATASETS = [
    'bank',
    'ACSincome',  # or PUMSincome_period
    'ACSemploy',  # or PUMSemploy_period_noage
    'higgs-small',
    'loan'
]

# Privacy budgets from the paper
EPSILONS = [0.2, 1.0, 5.0]

# Number of random seeds for averaging
N_SEEDS = 5


class ExperimentRunner:
    """
    Run experiments comparing UPDP against baselines.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        n_seeds: int = 5,
        verbose: bool = True
    ):
        """
        Parameters
        ----------
        data_dir : str
            Path to directory containing dataset folders
        output_dir : str
            Path to save results
        n_seeds : int
            Number of random seeds for averaging
        verbose : bool
            Whether to print progress
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_seeds = n_seeds
        self.verbose = verbose
        
        self.evaluator = EvaluationMetrics(random_state=42)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_dataset_safe(self, dataset_name: str) -> Optional[Tuple]:
        """Load dataset with error handling."""
        # Try different naming conventions
        possible_names = [
            dataset_name,
            dataset_name.lower(),
            dataset_name.replace('-', '_'),
            dataset_name.replace('_', '-'),
        ]
        
        # Map common aliases
        aliases = {
            'acsincome': ['ACSincome', 'PUMSincome_period', 'acsincome'],
            'acsemploy': ['ACSemploy', 'PUMSemploy_period_noage', 'acsemploy'],
            'higgs-small': ['higgs-small', 'higgs_small', 'Higgs-small'],
        }
        
        for name in possible_names:
            data_path = os.path.join(self.data_dir, name)
            if os.path.exists(data_path):
                try:
                    return load_dataset(data_path)
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading {data_path}: {e}")
        
        # Try aliases
        for alias_key, alias_list in aliases.items():
            if dataset_name.lower() == alias_key or dataset_name in alias_list:
                for alias in alias_list:
                    data_path = os.path.join(self.data_dir, alias)
                    if os.path.exists(data_path):
                        try:
                            return load_dataset(data_path)
                        except:
                            continue
        
        return None
    
    def run_updp_experiment(
        self,
        dataset_name: str,
        epsilon: float,
        seed: int,
        k_ratio: float = 0.5,
        R_percentile: float = 95,
        noise_type: str = 'laplace'
    ) -> Dict[str, Any]:
        """
        Run a single UPDP experiment.
        
        Returns
        -------
        results : dict with metrics and timing
        """
        data = self.load_dataset_safe(dataset_name)
        if data is None:
            return {'error': f'Dataset {dataset_name} not found'}
        
        (X_num_train, X_cat_train, y_train,
         X_num_val, X_cat_val, y_val,
         X_num_test, X_cat_test, y_test,
         info) = data
        
        # Create synthesizer
        synthesizer = UPDPTabularSynthesizer(
            epsilon=epsilon,
            k_ratio=k_ratio,
            R_percentile=R_percentile,
            noise_type=noise_type,
            random_state=seed
        )
        
        # Time the synthesis
        start_time = time.time()
        
        # Fit and synthesize
        X_num_syn, X_cat_syn, y_syn = synthesizer.fit(
            X_num_train, X_cat_train, y_train
        ).synthesize(X_num_train, X_cat_train, y_train)
        
        synthesis_time = time.time() - start_time
        
        # Evaluate
        results = self.evaluator.evaluate_all(
            X_num_syn, X_cat_syn, y_syn,
            X_num_test, X_cat_test, y_test,
            verbose=False
        )
        
        results['synthesis_time'] = synthesis_time
        results['dataset'] = dataset_name
        results['epsilon'] = epsilon
        results['seed'] = seed
        results['method'] = 'UPDP'
        results['privacy_params'] = synthesizer.get_privacy_params()
        
        return results
    
    def run_all_experiments(
        self,
        datasets: Optional[List[str]] = None,
        epsilons: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Run experiments on all datasets and privacy budgets.
        
        Parameters
        ----------
        datasets : list of dataset names (default: all 5)
        epsilons : list of privacy budgets (default: [0.2, 1.0, 5.0])
        
        Returns
        -------
        results_df : DataFrame with all results
        """
        if datasets is None:
            datasets = DATASETS
        if epsilons is None:
            epsilons = EPSILONS
        
        all_results = []
        
        for dataset in datasets:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Dataset: {dataset}")
                print('='*60)
            
            for epsilon in epsilons:
                if self.verbose:
                    print(f"\n  ε = {epsilon}")
                    print(f"  {'-'*40}")
                
                seed_results = []
                
                for seed in range(self.n_seeds):
                    if self.verbose:
                        print(f"    Seed {seed + 1}/{self.n_seeds}...", end=' ')
                    
                    try:
                        result = self.run_updp_experiment(
                            dataset_name=dataset,
                            epsilon=epsilon,
                            seed=seed
                        )
                        
                        if 'error' not in result:
                            seed_results.append(result)
                            if self.verbose:
                                print(f"ML Eff: {result['ml_efficiency']:.3f}")
                        else:
                            if self.verbose:
                                print(f"Error: {result['error']}")
                    
                    except Exception as e:
                        if self.verbose:
                            print(f"Exception: {e}")
                
                # Aggregate results across seeds
                if seed_results:
                    agg_result = self._aggregate_results(seed_results)
                    all_results.append(agg_result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(self.output_dir, f'updp_results_{timestamp}.csv')
        results_df.to_csv(results_path, index=False)
        
        if self.verbose:
            print(f"\nResults saved to {results_path}")
        
        return results_df
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across multiple seeds."""
        agg = {
            'dataset': results[0]['dataset'],
            'epsilon': results[0]['epsilon'],
            'method': results[0]['method'],
            'n_seeds': len(results)
        }
        
        # Metrics to aggregate
        metrics = [
            'ml_efficiency', 'query_error', 'fidelity_error',
            'distance_preservation', 'attribute_error', 'synthesis_time'
        ]
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                agg[f'{metric}_mean'] = np.mean(values)
                agg[f'{metric}_std'] = np.std(values)
        
        return agg
    
    def print_summary_table(self, results_df: pd.DataFrame):
        """Print a formatted summary table."""
        print("\n" + "="*80)
        print("UPDP EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        metrics = ['ml_efficiency_mean', 'query_error_mean', 'fidelity_error_mean']
        
        for dataset in results_df['dataset'].unique():
            print(f"\n{dataset}")
            print("-" * 60)
            
            df_sub = results_df[results_df['dataset'] == dataset]
            
            print(f"{'ε':<8} {'ML Eff (↑)':<15} {'Query Err (↓)':<15} {'Fidelity (↓)':<15}")
            print("-" * 60)
            
            for _, row in df_sub.iterrows():
                print(f"{row['epsilon']:<8.1f} "
                      f"{row.get('ml_efficiency_mean', 0):<15.4f} "
                      f"{row.get('query_error_mean', 0):<15.4f} "
                      f"{row.get('fidelity_error_mean', 0):<15.4f}")


def create_comparison_table(
    updp_results: pd.DataFrame,
    baseline_results: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Create a comparison table between UPDP and baseline methods.
    
    Parameters
    ----------
    updp_results : DataFrame with UPDP results
    baseline_results : dict with baseline results (from tab_bench)
    
    Returns
    -------
    comparison_df : DataFrame with side-by-side comparison
    """
    
    # These are approximate values for ε=1.0
    if baseline_results is None:
        baseline_results = {
            # Format: {dataset: {method: {metric: value}}}
            'bank': {
                'AIM': {'ml_efficiency': 0.71, 'query_error': 0.002, 'fidelity_error': 0.09},
                'PrivMRF': {'ml_efficiency': 0.69, 'query_error': 0.003, 'fidelity_error': 0.07},
                'PrivSyn': {'ml_efficiency': 0.47, 'query_error': 0.004, 'fidelity_error': 0.10},
                'RAP++': {'ml_efficiency': 0.69, 'query_error': 0.006, 'fidelity_error': 0.36},
                'GEM': {'ml_efficiency': 0.56, 'query_error': 0.021, 'fidelity_error': 0.21},
                'DP-MERF': {'ml_efficiency': 0.57, 'query_error': 0.035, 'fidelity_error': 0.48},
                'TabDDPM': {'ml_efficiency': 0.47, 'query_error': 0.071, 'fidelity_error': 0.78},
            }
        }
    
    comparison_rows = []
    
    for _, row in updp_results.iterrows():
        dataset = row['dataset']
        epsilon = row['epsilon']
        
        comp_row = {
            'Dataset': dataset,
            'ε': epsilon,
            'UPDP_ML': row.get('ml_efficiency_mean', 0),
            'UPDP_QE': row.get('query_error_mean', 0),
            'UPDP_FE': row.get('fidelity_error_mean', 0),
        }
        
        # Add baseline comparisons if available
        if dataset in baseline_results and abs(epsilon - 1.0) < 0.1:
            for method, metrics in baseline_results[dataset].items():
                comp_row[f'{method}_ML'] = metrics.get('ml_efficiency', 0)
                comp_row[f'{method}_QE'] = metrics.get('query_error', 0)
                comp_row[f'{method}_FE'] = metrics.get('fidelity_error', 0)
        
        comparison_rows.append(comp_row)
    
    return pd.DataFrame(comparison_rows)


def main():
    parser = argparse.ArgumentParser(
        description='Run UPDP experiments and compare with baselines'
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to tab_bench data directory')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Datasets to evaluate (default: all)')
    parser.add_argument('--epsilons', nargs='+', type=float, default=None,
                       help='Privacy budgets (default: 0.2, 1.0, 5.0)')
    parser.add_argument('--n_seeds', type=int, default=5,
                       help='Number of random seeds')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_seeds=args.n_seeds,
        verbose=not args.quiet
    )
    
    results = runner.run_all_experiments(
        datasets=args.datasets,
        epsilons=args.epsilons
    )
    
    runner.print_summary_table(results)
    
    # Create comparison table
    comparison = create_comparison_table(results)
    comparison_path = os.path.join(args.output_dir, 'comparison_table.csv')
    comparison.to_csv(comparison_path, index=False)
    
    print(f"\nComparison table saved to {comparison_path}")
    
    return results


if __name__ == "__main__":
    main()