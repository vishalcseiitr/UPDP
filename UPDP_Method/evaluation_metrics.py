"""
UPDP Evaluation Metrics

This module implements evaluation metrics for comparing differentially private
tabular data synthesis methods, following the tab_bench evaluation framework.

Metrics implemented:
1. ML Efficiency (F1 score on downstream ML tasks)
2. Query Error (3-way marginal query error)
3. Fidelity Error (TVD on 2-way marginals)
4. Distance Preservation (pairwise distance correlation)
5. Attribute Distribution Error (per-column distribution similarity)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.stats import wasserstein_distance
from itertools import combinations
import warnings


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for DP tabular data synthesis.
    
    Computes multiple metrics to compare synthetic data quality against
    real data, following the evaluation framework from tab_bench.
    """
    
    def __init__(
        self,
        n_bins: int = 20,
        random_state: Optional[int] = None
    ):
        """
        Parameters
        ----------
        n_bins : int
            Number of bins for discretizing numerical features
        random_state : int, optional
            Random seed for reproducibility
        """
        self.n_bins = n_bins
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
    
    # =========================================================================
    # 1. ML Efficiency (Higher is Better)
    # =========================================================================
    
    def compute_ml_efficiency(
        self,
        X_num_syn: Optional[np.ndarray],
        X_cat_syn: Optional[np.ndarray],
        y_syn: np.ndarray,
        X_num_test: Optional[np.ndarray],
        X_cat_test: Optional[np.ndarray],
        y_test: np.ndarray,
        models: List[str] = ['rf', 'mlp', 'lr']
    ) -> Dict[str, float]:
        """
        
        Parameters
        ----------
        X_num_syn, X_cat_syn, y_syn : synthetic training data
        X_num_test, X_cat_test, y_test : real test data
        models : list of model names to evaluate
        
        Returns
        -------
        results : dict with F1 scores for each model and average
        """
        # Prepare features
        X_syn = self._prepare_features(X_num_syn, X_cat_syn)
        X_test = self._prepare_features(X_num_test, X_cat_test)
        
        # Encode labels
        le = LabelEncoder()
        y_syn_enc = le.fit_transform(y_syn.astype(str))
        y_test_enc = le.transform(y_test.astype(str))
        
        # Scale features
        scaler = StandardScaler()
        X_syn_scaled = scaler.fit_transform(X_syn)
        X_test_scaled = scaler.transform(X_test)
        
        # Handle NaN/Inf
        X_syn_scaled = np.nan_to_num(X_syn_scaled, nan=0, posinf=0, neginf=0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
        
        results = {}
        model_scores = []
        
        for model_name in models:
            try:
                model = self._get_model(model_name)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_syn_scaled, y_syn_enc)
                    y_pred = model.predict(X_test_scaled)
                
                # Compute F1 score
                n_classes = len(np.unique(y_test_enc))
                avg = 'binary' if n_classes == 2 else 'macro'
                f1 = f1_score(y_test_enc, y_pred, average=avg, zero_division=0)
                
                results[f'f1_{model_name}'] = f1
                model_scores.append(f1)
                
            except Exception as e:
                results[f'f1_{model_name}'] = 0.0
                model_scores.append(0.0)
        
        # Average F1
        results['f1_avg'] = np.mean(model_scores)
        results['ml_efficiency'] = results['f1_avg']
        
        return results
    
    def _get_model(self, model_name: str):
        """Get ML model by name."""
        if model_name == 'rf':
            return RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=self.random_state, n_jobs=-1
            )
        elif model_name == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500,
                random_state=self.random_state, early_stopping=True
            )
        elif model_name == 'lr':
            return LogisticRegression(
                max_iter=1000, random_state=self.random_state
            )
        elif model_name == 'gb':
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=5,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _prepare_features(
        self,
        X_num: Optional[np.ndarray],
        X_cat: Optional[np.ndarray]
    ) -> np.ndarray:
        """Concatenate numerical and categorical features."""
        parts = []
        if X_num is not None and X_num.size > 0:
            parts.append(X_num.astype(float))
        if X_cat is not None and X_cat.size > 0:
            parts.append(X_cat.astype(float))
        
        if not parts:
            raise ValueError("No features provided")
        
        return np.hstack(parts)
    
    # =========================================================================
    # 2. Query Error (Lower is Better)
    # =========================================================================
    
    def compute_query_error(
        self,
        X_num_syn: Optional[np.ndarray],
        X_cat_syn: Optional[np.ndarray],
        y_syn: np.ndarray,
        X_num_real: Optional[np.ndarray],
        X_cat_real: Optional[np.ndarray],
        y_real: np.ndarray,
        n_queries: int = 100,
        marginal_order: int = 3
    ) -> Dict[str, float]:
        """
        Compute query error on k-way marginal queries.
        
        Parameters
        ----------
        X_num_syn, X_cat_syn, y_syn : synthetic data
        X_num_real, X_cat_real, y_real : real data
        n_queries : number of random queries to sample
        marginal_order : order of marginals (default: 3 for 3-way)
        
        Returns
        -------
        results : dict with mean and std query error
        """
        # Discretize and combine all features
        data_syn = self._discretize_data(X_num_syn, X_cat_syn, y_syn)
        data_real = self._discretize_data(X_num_real, X_cat_real, y_real)
        
        n_cols = data_syn.shape[1]
        
        if n_cols < marginal_order:
            marginal_order = n_cols
        
        # Sample random k-way marginal queries
        all_combinations = list(combinations(range(n_cols), marginal_order))
        if len(all_combinations) > n_queries:
            query_indices = self._rng.choice(
                len(all_combinations), n_queries, replace=False
            )
            queries = [all_combinations[i] for i in query_indices]
        else:
            queries = all_combinations
        
        errors = []
        for cols in queries:
            # Compute marginal distributions
            marg_syn = self._compute_marginal(data_syn, cols)
            marg_real = self._compute_marginal(data_real, cols)
            
            # L1 error between marginals
            error = self._marginal_l1_error(marg_syn, marg_real)
            errors.append(error)
        
        return {
            'query_error': np.mean(errors),
            'query_error_std': np.std(errors),
            'n_queries': len(queries)
        }
    
    def _discretize_data(
        self,
        X_num: Optional[np.ndarray],
        X_cat: Optional[np.ndarray],
        y: np.ndarray
    ) -> np.ndarray:
        """Discretize numerical features and combine all columns."""
        parts = []
        
        if X_num is not None and X_num.size > 0:
            # Discretize numerical features
            X_num_disc = np.zeros_like(X_num, dtype=int)
            for j in range(X_num.shape[1]):
                col = X_num[:, j]
                # Handle NaN
                col = np.nan_to_num(col, nan=np.nanmedian(col) if len(col) > 0 else 0)
                # Discretize into bins
                percentiles = np.percentile(col, np.linspace(0, 100, self.n_bins + 1))
                X_num_disc[:, j] = np.digitize(col, percentiles[1:-1])
            parts.append(X_num_disc)
        
        if X_cat is not None and X_cat.size > 0:
            parts.append(X_cat.astype(int))
        
        # Add labels
        parts.append(y.reshape(-1, 1).astype(int))
        
        return np.hstack(parts)
    
    def _compute_marginal(self, data: np.ndarray, cols: tuple) -> Dict:
        """Compute marginal distribution for given columns."""
        subset = data[:, cols]
        
        # Count occurrences of each unique tuple
        tuples = [tuple(row) for row in subset]
        counts = {}
        for t in tuples:
            counts[t] = counts.get(t, 0) + 1
        
        # Normalize to probabilities
        total = len(tuples)
        return {k: v / total for k, v in counts.items()}
    
    def _marginal_l1_error(self, marg1: Dict, marg2: Dict) -> float:
        """Compute L1 error between two marginal distributions."""
        all_keys = set(marg1.keys()) | set(marg2.keys())
        error = 0.0
        for k in all_keys:
            p1 = marg1.get(k, 0.0)
            p2 = marg2.get(k, 0.0)
            error += abs(p1 - p2)
        return error / 2  # Normalize to [0, 1]
    
    # =========================================================================
    # 3. Fidelity Error / TVD (Lower is Better)
    # =========================================================================
    
    def compute_fidelity_error(
        self,
        X_num_syn: Optional[np.ndarray],
        X_cat_syn: Optional[np.ndarray],
        y_syn: np.ndarray,
        X_num_real: Optional[np.ndarray],
        X_cat_real: Optional[np.ndarray],
        y_real: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute fidelity error using Total Variation Distance on 2-way marginals.
        
        Parameters
        ----------
        Synthetic and real data arrays
        
        Returns
        -------
        results : dict with TVD statistics
        """
        # Discretize data
        data_syn = self._discretize_data(X_num_syn, X_cat_syn, y_syn)
        data_real = self._discretize_data(X_num_real, X_cat_real, y_real)
        
        n_cols = data_syn.shape[1]
        
        # Compute TVD for all 2-way marginals
        tvd_values = []
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                marg_syn = self._compute_marginal(data_syn, (i, j))
                marg_real = self._compute_marginal(data_real, (i, j))
                tvd = self._marginal_l1_error(marg_syn, marg_real)
                tvd_values.append(tvd)
        
        return {
            'fidelity_error': np.mean(tvd_values),
            'fidelity_error_std': np.std(tvd_values),
            'tvd_max': np.max(tvd_values) if tvd_values else 0,
            'tvd_min': np.min(tvd_values) if tvd_values else 0,
            'n_pairs': len(tvd_values)
        }
    
    # =========================================================================
    # 4. Distance Preservation (Higher is Better)
    # =========================================================================
    
    def compute_distance_preservation(
        self,
        X_num_syn: Optional[np.ndarray],
        X_cat_syn: Optional[np.ndarray],
        X_num_real: Optional[np.ndarray],
        X_cat_real: Optional[np.ndarray],
        n_samples: int = 500
    ) -> Dict[str, float]:
        """
        Compute how well pairwise distances are preserved.
        
        Uses correlation between pairwise distances in synthetic and real data.
        
        Parameters
        ----------
        Synthetic and real feature arrays
        n_samples : number of samples for distance computation
        
        Returns
        -------
        results : dict with distance preservation metrics
        """
        X_syn = self._prepare_features(X_num_syn, X_cat_syn)
        X_real = self._prepare_features(X_num_real, X_cat_real)
        
        # Subsample if needed
        n_syn = min(n_samples, X_syn.shape[0])
        n_real = min(n_samples, X_real.shape[0])
        
        idx_syn = self._rng.choice(X_syn.shape[0], n_syn, replace=False)
        idx_real = self._rng.choice(X_real.shape[0], n_real, replace=False)
        
        X_syn_sub = X_syn[idx_syn]
        X_real_sub = X_real[idx_real]
        
        # Normalize
        scaler = StandardScaler()
        X_syn_norm = scaler.fit_transform(X_syn_sub)
        X_real_norm = scaler.fit_transform(X_real_sub)
        
        # Compute pairwise distances
        def pairwise_distances(X):
            n = X.shape[0]
            dists = []
            for i in range(min(n, 100)):
                for j in range(i + 1, min(n, 100)):
                    d = np.linalg.norm(X[i] - X[j])
                    dists.append(d)
            return np.array(dists)
        
        dists_syn = pairwise_distances(X_syn_norm)
        dists_real = pairwise_distances(X_real_norm)
        
        # Compare distributions of distances
        if len(dists_syn) > 0 and len(dists_real) > 0:
            # Wasserstein distance between distance distributions
            wd = wasserstein_distance(dists_syn, dists_real)
            
            # Correlation of distance statistics
            stats_syn = [np.mean(dists_syn), np.std(dists_syn), np.median(dists_syn)]
            stats_real = [np.mean(dists_real), np.std(dists_real), np.median(dists_real)]
            
            # Normalized similarity
            similarity = 1.0 / (1.0 + wd)
        else:
            similarity = 0.0
            wd = float('inf')
        
        return {
            'distance_preservation': similarity,
            'distance_wasserstein': wd,
            'mean_dist_syn': np.mean(dists_syn) if len(dists_syn) > 0 else 0,
            'mean_dist_real': np.mean(dists_real) if len(dists_real) > 0 else 0
        }
    
    # =========================================================================
    # 5. Attribute Distribution Error (Lower is Better)
    # =========================================================================
    
    def compute_attribute_error(
        self,
        X_num_syn: Optional[np.ndarray],
        X_cat_syn: Optional[np.ndarray],
        y_syn: np.ndarray,
        X_num_real: Optional[np.ndarray],
        X_cat_real: Optional[np.ndarray],
        y_real: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute per-attribute distribution error.
        
        Uses Wasserstein distance for numerical and TVD for categorical.
        
        Returns
        -------
        results : dict with attribute-level error statistics
        """
        errors_num = []
        errors_cat = []
        
        # Numerical attributes
        if X_num_syn is not None and X_num_real is not None:
            for j in range(X_num_syn.shape[1]):
                col_syn = X_num_syn[:, j]
                col_real = X_num_real[:, j]
                
                # Remove NaN
                col_syn = col_syn[~np.isnan(col_syn)]
                col_real = col_real[~np.isnan(col_real)]
                
                if len(col_syn) > 0 and len(col_real) > 0:
                    wd = wasserstein_distance(col_syn, col_real)
                    # Normalize by range
                    range_real = np.ptp(col_real) if np.ptp(col_real) > 0 else 1
                    errors_num.append(wd / range_real)
        
        # Categorical attributes
        if X_cat_syn is not None and X_cat_real is not None:
            for j in range(X_cat_syn.shape[1]):
                col_syn = X_cat_syn[:, j]
                col_real = X_cat_real[:, j]
                
                # Compute category distributions
                cats = np.unique(np.concatenate([col_syn, col_real]))
                dist_syn = {c: np.sum(col_syn == c) / len(col_syn) for c in cats}
                dist_real = {c: np.sum(col_real == c) / len(col_real) for c in cats}
                
                tvd = sum(abs(dist_syn.get(c, 0) - dist_real.get(c, 0)) for c in cats) / 2
                errors_cat.append(tvd)
        
        # Label distribution
        cats = np.unique(np.concatenate([y_syn, y_real]))
        dist_syn = {c: np.sum(y_syn == c) / len(y_syn) for c in cats}
        dist_real = {c: np.sum(y_real == c) / len(y_real) for c in cats}
        label_tvd = sum(abs(dist_syn.get(c, 0) - dist_real.get(c, 0)) for c in cats) / 2
        
        all_errors = errors_num + errors_cat + [label_tvd]
        
        return {
            'attribute_error': np.mean(all_errors) if all_errors else 0,
            'attribute_error_num': np.mean(errors_num) if errors_num else 0,
            'attribute_error_cat': np.mean(errors_cat) if errors_cat else 0,
            'label_error': label_tvd
        }
    
    # =========================================================================
    # Combined Evaluation
    # =========================================================================
    
    def evaluate_all(
        self,
        X_num_syn: Optional[np.ndarray],
        X_cat_syn: Optional[np.ndarray],
        y_syn: np.ndarray,
        X_num_test: Optional[np.ndarray],
        X_cat_test: Optional[np.ndarray],
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Parameters
        ----------
        Synthetic and test data arrays
        verbose : whether to print results
        
        Returns
        -------
        results : dict with all metrics
        """
        results = {}
        
        # 1. ML Efficiency
        if verbose:
            print("Computing ML Efficiency...")
        ml_results = self.compute_ml_efficiency(
            X_num_syn, X_cat_syn, y_syn,
            X_num_test, X_cat_test, y_test
        )
        results.update(ml_results)
        
        # 2. Query Error
        if verbose:
            print("Computing Query Error...")
        query_results = self.compute_query_error(
            X_num_syn, X_cat_syn, y_syn,
            X_num_test, X_cat_test, y_test
        )
        results.update(query_results)
        
        # 3. Fidelity Error
        if verbose:
            print("Computing Fidelity Error...")
        fidelity_results = self.compute_fidelity_error(
            X_num_syn, X_cat_syn, y_syn,
            X_num_test, X_cat_test, y_test
        )
        results.update(fidelity_results)
        
        # 4. Distance Preservation
        if verbose:
            print("Computing Distance Preservation...")
        distance_results = self.compute_distance_preservation(
            X_num_syn, X_cat_syn,
            X_num_test, X_cat_test
        )
        results.update(distance_results)
        
        # 5. Attribute Error
        if verbose:
            print("Computing Attribute Error...")
        attr_results = self.compute_attribute_error(
            X_num_syn, X_cat_syn, y_syn,
            X_num_test, X_cat_test, y_test
        )
        results.update(attr_results)
        
        if verbose:
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(f"ML Efficiency (↑):     {results['ml_efficiency']:.4f}")
            print(f"Query Error (↓):       {results['query_error']:.4f}")
            print(f"Fidelity Error (↓):    {results['fidelity_error']:.4f}")
            print(f"Distance Pres. (↑):    {results['distance_preservation']:.4f}")
            print(f"Attribute Error (↓):   {results['attribute_error']:.4f}")
            print("="*50)
        
        return results


def evaluate_synthetic_data(
    X_num_syn: Optional[np.ndarray],
    X_cat_syn: Optional[np.ndarray],
    y_syn: np.ndarray,
    X_num_test: Optional[np.ndarray],
    X_cat_test: Optional[np.ndarray],
    y_test: np.ndarray,
    random_state: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for evaluating synthetic data.
    
    Parameters
    ----------
    Synthetic and test data arrays
    random_state : random seed
    verbose : whether to print results
    
    Returns
    -------
    results : dict with all metrics
    """
    evaluator = EvaluationMetrics(random_state=random_state)
    return evaluator.evaluate_all(
        X_num_syn, X_cat_syn, y_syn,
        X_num_test, X_cat_test, y_test,
        verbose=verbose
    )


if __name__ == "__main__":
    print("Testing Evaluation Metrics...")
    
    np.random.seed(42)
    
    # Create test data
    n_train = 1000
    n_test = 200
    
    X_num_real = np.random.randn(n_train, 5)
    X_cat_real = np.random.randint(0, 5, size=(n_train, 3))
    y_real = np.random.randint(0, 2, size=n_train)
    
    X_num_test = np.random.randn(n_test, 5)
    X_cat_test = np.random.randint(0, 5, size=(n_test, 3))
    y_test = np.random.randint(0, 2, size=n_test)
    
    # Synthetic = real + noise (for testing)
    X_num_syn = X_num_real + np.random.randn(*X_num_real.shape) * 0.5
    X_cat_syn = X_cat_real.copy()
    # Add some noise to categorical
    flip_mask = np.random.random(X_cat_syn.shape) < 0.1
    X_cat_syn[flip_mask] = np.random.randint(0, 5, size=np.sum(flip_mask))
    y_syn = y_real.copy()
    
    # Evaluate
    results = evaluate_synthetic_data(
        X_num_syn, X_cat_syn, y_syn,
        X_num_test, X_cat_test, y_test,
        random_state=42
    )
    
    print("\nEvaluation Metrics tests passed!")