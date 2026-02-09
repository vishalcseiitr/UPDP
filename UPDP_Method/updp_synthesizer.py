"""
UPDP Tabular Data Synthesizer

This module provides a synthesizer class that applies UPDP mechanism
to tabular data with mixed numerical and categorical features.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import os

from updp import UPDPMechanism, UPDPGaussian


class UPDPTabularSynthesizer:
    """
    UPDP-based Tabular Data Synthesizer.
    
    Applies UPDP mechanism to tabular data with mixed numerical and
    categorical features for differentially private data release.
    
    Parameters
    ----------
    epsilon : float
        Privacy budget (ε > 0)
    delta : float, default=1e-5
        Privacy parameter for approximate DP
    k_ratio : float, default=0.5
        Ratio of projection dimension to encoded dimension (k = k_ratio * d)
    R_percentile : float, default=95
        Percentile of projected norms for clipping radius estimation
    projection_type : str, default='gaussian'
        Type of random projection: 'gaussian', 'sparse', 'orthogonal'
    use_uniformization : bool, default=True
        Whether to apply radial uniformization
    noise_type : str, default='laplace'
        Noise type: 'laplace' for ε-LDP, 'gaussian' for (ε,δ)-LDP
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        k_ratio: float = 0.5,
        R_percentile: float = 95,
        projection_type: str = 'gaussian',
        use_uniformization: bool = True,
        noise_type: str = 'laplace',
        random_state: Optional[int] = None
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.k_ratio = k_ratio
        self.R_percentile = R_percentile
        self.projection_type = projection_type
        self.use_uniformization = use_uniformization
        self.noise_type = noise_type
        self.random_state = random_state
        
        # Internal state
        self.updp = None
        self.num_scaler = None
        self.n_num_features = 0
        self.n_cat_features = 0
        self.cat_cardinalities = []  # Number of categories per categorical feature
        self.cat_mappings = []  # Value-to-index mappings for categorical features
        self.unique_labels = None
        self.n_classes = 0
        self.total_encoded_dim = 0
        self._is_fitted = False
        
        self._rng = np.random.RandomState(random_state)
    
    def _encode_numerical(self, X_num: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale numerical features to [-1, 1]."""
        if X_num is None or X_num.shape[1] == 0:
            return np.array([]).reshape(X_num.shape[0] if X_num is not None else 0, 0)
        
        # Convert to float, handling string values
        X_num_clean = np.zeros_like(X_num, dtype=float)
        for j in range(X_num.shape[1]):
            col = X_num[:, j]
            for i in range(len(col)):
                try:
                    X_num_clean[i, j] = float(col[i])
                except (ValueError, TypeError):
                    X_num_clean[i, j] = np.nan
        
        # Replace NaN with column median
        for j in range(X_num_clean.shape[1]):
            col = X_num_clean[:, j]
            mask = np.isnan(col)
            if np.any(mask):
                median_val = np.nanmedian(col)
                if np.isnan(median_val):
                    median_val = 0.0
                X_num_clean[mask, j] = median_val
        
        if fit:
            self.num_scaler = MinMaxScaler(feature_range=(-1, 1))
            return self.num_scaler.fit_transform(X_num_clean)
        else:
            return self.num_scaler.transform(X_num_clean)
    
    def _decode_numerical(self, X_num_encoded: np.ndarray) -> np.ndarray:
        """Inverse scale numerical features."""
        if X_num_encoded.shape[1] == 0 or self.num_scaler is None:
            return X_num_encoded
        
        # Clip to valid range before inverse transform
        X_clipped = np.clip(X_num_encoded, -1, 1)
        return self.num_scaler.inverse_transform(X_clipped)
    
    def _encode_categorical(self, X_cat: np.ndarray, fit: bool = False) -> np.ndarray:
        """One-hot encode categorical features."""
        if X_cat is None or X_cat.shape[1] == 0:
            return np.array([]).reshape(X_cat.shape[0] if X_cat is not None else 0, 0)
        
        n_samples = X_cat.shape[0]
        
        if fit:
            self.cat_cardinalities = []
            self.cat_mappings = []  # Store value-to-index mappings
            for j in range(X_cat.shape[1]):
                unique_vals = np.unique(X_cat[:, j])
                self.cat_cardinalities.append(len(unique_vals))
                # Create mapping from value to index
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                self.cat_mappings.append(mapping)
        
        # Create one-hot encoding
        encoded_parts = []
        for j in range(X_cat.shape[1]):
            n_cats = self.cat_cardinalities[j]
            mapping = self.cat_mappings[j]
            one_hot = np.zeros((n_samples, n_cats))
            for i in range(n_samples):
                val = X_cat[i, j]
                # Handle both string and numeric values
                if val in mapping:
                    cat_idx = mapping[val]
                else:
                    # Try converting to handle numeric stored as different types
                    try:
                        cat_idx = int(float(str(val))) % n_cats
                    except:
                        cat_idx = 0  # Default to first category if unknown
                one_hot[i, cat_idx] = 1.0
            encoded_parts.append(one_hot)
        
        return np.hstack(encoded_parts) if encoded_parts else np.zeros((n_samples, 0))
    
    def _decode_categorical(self, X_cat_encoded: np.ndarray) -> np.ndarray:
        """Decode one-hot to categorical values."""
        if X_cat_encoded.shape[1] == 0 or len(self.cat_cardinalities) == 0:
            return np.array([]).reshape(X_cat_encoded.shape[0], 0)
        
        n_samples = X_cat_encoded.shape[0]
        decoded_parts = []
        
        idx = 0
        for j, n_cats in enumerate(self.cat_cardinalities):
            one_hot_slice = X_cat_encoded[:, idx:idx + n_cats]
            # Use argmax to find most likely category
            cat_indices = np.argmax(one_hot_slice, axis=1)
            
            # Reverse mapping from index to original value
            idx_to_val = {v: k for k, v in self.cat_mappings[j].items()}
            decoded_vals = np.array([idx_to_val.get(ci, list(idx_to_val.values())[0]) 
                                     for ci in cat_indices])
            decoded_parts.append(decoded_vals.reshape(-1, 1))
            idx += n_cats
        
        return np.hstack(decoded_parts)
    
    def _encode_labels(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """One-hot encode labels."""
        n_samples = len(y)
        
        if fit:
            self.unique_labels = np.unique(y)
            self.n_classes = len(self.unique_labels)
            self._label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        
        one_hot = np.zeros((n_samples, self.n_classes))
        for i, label in enumerate(y):
            # Handle both exact match and type-converted match
            if label in self._label_to_idx:
                one_hot[i, self._label_to_idx[label]] = 1.0
            else:
                # Try to find matching label with type conversion
                found = False
                for stored_label, idx in self._label_to_idx.items():
                    if str(label) == str(stored_label):
                        one_hot[i, idx] = 1.0
                        found = True
                        break
                if not found:
                    # Unknown label - assign to first class
                    one_hot[i, 0] = 1.0
        
        return one_hot
    
    def _decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode one-hot to label values."""
        label_indices = np.argmax(y_encoded, axis=1)
        return np.array([self.unique_labels[idx] for idx in label_indices])
    
    def _encode_all(
        self,
        X_num: Optional[np.ndarray],
        X_cat: Optional[np.ndarray],
        y: np.ndarray,
        fit: bool = False
    ) -> np.ndarray:
        """Encode all features and labels into a single matrix."""
        parts = []
        
        # Encode numerical
        if X_num is not None and X_num.shape[1] > 0:
            X_num_enc = self._encode_numerical(X_num, fit=fit)
            parts.append(X_num_enc)
            if fit:
                self.n_num_features = X_num.shape[1]
        elif fit:
            self.n_num_features = 0
        
        # Encode categorical
        if X_cat is not None and X_cat.shape[1] > 0:
            X_cat_enc = self._encode_categorical(X_cat, fit=fit)
            parts.append(X_cat_enc)
            if fit:
                self.n_cat_features = X_cat.shape[1]
        elif fit:
            self.n_cat_features = 0
        
        # Encode labels
        y_enc = self._encode_labels(y, fit=fit)
        parts.append(y_enc)
        
        X_full = np.hstack(parts) if parts else np.array([])
        
        if fit:
            self.total_encoded_dim = X_full.shape[1]
        
        return X_full
    
    def _decode_all(
        self,
        X_full: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Decode encoded matrix back to original format."""
        idx = 0
        
        # Decode numerical
        X_num = None
        if self.n_num_features > 0:
            X_num_enc = X_full[:, idx:idx + self.n_num_features]
            X_num = self._decode_numerical(X_num_enc)
            idx += self.n_num_features
        
        # Decode categorical
        X_cat = None
        if self.n_cat_features > 0:
            n_cat_encoded = sum(self.cat_cardinalities)
            X_cat_enc = X_full[:, idx:idx + n_cat_encoded]
            X_cat = self._decode_categorical(X_cat_enc)
            idx += n_cat_encoded
        
        # Decode labels
        y_enc = X_full[:, idx:idx + self.n_classes]
        y = self._decode_labels(y_enc)
        
        return X_num, X_cat, y
    
    def fit(
        self,
        X_num: Optional[np.ndarray],
        X_cat: Optional[np.ndarray],
        y: np.ndarray
    ) -> 'UPDPTabularSynthesizer':
        """
        Fit the synthesizer on training data.
        
        Parameters
        ----------
        X_num : ndarray or None
            Numerical features of shape (n_samples, n_num_features)
        X_cat : ndarray or None
            Categorical features of shape (n_samples, n_cat_features)
        y : ndarray
            Target labels of shape (n_samples,)
            
        Returns
        -------
        self : UPDPTabularSynthesizer
        """
        # Encode all data
        X_encoded = self._encode_all(X_num, X_cat, y, fit=True)
        
        # Compute projection dimension
        k = max(2, int(self.k_ratio * self.total_encoded_dim))
        
        # Estimate clipping radius R using projected norms
        temp_P = self._rng.randn(k, self.total_encoded_dim) / np.sqrt(k)
        projected = X_encoded @ temp_P.T
        norms = np.linalg.norm(projected, axis=1)
        R = np.percentile(norms, self.R_percentile)
        R = max(R, 0.1)  # Ensure R is not too small
        
        # Create UPDP mechanism
        if self.noise_type == 'gaussian':
            self.updp = UPDPGaussian(
                k=k,
                R=R,
                epsilon=self.epsilon,
                delta=self.delta,
                projection_type=self.projection_type,
                use_uniformization=self.use_uniformization,
                random_state=self.random_state
            )
        else:
            self.updp = UPDPMechanism(
                k=k,
                R=R,
                epsilon=self.epsilon,
                projection_type=self.projection_type,
                use_uniformization=self.use_uniformization,
                random_state=self.random_state
            )
        
        # Fit the UPDP mechanism
        self.updp.fit(X_encoded)
        
        # Store statistics for reconstruction
        self._X_mean = np.mean(X_encoded, axis=0)
        self._X_std = np.std(X_encoded, axis=0) + 1e-8
        
        self._is_fitted = True
        return self
    
    def _reconstruct_from_projection(self, X_projected: np.ndarray) -> np.ndarray:
        """
        Reconstruct full-dimensional data from projected representation.
        
        Uses pseudo-inverse of projection matrix.
        """
        P_pinv = np.linalg.pinv(self.updp.P)
        X_reconstructed = X_projected @ P_pinv.T
        return X_reconstructed
    
    def transform(
        self,
        X_num: Optional[np.ndarray],
        X_cat: Optional[np.ndarray],
        y: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Apply UPDP mechanism to privatize data.
        
        Parameters
        ----------
        X_num : ndarray or None
            Numerical features
        X_cat : ndarray or None
            Categorical features
        y : ndarray
            Target labels
            
        Returns
        -------
        X_num_private : ndarray or None
        X_cat_private : ndarray or None
        y_private : ndarray
        """
        if not self._is_fitted:
            raise RuntimeError("Synthesizer not fitted. Call fit() first.")
        
        # Encode data
        X_encoded = self._encode_all(X_num, X_cat, y, fit=False)
        
        # Apply UPDP mechanism
        X_projected_private = self.updp.transform(X_encoded)
        
        # Reconstruct to original encoded dimension
        X_reconstructed = self._reconstruct_from_projection(X_projected_private)
        
        # Decode back to original format
        return self._decode_all(X_reconstructed)
    
    def fit_transform(
        self,
        X_num: Optional[np.ndarray],
        X_cat: Optional[np.ndarray],
        y: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Fit and transform in one step."""
        self.fit(X_num, X_cat, y)
        return self.transform(X_num, X_cat, y)
    
    def synthesize(
        self,
        X_num: Optional[np.ndarray],
        X_cat: Optional[np.ndarray],
        y: np.ndarray,
        n_samples: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Generate synthetic privatized data.
        
        This is the main interface for generating DP synthetic data,
        compatible with the tab_bench framework.
        
        Parameters
        ----------
        X_num : ndarray or None
            Original numerical features
        X_cat : ndarray or None
            Original categorical features
        y : ndarray
            Original labels
        n_samples : int, optional
            Number of synthetic samples (default: same as input)
            
        Returns
        -------
        X_num_syn : ndarray or None
        X_cat_syn : ndarray or None
        y_syn : ndarray
        """
        if not self._is_fitted:
            self.fit(X_num, X_cat, y)
        
        if n_samples is None:
            n_samples = len(y)
        
        # Sample with replacement if needed
        n_original = len(y)
        if n_samples <= n_original:
            indices = self._rng.choice(n_original, n_samples, replace=False)
        else:
            indices = self._rng.choice(n_original, n_samples, replace=True)
        
        X_num_sample = X_num[indices] if X_num is not None else None
        X_cat_sample = X_cat[indices] if X_cat is not None else None
        y_sample = y[indices]
        
        return self.transform(X_num_sample, X_cat_sample, y_sample)
    
    def get_params(self) -> Dict[str, Any]:
        """Get synthesizer parameters."""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'k_ratio': self.k_ratio,
            'R_percentile': self.R_percentile,
            'projection_type': self.projection_type,
            'use_uniformization': self.use_uniformization,
            'noise_type': self.noise_type,
            'random_state': self.random_state
        }
    
    def get_privacy_params(self) -> Dict[str, float]:
        """Get privacy-related parameters after fitting."""
        if not self._is_fitted:
            return {'epsilon': self.epsilon, 'delta': self.delta}
        
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'k': self.updp.k,
            'R': self.updp.R,
            'delta_1': self.updp.delta_1,
            'delta_2': self.updp.delta_2,
            'laplace_scale': self.updp.laplace_scale if self.noise_type == 'laplace' else None,
            'gaussian_sigma': self.updp.gaussian_sigma if self.noise_type == 'gaussian' else None
        }


def updp_synthesize(
    X_num: Optional[np.ndarray],
    X_cat: Optional[np.ndarray],
    y: np.ndarray,
    epsilon: float,
    delta: float = 1e-5,
    n_samples: Optional[int] = None,
    **kwargs
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Convenience function for one-shot UPDP synthesis.
    
    Parameters
    ----------
    X_num : ndarray or None
        Numerical features
    X_cat : ndarray or None  
        Categorical features
    y : ndarray
        Labels
    epsilon : float
        Privacy budget
    delta : float
        Privacy parameter
    n_samples : int, optional
        Number of synthetic samples
    **kwargs
        Additional arguments passed to UPDPTabularSynthesizer
        
    Returns
    -------
    X_num_syn, X_cat_syn, y_syn : tuple
        Synthesized data
    """
    synthesizer = UPDPTabularSynthesizer(
        epsilon=epsilon,
        delta=delta,
        **kwargs
    )
    return synthesizer.fit(X_num, X_cat, y).synthesize(X_num, X_cat, y, n_samples)


if __name__ == "__main__":
    print("Testing UPDP Tabular Synthesizer...")
    
    np.random.seed(42)
    
    # Create synthetic test data
    n_samples = 1000
    X_num = np.random.randn(n_samples, 5)  # 5 numerical features
    X_cat = np.random.randint(0, 10, size=(n_samples, 3))  # 3 categorical features with 10 categories each
    y = np.random.randint(0, 2, size=n_samples)  # Binary classification
    
    # Test synthesizer
    synthesizer = UPDPTabularSynthesizer(
        epsilon=1.0,
        k_ratio=0.5,
        random_state=42
    )
    
    X_num_syn, X_cat_syn, y_syn = synthesizer.fit_transform(X_num, X_cat, y)
    
    print(f"Original numerical shape: {X_num.shape}")
    print(f"Synthetic numerical shape: {X_num_syn.shape}")
    print(f"Original categorical shape: {X_cat.shape}")
    print(f"Synthetic categorical shape: {X_cat_syn.shape}")
    print(f"Original label distribution: {np.bincount(y)}")
    print(f"Synthetic label distribution: {np.bincount(y_syn)}")
    
    print("\nPrivacy parameters:")
    for k, v in synthesizer.get_privacy_params().items():
        print(f"  {k}: {v}")
    
    print("\nUPDP Tabular Synthesizer tests passed!")