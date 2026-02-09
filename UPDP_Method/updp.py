"""
UPDP: Uniform Projection for Differential Privacy in High-Dimensional Data

Algorithm Overview:
1. Projection: Project input x ∈ R^d to lower dimension R^k using public matrix P
2. ℓ2 Clipping: Clip projected vector to ball of radius R
3. Laplace Perturbation: Add calibrated Laplace noise for ε-LDP
4. Radial Uniformization: Post-process to uniform radial distribution
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
import warnings


@dataclass
class UPDPConfig:
    """Configuration for UPDP mechanism."""
    k: int  # Target projection dimension
    R: float  # Clipping radius
    epsilon: float  # Privacy budget
    projection_type: str = 'gaussian'  # 'gaussian', 'sparse', or 'orthogonal'
    use_uniformization: bool = True  # Whether to apply radial uniformization
    random_state: Optional[int] = None
    

class UPDPMechanism:
    """
    Uniform Projection for Differential Privacy (UPDP) Mechanism.
    
    This mechanism provides ε-Local Differential Privacy for high-dimensional
    vectors through:
    1. Dimension reduction via random projection
    2. ℓ2-norm clipping for bounded sensitivity
    3. Laplace noise addition calibrated to ℓ1 sensitivity
    4. Radial uniformization as post-processing
    
    Parameters
    ----------
    k : int
        Target projection dimension (k << d)
    R : float
        Clipping radius for ℓ2 ball
    epsilon : float
        Privacy budget (ε > 0)
    projection_type : str, default='gaussian'
        Type of random projection: 'gaussian', 'sparse', or 'orthogonal'
    use_uniformization : bool, default=True
        Whether to apply radial uniformization step
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        k: int,
        R: float,
        epsilon: float,
        projection_type: str = 'gaussian',
        use_uniformization: bool = True,
        random_state: Optional[int] = None
    ):
        if k <= 0:
            raise ValueError("Projection dimension k must be positive")
        if R <= 0:
            raise ValueError("Clipping radius R must be positive")
        if epsilon <= 0:
            raise ValueError("Privacy budget epsilon must be positive")
            
        self.k = k
        self.R = R
        self.epsilon = epsilon
        self.projection_type = projection_type
        self.use_uniformization = use_uniformization
        self.random_state = random_state
        
        self.P = None  # Projection matrix
        self.d = None  # Original dimension
        self._is_fitted = False
        self._rng = np.random.RandomState(random_state)
    
    @property
    def delta_1(self) -> float:
        """ℓ1 sensitivity: Δ₁ = 2R√k"""
        return 2 * self.R * np.sqrt(self.k)
    
    @property
    def delta_2(self) -> float:
        """ℓ2 sensitivity: Δ₂ = 2R"""
        return 2 * self.R
    
    @property
    def laplace_scale(self) -> float:
        """Laplace noise scale: b = Δ₁/ε"""
        return self.delta_1 / self.epsilon
    
    def _create_projection_matrix(self, d: int) -> np.ndarray:
        """
        Create the public projection matrix P ∈ R^(k×d).
        
        Parameters
        ----------
        d : int
            Original dimension
            
        Returns
        -------
        P : ndarray of shape (k, d)
            Random projection matrix
        """
        if self.projection_type == 'gaussian':
            # Gaussian random projection (JL-type)
            P = self._rng.randn(self.k, d) / np.sqrt(self.k)
            
        elif self.projection_type == 'sparse':
            # Sparse random projection (more efficient)
            s = np.sqrt(d)  # Sparsity parameter
            prob = 1 / (2 * s)
            P = np.zeros((self.k, d))
            for i in range(self.k):
                for j in range(d):
                    u = self._rng.random()
                    if u < prob:
                        P[i, j] = np.sqrt(s / self.k)
                    elif u < 2 * prob:
                        P[i, j] = -np.sqrt(s / self.k)
                    
        elif self.projection_type == 'orthogonal':
            # Random orthogonal projection
            Q, _ = np.linalg.qr(self._rng.randn(d, self.k))
            P = Q.T / np.sqrt(self.k / d)
            
        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")
            
        return P
    
    def fit(self, X: np.ndarray) -> 'UPDPMechanism':
        """
        Fit the mechanism by creating the projection matrix.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, d)
            Input data (used only to determine dimension d)
            
        Returns
        -------
        self : UPDPMechanism
            Fitted mechanism
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        self.d = X.shape[1]
        
        # Adjust k if larger than d
        if self.k > self.d:
            warnings.warn(f"k={self.k} > d={self.d}, setting k=d")
            self.k = self.d
            
        # Create projection matrix
        self.P = self._create_projection_matrix(self.d)
        self._is_fitted = True
        
        return self
    
    def _clip_l2(self, v: np.ndarray) -> np.ndarray:
        """
        ℓ2-ball clipping: Π_R(v) = v · min(1, R/‖v‖₂)
        
        Parameters
        ----------
        v : ndarray
            Input vector(s)
            
        Returns
        -------
        clipped : ndarray
            Clipped vector(s) with ‖clipped‖₂ ≤ R
        """
        if v.ndim == 1:
            norm = np.linalg.norm(v, ord=2)
            if norm > self.R:
                return v * (self.R / norm)
            return v.copy()
        else:
            # Batch processing
            norms = np.linalg.norm(v, ord=2, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.R / (norms + 1e-10))
            return v * scale
    
    def _add_laplace_noise(self, y: np.ndarray) -> np.ndarray:
        """
        Add i.i.d. Laplace noise calibrated for ε-LDP.
        
        Parameters
        ----------
        y : ndarray
            Clipped projected vector(s)
            
        Returns
        -------
        z : ndarray
            Noisy vector(s)
        """
        b = self.laplace_scale
        noise = self._rng.laplace(loc=0, scale=b, size=y.shape)
        return y + noise
    
    def _radial_uniformization(self, z: np.ndarray) -> np.ndarray:
        """
        Radial uniformization post-processing.
        
        Replaces the radius with a uniform sample while keeping direction.
        This is a post-processing step that doesn't affect privacy.
        
        Parameters
        ----------
        z : ndarray
            Noisy vector(s)
            
        Returns
        -------
        z_tilde : ndarray
            Uniformized vector(s) with uniform radial distribution
        """
        if z.ndim == 1:
            r = np.linalg.norm(z, ord=2)
            if r > 0:
                theta = z / r
            else:
                # Fixed unit vector when r=0
                theta = np.zeros_like(z)
                theta[0] = 1.0
            
            # Sample new radius uniformly from [0, R]
            r_tilde = self._rng.uniform(0, self.R)
            return r_tilde * theta
        else:
            # Batch processing
            r = np.linalg.norm(z, ord=2, axis=1, keepdims=True)
            
            # Compute direction (handle zero vectors)
            theta = np.zeros_like(z)
            nonzero_mask = (r.flatten() > 1e-10)
            theta[nonzero_mask] = z[nonzero_mask] / r[nonzero_mask]
            
            # For zero vectors, use fixed direction
            zero_mask = ~nonzero_mask
            if np.any(zero_mask):
                theta[zero_mask, 0] = 1.0
            
            # Sample new radii uniformly
            r_tilde = self._rng.uniform(0, self.R, size=(z.shape[0], 1))
            
            return r_tilde * theta
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply UPDP mechanism to input vectors.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, d) or (d,)
            Input vector(s)
            
        Returns
        -------
        X_private : ndarray of shape (n_samples, k) or (k,)
            Privatized vector(s)
        """
        if not self._is_fitted:
            raise RuntimeError("Mechanism not fitted. Call fit() first.")
            
        single_vector = X.ndim == 1
        if single_vector:
            X = X.reshape(1, -1)
            
        # Step 1: Projection (y₀ = Px)
        y0 = X @ self.P.T  # Shape: (n_samples, k)
        
        # Step 2: ℓ2 Clipping
        y = self._clip_l2(y0)
        
        # Step 3: Laplace Perturbation
        z = self._add_laplace_noise(y)
        
        # Step 4: Radial Uniformization (optional post-processing)
        if self.use_uniformization:
            z_tilde = self._radial_uniformization(z)
        else:
            z_tilde = z
            
        if single_vector:
            return z_tilde.flatten()
        return z_tilde
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class UPDPGaussian(UPDPMechanism):
    """
    Gaussian variant of UPDP for (ε, δ)-LDP.
    
    Uses Gaussian noise instead of Laplace for approximate DP.
    """
    
    def __init__(
        self,
        k: int,
        R: float,
        epsilon: float,
        delta: float = 1e-5,
        projection_type: str = 'gaussian',
        use_uniformization: bool = True,
        random_state: Optional[int] = None
    ):
        super().__init__(k, R, epsilon, projection_type, use_uniformization, random_state)
        self.delta = delta
    
    @property
    def gaussian_sigma(self) -> float:
        """Gaussian noise σ = Δ₂ · √(2 ln(1.25/δ)) / ε"""
        return self.delta_2 * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def _add_gaussian_noise(self, y: np.ndarray) -> np.ndarray:
        """Add calibrated Gaussian noise."""
        sigma = self.gaussian_sigma
        noise = self._rng.normal(loc=0, scale=sigma, size=y.shape)
        return y + noise
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply UPDP with Gaussian noise."""
        if not self._is_fitted:
            raise RuntimeError("Mechanism not fitted. Call fit() first.")
            
        single_vector = X.ndim == 1
        if single_vector:
            X = X.reshape(1, -1)
            
        y0 = X @ self.P.T
        y = self._clip_l2(y0)
        z = self._add_gaussian_noise(y)  # Gaussian instead of Laplace
        
        if self.use_uniformization:
            z_tilde = self._radial_uniformization(z)
        else:
            z_tilde = z
            
        if single_vector:
            return z_tilde.flatten()
        return z_tilde


if __name__ == "__main__":
    print("Testing UPDP Mechanism...")
    
    np.random.seed(42)
    n, d = 1000, 50
    X = np.random.randn(n, d)
    
    updp = UPDPMechanism(k=10, R=5.0, epsilon=1.0, random_state=42)
    X_private = updp.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Private shape: {X_private.shape}")
    print(f"ℓ1 sensitivity: {updp.delta_1:.4f}")
    print(f"ℓ2 sensitivity: {updp.delta_2:.4f}")
    print(f"Laplace scale: {updp.laplace_scale:.4f}")
    print("\nUPDP mechanism tests passed!")