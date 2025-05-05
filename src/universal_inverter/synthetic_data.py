import numpy as np

import jax.numpy as jnp
from jax import jit
import jax

#@jit

def compute_s_np(g, tau, b, tau2, d, change_point=5):
    # Ensure positive time constants with safe minimum
    tau_safe = np.maximum(tau, 1e-7)
    tau2_safe = np.maximum(tau2, 1e-7)
    
    # Safe exponential with input clipping
    def safe_exp(x):
        return np.exp(np.clip(x, -50, 50))  # Prevent overflow/underflow
    
    # Precompute common terms safely
    c = (1 - safe_exp(-change_point / tau_safe)) * b
    
# Compute intermediate terms with stabilization
    exp_term1 = safe_exp(-g / tau_safe)
    exp_term2 = safe_exp(-(g - change_point) / tau2_safe)
        
    m1 = (1 - exp_term1) * b
    m2 = c + (1 - exp_term2) * (d - c)
    
    # Use log-space for small values
    mask_neg = g < 0
    mask_mid = (g >= 0) & (g < change_point)
    
    return np.where(
        mask_neg,
        0.0,
        np.where(
            mask_mid,
            m1,
            np.where(
                d > c,  # Prevent reversed subtraction
                m2,
                c + (exp_term2 - 1) * (c - d)  # Alternative form when d < c
            )
        )
    )

def compute_s(g, tau, b, tau2, d, change_point=5):
    # Ensure positive time constants with safe minimum
    tau_safe = jnp.maximum(tau, 1e-7)
    tau2_safe = jnp.maximum(tau2, 1e-7)
    
    # Safe exponential with input clipping
    def safe_exp(x):
        return jnp.exp(jnp.clip(x, -50, 50))  # Prevent overflow/underflow
    
    # Precompute common terms safely
    c = (1 - safe_exp(-change_point / tau_safe)) * b
    
    # Compute intermediate terms with stabilization
    with jax.numpy_dtype_promotion('standard'):
        exp_term1 = safe_exp(-g / tau_safe)
        exp_term2 = safe_exp(-(g - change_point) / tau2_safe)
        
    m1 = (1 - exp_term1) * b
    m2 = c + (1 - exp_term2) * (d - c)
    
    # Use log-space for small values
    mask_neg = g < 0
    mask_mid = (g >= 0) & (g < change_point)
    
    return jnp.where(
        mask_neg,
        0.0,
        jnp.where(
            mask_mid,
            m1,
            jnp.where(
                d > c,  # Prevent reversed subtraction
                m2,
                c + (exp_term2 - 1) * (c - d)  # Alternative form when d < c
            )
        )
    )

class PiecewiseLinearSimulator:
    def __init__(self, signal_length=572, tau=1.0, tau2=2, b=.8, d=0.2,change_point=5,binomial_noise=100):
        self.signal_length = signal_length
        self.tau = tau
        self.tau2 = tau2
        self.b = b
        self.d = d
        self.change_point=change_point
        self.binomial_noise=binomial_noise

    def generate_piecewise_linear(self):
        """Generate piecewise linear signal with specified constraints"""
        t = 0
        segments = []
        current_value = 0.0

        while t < self.signal_length:
            # Random segment length between 20-50
            seg_length = np.random.randint(50,100)
            seg_length = min(seg_length, self.signal_length - t)

            # Choose slope direction and magnitude
            slope_sign = np.random.choice([1, -1])
            slope_mag = np.random.uniform(0.01, 0.05)
            slope = slope_sign * slope_mag

            # Generate segment
            segment = current_value + slope * np.arange(seg_length)
            segments.append(segment)

            current_value = segment[-1]
            t += seg_length

        # Combine segments and truncate to exact length
        signal = np.concatenate(segments)[:self.signal_length]

        # Adjust minimum to be between -10 and 20
        min_val = np.random.uniform(-10, 20)
        signal += (min_val - signal.min())

        return signal.astype(np.float32)

    def compose_function(self, g):
        """Apply the specified composition function"""


        # Clip to [0, 1] range
        return np.array(compute_s(g, self.tau, self.b, self.tau2, self.d, self.change_point)) #np.clip(s, 0, 1).astype(np.float32)

    def add_noise(self, s,):
        """Add binomial noise"""
        return np.random.binomial(self.binomial_noise, s).astype(np.float32) / self.binomial_noise
    def generate_sample(self,delta=0):
        while True:
            sample = self.generate_one_sample()
            if sample['s'].max() - sample['s'].min() >= delta:
                break
        return sample
            #print("try")


    def generate_one_sample(self):
        """Generate full sample"""
        g = self.generate_piecewise_linear()
        s = self.compose_function(g)
        noisy_s = self.add_noise(s)
        return {
            'g': g,
            's': s,
            'noisy_s': noisy_s
        }