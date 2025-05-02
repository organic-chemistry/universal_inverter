from flax import nnx
import jax
import jax.numpy as jnp

from universal_inverter.interpolate import interpolate_active

class PiecewiseLinear(nnx.Module):
   
    def __init__(self, coarse_bins: int, xmin: float = 0, xmax: float = 0 ,default_value: float = 0.0,activation=True):
        self.coarse_bins=coarse_bins
        self.xmin =xmin
        self.xmax =xmax
        self.default_value = default_value
        self.activation = activation
        
    def coarse_positions(self,inputs):
        batch_size = inputs.shape[0]
        
        # Compute bin centers in the normalized range
        bin_width = (self.xmax - self.xmin) / self.coarse_bins
        centers = jnp.linspace(self.xmin + bin_width / 2, self.xmax - bin_width / 2, self.coarse_bins)
        centers = jnp.broadcast_to(centers, (batch_size, self.coarse_bins))

        x_abs = centers + nnx.tanh(inputs[..., 0])*0.5*bin_width  # inputs[..., 0] represents offsets
        x_abs = jnp.clip(x_abs, self.xmin, self.xmax)

        return x_abs
    
    def is_activ(self,inputs):
        return nnx.sigmoid(inputs[..., 2]) > 0.5
    
    def __call__(self, inputs,x):

        batch_size, num_coarse, num_params = inputs.shape
        #print(self.activation)
        if self.activation:
            assert(num_params == 3)
        else:
            assert(num_params == 2)
        
        x_abs = self.coarse_positions(inputs)
        t_values = inputs[..., 1]
        
        
        if self.activation:
            active = self.is_activ(inputs)  # Convert to boolean mask

            # Double vmap: first over batch, then over x_eval points
            return jax.vmap(interpolate_active)(x, x_abs, t_values, active )
        else:
            #print(x.shape, x_abs.shape, t_values.shape)
            return jax.vmap(lambda x_inte,x_c,y_c: jnp.interp( x_inte,x_c,y_c, left="extrapolate", right="extrapolate"))(x, x_abs, t_values) 
        
        

from typing import Optional

class ReplicationActivation(PiecewiseLinear):
    """
    For each fine index i, computes:
       activation(i) = min_j [ t_j + |i - x_j_abs| / v_j ]

    Args:
        fine_resolution: number of fine positions (e.g. 1024)
        coarse_bins: number of coarse bins (e.g. 16)
        window_radius: optional integer restricting which bins can influence i Not implemented
    """

    def __init__(self, coarse_bins: int, xmin: float = 0, xmax: float = 0 , window_radius: Optional[int] = None):
        super().__init__(coarse_bins,xmin,xmax)
        self.window_radius = window_radius

    def __call__(self, inputs,velocity,x):
        """
        Args:
            inputs: shape (batch, coarse_bins, num_params)
                Where num_params can be:
                  2 => [time, position]


        Returns:
            activations: (batch, fine_resolution)
        """
        batch_size, num_coarse, num_params = inputs.shape

        assert(num_params == 2)

        x_abs = self.coarse_positions(inputs)
        t_values = inputs[..., 1]
        v_j = nnx.relu(velocity)   #Ensure positiveness  

        #    If we have only 2 params => we must use constant_velocity or a learned velocity

        v_j = v_j * jnp.ones_like(t_values)  # (batch, coarse_bins)
      
        # 4. For each fine position i, compute the candidate activation from each bin j:
        #       candidate(i, j) = t_j + |i - x_j_abs| / v_j
        
        i_grid = x[:, :, None]                      # (1, fine, 1)
        x_j_abs_expanded = x_abs[:, None, :]              # (batch, 1, coarse_bins)
        dist = jnp.abs(i_grid - x_j_abs_expanded)           # (batch, fine, coarse_bins)

        v_j_expanded = v_j[:, None, :] + 1e-8  # to avoid div by zero
        candidates = t_values[:, None, :] + dist / v_j_expanded

        # 5. Optional: restrict influence using a window in coarse‐bin space
        """
        if self.window_radius is not None:
            fine_per_coarse = self.fine_resolution / self.coarse_bins

            # Convert the window radius from “coarse bins” to “fine” units
            window_fine_radius = self.window_radius * fine_per_coarse
            valid_mask = dist <= window_fine_radius  # (batch, fine, coarse_bins)
            candidates = jnp.where(valid_mask, candidates, jnp.inf)  # set out‐of‐window to +inf
        """
        # 6. The activation is the min over all coarse bins
        activation = jnp.min(candidates, axis=-1)  # (batch, fine)

        return activation
    


import jax
import jax.numpy as jnp
import jax.scipy.special
from typing import Optional

class ReplicationActivationBase(PiecewiseLinear):
    """
    Base class containing shared functionality for all replication activation variants
    """
    def __init__(self, coarse_bins: int, lambda_param: float = 1.0, 
                 xmin: float = 0, xmax: float = 0, window_radius: Optional[int] = None):
        super().__init__(coarse_bins, xmin, xmax)
        self.window_radius = window_radius
        self.lambda_param = lambda_param + 1e-8  # Prevent division by zero

    def compute_mrt_from_arrival_time(self, candidates):
        sorted_candidates = jnp.sort(candidates, axis=-1)
        K = sorted_candidates.shape[-1]

        t1 = sorted_candidates[..., 0]
        sorted_candidates = sorted_candidates-t1[...,None]

        if K == 1:
            return t1

        i_indices = jnp.arange(K-1)
        i_plus_1 = (i_indices + 1).astype(jnp.float32)
        i_plus_2 = i_plus_1 + 1

        t_i = sorted_candidates[..., i_indices]
        t_next = sorted_candidates[..., i_indices + 1]
        cum_sums = jnp.cumsum(sorted_candidates, axis=-1)
        S_i = cum_sums[..., i_indices]

        # Numerical stability improvements
        lambda_safe = jnp.maximum(self.lambda_param, 1e-8)
        
        # 1. Prevent large exponents using safe scaling
        A = lambda_safe * (S_i - i_plus_1 * t_i)
        B = lambda_safe * (i_plus_1 * t_i - i_plus_2 * t_next)
        
        # 2. Log-space computation with stability guards
        log_terms = (
            -jnp.log(i_plus_1 + 1e-12)  # Prevent log(0)
            + A
            + jnp.log(-jnp.expm1(B)+ 1e-12)  # Safer than log(-expm1(B))
        )

        # 3. Log-sum-exp trick for numerical stability
        max_log = jnp.max(log_terms, axis=-1, keepdims=True)
        log_terms_shifted = log_terms - max_log
        terms = jnp.exp(log_terms_shifted)
        sum_terms = jnp.sum(terms, axis=-1) * jnp.exp(max_log.squeeze(-1))

        # 4. Final computation with output stabilization
        return t1 + (1.0 / lambda_safe) * jnp.nan_to_num(sum_terms, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)


class ConstantSpeedReplication(ReplicationActivationBase):
    """
    Handles 2-parameter inputs with a constant speed parameter
    """
    def __init__(self, coarse_bins: int, speed: float, lambda_param: float = 1.0, 
                 xmin: float = 0, xmax: float = 0, window_radius: Optional[int] = None):
        super().__init__(coarse_bins, lambda_param, xmin, xmax, window_radius)
        self.speed = speed + 1e-8  # Ensure positive speed

    def __call__(self, inputs, x):
        batch_size, num_coarse, num_params = inputs.shape
        assert num_params == 2, "Inputs must have 2 parameters (position, time)"

        x_abs = self.coarse_positions(inputs)
        t_values = inputs[..., 1]
        i_grid = x[..., None]
        x_j_abs = x_abs[:, None, :]

        dist = jnp.abs(i_grid - x_j_abs)
        candidates = t_values[:, None, :] + dist / self.speed
        return self.compute_mrt_from_arrival_time(candidates)

class AverageSpeedReplication(ReplicationActivationBase):
    """
    Handles 3-parameter inputs with speed averaging
    """
    def __call__(self, inputs, x):
        batch_size, num_coarse, num_params = inputs.shape
        assert num_params == 3, "Inputs must have 3 parameters (position, time, speed)"

        x_abs = self.coarse_positions(inputs)
        t_values = inputs[..., 1]
        speed_per_coarse = jax.nn.relu(inputs[..., 2]) + 1e-8

        # Compute average speed across coarse bins
        average_speed = jnp.mean(speed_per_coarse, axis=1, keepdims=True)
        #print(average_speed)
        i_grid = x[..., None]
        x_j_abs = x_abs[:, None, :]
        dist = jnp.abs(i_grid - x_j_abs)
        candidates = t_values[:, None, :] + dist / average_speed[:, None, :]
        #candidates  = jnp.clip(candidates,-1e4,1e4)
        return self.compute_mrt_from_arrival_time(candidates)

class DirectionalSpeedReplication(ReplicationActivationBase):
    """
    Original implementation handling 3 or 4 parameters with directional speeds
    """
    def __call__(self, inputs, x):
        batch_size, num_coarse, num_params = inputs.shape
        assert num_params in (3, 4), "Inputs must have 3 or 4 parameters"

        x_abs = self.coarse_positions(inputs)
        t_values = inputs[..., 1]
        i_grid = x[..., None]
        x_j_abs = x_abs[:, None, :]

        if num_params == 3:
            v = jax.nn.relu(inputs[..., 2]) + 1e-8
            dist = jnp.abs(i_grid - x_j_abs)
            candidates = t_values[:, None, :] + dist / v[:, None, :]
        else:
            v_left = jax.nn.relu(inputs[..., 2]) + 1e-8
            v_right = jax.nn.relu(inputs[..., 3]) + 1e-8
            delta = i_grid - x_j_abs
            candidates = t_values[:, None, :] + jnp.where(
                delta >= 0,
                delta / v_right[:, None, :],
                -delta / v_left[:, None, :]
            )

        return self.compute_mrt_from_arrival_time(candidates)