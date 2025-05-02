import jax.numpy as jnp
import jax

from jax import lax,jit
@jit
def find_active_neighbors(active):
    """
    Find indices of the previous and next active control points, supporting boundary extrapolation.
    
    Args:
        active (array): Boolean array indicating which control points are active.
        
    Returns:
        prev_active, next_active: Arrays containing the index of the nearest active point 
                                  to the left (previous) and to the right (next) for each point.
    """
    def forward_scan(carry, idx):
        # If current point is active, update the last active index; otherwise, keep the previous one.
        last_active = jnp.where(active[idx], idx, carry)
        return last_active, last_active
    
    def backward_scan(carry, idx):
        # In the backward pass, update the next active index.
        next_active = jnp.where(active[idx], idx, carry)
        return next_active, next_active

    # Forward scan: for each point, get the previous active index.
    _, prev_active = lax.scan(forward_scan, -1, jnp.arange(active.shape[0]))
    
    # Backward scan: for each point, get the next active index (scanning in reverse).
    _, next_active = lax.scan(
        backward_scan, -1, jnp.arange(active.shape[0] - 1, -1, -1)
    )
    next_active = jnp.flip(next_active)
    
    return prev_active, next_active
@jit
def handle_multiple_active_extrapolation(x_sorted, t_sorted, active_sorted):
    """
    Calculate boundary extrapolation parameters using a safe method that avoids Boolean indexing.
    
    The function finds the first two and last two active points in the sorted arrays, then computes
    slopes for left and right extrapolation.
    
    Args:
        x_sorted (array): Sorted x-coordinates.
        t_sorted (array): Sorted t-values.
        active_sorted (array): Sorted boolean mask of active control points.
    
    Returns:
        A tuple containing:
            - (x1, t1, slope_start): Parameters for left extrapolation (first active point, its t-value, and computed slope).
            - (xl1, tl1, slope_end): Parameters for right extrapolation (last active point, its t-value, and computed slope).
    """
    indices = jnp.arange(active_sorted.shape[0])
    
    # Use a large value to ignore inactive indices when taking min.
    large_value = jnp.iinfo(jnp.int32).max
    
    # Find first active index.
    first_idx = jnp.argmin(jnp.where(active_sorted, indices, large_value))
    
    # Find second active index by temporarily marking the first as inactive.
    temp_mask = active_sorted.at[first_idx].set(False)
    second_idx = jnp.argmin(jnp.where(temp_mask, indices, large_value))
    
    # Find last active index.
    last_idx = jnp.argmax(jnp.where(active_sorted, indices, -1))
    
    # Find second last active index by marking the last as inactive.
    temp_mask2 = active_sorted.at[last_idx].set(False)
    second_last_idx = jnp.argmax(jnp.where(temp_mask2, indices, -1))
    
    # Extract coordinates and values for extrapolation.
    x1 = x_sorted[first_idx]
    t1 = t_sorted[first_idx]
    x2 = x_sorted[second_idx]
    t2 = t_sorted[second_idx]
    
    xl1 = x_sorted[last_idx]
    tl1 = t_sorted[last_idx]
    xl2 = x_sorted[second_last_idx]
    tl2 = t_sorted[second_last_idx]
    
    # Calculate slopes for left and right extrapolation.
    slope_start = (t2 - t1) / (x2 - x1 + 1e-8)
    slope_end = (tl1 - tl2) / (xl1 - xl2 + 1e-8)
    
    return (x1, t1, slope_start), (xl1, tl1, slope_end)
@jit
def handle_no_active(x, t, active, x_grid, default_value):
    """
    Handle the case where there are no active control points.
    
    Returns a constant array filled with the default value.
    """
    return jnp.full_like(x_grid, default_value)
@jit
def handle_one_active(x, t, active, x_grid, default_value):
    """
    Handle the case where exactly one control point is active.
    
    Returns a constant array using the t-value of the active control point.
    """
    active_idx = jnp.argmax(active)
    return jnp.full_like(x_grid, t[active_idx])

@jit
def handle_multiple_active(x, t, active, x_grid, default_value):
    """
    Handle interpolation when there are multiple active control points.
    
    The method works as follows:
    
    1. **Sorting:**  
       The control points are sorted by their x-coordinates so that interpolation is performed
       in order.
       
    2. **Neighbor Finding:**  
       For each point, the nearest active control point to its left and right is found using the 
       `find_active_neighbors` function.
       
    3. **Extrapolation Setup:**  
       The boundary extrapolation parameters are computed using the first two and last two active
       points via `handle_multiple_active_extrapolation`. This provides slopes for left and right 
       extrapolation.
       
    4. **Propagation and Local Interpolation:**  
       The function creates an initial propagation array (`t_propagated`), preserving values at 
       active points and setting inactive points to zero. It then processes each inactive control point:
         - If the point is **before** the first active point, it uses left extrapolation.
         - If the point is **after** the last active point, it uses right extrapolation.
         - If the point is **between** active points, it performs linear interpolation between the 
           nearest active neighbors.
       
    5. **Grid Interpolation:**  
       Finally, it uses `jnp.interp` with the mode `"extrapolate"` to map the propagated control point 
       values onto a fine grid.
    
    Args:
        x (array): x-coordinates of the control points.
        t (array): t-values (y-values) at the control points.
        active (array): Boolean array indicating active control points.
        x_grid (array): Fine grid over which to perform interpolation.
        default_value: Default value used when there are no active points (not used in this branch).
        
    Returns:
        Interpolated values over `x_grid`.
    """
    # Sort coordinates and values by x position.
    sort_idx = jnp.argsort(x)
    x_sorted = jnp.take(x, sort_idx)
    t_sorted = jnp.take(t, sort_idx)
    active_sorted = jnp.take(active, sort_idx)
    
    # Get previous/next active indices.
    prev_idx, next_idx = find_active_neighbors(active_sorted)
    
    # Compute boundary extrapolation parameters.
    (x_start, t_start, slope_start), (x_end, t_end, slope_end) = \
        handle_multiple_active_extrapolation(x_sorted, t_sorted, active_sorted)
    
    # Initialize propagated values.
    t_propagated = jnp.where(active_sorted, t_sorted, 0.0)
    
    # Process each point for interpolation or extrapolation.
    def process_inactive_points(t_prop, x_idx):
        i = x_idx  # For clarity
        
        # Process only if the point is inactive.
        is_inactive = ~active_sorted[i]
        
        # Determine if the point is before the first active control point.
        before_first = x_sorted[i] < x_start
        t_before = t_start + slope_start * (x_sorted[i] - x_start)
        
        # Determine if the point is after the last active control point.
        after_last = x_sorted[i] > x_end
        t_after = t_end + slope_end * (x_sorted[i] - x_end)
        
        # For points between active controls, compute linear interpolation.
        prev = prev_idx[i]
        next_ = next_idx[i]
        has_valid_neighbors = (prev >= 0) & (next_ >= 0)
        denom = x_sorted[next_] - x_sorted[prev] + 1e-8
        alpha = jnp.where(has_valid_neighbors, (x_sorted[i] - x_sorted[prev]) / denom, 0.0)
        t_between = jnp.where(has_valid_neighbors,
                              (1 - alpha) * t_sorted[prev] + alpha * t_sorted[next_],
                              0.0)
        
        # Choose the region-specific value.
        region_value = jnp.where(before_first, t_before,
                                 jnp.where(after_last, t_after, t_between))
        
        # Only update the point if it is inactive.
        new_value = jnp.where(is_inactive, region_value, t_prop[i])
        return t_prop.at[i].set(new_value)
    
    # Process each point using a for-loop (JAX compatible).
    t_propagated = lax.fori_loop(
        0, len(x_sorted),
        lambda i, t_prop: process_inactive_points(t_prop, i),
        t_propagated
    )
    
    # Map the processed values onto the fine grid, with extrapolation enabled.
    return jnp.interp(x_grid, x_sorted, t_propagated, left="extrapolate", right="extrapolate")

@jit
def interpolate_active(x_grid, x, t, active, default_value=0.0):
    """
    Perform piecewise linear interpolation with proper boundary handling.
    
    Depending on the number of active control points, this function:
      - Returns a constant value (if none or one active point), or
      - Uses `handle_multiple_active` for full interpolation.
    
    Args:
        x: x-coordinates of control points.
        t: t-values at the control points.
        active: Boolean mask for active control points.
        x_grid: Fine grid positions where interpolation is computed.
        default_value: Default value used when no active control points exist.
    
    Returns:
        Interpolated values along `x_grid`.
    """
    n_active = jnp.sum(active)
    result = lax.cond(
        n_active == 0,
        lambda args: handle_no_active(*args),
        lambda args: lax.cond(
            n_active == 1,
            lambda inner_args: handle_one_active(*inner_args),
            lambda inner_args: handle_multiple_active(*inner_args),
            args
        ),
        (x, t, active, x_grid, default_value)
    )
    
    return result