"""Configuration values for jax kernels."""

MEMORY_SAVING_CONFIGS = [
    'KERNEL_MATRIX_USE_LOOP',
    'NORMALIZED_KERNEL_PUSH_DOWN'
]

SAVE_MEMORY = False
"""bool: Apply multiple tricks to reduce memory overhead.

This could slow down computation significantly but may allow the direct
application to larger problems when running into GPU memory issues.

This variable activates all configs in `MEMORY_SAVING_CONFIGS`.
"""

KERNEL_MATRIX_USE_LOOP = True
"""bool: Use loop when constructing kernel matrix

Compute kernel matrix using a dynamically unrolled loop instead of
vectorization. Reduces memory overhead as not all intermediate tensors have to
be allocated at once.
"""

NORMALIZED_KERNEL_PUSH_DOWN = False
"""bool: Push down the computation of normalization factor

Moves the computation of normalization to each individual instance. This can
reduce memory overhead, especially when computing gradients with respect to
kernel parameters at the cost of higher computation time (approx. 3 fold).  In
this case k(x,x) and k(y,y) needed to be recomputed for each instance.
"""


def config_value(config_var_name):
    """Get value of a config variable.

    Also accounts for config variables that activate other variables such as
    `SAVE_MEMORY`.
    """
    if config_var_name in MEMORY_SAVING_CONFIGS and SAVE_MEMORY:
        return True
    return globals()[config_var_name]
