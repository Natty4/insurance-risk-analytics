import numpy as np

def compute_optimized_premium(p_claim, expected_claim, loading_factor=1.2):
    return p_claim * expected_claim * loading_factor