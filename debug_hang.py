import sys
sys.path.insert(0, './src')
import numpy as np
import time

def print_time(msg):
    pass

from demo_sindy_nonlinear_test import generate_hill_data
from opera.discovery.pipeline import run_discovery, DiscoveryConfig
print("Generating data...")
spectra, factors, wavelengths, f1, f2, P1, P2 = generate_hill_data()
cfg = DiscoveryConfig(k_mode='fixed', k_value=2, sparsity_threshold=0.01)
cfg.matrix.include_D_squared = False

print("running discovery...")
import opera.discovery.pipeline as pipe
original_decode = pipe.decode_physical_manifolds
def mock_decode(*args, **kwargs):
    print("Reached decode_physical_manifolds!")
    return original_decode(*args, **kwargs)
pipe.decode_physical_manifolds = mock_decode

original_build_lib = pipe.build_observable_library
def mock_build_lib(*args, **kwargs):
    print("Reached build_observable_library!")
    return original_build_lib(*args, **kwargs)
pipe.build_observable_library = mock_build_lib

original_nullspace = pipe.find_joint_nullspace
def mock_nullspace(*args, **kwargs):
    print("Reached find_joint_nullspace!")
    return original_nullspace(*args, **kwargs)
pipe.find_joint_nullspace = mock_nullspace

t0 = time.time()
res = run_discovery(spectra, factors, wavelengths, cfg)
print(f"done in {time.time()-t0:.2f}s")
