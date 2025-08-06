"""
Example demonstrating rotationally invariant CNN SUBNET usage in deepSI.

This example shows how to use the RotInvariant_CNN_SUBNET for learning dynamics
from image data while maintaining invariance to rotations.
"""

import numpy as np
import sys
import os

# Add the parent directory to Python path to import local deepSI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
except ImportError:
    print("Error: PyTorch is not installed. Please install it with:")
    print("pip install torch")
    sys.exit(1)

try:
    import deepSI as dsi
    from deepSI.models import RotInvariant_CNN_SUBNET
    from deepSI.networks import RotationallyInvariantCNN
    from nonlinear_benchmarks import Input_output_data
    print("✓ Successfully imported deepSI modules")
except ImportError as e:
    print(f"Error importing deepSI modules: {e}")
    print("Make sure you're running this from the deepSI directory")
    sys.exit(1)

def create_rotating_image_data(n_samples=1000, image_size=(32, 32), rotation_rate=0.1):
    """
    Create synthetic data with rotating patterns for demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of time samples
    image_size : tuple
        (height, width) of images
    rotation_rate : float
        Rate of rotation per time step
    
    Returns:
    --------
    data : Input_output_data
        Synthetic dataset with input and rotating image outputs
    """
    np.random.seed(42)
    
    # Generate input sequence
    u = np.random.randn(n_samples, 1) * 0.5
    
    # Generate rotating image patterns
    h, w = image_size
    y = np.zeros((n_samples, h, w))
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    center_h, center_w = h // 2, w // 2
    
    for t in range(n_samples):
        # Create a simple rotating pattern using numpy's built-in rotation
        angle = t * rotation_rate + u[t, 0] * 0.5  # Input affects rotation
        
        # Create a simple pattern with diagonal stripes
        x_centered = x_coords - center_w
        y_centered = y_coords - center_h
        
        # Rotate coordinates
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = x_centered * cos_a - y_centered * sin_a
        y_rot = x_centered * sin_a + y_centered * cos_a
        
        # Create rotating pattern (diagonal stripes)
        pattern = np.sin(0.2 * (x_rot + y_rot)) * np.exp(-(x_rot**2 + y_rot**2) / (h*w/8))
        
        # Add some structure
        pattern += 0.3 * np.sin(0.15 * x_rot) * np.exp(-y_rot**2 / (h/4)**2)
        
        # Add some noise
        pattern += np.random.randn(h, w) * 0.05
        
        y[t] = pattern
    
    return Input_output_data(u=u, y=y)


def main():
    """Main example demonstrating rotationally invariant CNN SUBNET."""
    
    print("Creating synthetic rotating image data...")
    
    # Create synthetic data
    data = create_rotating_image_data(n_samples=1000, image_size=(32, 32))
    
    # Split data
    train_data = data[:600]
    val_data = data[600:800] 
    test_data = data[800:]
    
    print(f"Data shapes: u={data.u.shape}, y={data.y.shape}")
    
    # Get data characteristics and normalization (now supports image data)
    nu, ny, norm = dsi.get_nu_ny_and_auto_norm(train_data)
    print(f"nu={nu}, ny={ny}")
    
    # Create standard CNN SUBNET for comparison
    print("\nTraining standard CNN SUBNET...")
    try:
        model_standard = dsi.CNN_SUBNET(nu, ny, norm, nx=8, nb=10, na=10)
    except Exception as e:
        print(f"Error creating CNN_SUBNET: {e}")
        print("Using regular SUBNET instead...")
        model_standard = dsi.SUBNET(nu, ny, norm, nx=8, nb=10, na=10)
    
    # Train standard model
    train_dict_standard = dsi.fit(
        model_standard, train_data, val_data, 
        n_its=100,  # Reduced for quick demo
        T=20, 
        batch_size=32, 
        val_freq=25
    )
    
    # Create rotationally invariant CNN SUBNET
    print("\nTraining rotationally invariant CNN SUBNET...")
    model_invariant = RotInvariant_CNN_SUBNET(
        nu, ny, norm, nx=8, nb=10, na=10,
        invariant_type='rotation_avg',  # or 'steerable'
        n_rotations=4,
        cnn_layers=2,  # Reduced for quick demo
        base_channels=16
    )
    
    # Train invariant model
    train_dict_invariant = dsi.fit(
        model_invariant, train_data, val_data,
        n_its=100,  # Reduced for quick demo  
        T=20,
        batch_size=32,
        val_freq=25
    )
    
    # Test both models
    print("\nTesting models...")
    
    # Simulate on test data
    test_sim_standard = model_standard.simulate(test_data)
    test_sim_invariant = model_invariant.simulate(test_data)
    
    # Compute NRMS errors
    def compute_nrms(y_true, y_pred):
        error = np.mean((y_true - y_pred)**2)
        variance = np.var(y_true)
        return np.sqrt(error / variance)
    
    nrms_standard = compute_nrms(test_data.y, test_sim_standard.y)
    nrms_invariant = compute_nrms(test_data.y, test_sim_invariant.y)
    
    print(f"\nResults:")
    print(f"Standard CNN SUBNET NRMS: {nrms_standard:.4f}")
    print(f"Rotationally Invariant CNN SUBNET NRMS: {nrms_invariant:.4f}")
    
    # Test rotation invariance by rotating test data
    print("\nTesting rotation invariance...")
    
    # Create rotated version of first test sample
    test_sample = test_data[:50]  # First 50 samples
    
    # Rotate test images by 90 degrees
    rotated_y = np.rot90(test_sample.y, k=1, axes=(1, 2))
    rotated_test = Input_output_data(u=test_sample.u, y=rotated_y)
    
    # Simulate rotated data
    rotated_sim_standard = model_standard.simulate(rotated_test)
    rotated_sim_invariant = model_invariant.simulate(rotated_test)
    
    # Compare outputs (invariant model should have similar performance)
    original_sim_standard = model_standard.simulate(test_sample)
    original_sim_invariant = model_invariant.simulate(test_sample)
    
    # Compute differences in simulation quality
    def output_consistency(sim1, sim2):
        return np.mean((sim1.y - sim2.y)**2)
    
    consistency_standard = output_consistency(original_sim_standard, rotated_sim_standard)
    consistency_invariant = output_consistency(original_sim_invariant, rotated_sim_invariant)
    
    print(f"Standard model output consistency (lower is better): {consistency_standard:.6f}")
    print(f"Invariant model output consistency (lower is better): {consistency_invariant:.6f}")
    
    if consistency_invariant < consistency_standard:
        print("✓ Rotationally invariant model shows better consistency!")
    else:
        print("⚠ Results may vary due to random initialization and limited training")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()