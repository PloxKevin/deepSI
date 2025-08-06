#!/usr/bin/env python3
"""
Simple test script to verify rotationally invariant CNN modules can be imported.
"""

import sys
import os

# Add current directory to path for local deepSI import
sys.path.insert(0, '.')

def test_imports():
    """Test importing the rotationally invariant modules."""
    
    print("Testing rotationally invariant CNN imports...")
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        print("Please install PyTorch: pip install torch")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from deepSI.networks import RotationallyInvariantCNN, RotInvariant_CNN_encoder
        print("✓ Rotationally invariant network modules imported successfully")
    except ImportError as e:
        print(f"✗ Network module import failed: {e}")
        return False
    
    try:
        from deepSI.models import RotInvariant_CNN_SUBNET
        print("✓ RotInvariant_CNN_SUBNET imported successfully")
    except ImportError as e:
        print(f"✗ Model import failed: {e}")
        return False
    
    # Test basic instantiation
    try:
        # Test creating a simple rotationally invariant CNN
        input_shape = (3, 32, 32)  # 3 channels, 32x32 image
        cnn = RotationallyInvariantCNN(
            input_shape=input_shape,
            n_layers=2,
            base_channels=16,
            invariant_type='rotation_avg',
            n_rotations=4
        )
        print("✓ RotationallyInvariantCNN instantiated successfully")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(2, 3, 32, 32)  # batch_size=2
        output = cnn(dummy_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ CNN instantiation/forward pass failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Rotationally invariant CNN modules are working correctly.")
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("Testing deepSI Rotationally Invariant CNN Implementation")
    print("=" * 60)
    
    success = test_imports()
    
    if success:
        print("\nYou can now use the rotationally invariant CNN modules:")
        print("- RotationallyInvariantCNN")
        print("- RotInvariant_CNN_encoder") 
        print("- RotInvariant_CNN_SUBNET")
        print("\nExample usage:")
        print("from deepSI.models import RotInvariant_CNN_SUBNET")
        print("model = RotInvariant_CNN_SUBNET(nu, ny, norm, nx=8, nb=10, na=10)")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())