"""
Unit tests for encoding library.
"""

import torch
from src.encoding.library import EncodingLibrary, get_encoder


def test_encoding_shapes():
    """Test that all encodings produce correct output shape."""
    print("Testing encoding output shapes...")
    
    batch, seq, dim, T = 2, 10, 256, 4
    x = torch.randn(batch, seq, dim)
    expected_shape = (batch, seq, dim, T)
    
    encoders = [
        ('rate', EncodingLibrary.rate_encoding),
        ('temporal', EncodingLibrary.temporal_encoding),
        ('population', EncodingLibrary.population_encoding),
        ('burst', EncodingLibrary.burst_encoding),
        ('adaptive', EncodingLibrary.adaptive_threshold_encoding),
    ]
    
    for name, encoder in encoders:
        spikes = encoder(x, T=T)
        assert spikes.shape == expected_shape, (
            f"{name} failed: expected {expected_shape}, got {spikes.shape}"
        )
        print(f"  ✓ {name}: {spikes.shape}")
    
    print("✓ All encodings produce correct shape\n")


def test_encoding_values():
    """Test that encodings produce valid spike values."""
    print("Testing encoding value ranges...")
    
    x = torch.randn(2, 10, 256)
    T = 4
    
    encoders = {
        'rate': EncodingLibrary.rate_encoding,
        'temporal': EncodingLibrary.temporal_encoding,
        'population': EncodingLibrary.population_encoding,
        'burst': EncodingLibrary.burst_encoding,
        'adaptive': EncodingLibrary.adaptive_threshold_encoding,
    }
    
    for name, encoder in encoders.items():
        spikes = encoder(x, T=T)
        
        # Check values are in valid range [0, 1]
        assert spikes.min() >= 0, f"{name}: has negative values"
        assert spikes.max() <= 1, f"{name}: has values > 1"
        
        # Check not all zeros
        assert spikes.sum() > 0, f"{name}: produces all zeros"
        
        sparsity = 1 - spikes.mean().item()
        print(f"  ✓ {name:12} - Range: [{spikes.min():.3f}, {spikes.max():.3f}], Sparsity: {sparsity:.1%}")
    
    print("✓ All encodings produce valid values\n")


def test_get_encoder():
    """Test the get_encoder convenience function."""
    print("Testing get_encoder function...")
    
    x = torch.randn(2, 10, 256)
    T = 4
    
    for name in ['rate', 'temporal', 'population', 'burst', 'adaptive']:
        encoder = get_encoder(name)
        spikes = encoder(x, T=T)
        assert spikes.shape == (2, 10, 256, T)
        print(f"  ✓ {name}")
    
    print("✓ get_encoder works correctly\n")


def test_encoding_consistency():
    """Test that encodings are consistent across calls."""
    print("Testing encoding consistency...")
    
    torch.manual_seed(42)
    x = torch.randn(2, 10, 256)
    T = 4
    
    # Deterministic encodings should be identical
    temp1 = EncodingLibrary.temporal_encoding(x, T=T)
    temp2 = EncodingLibrary.temporal_encoding(x, T=T)
    
    assert torch.allclose(temp1, temp2), "Temporal encoding not consistent"
    print("  ✓ Deterministic encodings are consistent")
    
    # Stochastic encodings should be different (with different seeds)
    torch.manual_seed(42)
    rate1 = EncodingLibrary.rate_encoding(x, T=T)
    torch.manual_seed(43)
    rate2 = EncodingLibrary.rate_encoding(x, T=T)
    
    assert not torch.allclose(rate1, rate2), "Rate encoding should be stochastic"
    print("  ✓ Stochastic encodings are different with different seeds")
    
    print("✓ Encoding consistency verified\n")


if __name__ == '__main__':
    print("="*60)
    print("Running Encoding Library Tests")
    print("="*60 + "\n")
    
    test_encoding_shapes()
    test_encoding_values()
    test_get_encoder()
    test_encoding_consistency()
    
    print("="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)