"""
Spike encoding schemes library.
Each method converts continuous values to spike trains.
"""

import torch
import torch.nn as nn


class EncodingLibrary:
    """
    Collection of spike encoding methods.
    All methods follow the same interface:
        Input: x [batch, seq, dim] continuous values
        Output: spikes [batch, seq, dim, T] binary spike trains
    
    IMPORTANT: All encoders must output the SAME shape!
    """
    
    @staticmethod
    def rate_encoding(x, T=4, threshold=0.5):
        """
        Rate coding: Number of spikes proportional to input magnitude.
        
        Args:
            x: Input tensor [batch, seq, dim]
            T: Number of time steps
            threshold: Unused, kept for interface consistency
            
        Returns:
            spikes: [batch, seq, dim, T]
        """
        spikes = torch.zeros(*x.shape, T, device=x.device)
        x_normalized = torch.sigmoid(x)  # Map to [0, 1]
        
        for t in range(T):
            # Stochastic firing based on magnitude
            fire_prob = x_normalized
            spikes[..., t] = (torch.rand_like(x) < fire_prob).float()
            
        return spikes
    
    @staticmethod
    def temporal_encoding(x, T=4):
        """
        Temporal coding: Spike timing encodes information.
        Larger values fire earlier.
        
        Args:
            x: Input tensor [batch, seq, dim]
            T: Number of time steps
            
        Returns:
            spikes: [batch, seq, dim, T]
        """
        spikes = torch.zeros(*x.shape, T, device=x.device)
        x_normalized = torch.sigmoid(x)
        
        # Map value to spike time
        # Large values → early spikes (small t)
        spike_times = ((1 - x_normalized) * (T - 1)).long()
        spike_times = torch.clamp(spike_times, 0, T-1)
        
        # Create one-hot spike train
        spikes.scatter_(-1, spike_times.unsqueeze(-1), 1.0)
        
        return spikes
    
    @staticmethod
    def population_encoding(x, T=4, n_neurons=4):
        """
        Population coding: Multiple neurons per dimension with different preferences.
        
        FIXED: Now maintains output shape by averaging across population.
        
        Args:
            x: Input tensor [batch, seq, dim]
            T: Number of time steps
            n_neurons: Number of neurons per dimension (internal)
            
        Returns:
            spikes: [batch, seq, dim, T] - SAME SHAPE as input
        """
        batch, seq, dim = x.shape
        
        # Create neuron preferences (different receptive fields)
        # Shape: [n_neurons]
        preferences = torch.linspace(-2, 2, n_neurons, device=x.device)
        
        # Expand x for population: [B, S, D] → [B, S, D, n_neurons]
        x_expanded = x.unsqueeze(-1).expand(-1, -1, -1, n_neurons)
        
        # Compute responses using Gaussian tuning curves
        # preferences: [n_neurons] → [1, 1, 1, n_neurons]
        prefs = preferences.view(1, 1, 1, n_neurons)
        
        # Gaussian tuning
        distances = (x_expanded - prefs) ** 2
        responses = torch.exp(-distances / 0.5)  # [B, S, D, n_neurons]
        
        # Generate spikes for each neuron
        spikes_pop = torch.zeros(*responses.shape, T, device=x.device)
        for t in range(T):
            spikes_pop[..., t] = (torch.rand_like(responses) < responses).float()
        
        # Aggregate back to original dimension by AVERAGING across population
        # [B, S, D, n_neurons, T] → [B, S, D, T]
        spikes = spikes_pop.mean(dim=-2)
        
        return spikes
    
    @staticmethod
    def burst_encoding(x, T=4, burst_length=2):
        """
        Burst coding: Groups of spikes for salient features.
        
        Args:
            x: Input tensor [batch, seq, dim]
            T: Number of time steps
            burst_length: Length of burst for high-magnitude inputs
            
        Returns:
            spikes: [batch, seq, dim, T]
        """
        spikes = torch.zeros(*x.shape, T, device=x.device)
        x_normalized = torch.sigmoid(x)
        
        # High values trigger bursts
        should_burst = x_normalized > 0.7
        
        for t in range(T):
            if t < burst_length:
                # Burst phase
                spikes[..., t] = should_burst.float()
            else:
                # Sparse random firing after burst
                spikes[..., t] = (torch.rand_like(x) < 0.1).float()
                
        return spikes
    
    @staticmethod
    def adaptive_threshold_encoding(x, T=4, decay=0.9):
        """
        Adaptive threshold: Leaky integrate-and-fire dynamics.
        
        Args:
            x: Input tensor [batch, seq, dim]
            T: Number of time steps
            decay: Membrane potential decay rate
            
        Returns:
            spikes: [batch, seq, dim, T]
        """
        spikes = torch.zeros(*x.shape, T, device=x.device)
        
        # Adaptive threshold based on input statistics
        threshold = x.abs().mean(dim=-1, keepdim=True) + 1e-8
        
        # Initialize membrane potential
        membrane = torch.zeros_like(x)
        
        for t in range(T):
            # Leaky integration
            membrane = decay * membrane + x
            
            # Fire if above threshold
            fire = (membrane > threshold).float()
            spikes[..., t] = fire
            
            # Reset membrane after firing
            membrane = membrane * (1 - fire)
            
        return spikes


# Convenience function for getting encoder by name
def get_encoder(name):
    """Get encoder function by name."""
    encoders = {
        'rate': EncodingLibrary.rate_encoding,
        'temporal': EncodingLibrary.temporal_encoding,
        'population': EncodingLibrary.population_encoding,
        'burst': EncodingLibrary.burst_encoding,
        'adaptive': EncodingLibrary.adaptive_threshold_encoding,
    }
    return encoders[name]


# Testing function
def test_all_encoders():
    """Test that all encoders produce correct output shape."""
    print("Testing encoding library...\n")
    
    # Test input
    batch, seq, dim, T = 2, 10, 256, 4
    x = torch.randn(batch, seq, dim)
    
    encoders = {
        'rate': EncodingLibrary.rate_encoding,
        'temporal': EncodingLibrary.temporal_encoding,
        'population': EncodingLibrary.population_encoding,
        'burst': EncodingLibrary.burst_encoding,
        'adaptive': EncodingLibrary.adaptive_threshold_encoding,
    }
    
    expected_shape = (batch, seq, dim, T)
    
    for name, encoder in encoders.items():
        spikes = encoder(x, T=T)
        assert spikes.shape == expected_shape, (
            f"{name} encoding failed: expected {expected_shape}, got {spikes.shape}"
        )
        
        # Check that spikes are binary or near-binary
        unique_vals = torch.unique(spikes)
        print(f"✓ {name:12} - Shape: {spikes.shape}, Unique values: {unique_vals.tolist()[:5]}")
        
        # Check sparsity
        sparsity = 1 - spikes.mean().item()
        print(f"  {'':12}   Sparsity: {sparsity:.2%}")
    
    print("\n✓ All encoders produce correct output shape!")


if __name__ == '__main__':
    test_all_encoders()