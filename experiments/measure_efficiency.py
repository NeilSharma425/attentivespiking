"""
Measure spike counts, memory, and operations.
This could be your key differentiator!
"""

import torch
from src.models.transformer import create_small_model, create_fixed_rate_model
from transformers import GPT2Tokenizer

def count_spikes(model, input_ids):
    """Count total spikes generated."""
    total_spikes = 0
    total_possible = 0
    
    # Hook to capture spikes
    spike_counts = []
    
    def spike_hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.dim() == 4:
            # Spike tensor: [B, S, D, T]
            spike_counts.append(output.sum().item())
            total_possible = output.numel()
    
    # Register hooks
    hooks = []
    for layer in model.layers:
        hooks.append(layer.encoder_attn.register_forward_hook(spike_hook))
        hooks.append(layer.encoder_ffn.register_forward_hook(spike_hook))
    
    # Forward pass
    with torch.no_grad():
        model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    total_spikes = sum(spike_counts)
    return total_spikes, spike_counts

def compare_efficiency():
    """Compare computational efficiency."""
    
    print("="*60)
    print("EFFICIENCY COMPARISON")
    print("="*60 + "\n")
    
    # Load models
    adaptive = create_small_model()
    adaptive.load_state_dict(torch.load('checkpoints/adaptive.pt', map_location='cpu'))
    adaptive.eval()
    
    fixed_rate = create_fixed_rate_model()
    fixed_rate.load_state_dict(torch.load('checkpoints/fixed_rate.pt', map_location='cpu'))
    fixed_rate.eval()
    
    # Test input
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text = "The quick brown fox jumps over the lazy dog" * 5
    input_ids = tokenizer.encode(text, return_tensors='pt')
    
    # Count spikes
    print("Counting spikes...\n")
    
    adaptive_spikes, _ = count_spikes(adaptive, input_ids)
    rate_spikes, _ = count_spikes(fixed_rate, input_ids)
    
    print(f"Adaptive:    {adaptive_spikes:,} spikes")
    print(f"Fixed Rate:  {rate_spikes:,} spikes")
    print(f"Reduction:   {(1 - adaptive_spikes/rate_spikes)*100:.1f}%")
    
    print("\n" + "="*60)


# Add to measure_efficiency.py to break down by encoding type

def analyze_spikes_by_encoding(model, input_ids):
    """Detailed spike analysis."""
    
    print("\nDETAILED SPIKE ANALYSIS:")
    print("-"*60)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    pos_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    
    token_embeds = model.token_embedding(input_ids)
    pos_embeds = model.position_embedding(pos_ids)
    x = token_embeds + pos_embeds
    
    for i, layer in enumerate(model.layers):
        x_norm = layer.norm1(x)
        
        # Get encoding weights
        with torch.no_grad():
            weights = layer.encoder_attn.selector(x_norm, pos_ids, i)
            spikes = layer.encoder_attn(x_norm, pos_ids, i)
        
        avg_weights = weights.mean(dim=(0,1)).cpu().numpy()
        sparsity = 1 - spikes.mean().item()
        spike_count = spikes.sum().item()
        
        encoding_names = ['Rate', 'Temporal', 'Population', 'Burst', 'Adaptive']
        dominant = encoding_names[avg_weights.argmax()]
        
        print(f"\nLayer {i} ({dominant}):")
        print(f"  Spikes: {spike_count:,.0f}")
        print(f"  Sparsity: {sparsity:.1%}")
        print(f"  Encoding mix: {dict(zip(encoding_names, [f'{w:.1%}' for w in avg_weights]))}")
        
        # Forward through layer
        with torch.no_grad():
            x = layer(x, pos_ids)

if __name__ == '__main__':
    compare_efficiency()