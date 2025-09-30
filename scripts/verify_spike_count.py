"""
Verify that spike counting is consistent.
"""

import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import create_small_model, create_fixed_rate_model
from transformers import GPT2Tokenizer


def count_all_spikes(model, input_ids, model_name="Model"):
    """Count spikes using multiple methods to verify."""
    
    print(f"\n{model_name}:")
    print("-"*60)
    
    spike_counts = []
    
    # Method 1: Hook-based counting
    def spike_hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.dim() == 4:
            spike_counts.append(output.sum().item())
    
    hooks = []
    for layer in model.layers:
        hooks.append(layer.encoder_attn.register_forward_hook(spike_hook))
        hooks.append(layer.encoder_ffn.register_forward_hook(spike_hook))
    
    with torch.no_grad():
        model(input_ids)
    
    for hook in hooks:
        hook.remove()
    
    method1_total = sum(spike_counts)
    
    # Method 2: Manual forward pass
    pos_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    token_embeds = model.token_embedding(input_ids)
    pos_embeds = model.position_embedding(pos_ids)
    x = token_embeds + pos_embeds
    
    method2_total = 0
    with torch.no_grad():
        for layer in model.layers:
            x_norm = layer.norm1(x)
            spikes_attn = layer.encoder_attn(x_norm, pos_ids, layer.layer_idx)
            method2_total += spikes_attn.sum().item()
            
            x = layer(x, pos_ids)
            
            x_norm = layer.norm2(x)
            spikes_ffn = layer.encoder_ffn(x_norm, pos_ids, layer.layer_idx)
            method2_total += spikes_ffn.sum().item()
    
    print(f"  Method 1 (hooks):       {method1_total:,.0f} spikes")
    print(f"  Method 2 (manual):      {method2_total:,.0f} spikes")
    print(f"  Difference:             {abs(method1_total - method2_total):,.0f}")
    
    return method1_total


def verify_measurements():
    """Verify spike count measurements."""
    
    print("="*60)
    print("SPIKE COUNT VERIFICATION")
    print("="*60)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text = 'The quick brown fox jumps over the lazy dog' * 5
    input_ids = tokenizer.encode(text, return_tensors='pt')
    
    # Adaptive
    adaptive = create_small_model()
    adaptive.load_state_dict(torch.load('checkpoints/adaptive.pt', map_location='cpu'))
    adaptive.eval()
    
    adaptive_count = count_all_spikes(adaptive, input_ids, "Adaptive (Ours)")
    
    # Fixed Rate
    fixed_rate = create_fixed_rate_model()
    fixed_rate.load_state_dict(torch.load('checkpoints/fixed_rate.pt', map_location='cpu'))
    fixed_rate.eval()
    
    rate_count = count_all_spikes(fixed_rate, input_ids, "Fixed Rate")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Adaptive:       {adaptive_count:,.0f} spikes")
    print(f"Fixed Rate:     {rate_count:,.0f} spikes")
    print(f"Reduction:      {(1 - adaptive_count/rate_count)*100:.1f}%")
    print(f"Efficiency:     {rate_count/adaptive_count:.2f}×")
    print("="*60)


if __name__ == '__main__':
    verify_measurements()