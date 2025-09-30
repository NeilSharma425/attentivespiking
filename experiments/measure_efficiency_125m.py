"""
Measure spike efficiency at 125M scale.
"""

import torch
from pathlib import Path
import sys
sys.path.append('/content/attentivespiking')

from src.models.transformer import SpikingTransformer
from transformers import GPT2Tokenizer


def count_spikes_125m():
    """Count spikes for 125M models."""
    
    device = torch.device('cuda')
    
    # Load adaptive model
    print("Loading Adaptive 125M model...")
    adaptive = SpikingTransformer(
        vocab_size=50257, d_model=768, nhead=12, num_layers=12,
        dim_feedforward=3072, T=4, max_seq_len=512, dropout=0.1
    )
    adaptive.load_state_dict(torch.load(
        '/content/drive/MyDrive/spike_checkpoints_125m/best_model.pt',
        map_location=device
    ))
    adaptive.eval()
    
    # Load fixed rate model
    print("Loading Fixed Rate 125M model...")
    fixed_rate = SpikingTransformer(
        vocab_size=50257, d_model=768, nhead=12, num_layers=12,
        dim_feedforward=3072, T=4, max_seq_len=512, dropout=0.1,
        fixed_encoding='rate'
    )
    fixed_rate.load_state_dict(torch.load(
        '/content/drive/MyDrive/spike_checkpoints_125m_baselines/fixed_rate_best.pt',
        map_location=device
    ))
    fixed_rate.eval()
    
    # Test input
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text = 'The quick brown fox jumps over the lazy dog' * 10
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    # Count spikes
    def count_spikes(model):
        spike_counts = []
        
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                spike_counts.append(output.sum().item())
        
        hooks = []
        for layer in model.layers:
            hooks.append(layer.encoder_attn.register_forward_hook(hook))
            hooks.append(layer.encoder_ffn.register_forward_hook(hook))
        
        with torch.no_grad():
            model(input_ids)
        
        for hook in hooks:
            hook.remove()
        
        return sum(spike_counts)
    
    adaptive_spikes = count_spikes(adaptive)
    rate_spikes = count_spikes(fixed_rate)
    
    reduction = (1 - adaptive_spikes / rate_spikes) * 100
    
    print("\n" + "="*60)
    print("125M SPIKE EFFICIENCY")
    print("="*60)
    print(f"Adaptive:    {adaptive_spikes:,.0f} spikes")
    print(f"Fixed Rate:  {rate_spikes:,.0f} spikes")
    print(f"Reduction:   {reduction:.1f}%")
    print(f"Efficiency:  {rate_spikes/adaptive_spikes:.2f}×")
    print("="*60)


if __name__ == '__main__':
    count_spikes_125m()
