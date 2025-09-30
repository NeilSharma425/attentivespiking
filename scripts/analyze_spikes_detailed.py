"""
Detailed spike analysis by layer and encoding type.
"""

import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import create_small_model
from transformers import GPT2Tokenizer


def analyze_spikes_detailed(model_path='checkpoints/adaptive.pt'):
    """Analyze spikes layer by layer."""
    
    print("="*70)
    print("DETAILED SPIKE ANALYSIS BY LAYER")
    print("="*70 + "\n")
    
    # Load model
    model = create_small_model()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text = 'The quick brown fox jumps over the lazy dog' * 5
    input_ids = tokenizer.encode(text, return_tensors='pt')
    pos_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    
    # Get embeddings
    token_embeds = model.token_embedding(input_ids)
    pos_embeds = model.position_embedding(pos_ids)
    x = token_embeds + pos_embeds
    
    total_spikes = 0
    encoding_names = ['Rate', 'Temporal', 'Population', 'Burst', 'Adaptive']
    
    for i, layer in enumerate(model.layers):
        print(f"\nLAYER {i}:")
        print("-"*70)
        
        # Attention encoder
        x_norm = layer.norm1(x)
        
        with torch.no_grad():
            # Get encoding selection
            weights = layer.encoder_attn.selector(x_norm, pos_ids, i)
            avg_weights = weights.mean(dim=(0,1)).cpu().numpy()
            
            # Generate spikes
            spikes_attn = layer.encoder_attn(x_norm, pos_ids, i)
            
            # FFN encoder
            x_norm_ffn = layer.norm2(x)
            spikes_ffn = layer.encoder_ffn(x_norm_ffn, pos_ids, i)
        
        # Statistics
        spike_count_attn = spikes_attn.sum().item()
        spike_count_ffn = spikes_ffn.sum().item()
        layer_total = spike_count_attn + spike_count_ffn
        
        sparsity_attn = 1 - spikes_attn.mean().item()
        sparsity_ffn = 1 - spikes_ffn.mean().item()
        
        dominant = encoding_names[avg_weights.argmax()]
        
        print(f"  Encoding Selection:")
        for name, weight in zip(encoding_names, avg_weights):
            bar = '█' * int(weight * 30)
            print(f"    {name:12} {bar:30} {weight:5.1%}")
        
        print(f"\n  Spike Statistics:")
        print(f"    Attention:  {spike_count_attn:8,.0f} spikes ({sparsity_attn:.1%} sparsity)")
        print(f"    FFN:        {spike_count_ffn:8,.0f} spikes ({sparsity_ffn:.1%} sparsity)")
        print(f"    Layer Total:{layer_total:8,.0f} spikes")
        
        total_spikes += layer_total
        
        # Forward through layer for next iteration
        with torch.no_grad():
            x = layer(x, pos_ids)
    
    print("\n" + "="*70)
    print(f"TOTAL SPIKES ACROSS ALL LAYERS: {total_spikes:,.0f}")
    print("="*70)
    
    return total_spikes


if __name__ == '__main__':
    total = analyze_spikes_detailed()