"""
Optimized for Colab: 125M parameters, 8-10 hour training.
"""

def create_medium_model_optimized():
    """125M parameters - perfect for Colab."""
    return SpikingTransformer(
        vocab_size=50257,
        d_model=768,
        nhead=12,
        num_layers=12,
        dim_feedforward=3072,
        T=4,
        max_seq_len=512,  # Shorter for speed
        dropout=0.1,
        fixed_encoding=None
    )


def train_125m_colab(num_epochs=5, checkpoint_every=500):
    """
    Optimized training for Colab.
    Will complete in ~8 hours.
    """
    
    from google.colab import drive
    drive.mount('/content/drive')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Smaller, faster dataset
    train_data, val_data = prepare_dataset_fast(
        num_samples=10000,  # 10K samples
        max_length=256      # Shorter sequences
    )
    
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=2)
    
    # Model
    model = create_medium_model_optimized()
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')
    
    # Resume from checkpoint if exists
    checkpoint_dir = Path('/content/drive/MyDrive/spike_checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = checkpoint_dir / 'latest.pt'
    
    start_epoch = 0
    history = {'train_loss': [], 'val_loss': []}
    
    if resume_path.exists():
        print("Resuming from checkpoint...")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
    
    # Training
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1)
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # CHECKPOINT EVERY N BATCHES
            if (batch_idx + 1) % checkpoint_every == 0:
                torch.save({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'train_loss': train_loss / (batch_idx + 1)
                }, checkpoint_dir / 'latest.pt')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                with torch.cuda.amp.autocast():
                    logits = model(input_ids)
                    loss = nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        input_ids[:, 1:].reshape(-1)
                    )
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        
        print(f"Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}")
        
        # Save epoch checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, checkpoint_dir / f'epoch_{epoch+1}.pt')
    
    print("\nTraining complete!")
    return model, history


def prepare_dataset_fast(num_samples=10000, max_length=256):
    """Fast dataset preparation."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    def tokenize(examples):
        texts = [t for t in examples['text'] if len(t.strip()) > 100]
        if not texts:
            return {'input_ids': []}
        return tokenizer(texts, truncation=True, max_length=max_length, 
                        padding='max_length')
    
    train = dataset['train'].select(range(num_samples)).map(
        tokenize, batched=True, remove_columns=['text']
    )
    val = dataset['validation'].select(range(num_samples//10)).map(
        tokenize, batched=True, remove_columns=['text']
    )
    
    train.set_format('torch', columns=['input_ids'])
    val.set_format('torch', columns=['input_ids'])
    
    return train, val
