# Add this diagnostic code right after your training loop starts

print("\n" + "="*80)
print("DIAGNOSTIC CHECKS")
print("="*80)

# CHECK 1: Model Parameters
print("\n=== Model Parameters Check ===")
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")

print(f"\nTotal trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# CHECK 2: Data shapes
print("\n=== Data Shape Check ===")
print(f"Edge index shape: {edge_index_cpu.shape}")
print(f"Edge weight shape: {edge_weight_cpu.shape}")
print(f"Edge index max: {edge_index_cpu.max().item()}, min: {edge_index_cpu.min().item()}")
print(f"Num nodes: {n_providers + n_codes}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Edge index device: {edge_index_cpu.device}")
print(f"Edge weight device: {edge_weight_cpu.device}")

print("\n" + "="*80)
print("STARTING TRAINING WITH DIAGNOSTICS")
print("="*80)

epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Encode
    z = model.encode(edge_index_cpu, edge_weight_cpu)
    
    # CHECK 3: Z Statistics
    print(f"\n=== Epoch {epoch} ===")
    print(f"z shape: {z.shape}")
    print(f"z mean: {z.mean().item():.6f}, std: {z.std().item():.6f}")
    print(f"z min: {z.min().item():.6f}, max: {z.max().item():.6f}")
    print(f"z requires_grad: {z.requires_grad}")
    print(f"z contains NaN: {torch.isnan(z).any().item()}")
    print(f"z contains Inf: {torch.isinf(z).any().item()}")
    
    # Compute losses
    recon_loss = model.recon_loss(z, edge_index_cpu)
    kl_loss = model.kl_loss() / (n_providers + n_codes)
    
    print(f"Recon loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.6f}")
    
    loss = recon_loss + kl_loss
    
    loss.backward()
    
    # CHECK 4: Gradients
    print("\n=== Gradient Check ===")
    has_grad = False
    total_grad = 0
    num_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_mean = param.grad.abs().mean().item()
            total_grad += grad_mean
            num_params += 1
            print(f"{name}: grad mean = {grad_mean:.8f}")
        else:
            print(f"{name}: NO GRADIENT")
    
    if has_grad:
        print(f"\nAverage gradient: {total_grad/num_params:.8f}")
    else:
        print("\n!!! NO GRADIENTS - MODEL NOT LEARNING !!!")
    
    optimizer.step()
    
    print("-"*80)
