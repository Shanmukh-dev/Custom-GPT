import torch
from torch import nn
from torch.amp import autocast

def train_step(model, train_dataloader, train_iter, loss_fn, optimizer, scaler, device):
    try:
        X_train, y_train = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataloader)
        X_train, y_train = next(train_iter)

    X_train, y_train = X_train.to(device), y_train.to(device)
    
    model.train()
    optimizer.zero_grad()
    with autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
    
        logits = model(X_train)
        
        train_loss = loss_fn(logits.view(-1, logits.size(-1)), y_train.view(-1))
    
    scaler.scale(train_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return train_loss, train_iter

def test_step(model, test_dl, n_steps, loss_fn, device):
    from tqdm.auto import tqdm
    model.eval()
    test_loss = 0
    with torch.inference_mode():

        for idx, (X_test, y_test) in tqdm(enumerate(test_dl)):
            if idx == n_steps-1:
              break
            X_test, y_test = X_test.to(device), y_test.to(device)
            test_logits = model(X_test)
            
            curr_loss = loss_fn(test_logits.view(-1, test_logits.size(-1)), y_test.view(-1))
            
            test_loss += curr_loss.item()


    test_loss = test_loss / n_steps

    return test_loss

def calc_params(model):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")