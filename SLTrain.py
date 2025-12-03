# SLTrain.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
import numpy as np
import logging
import argparse
import os
import sys
import time

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [TRAIN] - %(message)s')

try:
    from config import CONF
    from model import AlphaCerberusNet
except ImportError:
    print("Error: Cannot import 'config' or 'model'.")
    sys.exit(1)

# --- RAM Dataset ---
class GomokuRAMDataset(Dataset):
    def __init__(self, npz_path, split_indices=None):
        logging.info("Loading data into RAM...")
        t0 = time.time()
        try:
            with np.load(npz_path) as f:
                self.states = f['states']
                self.policies = f['policies']
                self.values = f['values']
            
            if split_indices is not None:
                self.states = self.states[split_indices]
                self.policies = self.policies[split_indices]
                self.values = self.values[split_indices]
                
            logging.info(f"Data loaded! Time: {time.time()-t0:.2f}s. Samples: {len(self.states)}")
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            sys.exit(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # Convert to Tensor
        s = torch.from_numpy(self.states[idx]) 
        p = torch.from_numpy(self.policies[idx])
        v = torch.from_numpy(self.values[idx])
        return s, p, v

def create_dataloaders(npz_path, val_split_ratio, batch_size):
    try:
        with np.load(npz_path, mmap_mode='r') as f:
            total_len = len(f['states'])
    except FileNotFoundError:
        logging.error(f"File not found: {npz_path}")
        return None, None
    
    indices = np.arange(total_len)
    np.random.shuffle(indices)
    val_size = int(total_len * val_split_ratio)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    logging.info("--- Initializing Training Set ---")
    train_dataset = GomokuRAMDataset(npz_path, train_indices)
    
    logging.info("--- Initializing Validation Set ---")
    val_dataset = GomokuRAMDataset(npz_path, val_indices)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 4, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    return train_loader, val_loader

def validate(model, loader, p_loss_fn, v_loss_fn, device, limit_batches=None):
    model.eval()
    total_val_loss = 0.0
    steps_run = 0
    
    with torch.inference_mode():
        for i, (states, target_policies, target_values) in enumerate(loader):
            if limit_batches is not None and i >= limit_batches: break
            
            states = states.to(device, non_blocking=True).float()
            target_policies = target_policies.to(device, non_blocking=True)
            target_values = target_values.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                policy_logits, values = model(states)
                
                # Standard LogSoftmax + KLDiv
                log_probs = F.log_softmax(policy_logits, dim=1)
                loss_p = p_loss_fn(log_probs, target_policies)
                loss_v = v_loss_fn(values.view(-1), target_values.view(-1))
                loss = loss_p + loss_v
            
            total_val_loss += loss.item()
            steps_run += 1
            
    return total_val_loss / steps_run if steps_run > 0 else 0

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss):
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, path)

def main(args):
    device = CONF.DEVICE
    print(f"\n>>> Device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    log_dir = os.path.join("runs", f"sl_run_{int(time.time())}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # 1. Data
    train_loader, val_loader = create_dataloaders(args.data_path, args.val_split, args.batch_size)
    if not train_loader: return

    # 2. Model
    model = AlphaCerberusNet().to(device)
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=CONF.WEIGHT_DECAY)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # 4. Resume Logic
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if args.resume_path:
        if os.path.exists(args.resume_path):
            logging.info(f"Loading checkpoint: {args.resume_path}")
            try:
                checkpoint = torch.load(args.resume_path, map_location=device)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint:
                    try: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except: logging.warning("Scheduler load failed, resetting.")

                if 'scaler_state_dict' in checkpoint and device.type == 'cuda':
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                start_epoch = checkpoint['epoch'] + 1
                global_step = checkpoint['global_step']
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                
                logging.info(f"✅ Resumed! Start Epoch: {start_epoch}, Best Loss: {best_val_loss:.4f}")
            except Exception as e:
                logging.error(f"❌ Resume failed: {e}")
                logging.error("Check if checkpoint matches current architecture.")
                return
        else:
            logging.warning(f"Checkpoint not found: {args.resume_path}. Starting fresh.")
    else:
        logging.info("No checkpoint provided. Starting fresh.")

    # 5. Loss Function
    p_fn = nn.KLDivLoss(reduction='batchmean')
    v_fn = nn.MSELoss()

    # 6. Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for i, (states, target_policies, target_values) in enumerate(train_loader):
            global_step += 1
            
            states = states.to(device, non_blocking=True).float()
            target_policies = target_policies.to(device, non_blocking=True)
            target_values = target_values.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                policy_logits, values = model(states)
                
                log_probs = F.log_softmax(policy_logits, dim=1)
                loss_p = p_fn(log_probs, target_policies)
                loss_v = v_fn(values.view(-1), target_values.view(-1))
                loss = loss_p + loss_v
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if global_step % args.log_freq == 0:
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                print(f"\rStep {global_step} | Loss: {loss.item():.4f} (P:{loss_p.item():.4f} V:{loss_v.item():.4f}) | LR: {lr:.6f}", end="")

            if global_step % args.val_check_interval == 0:
                val_loss = validate(model, val_loader, p_fn, v_fn, device, limit_batches=args.val_limit)
                writer.add_scalar('Val/Loss', val_loss, global_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(args.best_model_path, model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss)
                    logging.info(f"\n[Save] Best: {best_val_loss:.4f}")

        save_checkpoint(args.last_model_path, model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss)
        logging.info(f"\nEpoch {epoch+1} Done.")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help="Path to .npz data file")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--val_check_interval', type=int, default=500)
    parser.add_argument('--val_limit', type=int, default=50)
    parser.add_argument('--log_freq', type=int, default=10)
    
    parser.add_argument('--resume_path', type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument('--best_model_path', type=str, default='sl_best.ckpt')
    parser.add_argument('--last_model_path', type=str, default='sl_last.ckpt')

    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist.")
        sys.exit(1)
    main(args)