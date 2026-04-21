import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
import os
import time

from dataset import ArmenianDataset
from model import ArmenianCRNN


# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #
DEVICE               = torch.device('cpu')
BATCH_SIZE           = 8
EPOCHS               = 500
LR                   = 3e-4   # was 1e-4
GRAD_CLIP            = 5.0          # max gradient norm — critical for RNNs
EARLY_STOP_PATIENCE  = 75           # stop if val loss doesn't improve for 50 epochs
SAVE_DIR             = 'models'
BEST_MODEL           = os.path.join(SAVE_DIR, 'model_best.pth')
LAST_MODEL           = os.path.join(SAVE_DIR, 'model_last.pth')

# Paths
ALPHABET_FILE   = 'dataset/alphabet.txt'
TRAIN_MANIFEST  = 'dataset/train_line_list.txt'
TRAIN_IMG_DIR   = 'dataset/train/'
VAL_MANIFEST    = 'dataset/val_line_list.txt'
VAL_IMG_DIR     = 'dataset/val/'


# --------------------------------------------------------------------------- #
#  Collate                                                                     #
# --------------------------------------------------------------------------- #
def collate_fn(batch):
    imgs, labels, label_lens = zip(*batch)

    max_width = max(img.size(2) for img in imgs)
    padded_imgs = [
        torch.nn.functional.pad(img, (0, max_width - img.size(2)))
        for img in imgs
    ]

    imgs       = torch.stack(padded_imgs, 0)
    labels     = pad_sequence(labels, batch_first=True, padding_value=0)
    label_lens = torch.IntTensor(label_lens)
    return imgs, labels, label_lens


# --------------------------------------------------------------------------- #
#  CTC greedy decoder                                                          #
# --------------------------------------------------------------------------- #
def ctc_decode(preds, alphabet):
    """
    Greedy CTC decode: collapse repeated indices, remove blank (index 0).

    Args:
        preds:    [T, B, C] raw logits tensor
        alphabet: string of characters (index 1-based)
    Returns:
        List of decoded strings, one per batch item.
    """
    indices = torch.argmax(preds, dim=2)   # [T, B]
    results = []
    for b in range(indices.size(1)):
        seq = indices[:, b].cpu().numpy()
        chars, last = [], 0
        for idx in seq:
            if idx != 0 and idx != last:
                chars.append(alphabet[idx - 1])
            last = idx
        results.append("".join(chars))
    return results


# --------------------------------------------------------------------------- #
#  One epoch of training or validation                                         #
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, optimizer, alphabet, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    skipped    = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (imgs, labels, label_lens) in enumerate(loader):
            imgs       = imgs.to(DEVICE)
            labels     = labels.to(DEVICE)
            label_lens = label_lens.to(DEVICE)

            preds      = model(imgs)                                     # [T, B, C]
            preds_size = torch.IntTensor([preds.size(0)] * preds.size(1)).to(DEVICE)

            # CTCLoss expects log-probabilities
            log_probs = torch.nn.functional.log_softmax(preds, dim=2)
            loss = criterion(log_probs, labels, preds_size, label_lens)

            # Guard: skip batches that produce inf/nan loss
            if torch.isinf(loss) or torch.isnan(loss):
                skipped += 1
                if train:
                    optimizer.zero_grad()
                continue

            if train:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()

            if train and i % 10 == 0:
                example = ctc_decode(preds[:, :1, :], alphabet)[0]
                print(f"  Batch {i:>4} | Loss: {loss.item():.4f} | Sample: '{example}'")

    avg_loss = total_loss / max(len(loader) - skipped, 1)
    if skipped:
        print(f"  ⚠️  Skipped {skipped} batch(es) due to inf/nan loss.")
    return avg_loss


# --------------------------------------------------------------------------- #
#  Main training loop                                                          #
# --------------------------------------------------------------------------- #
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load alphabet
    with open(ALPHABET_FILE, 'r', encoding='utf-8') as f:
        alphabet = f.read()

    # Datasets — augment only for training
    train_ds = ArmenianDataset(TRAIN_MANIFEST, ALPHABET_FILE, TRAIN_IMG_DIR, augment=True)
    val_ds   = ArmenianDataset(VAL_MANIFEST,   ALPHABET_FILE, VAL_IMG_DIR,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # Model
    model     = ArmenianCRNN(len(alphabet)).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # verbose removed — newer PyTorch versions don't support it
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    best_val_loss    = float('inf')
    epochs_no_improve = 0

    print(f"Training on {len(train_ds)} samples | Validating on {len(val_ds)} samples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, alphabet, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, alphabet, train=False)

        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:>4}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # LR scheduler step (based on val loss)
        scheduler.step(epoch)

        # Save last checkpoint every epoch (for resuming)
        torch.save({
            'epoch':      epoch,
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'val_loss':   val_loss,
        }, LAST_MODEL)

        # Save best checkpoint and track early stopping
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch':    epoch,
                'model':    model.state_dict(),
                'val_loss': val_loss,
            }, BEST_MODEL)
            print(f"  ✅ New best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.4f}")
                break

        print()  # blank line between epochs


# --------------------------------------------------------------------------- #
#  Resume training from last checkpoint                                        #
# --------------------------------------------------------------------------- #
def resume():
    """Call with --resume flag to continue from LAST_MODEL."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(ALPHABET_FILE, 'r', encoding='utf-8') as f:
        alphabet = f.read()

    train_ds = ArmenianDataset(TRAIN_MANIFEST, ALPHABET_FILE, TRAIN_IMG_DIR, augment=True)
    val_ds   = ArmenianDataset(VAL_MANIFEST,   ALPHABET_FILE, VAL_IMG_DIR,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model     = ArmenianCRNN(len(alphabet)).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    checkpoint  = torch.load(LAST_MODEL, map_location=DEVICE)
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['val_loss']

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    print(f"Resumed from epoch {start_epoch - 1} (val_loss={best_val_loss:.4f})\n")

    epochs_no_improve = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, alphabet, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, alphabet, train=False)

        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:>4}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        scheduler.step(val_loss)

        torch.save({
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss':  val_loss,
        }, LAST_MODEL)

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch':    epoch,
                'model':    model.state_dict(),
                'val_loss': val_loss,
            }, BEST_MODEL)
            print(f"  ✅ New best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.4f}")
                break

        print()


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--resume':
        resume()
    else:
        train()