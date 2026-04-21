import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os

from model import ArmenianCRNN


# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #
VAL_DIR       = 'dataset/val/'
LABEL_FILE    = 'dataset/val_line_list.txt'
MODEL_PATH    = 'models/model_best.pth'       # use the best checkpoint
ALPHABET_PATH = 'dataset/alphabet.txt'
TARGET_H      = 64


# --------------------------------------------------------------------------- #
#  Edit distance (for CER)                                                     #
# --------------------------------------------------------------------------- #
def edit_distance(a: str, b: str) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def cer(pred: str, target: str) -> float:
    """Character Error Rate: edit_distance / len(target). Returns 0 if both empty."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, target) / len(target)


# --------------------------------------------------------------------------- #
#  Preprocessing  (identical to dataset.py — no augmentation)                 #
# --------------------------------------------------------------------------- #
def preprocess(img: Image.Image, target_h: int = TARGET_H) -> torch.Tensor:
    """
    Resize to fixed height preserving aspect ratio, then normalize to [-1, 1].
    Returns a [1, 1, H, W] tensor ready for the model.
    """
    w, h = img.size
    new_w = max(1, int(w * (target_h / h)))
    img   = img.resize((new_w, target_h), Image.Resampling.LANCZOS)

    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0          # [-1, 1]

    tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return tensor


# --------------------------------------------------------------------------- #
#  CTC greedy decoder                                                          #
# --------------------------------------------------------------------------- #
def ctc_decode(preds: torch.Tensor, alphabet: str) -> str:
    """
    Greedy decode: argmax over classes, collapse repeats, remove blank (0).

    Args:
        preds:    [T, 1, C] logits for a single image
        alphabet: character string (1-indexed, index 0 = blank)
    Returns:
        Decoded string.
    """
    indices = torch.argmax(preds, dim=2).squeeze(1).cpu().numpy()   # [T]
    chars, last = [], 0
    for idx in indices:
        if idx != 0 and idx != last:
            chars.append(alphabet[idx - 1])
        last = idx
    return "".join(chars)


# --------------------------------------------------------------------------- #
#  Main evaluation                                                             #
# --------------------------------------------------------------------------- #
def evaluate():
    # --- Load alphabet ---
    with open(ALPHABET_PATH, 'r', encoding='utf-8') as f:
        alphabet = f.read()

    # --- Load ground-truth labels ---
    true_labels = {}
    with open(LABEL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                true_labels[parts[0]] = parts[1]

    # --- Load model ---
    device = torch.device('cpu')
    model  = ArmenianCRNN(len(alphabet)).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Support both raw state_dict and checkpoint dicts saved by train_cpu.py
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    saved_epoch = checkpoint.get('epoch', '?')
    saved_loss  = checkpoint.get('val_loss', '?')
    print(f"Loaded checkpoint — epoch: {saved_epoch} | val_loss: {saved_loss}")
    print(f"Evaluating on {len(true_labels)} samples...\n")

    # --- Metrics ---
    exact_correct = 0
    total_cer     = 0.0
    total         = 0
    errors        = []   # collect mismatches for summary at the end

    for img_name, true_text in true_labels.items():
        img_path = os.path.join(VAL_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"  ⚠️  Missing: {img_path}")
            continue

        img   = Image.open(img_path).convert('L')
        img_t = preprocess(img).to(device)          # single fixed preprocessing

        with torch.no_grad():
            preds     = model(img_t)                # [T, 1, C]
            pred_text = ctc_decode(preds, alphabet)

        sample_cer   = cer(pred_text, true_text)
        is_exact     = (pred_text == true_text)

        if is_exact:
            exact_correct += 1
            print(f"  ✅ {img_name:<30} | '{true_text}'")
        else:
            errors.append((img_name, true_text, pred_text, sample_cer))
            print(f"  ❌ {img_name:<30} | Expected: '{true_text}' | Got: '{pred_text}' | CER: {sample_cer:.2%}")

        total_cer += sample_cer
        total     += 1

    if total == 0:
        print("No samples evaluated — check your val directory and label file.")
        return

    # --- Summary ---
    avg_cer          = total_cer / total
    exact_acc        = exact_correct / total
    worst_errors     = sorted(errors, key=lambda x: x[3], reverse=True)[:10]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total samples       : {total}")
    print(f"  Exact match         : {exact_correct} / {total}  ({exact_acc:.2%})")
    print(f"  Average CER         : {avg_cer:.2%}")
    print(f"  Perfect CER (0%)    : {sum(1 for e in errors if e[3] == 0.0) + exact_correct} samples")

    if worst_errors:
        print(f"\n  Worst predictions (top {len(worst_errors)}):")
        for img_name, true_text, pred_text, sample_cer in worst_errors:
            print(f"    {img_name:<30} | CER {sample_cer:.2%} | '{true_text}' → '{pred_text}'")

    print("=" * 60)


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    evaluate()