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
MODEL_PATH    = 'models/model_best.pth'
ALPHABET_PATH = 'dataset/alphabet.txt'
TARGET_H      = 64
BEAM_WIDTH    = 10   # increase for better accuracy at cost of speed (try 10-25)


# --------------------------------------------------------------------------- #
#  Edit distance (for CER) — no external dependencies                         #
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
    """Character Error Rate: edit_distance / len(target)."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, target) / len(target)


# --------------------------------------------------------------------------- #
#  Preprocessing — identical to dataset.py, no augmentation                   #
# --------------------------------------------------------------------------- #
def preprocess(img: Image.Image, target_h: int = TARGET_H) -> torch.Tensor:
    """
    Resize to fixed height preserving aspect ratio, normalize to [-1, 1].
    Returns [1, 1, H, W] tensor.
    """
    w, h = img.size
    new_w = max(1, int(w * (target_h / h)))
    img   = img.resize((new_w, target_h), Image.Resampling.LANCZOS)

    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0          # [-1, 1]

    return torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


# --------------------------------------------------------------------------- #
#  Greedy decoder (kept for comparison)                                        #
# --------------------------------------------------------------------------- #
def ctc_greedy(preds: torch.Tensor, alphabet: str) -> str:
    """
    Greedy CTC decode: argmax at each timestep, collapse repeats, remove blank.

    Args:
        preds:    [T, 1, C] logits
        alphabet: character string (1-indexed, index 0 = blank)
    """
    indices = torch.argmax(preds, dim=2).squeeze(1).cpu().numpy()
    chars, last = [], 0
    for idx in indices:
        if idx != 0 and idx != last:
            chars.append(alphabet[idx - 1])
        last = idx
    return "".join(chars)


# --------------------------------------------------------------------------- #
#  Beam search decoder                                                         #
# --------------------------------------------------------------------------- #
def ctc_beam_search(preds: torch.Tensor, alphabet: str, beam_width: int = BEAM_WIDTH) -> str:
    """
    Beam search CTC decoder — considers multiple character paths simultaneously,
    consistently outperforms greedy on visually similar Armenian letterforms.

    Args:
        preds:      [T, 1, C] logits for a single image
        alphabet:   character string (1-indexed, index 0 = blank)
        beam_width: number of beams to keep at each timestep (default 10)
    Returns:
        Best decoded string.
    """
    log_probs = F.log_softmax(preds.squeeze(1), dim=-1)  # [T, C]
    T, C = log_probs.shape

    # Each beam stored as (score, last_char_idx, decoded_text)
    beams = [(0.0, 0, "")]

    for t in range(T):
        new_beams = {}

        for score, last_char, text in beams:
            for c in range(C):
                c_score   = log_probs[t, c].item()
                new_score = score + c_score

                if c == 0:
                    # Blank token — text doesn't change, last_char resets to 0
                    key = (0, text)
                    if key not in new_beams or new_beams[key][0] < new_score:
                        new_beams[key] = (new_score, 0, text)

                elif c == last_char:
                    # Same character repeated without a blank in between — don't extend
                    key = (c, text)
                    if key not in new_beams or new_beams[key][0] < new_score:
                        new_beams[key] = (new_score, c, text)

                else:
                    # New character — extend the text
                    new_text = text + alphabet[c - 1]
                    key      = (c, new_text)
                    if key not in new_beams or new_beams[key][0] < new_score:
                        new_beams[key] = (new_score, c, new_text)

        # Keep only the top beam_width beams by score
        beams = sorted(new_beams.values(), key=lambda x: x[0], reverse=True)[:beam_width]

    return beams[0][2] if beams else ""


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
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    saved_epoch = checkpoint.get('epoch', '?')
    saved_loss  = checkpoint.get('val_loss', '?')
    print(f"Loaded checkpoint — epoch: {saved_epoch} | val_loss: {saved_loss}")
    print(f"Beam width: {BEAM_WIDTH}")
    print(f"Evaluating on {len(true_labels)} samples...\n")

    # --- Metrics ---
    greedy_exact  = 0
    beam_exact    = 0
    greedy_cer    = 0.0
    beam_cer      = 0.0
    total         = 0
    errors        = []

    for img_name, true_text in true_labels.items():
        img_path = os.path.join(VAL_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"  ⚠️  Missing: {img_path}")
            continue

        img   = Image.open(img_path).convert('L')
        img_t = preprocess(img).to(device)

        with torch.no_grad():
            preds = model(img_t)      
            
        preds = preds / 1.5   # temperature — try values 1.2 to 2.0                      # [T, 1, C]

        greedy_pred = ctc_greedy(preds, alphabet)
        beam_pred   = ctc_beam_search(preds, alphabet, BEAM_WIDTH)

        sample_greedy_cer = cer(greedy_pred, true_text)
        sample_beam_cer   = cer(beam_pred,   true_text)
        beam_is_exact     = (beam_pred == true_text)

        greedy_exact += int(greedy_pred == true_text)
        beam_exact   += int(beam_is_exact)
        greedy_cer   += sample_greedy_cer
        beam_cer     += sample_beam_cer

        if beam_is_exact:
            print(f"  ✅ {img_name:<35} | '{true_text}'")
        else:
            errors.append((img_name, true_text, beam_pred, sample_beam_cer))
            improved = " ↑" if sample_beam_cer < sample_greedy_cer else ""
            print(
                f"  ❌ {img_name:<35} | "
                f"Expected: '{true_text}' | "
                f"Got: '{beam_pred}' | "
                f"CER: {sample_beam_cer:.2%}{improved}"
            )

        total += 1

    if total == 0:
        print("No samples evaluated — check your val directory and label file.")
        return

    # --- Summary ---
    avg_greedy_cer = greedy_cer / total
    avg_beam_cer   = beam_cer   / total
    worst_errors   = sorted(errors, key=lambda x: x[3], reverse=True)[:10]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total samples       : {total}")
    print(f"")
    print(f"  Greedy decoder:")
    print(f"    Exact match       : {greedy_exact} / {total}  ({greedy_exact/total:.2%})")
    print(f"    Average CER       : {avg_greedy_cer:.2%}")
    print(f"")
    print(f"  Beam search (w={BEAM_WIDTH}):")
    print(f"    Exact match       : {beam_exact} / {total}  ({beam_exact/total:.2%})")
    print(f"    Average CER       : {avg_beam_cer:.2%}")
    print(f"")
    cer_gain   = avg_greedy_cer - avg_beam_cer
    exact_gain = beam_exact - greedy_exact
    print(f"  Beam search gain    : +{exact_gain} exact matches | CER -{cer_gain:.2%}")

    if worst_errors:
        print(f"\n  Worst predictions (top {len(worst_errors)}):")
        for img_name, true_text, pred_text, sample_cer in worst_errors:
            print(f"    {img_name:<35} | CER {sample_cer:.2%} | '{true_text}' → '{pred_text}'")

    print("=" * 60)


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    evaluate()