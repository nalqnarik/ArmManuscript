import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import numpy as np
import random


class ArmenianDataset(Dataset):
    def __init__(self, manifest_file, alphabet_file, img_dir, augment=False):
        """
        Armenian Handwriting Dataset.

        Args:
            manifest_file: Path to a tab-separated file: <img_name>\\t<label>
            alphabet_file: Path to a plain-text file containing all characters.
            img_dir:       Directory that contains the image files.
            augment:       If True, applies random augmentation (use for training only,
                           set to False for validation / evaluation).
        """
        with open(alphabet_file, 'r', encoding='utf-8') as f:
            self.alphabet = f.read()

        # Index 0 is reserved for the CTC blank token
        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.alphabet)}

        self.data = []
        unknown_chars = set()
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name, text = parts
                    # Validate every character at load time, not at __getitem__
                    for char in text:
                        if char not in self.char_to_idx:
                            unknown_chars.add(char)
                    self.data.append((img_name, text))

        if unknown_chars:
            print(
                f"WARNING: {len(unknown_chars)} character(s) found in labels "
                f"that are NOT in the alphabet and will be skipped:\n"
                f"  {sorted(unknown_chars)}"
            )

        self.img_dir = img_dir
        self.augment = augment
        self.target_h = 64   # fixed height expected by the CNN

    # ---------------------------------------------------------------------- #
    #  Augmentation helpers                                                    #
    # ---------------------------------------------------------------------- #
    def _random_affine(self, img: Image.Image) -> Image.Image:
        """Random rotation ±5° to mimic natural writing tilt."""
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
        return img

    def _random_blur(self, img: Image.Image) -> Image.Image:
        """Occasional blur (50% chance) to simulate smudging or low resolution."""
        if random.random() < 0.5:
            radius = random.uniform(0.3, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

    def _random_brightness(self, img_array: np.ndarray) -> np.ndarray:
        """Randomly shift pixel brightness ±40 to simulate ink/paper variation."""
        if random.random() < 0.7:
            delta = random.uniform(-40, 40)
            img_array = np.clip(img_array + delta, 0, 255)
        return img_array

    def _random_noise(self, img_array: np.ndarray) -> np.ndarray:
        """Add Gaussian noise (50% chance, σ=8) to prevent overfitting to clean images."""
        if random.random() < 0.5:
            noise = np.random.normal(0, 8, img_array.shape).astype(np.float32)
            img_array = np.clip(img_array + noise, 0, 255)
        return img_array

    def _random_scale_width(self, img: Image.Image) -> Image.Image:
        """Randomly stretch or compress the image horizontally (±25%)."""
        if random.random() < 0.6:
            factor = random.uniform(0.75, 1.25)
            new_w = max(1, int(img.width * factor))
            img = img.resize((new_w, img.height), Image.Resampling.LANCZOS)
        return img

    def _random_erode(self, img_array: np.ndarray) -> np.ndarray:
        """
        Randomly thin or thicken strokes (30% chance) to simulate different
        pen pressures and writing instruments.
        Requires scipy — install with: pip install scipy
        """
        if random.random() < 0.3:
            try:
                from scipy.ndimage import binary_erosion, binary_dilation
                binary = img_array > 127
                if random.random() < 0.5:
                    binary = binary_erosion(binary)   # thinner strokes
                else:
                    binary = binary_dilation(binary)  # thicker strokes
                img_array = binary.astype(np.float32) * 255
            except ImportError:
                pass  # scipy not installed — skip silently
        return img_array

    # ---------------------------------------------------------------------- #
    #  Core preprocessing                                                      #
    # ---------------------------------------------------------------------- #
    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        """
        Resize to target height (preserving aspect ratio), optionally augment,
        then normalize to [-1, 1] and return a [1, H, W] float tensor.
        """
        # --- Resize to fixed height ---
        w, h = img.size
        new_w = max(1, int(w * (self.target_h / h)))
        img = img.resize((new_w, self.target_h), Image.Resampling.LANCZOS)

        # --- Image-level augmentation (before converting to array) ---
        if self.augment:
            img = self._random_affine(img)
            img = self._random_blur(img)
            img = self._random_scale_width(img)

        img_array = np.array(img, dtype=np.float32)

        # --- Array-level augmentation ---
        if self.augment:
            img_array = self._random_brightness(img_array)
            img_array = self._random_noise(img_array)
            img_array = self._random_erode(img_array)

        # --- Normalize to [-1, 1] instead of hard binarization ---
        # Hard thresholding discards gradient information in ambiguous pixels.
        # Normalization lets the model learn what "ink" looks like.
        img_array = (img_array / 127.5) - 1.0   # [0, 255] → [-1.0, 1.0]

        return torch.FloatTensor(img_array).unsqueeze(0)  # [1, H, W]

    # ---------------------------------------------------------------------- #
    #  Dataset interface                                                       #
    # ---------------------------------------------------------------------- #
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, text = self.data[idx]
        img_path = self.img_dir + img_name

        img = Image.open(img_path).convert('L')
        img_tensor = self._preprocess(img)          # [1, 64, W]

        # Encode label — unknown characters are skipped (warned at __init__)
        label = [
            self.char_to_idx[char]
            for char in text
            if char in self.char_to_idx
        ]

        if len(label) == 0:
            raise ValueError(
                f"Label for '{img_name}' is empty after encoding. "
                f"Check that the text '{text}' contains valid alphabet characters."
            )

        label_tensor = torch.LongTensor(label)
        return img_tensor, label_tensor, len(label)


# --------------------------------------------------------------------------- #
#  Quick sanity-check                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os

    MANIFEST  = 'dataset/train_line_list.txt'
    ALPHABET  = 'dataset/alphabet.txt'
    IMG_DIR   = 'dataset/train/'

    if not os.path.exists(MANIFEST):
        print("Dataset files not found — skipping sanity check.")
    else:
        ds_train = ArmenianDataset(MANIFEST, ALPHABET, IMG_DIR, augment=True)
        ds_val   = ArmenianDataset(MANIFEST, ALPHABET, IMG_DIR, augment=False)

        img, label, length = ds_train[0]
        print(f"Image tensor shape : {img.shape}")
        print(f"Pixel range        : [{img.min():.2f}, {img.max():.2f}]")
        print(f"Label indices      : {label.tolist()}")
        print(f"Label length       : {length}")
        print("Sanity check passed ✅")