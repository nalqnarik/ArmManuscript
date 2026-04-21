import torch
import torch.nn as nn


class ArmenianCRNN(nn.Module):
    def __init__(self, num_chars, rnn_hidden=128, rnn_layers=2, rnn_dropout=0.5):
        """
        Armenian Handwriting CRNN with CTC loss support.

        Args:
            num_chars:    Number of characters in the alphabet (excluding CTC blank).
            rnn_hidden:   Hidden size for each LSTM direction (128 suits ~2500 samples).
            rnn_layers:   Number of stacked LSTM layers.
            rnn_dropout:  Dropout applied between LSTM layers (only when rnn_layers > 1).
        """
        super(ArmenianCRNN, self).__init__()

        # ------------------------------------------------------------------ #
        #  CNN backbone                                                        #
        #  Input:  [B, 1, 64, W]                                              #
        #  Output: [B, 256, 1, W']  (height collapsed by AdaptiveAvgPool2d)  #
        # ------------------------------------------------------------------ #
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # [B, 32, 32, W/2]
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # [B, 64, 16, W/4]
            nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),                          # [B, 128, 8, W/4]
            nn.Dropout2d(0.2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Collapse height to 1 safely regardless of input width
            nn.AdaptiveAvgPool2d((1, None)),               # [B, 256, 1, W/4]
        )

        # ------------------------------------------------------------------ #
        #  RNN head                                                            #
        # ------------------------------------------------------------------ #
        # dropout is applied between layers, so only works when rnn_layers > 1
        effective_dropout = rnn_dropout if rnn_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=False,          # expects [T, B, C]
            dropout=effective_dropout,
        )

        # Output projection: bidirectional → rnn_hidden * 2
        # +1 for CTC blank token (index 0)
        self.fc = nn.Linear(rnn_hidden * 2, num_chars + 1)

    # ---------------------------------------------------------------------- #
    #  Forward pass                                                            #
    # ---------------------------------------------------------------------- #
    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W]  — grayscale image batch, H should be 64.
        Returns:
            logits: [T, B, num_chars + 1]  — raw (pre-softmax) CTC scores.
        """
        x = self.cnn(x)        # [B, 256, 1, W']
        x = x.squeeze(2)       # [B, 256, W']
        x = x.permute(2, 0, 1) # [W', B, 256]  — time-first for LSTM
        x, _ = self.rnn(x)     # [W', B, rnn_hidden*2]
        return self.fc(x)      # [W', B, num_chars+1]


# --------------------------------------------------------------------------- #
#  Quick sanity-check                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ALPHABET_SIZE = 39  # adjust to your actual alphabet length

    model = ArmenianCRNN(num_chars=ALPHABET_SIZE)
    model.eval()

    dummy = torch.zeros(2, 1, 64, 256)   # batch=2, H=64, W=256
    with torch.no_grad():
        out = model(dummy)

    T, B, C = out.shape
    print(f"Output shape : T={T}, B={B}, C={C}")
    print(f"  T (time steps) = W/4 = {256 // 4}")
    print(f"  C (classes)    = alphabet + blank = {ALPHABET_SIZE + 1}")
    print(f"  Parameters     : {sum(p.numel() for p in model.parameters()):,}")
    print("Sanity check passed ✅")