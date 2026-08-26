"""Tiny Spectral Language Model (Prompt 5).

Defines ``TinySpectralLM`` — a 2-layer spectral transformer language model
targeting roughly **100K trainable parameters** — together with training and
text-generation utilities and a command-line entry point.

Architecture::

    tokens → token embedding → 2× SpectralTransformerBlock → LayerNorm
           → unembedding (d_model → vocab_size) → logits

The model is a *character-level* language model: the vocabulary is the set of
unique bytes/characters seen in the training data.  Because the transformer
blocks use spectral mixing (AFNO by default), the model enjoys **resolution
invariance** — it can be trained on one context length and sampled on a
longer one without parameter changes.

Example
-------
Run from the repository root::

    python -m spectral_silicon.model --train --steps 1000
    python -m spectral_silicon.model --prompt "ROMEO:"

The training entry point uses a tiny inline dataset (a short Shakespeare
excerpt) so the model is self-contained and runnable with no external files.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_silicon.transformer import SpectralTransformerBlock


__all__ = [
    "TinySpectralLM",
    "count_parameters",
    "train_model",
    "generate",
    "main",
]


# ---------------------------------------------------------------------------
# Default hyper-parameters — tuned for ~100K parameters.
# ---------------------------------------------------------------------------
DEFAULT_VOCAB_SIZE = 128      # byte-level vocabulary
DEFAULT_D_MODEL = 64
DEFAULT_N_LAYERS = 2
DEFAULT_N_HEADS = 4            # unused by spectral mixer, kept for API compat
DEFAULT_SEQ_LEN = 512
DEFAULT_NUM_MODES = 13         # retained Fourier modes (≈100K params w/ AFNO)
DEFAULT_BLOCK_SIZE = 16
DEFAULT_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Tiny inline training corpus (a short Shakespeare excerpt).  Using bytes as
# the token space keeps vocab_size ≤ 128 and avoids any external data files.
# ---------------------------------------------------------------------------
TINY_CORPUS = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. To die—to sleep,\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to: 'tis a consummation\n"
    "Devoutly to be wish'd. To die, to sleep;\n"
    "To sleep, perchance to dream—ay, there's the rub:\n"
    "For in that sleep of death what dreams may come,\n"
    "When we have shuffled off this mortal coil,\n"
    "Must give us pause. There's the respect\n"
    "That makes calamity of so long a life.\n"
    "For who would bear the whips and scorns of time,\n"
    "The oppressor's wrong, the proud man's contumely,\n"
    "The pangs of despised love, the law's delay,\n"
    "The insolence of office and the spurns\n"
    "That patient merit of the unworthy takes,\n"
    "When he himself might his quietus make\n"
    "With a bare bodkin? Who would these fardels bear,\n"
    "To grunt and sweat under a weary life,\n"
    "But that the dread of something after death,\n"
    "The undiscover'd country, from whose bourn\n"
    "No traveller returns, puzzles the will,\n"
    "And makes us rather bear those ills we have\n"
    "Than fly to others that we know not of?\n"
)


class TinySpectralLM(nn.Module):
    """A tiny character-level spectral-transformer language model.

    The model stacks ``n_layers`` :class:`SpectralTransformerBlock` instances
    between a token embedding and a linear unembedding head.  With the default
    configuration (``vocab=128, d_model=64, 2 layers, num_modes=13,
    block_size=16``) the total parameter count is ~100K.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary (e.g. 128 for byte-level).
    d_model : int
        Hidden width of the transformer.
    n_layers : int
        Number of spectral transformer blocks.
    n_heads : int
        Number of heads (unused by the spectral mixer, kept for API
        compatibility with standard transformer configs).
    seq_len : int
        Nominal sequence length used when constructing the blocks.
    mixer_type : {"afno", "fftnet"}
        Which spectral mixing layer the blocks use.
    num_modes : int
        Number of retained Fourier modes in the spectral mixer.
    block_size : int
        Block-diagonal width for the AFNO mixer.
    threshold : float
        Soft-thresholding cut-off for spectral coefficients.
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        d_model: int = DEFAULT_D_MODEL,
        n_layers: int = DEFAULT_N_LAYERS,
        n_heads: int = DEFAULT_N_HEADS,
        seq_len: int = DEFAULT_SEQ_LEN,
        mixer_type: str = "afno",
        num_modes: int = DEFAULT_NUM_MODES,
        block_size: int = DEFAULT_BLOCK_SIZE,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.mixer_type = mixer_type

        # Token embedding (no positional embedding needed — spectral mixing
        # operates on the whole sequence and does not rely on absolute
        # positions).
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Spectral transformer blocks.
        self.blocks = nn.ModuleList([
            SpectralTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                seq_len=seq_len,
                mixer_type=mixer_type,
                num_modes=num_modes,
                block_size=block_size,
                threshold=threshold,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm before unembedding.
        self.norm = nn.LayerNorm(d_model)

        # Unembedding head: d_model -> vocab_size.  Weight-tied to the token
        # embedding is an option, but we keep a separate head so the parameter
        # count lands near the ~100K target with the default config.
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of token sequences.

        Parameters
        ----------
        tokens : torch.Tensor
            Long tensor of shape ``(batch, seq_len)`` containing token ids.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        x = self.token_embed(tokens)            # (B, S, d_model)
        for block in self.blocks:
            x = block(x)                        # (B, S, d_model)
        x = self.norm(x)                        # (B, S, d_model)
        logits = self.unembed(x)                # (B, S, vocab_size)
        return logits

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters of ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data helpers (character-level, byte vocabulary)
# ---------------------------------------------------------------------------
def build_vocab(text: str) -> Tuple[List[str], dict, dict]:
    """Build a character-level vocabulary from ``text``.

    Returns
    -------
    chars : list of str
        Sorted list of unique characters.
    stoi : dict
        Character → token id.
    itos : dict
        Token id → character.
    """
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


def encode(text: str, stoi: dict) -> List[int]:
    """Encode a string into a list of token ids."""
    return [stoi[c] for c in text if c in stoi]


def decode(ids: List[int], itos: dict) -> str:
    """Decode a list of token ids back into a string."""
    return "".join(itos[i] for i in ids if i in itos)


def get_batches(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Slice a 1-D token tensor into contiguous training sequences.

    Parameters
    ----------
    data : torch.Tensor
        1-D long tensor of token ids.
    batch_size : int
        Number of sequences per batch.
    seq_len : int
        Length of each sequence.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch_size, seq_len)``.
    """
    n = data.numel()
    if n < batch_size * seq_len + 1:
        raise ValueError(
            f"Corpus too small ({n} tokens) for batch_size={batch_size} "
            f"and seq_len={seq_len}."
        )
    # Random offsets for each sample in the batch.
    starts = torch.randint(0, n - seq_len - 1, (batch_size,))
    batch = torch.stack([data[s:s + seq_len] for s in starts])
    return batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    steps: int = 1000,
    *,
    text: str = TINY_CORPUS,
    batch_size: int = 16,
    seq_len: int = 64,
    lr: float = 3e-4,
    device: Optional[str] = None,
    verbose: bool = True,
    model: Optional[TinySpectralLM] = None,
    vocab: Optional[Tuple[List[str], dict, dict]] = None,
) -> Tuple[TinySpectralLM, dict, List[float]]:
    """Train ``TinySpectralLM`` for a fixed number of steps.

    The model is trained as a character-level language model with next-token
    cross-entropy loss and the Adam optimizer.

    Parameters
    ----------
    steps : int
        Number of gradient-update steps.
    text : str
        Training corpus (character-level).
    batch_size : int
        Sequences per mini-batch.
    seq_len : int
        Length of each training sequence.
    lr : float
        Learning rate for Adam.
    device : str, optional
        ``"cpu"`` or ``"cuda"``.  Defaults to ``"cuda"`` if available.
    verbose : bool
        Print progress every 100 steps.
    model : TinySpectralLM, optional
        Pre-constructed model to continue training.  If ``None`` a fresh model
        sized to the corpus vocabulary is created.
    vocab : tuple, optional
        Pre-built ``(chars, stoi, itos)`` triple.  If ``None`` it is built
        from ``text``.

    Returns
    -------
    model : TinySpectralLM
        The trained model.
    vocab : dict-tuple
        ``(chars, stoi, itos)`` used during training.
    losses : list of float
        Recorded training losses (one per 100 steps + final).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Vocabulary ---------------------------------------------------
    if vocab is None:
        chars, stoi, itos = build_vocab(text)
    else:
        chars, stoi, itos = vocab
    vocab_size = len(chars)

    # --- Model --------------------------------------------------------
    if model is None:
        model = TinySpectralLM(vocab_size=vocab_size)
    model = model.to(device)

    # --- Data ---------------------------------------------------------
    data_ids = encode(text, stoi)
    if len(data_ids) < batch_size * seq_len + 1:
        raise ValueError(
            f"Corpus too small ({len(data_ids)} tokens) for the requested "
            f"batch/sequence configuration."
        )
    data = torch.tensor(data_ids, dtype=torch.long, device=device)

    # --- Optimizer & loss --------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []

    model.train()
    for step in range(steps):
        xb = get_batches(data, batch_size, seq_len)             # (B, S)
        yb = get_batches(data, batch_size, seq_len)             # next-token target
        # Shift targets to be next-token: yb[b, t] should predict xb[b, t+1].
        # We use a simpler approach: sample contiguous blocks and offset.
        # Rebuild xb/yb properly:
        n = data.numel()
        starts = torch.randint(0, n - seq_len - 1, (batch_size,))
        xb = torch.stack([data[s:s + seq_len] for s in starts])
        yb = torch.stack([data[s + 1:s + 1 + seq_len] for s in starts])

        logits = model(xb)                                       # (B, S, V)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            yb.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if verbose and (step % 100 == 0 or step == steps - 1):
            losses.append(loss.item())
            print(
                f"step {step:5d}/{steps}  loss={loss.item():.4f}  "
                f"params={count_parameters(model)}"
            )

    # Always record the final loss.
    final_loss = loss.item()
    if not losses or losses[-1] != final_loss:
        losses.append(final_loss)

    return model, (chars, stoi, itos), losses


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(
    model: TinySpectralLM,
    prompt: str,
    max_len: int = 128,
    *,
    stoi: Optional[dict] = None,
    itos: Optional[dict] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: Optional[str] = None,
) -> str:
    """Generate text from a trained model.

    Autoregressively samples ``max_len`` characters, conditioned on ``prompt``.

    Parameters
    ----------
    model : TinySpectralLM
        A trained model.
    prompt : str
        Seed text.  Characters not in the vocabulary are skipped.
    max_len : int
        Number of new characters to generate.
    stoi, itos : dict
        Token mappings.  Required if not attached to ``model``.
    temperature : float
        Sampling temperature; lower → more greedy.
    top_k : int, optional
        If set, restrict sampling to the top-k logits at each step.
    device : str, optional
        Device override; defaults to the model's device.

    Returns
    -------
    str
        The full text (prompt + generated continuation).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # If vocab not provided, we cannot decode — caller must supply.
    if stoi is None or itos is None:
        raise ValueError("stoi and itos must be provided for generation.")

    # Encode the prompt.
    idx = encode(prompt, stoi)
    if len(idx) == 0:
        # Fall back to a single arbitrary token if the prompt is empty/unknown.
        idx = [0]
    idx = torch.tensor([idx], dtype=torch.long, device=device)  # (1, T)

    generated: List[int] = idx[0].tolist()

    for _ in range(max_len):
        # Crop context to the model's nominal seq_len to bound compute.
        context = idx[:, -model.seq_len:] if idx.size(1) > model.seq_len else idx
        logits = model(context)                       # (1, T, V)
        logits = logits[:, -1, :] / max(temperature, 1e-5)  # (1, V)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, top_k, dim=-1)
            logits = torch.where(
                logits < v[:, [-1]],
                torch.full_like(logits, float("-inf")),
                logits,
            )

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)
        idx = torch.cat([idx, next_id], dim=1)
        generated.append(next_id.item())

    return decode(generated, itos)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point.

    Usage::

        python -m spectral_silicon.model --train --steps 1000
        python -m spectral_silicon.model --prompt "To be"
    """
    parser = argparse.ArgumentParser(
        prog="spectral_silicon.model",
        description="Tiny spectral-transformer language model (Prompt 5).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a fresh model for --steps iterations.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of training steps (default: 1000).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="If given (and not --train), generate text from this prompt "
             "using a freshly-initialized model (for smoke testing).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=128,
        help="Number of characters to generate (default: 128).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Training sequence length (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--mixer",
        type=str,
        default="afno",
        choices=["afno", "fftnet"],
        help="Spectral mixer type (default: afno).",
    )
    args = parser.parse_args(argv)

    if args.train:
        print(f"=== Training TinySpectralLM for {args.steps} steps ===")
        model, vocab, losses = train_model(
            steps=args.steps,
            seq_len=args.seq_len,
            lr=args.lr,
        )
        _, stoi, itos = vocab
        print(f"\nFinal loss: {losses[-1]:.4f}")
        print(f"Parameters: {count_parameters(model)}")
        sample = generate(
            model,
            prompt="To be",
            max_len=args.max_len,
            stoi=stoi,
            itos=itos,
        )
        print("\n--- Sample generation ---")
        print(sample)
        return 0

    if args.prompt:
        print(f"=== Generating from prompt: {args.prompt!r} ===")
        # Smoke test: build a fresh model and generate (untrained — random text).
        text = TINY_CORPUS
        chars, stoi, itos = build_vocab(text)
        model = TinySpectralLM(vocab_size=len(chars), mixer_type=args.mixer)
        out = generate(
            model,
            prompt=args.prompt,
            max_len=args.max_len,
            stoi=stoi,
            itos=itos,
        )
        print(out)
        return 0

    # No flags: print model info.
    model = TinySpectralLM(mixer_type=args.mixer)
    print("TinySpectralLM")
    print(f"  parameters : {count_parameters(model)}")
    print(f"  layers     : {model.n_layers}")
    print(f"  d_model    : {model.d_model}")
    print(f"  vocab_size : {model.vocab_size}")
    print(f"  mixer      : {model.mixer_type}")
    print("\nUse --train to train, --prompt 'TEXT' to generate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())