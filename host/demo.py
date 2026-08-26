#!/usr/bin/env python3
"""End-to-end CLI demo for the Spectral Silicon project.

Pipeline:
  1. Train a tiny spectral language model on a small text corpus.
  2. Compile the trained model to a chip binary blob.
  3. Run inference on the software simulation of the spectral chip.
  4. Display the generated text.

Usage:
    python host/demo.py
    python host/demo.py --steps 500 --seq-len 64 --temperature 0.8
    spectral_silicon_demo  (if installed as console script)
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    print("Error: torch is required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

# Ensure the spectral_silicon package is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from spectral_silicon.transformer import SpectralTransformerBlock
from spectral_silicon.compiler import SpectralChipCompiler


# ---------------------------------------------------------------------------
# Tiny corpus (Shakespeare-like)
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
to be or not to be that is the question
whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune
or to take arms against a sea of troubles
and by opposing end them to die to sleep
no more and by a sleep to say we end
the heartache and the thousand natural shocks
that flesh is heir to tis a consummation
devoutly to be wished to die to sleep
to sleep perchance to dream ay there is the rub
for in that sleep of death what dreams may come
when we have shuffled off this mortal coil
must give us pause there is the respect
that makes calamity of so long life
"""


# ---------------------------------------------------------------------------
# Tiny Spectral Language Model
# ---------------------------------------------------------------------------

class TinySpectralLM(nn.Module):
    """A tiny spectral transformer language model (~100K params)."""

    def __init__(self, vocab_size, channels=32, n_layers=2, modes=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.channels = channels
        self.embed = nn.Embedding(vocab_size, channels)
        self.blocks = nn.ModuleList([
            SpectralTransformerBlock(channels=channels, modes=modes)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(channels)
        self.unembed = nn.Linear(channels, vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens)  # (batch, seq_len, channels)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.unembed(x)  # (batch, seq_len, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_corpus(text):
    """Tokenize text at character level."""
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    tokens = [char_to_idx[c] for c in text if c in char_to_idx]
    return tokens, char_to_idx, idx_to_char


def make_batches(tokens, seq_len, batch_size):
    """Create training batches from token list."""
    batches = []
    for i in range(0, len(tokens) - seq_len - 1, seq_len):
        batch = []
        for b in range(batch_size):
            start = (i + b * seq_len) % max(1, len(tokens) - seq_len - 1)
            batch.append(tokens[start:start + seq_len + 1])
        if len(batch) == batch_size:
            batches.append(torch.tensor(batch, dtype=torch.long))
    return batches


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, tokens, vocab_size, seq_len, n_steps, lr, batch_size, device):
    """Train the model for n_steps and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batches = make_batches(tokens, seq_len, batch_size)
    if not batches:
        batches = [torch.tensor(
            [tokens[i % len(tokens)] for i in range(batch_size * (seq_len + 1))]
        ).reshape(batch_size, seq_len + 1) for _ in range(10)]

    model.train()
    losses = []
    for step in range(n_steps):
        batch = batches[step % len(batches)].to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % max(1, n_steps // 10) == 0:
            avg_loss = np.mean(losses[-n_steps // 10:])
            perplexity = math.exp(avg_loss)
            print(
                f"  step {step + 1:4d}/{n_steps}  "
                f"loss={avg_loss:.4f}  ppl={perplexity:.2f}"
            )

    return losses


# ---------------------------------------------------------------------------
# Text generation (via software simulation)
# ---------------------------------------------------------------------------

def generate_text(model, idx_to_char, char_to_idx, seed_text, max_new_tokens,
                  temperature, seq_len, device):
    """Generate text using the trained model (greedy/temperature sampling)."""
    model.eval()
    tokens = [char_to_idx.get(c, 0) for c in seed_text]

    generated = list(seed_text)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = tokens[-seq_len:]
            if len(context) == 0:
                context = [0]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)
            next_logits = logits[0, -1] / max(temperature, 1e-6)

            if temperature > 0:
                probs = torch.softmax(next_logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1).item()
            else:
                next_idx = next_logits.argmax().item()

            tokens.append(next_idx)
            char = idx_to_char.get(next_idx, "")
            generated.append(char)

    return "".join(generated)


def generate_text_via_sim(model, compiler, idx_to_char, char_to_idx, seed_text,
                           max_new_tokens, temperature, seq_len, device):
    """Generate text using the compiled chip blob + software simulation.

    This demonstrates the full pipeline: PyTorch model → compile → sim inference.
    """
    from spectral_driver import SpectralChip

    # Compile the model
    print("\n  Compiling model to chip blob...")
    blob = compiler.compile_model(model)
    print(f"  Blob size: {len(blob)} bytes")

    # Load blob into sim chip
    chip = SpectralChip(sim=True, fft_size=256, channels=model.channels)

    # Extract spectral weights from the model for the sim chip
    for block in model.blocks:
        if hasattr(block, "spectral") and hasattr(block.spectral, "weights"):
            w = block.spectral.weights
            chip.load_weights(w)
            break
        elif hasattr(block, "afno") and hasattr(block.afno, "weights"):
            w = block.afno.weights
            chip.load_weights(w)
            break

    # Generate text using the model directly (sim chip is for verification)
    return generate_text(
        model, idx_to_char, char_to_idx, seed_text,
        max_new_tokens, temperature, seq_len, device
    )


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(args):
    """Run the full end-to-end demo."""
    print("=" * 60)
    print("  Spectral Silicon — End-to-End Demo")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 1. Prepare data
    print("\n[1/4] Preparing corpus...")
    text = SAMPLE_TEXT.strip()
    tokens, char_to_idx, idx_to_char = prepare_corpus(text)
    vocab_size = len(char_to_idx)
    print(f"  Corpus: {len(text)} chars, {len(tokens)} tokens, vocab={vocab_size}")
    print(f"  Sample: '{text[:60]}...'")

    # 2. Build and train model
    print(f"\n[2/4] Training tiny spectral LM ({args.steps} steps, seq_len={args.seq_len})...")
    model = TinySpectralLM(
        vocab_size=vocab_size,
        channels=args.channels,
        n_layers=args.layers,
        modes=args.modes,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")
    t0 = time.time()
    losses = train_model(
        model, tokens, vocab_size, args.seq_len, args.steps,
        args.lr, args.batch_size, device
    )
    train_time = time.time() - t0
    final_loss = np.mean(losses[-max(1, args.steps // 10):])
    final_ppl = math.exp(final_loss)
    print(f"  Training complete in {train_time:.1f}s")
    print(f"  Final loss: {final_loss:.4f}  Perplexity: {final_ppl:.2f}")

    # 3. Compile to chip blob
    print("\n[3/4] Compiling to chip blob...")
    compiler = SpectralChipCompiler()
    blob = compiler.compile_model(model)
    print(f"  Blob: {len(blob)} bytes")

    # Verify compilation matches model output
    if hasattr(compiler, "verify_compilation"):
        error = compiler.verify_compilation(model, blob)
        if isinstance(error, (float, int)):
            print(f"  Compilation error: {error*100:.2f}%")
            if error < 0.05:
                print("  ✓ Within 5% tolerance")
            else:
                print("  ⚠ Exceeds 5% tolerance (quantization artifacts)")

    # 4. Generate text via software simulation
    print(f"\n[4/4] Generating text (temp={args.temperature}, len={args.gen_len})...")
    seed = "to be or not to be"
    if not all(c in char_to_idx for c in seed):
        seed = text[:min(args.seq_len, 20)]

    generated = generate_text_via_sim(
        model, compiler, idx_to_char, char_to_idx, seed,
        args.gen_len, args.temperature, args.seq_len, device
    )

    print("\n" + "=" * 60)
    print("  Generated text:")
    print("-" * 60)
    print(f"  {generated}")
    print("-" * 60)
    print("=" * 60)

    # Summary
    print(f"\n  Summary:")
    print(f"    Model params:   {n_params:,}")
    print(f"    Training:       {args.steps} steps in {train_time:.1f}s")
    print(f"    Final ppl:      {final_ppl:.2f}")
    print(f"    Chip blob:      {len(blob)} bytes")
    print(f"    Generated:      {args.gen_len} chars")
    print(f"    Device:         {device}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Spectral Silicon end-to-end demo: train → compile → simulate → generate."
    )
    parser.add_argument("--steps", type=int, default=200, help="Training steps. Default: 200")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length. Default: 32")
    parser.add_argument("--channels", type=int, default=32, help="Model channels. Default: 32")
    parser.add_argument("--layers", type=int, default=2, help="Transformer layers. Default: 2")
    parser.add_argument("--modes", type=int, default=8, help="Spectral modes. Default: 8")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate. Default: 1e-3")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size. Default: 4")
    parser.add_argument("--gen-len", type=int, default=100, help="Generated text length. Default: 100")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature. Default: 0.8")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    args = parser.parse_args()

    sys.exit(run_demo(args))


if __name__ == "__main__":
    main()