# NanoGPT Arithmetic — Laplacian Eigenvector Visualization

Interactive scrollytelling visualization of how a 1.2M-parameter
char-level NanoGPT (6 layers, 128-dim, 4-head) organizes arithmetic
across its `=`-token hidden states at each layer.

**Live demo:** https://jdonaldson.github.io/nanogpt-arithmetic-viz/

## What it shows

A 6-block / 128-dim char-level GPT was trained on the operators `+`,
`-`, `*` with operands 0–9 (plus filler text for vocab variety). It
reaches 100% accuracy on all three operators. The visualization
probes what each layer is doing geometrically by:

1. **Joint Laplacian projection** (overview, top of page): pool all
   300 (a, b, op) hidden states per layer, compute one Laplacian
   eigendecomposition, project to 3 dimensions, stack the 6 layers
   on the z-axis. Each pair's path through the network becomes a
   6-point polyline. At L0 the three operators form clean
   separable clusters (100% 1-NN op-recovery, 5-fold CV); by L5
   they overlap (~51%, near random for 3 classes).

2. **Per-operator Laplacian** (scroll-driven, below the overview):
   for each operator independently, compute the kNN graph
   Laplacian's smallest non-trivial eigenvectors at each layer.
   Procrustes-align across consecutive layers so the animation
   has minimal spurious rotation; per-layer unit-variance
   normalize so scale stays consistent. Scroll to morph the cluster
   from L0 → L5; click any feature pill in the prose to recolor
   every point by that feature.

## How to read it

- **Pick an operator** (×, +, −) at the top of the scrolly section.
- **Scroll** to advance through layers L0..L5; the right-hand 3D
  plot smoothly interpolates between layer positions.
- **Click feature pills** inline (`max`, `log_prod`, `signed_diff`,
  `predicted_char`, ...) to recolor every point by that feature.
- **Hover** any point for its (a, b) and predicted character.
- **Drag** on the 3D plot to rotate the camera.
- **Vertical minimaps** on the left of the main plot show each
  layer's projection at small scale — click any to jump to it.

## Key findings (visible in the viz)

- **L0 = (min, max) triangle.** Each operator's L0 cluster is a
  triangulated 2D sub-lattice — the quotient of the 10×10 operand
  grid by the swap action. Swap pairs (a, b) and (b, a) collapse to
  within 8% of random-pair distance. Probes for `min` / `max` hit
  94–96% decodability; the spectrum shows it *is* the cluster's
  geometry, not just decodable from it.

- **L1 = operator dispatcher.** Activation patching shows ≥91%
  operator-flip rate after block 1, but only for sub does this
  show as a visible basis change within a single-op view (sub's
  u₁ locks onto operand b with R²=0.78). Mul and add preserve the
  symmetric basis here.

- **L2 = shared zero detection.** All three operators have u₁ ≈
  `zero_flag` with R² ≈ 1.0. The Laplacian's Fiedler vector
  becomes the cut separating zero pairs from non-zero pairs.
  Cluster topology shifts: 2D-LATTICE → HUB.

- **L3 = operator-specific commit.** Each operator's u₁ shifts to
  its own answer dimension: mul stays on `zero_flag` (commit
  fork = zero-vs-nonzero, the "annihilator gutter"), add commits
  to `sum`, sub commits to `signed_diff`. For sub, the joint R²
  of (u₁, u₂) onto (min, max) collapses 0.65 → 0.05.

- **L4-L5 = three-vertex simplex for sub.** Sub L5's plane has
  three discrete vertices (between/within ratio 10.5×) — a
  neural-collapse ETF pattern. The vertices correspond *exactly*
  to three predicted-character classes: `−` (45 pairs), `0` (10
  pairs), and digits 1–9 (45 pairs). Sub is the only operator
  that ever emits a non-digit at the `=` position.

## Methodology

- **kNN graph**: per-layer, k=10, symmetric-normalized Laplacian
  over 100 (a, b) pairs (per-op view) or 300 pooled (joint view).
- **Procrustes alignment**: orthogonal 3×3 R minimizing
  `||X_L · R − X_{L-1}||_F` so consecutive layers don't flip
  spuriously. Allows rotation + reflection.
- **Cross-op L0 alignment**: each op's L0 is Procrustes-aligned
  to mul's L0 (the same R propagated to all that op's layers),
  so single-op views start in a comparable orientation.
- **Predicted characters**: the actual model run on the prompt
  `\n{a}{op}{b}=` for each (a, b) — predicted next character is
  cached and used as a categorical coloring.

## Architecture sources

The findings the viz illustrates are established empirically by
companion experiments and writeups (gutter direction, log-add
combination rule, contrastive digit selector at L5, dispatcher
generalization to SmolLM2). Models trained from scratch following
the [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT) recipe;
all data is the model's own hidden states, no external corpus.

## Tech

- Plotly.js (via CDN) for all interactive plots
- Scrollama for scroll-step detection
- HTML5 Canvas for the vertical minimap strip
- Procrustes alignment + per-layer normalization for smooth animation
- `requestAnimationFrame`-driven interpolation (avoids Plotly's
  WebGL transition machinery for 3D, which can stutter)

Single self-contained HTML file (~1 MB) — no backend, no data
fetches at runtime beyond Plotly + Scrollama from CDN.

## Reproducing the build

The full source for the build is in this repo. To regenerate
`index.html` and `og.png` from the cached hidden states:

```bash
pip install -r requirements.txt
python build.py
```

Or with `uv`:

```bash
uv run --with-requirements requirements.txt python build.py
```

This re-runs all spectral analysis, Procrustes alignment, and
figure generation from the data in `data/`. No model required —
the hidden states are cached `.npz` files (one per operator,
each ~300 KB).

### Repository layout

```
nanogpt-arithmetic-viz/
├── index.html              # the published site
├── og.png                  # Open Graph preview image
├── build.py                # full build pipeline (Laplacian → HTML)
├── data/
│   ├── hidden_states.npz       # mul: =-token state at every layer
│   ├── hidden_states_add.npz   # add
│   ├── hidden_states_sub.npz   # sub
│   └── predicted_chars.json    # cached model predictions per pair
├── requirements.txt
└── README.md
```

### How the data was generated

The hidden states were captured by training a 1.2M-parameter
char-level NanoGPT (config: 6 layers, 4 heads, 128-dim, block size
32) on a corpus of arithmetic prompts mixed with SVO filler text.
At inference time, the model was run on each `\n{a}{op}{b}=`
prompt for (a, b) ∈ {0..9}² and the hidden state at the `=`-token
was captured at every layer (0..5). The cached `.npz` files store
these states in shape (100 pairs × 6 layers, 128 dims) along
with the (a, b) integer arrays. `predicted_chars.json` records
the argmax prediction at the `=` position for each pair —
generated by running the trained model forward on each prompt.

The NanoGPT training code and checkpoint live in a separate
repository; here we ship only the cached outputs needed to
reproduce the visualization.
