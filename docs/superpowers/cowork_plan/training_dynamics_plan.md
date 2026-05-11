# Experimental Design: Understanding FURA Through Learning Dynamics

## Motivation and Hypothesis

Recent work on learning dynamics (Ren & Sutherland, ICLR 2025) decomposes the per-step change in an LLM's predictions as:

$$\Delta\log \pi_t(y \mid x_o) = -\eta \cdot \mathcal{A}_t(x_o) \cdot \mathcal{K}_t(x_o, x_u) \cdot \mathcal{G}_t(x_u, y_u)$$

where $\mathcal{K}_t$ (the empirical NTK) governs cross-example interference: learning from $(x_u, y_u)$ shifts predictions on $x_o$ in proportion to $\|\mathcal{K}_t(x_o, x_u)\|_F$. The framework shows that unconstrained gradient updates during SFT cause hallucination (using facts from one example to answer another) and that DPO's negative gradient triggers a "squeezing effect" concentrating mass on degenerate predictions.

**Core hypothesis.** FURA's spectral preconditioning—projecting every gradient step through $\mathbf{U}_k \mathbf{U}_k^\top$—stabilizes $\mathcal{K}_t$ relative to its pretrained value $\mathcal{K}_0$, confining adaptation to *recombination* of existing features rather than creation of new, ungrounded feature directions. This reduces cross-example interference and preserves the pretrained knowledge structure during fine-tuning.

Concretely:
1. Full FT can create feature directions orthogonal to the pretrained manifold. These distort $\mathcal{K}_t$, making previously dissimilar examples suddenly "look similar" and amplifying cross-contamination.
2. LoRA initializes at zero and can drift arbitrarily during training; the update subspace is unconstrained.
3. FURA constrains $\Delta \mathbf{W}_k \in \text{col}(\mathbf{U}_k)$ by construction. The internal representations—and thus $\mathcal{K}_t$—evolve only within the pretrained feature manifold.

---

## Experiment 1: Cross-Example Influence Tracking

**Goal.** Directly measure how learning one example influences predictions on other examples, comparing Full FT, LoRA-64, and FURA under matched training budgets.

### Setup

| Component | Choice |
|-----------|--------|
| Model | Qwen3-1.7B (main) or LLaMA-3-8B (scale check) |
| Training data | Commonsense-170K or MATH-10K (SFT), subset of 5000 examples |
| Probing set $D_\text{prob}$ | 500 examples randomly sampled from $D_\text{train}$ |
| Methods | Full FT, LoRA-64, FURA (default: output-one-block, $\mathbf{S}$ separate trainable) |
| Evaluation frequency | Every 25 optimizer steps (batch size 4 → every 100 examples) |

### Response Types to Track

For each prompt $x_u$ in $D_\text{prob}$, generate/identify:

| Symbol | Description | Interference signal |
|--------|-------------|---------------------|
| $y_u^+$ | Target response (from training data) | Direct learning target |
| $y_{u,\text{gpt}}^+$ | GPT-4 rephrase of $y_u^+$ | Similar-response interference |
| $y_{j \neq u}^+$ | Target response for a *different* question $x_j$ | **Cross-question contamination (hallucination channel)** |
| $y_u^-$ | Rejected/incorrect response for $x_u$ | Negative-pressure target |
| $y_\text{rnd}$ | Random English sentence (same token count) | Baseline: dissimilar control |
| $y_{u,\text{perm}}^+$ | Random word-permutation of $y_u^+$ | Baseline: same tokens, no structure |

### Metric

For each response type, record the average log-likelihood across the probing set:
$$\bar{L}_t(\text{type}) = \frac{1}{|D_\text{prob}|} \sum_{(x_u, \cdot) \in D_\text{prob}} \log \pi_{\theta_t}(y_\text{type} \mid x_u)$$

### Expected Observations

1. **$y_u^+$ (target):** All methods pull this up. FURA should match or exceed Full FT in convergence speed (since full-rank capacity is available).

2. **$y_{j \neq u}^+$ (cross-contamination):** Under Full FT, this rises significantly early in training (the model starts "borrowing" answers from other questions). Under FURA, the rise should be **smaller and shorter-lived** because the eNTK similarity between unrelated question-answer pairs is more stable—the model doesn't create new features that spuriously link them.

3. **$y_{u,\text{gpt}}^+$ (benign similar responses):** Some rise is expected and harmless. FURA and Full FT may be similar here, since the pretrained similarity between $y_u^+$ and its rephrase is already high.

4. **$y_\text{rnd}$ and $y_{u,\text{perm}}^+$ (dissimilar controls):** Should monotonically decrease for all methods. If Full FT shows a temporary *rise* in these while FURA does not, that's evidence of feature-space distortion.

### Key Plots

- **Figure A (main result):** 3-column panel (Full FT | LoRA | FURA), each showing curves for all response types over training epochs. Highlight the $y_{j \neq u}^+$ curve—it should peak lower and earlier for FURA.
- **Figure B (summary):** Bar chart of $\max_t \bar{L}_t(y_{j \neq u}^+) - \bar{L}_0(y_{j \neq u}^+)$ (peak cross-contamination) across methods. FURA should have the smallest bar.

---

## Experiment 2: Feature Similarity Drift (CKA / Representation Stability)

**Goal.** Show that FURA's internal representations drift less from the pretrained model than Full FT and LoRA, validating the "eNTK stability" mechanism.

### Setup

| Component | Choice |
|-----------|--------|
| Probe dataset | 1000 held-out examples (mix of domains) |
| Layers measured | Every 4th transformer layer |
| Metric | Linear CKA between pretrained and fine-tuned hidden states |
| Checkpoints | Every epoch (or every 500 steps) |

### Method

At each checkpoint and layer $l$:
1. Forward-pass the 1000 probe examples through both the pretrained model $\theta_0$ and the current checkpoint $\theta_t$.
2. Collect hidden representations $H_0^{(l)} \in \mathbb{R}^{1000 \times d}$ and $H_t^{(l)} \in \mathbb{R}^{1000 \times d}$.
3. Compute linear CKA:
   $$\text{CKA}(H_0, H_t) = \frac{\|H_0^\top H_t\|_F^2}{\|H_0^\top H_0\|_F \cdot \|H_t^\top H_t\|_F}$$

### Expected Result

FURA maintains higher CKA throughout training (closer to 1.0), especially in middle/later layers where the spectral constraint most directly acts. Full FT shows the largest CKA drop. LoRA falls in between (constrained by low-rank but not by subspace).

### Key Plot

Line plot: CKA vs. training step, one line per method, averaged over layers. Optionally a heatmap (method × layer) showing final CKA values.

---

## Experiment 3: Catastrophic Forgetting on Held-Out Capabilities

**Goal.** Demonstrate that reduced cross-example interference translates to better preservation of pretrained knowledge on unrelated tasks.

### Setup

Fine-tune on **Task A** (narrow domain), evaluate on **Task B** (unrelated, where the pretrained model is already competent).

| Fine-tuning task (A) | Evaluation task (B) |
|----------------------|---------------------|
| MATH-10K (arithmetic reasoning) | Commonsense (BoolQ, PIQA, HellaSwag) |
| Commonsense-170K | MMLU (knowledge-intensive) |
| Medical QA subset | Legal/general QA |

Alternatively, use perplexity on a general corpus (WikiText-103 or C4 validation) as a domain-agnostic measure of knowledge retention.

### Metric

$$\text{Retention}(B) = \frac{\text{Acc}_{\theta_t}(B)}{\text{Acc}_{\theta_0}(B)}$$

or equivalently the perplexity ratio $\text{PPL}_{\theta_0} / \text{PPL}_{\theta_t}$ (values < 1 indicate forgetting).

### Expected Result

FURA achieves comparable Task A performance while maintaining higher Task B accuracy / lower perplexity increase. The Pareto frontier (Task A accuracy vs. Task B retention) is better for FURA.

### Key Plot

- **Pareto plot:** x-axis = Task A accuracy, y-axis = Task B retention. Each point is a checkpoint. FURA's curve should dominate (up and to the right).
- **Dual-axis line plot:** Task A accuracy (rising) and Task B accuracy (declining) over training steps, one subplot per method.

---

## Experiment 4: Spectral Analysis of $\Delta \mathbf{W}$ (Mechanistic Smoking Gun)

**Goal.** Directly show that Full FT creates significant off-subspace update components while FURA (by construction) does not, and that the off-subspace component correlates with interference.

### Setup

After fine-tuning, for each linear layer, compute $\Delta \mathbf{W} = \mathbf{W}' - \mathbf{W}_0$.

### Measurements

1. **Off-subspace energy ratio:**
   $$\rho = \frac{\|(\mathbf{I} - \mathbf{U}\mathbf{U}^\top) \Delta\mathbf{W}\|_F}{\|\Delta\mathbf{W}\|_F}$$
   where $\mathbf{U}$ is the pretrained left-singular basis. For FURA this is 0 by construction; for Full FT and LoRA, measure how large it is.

2. **Alignment with pretrained spectrum:**
   Project $\Delta \mathbf{W}$ into the SVD basis: $\mathbf{C} = \mathbf{U}^\top \Delta\mathbf{W} \mathbf{V}$. Plot the energy distribution $\|C_{ij}\|^2$ as a heatmap (rows = left-singular directions, columns = right-singular directions).

3. **Correlation analysis:** Across layers, correlate $\rho$ (off-subspace fraction) with the degree of cross-example interference measured in Experiment 1. If high $\rho$ predicts high interference, that's the causal link.

### Expected Result

- Full FT: $\rho$ is 20–40% across layers. High-$\rho$ layers show more cross-contamination in Experiment 1.
- LoRA: $\rho$ is moderate (10–25%), growing with training duration as the adapter drifts from initialization.
- FURA: $\rho = 0$ by construction. The energy in $\Delta \mathbf{W}$ is entirely within the pretrained feature manifold.

### Key Plot

- **Bar chart:** $\rho$ per layer, grouped by method (Full FT | LoRA | FURA).
- **Scatter plot:** $\rho$ (x-axis) vs. peak cross-contamination $\max_t \bar{L}_t(y_{j \neq u}^+)$ (y-axis), pooled across layers and methods. Expect positive correlation.

---

## Experiment 5: Controlled Topic-Isolation Test (Pedagogical / Intro Figure)

**Goal.** A clean, small-scale demonstration that FURA's updates are more "local" in the knowledge graph—learning one topic doesn't bleed into unrelated topics.

### Setup

1. Curate a 5-topic subset of Alpaca or a similar instruction dataset: **Math**, **Code**, **History**, **Biology**, **Grammar**. 200 examples per topic, balanced.
2. Fine-tune on **one topic only** (e.g., Math, 200 examples).
3. Evaluate prediction shift on all 5 topics.

### Metric

For each topic $T$:
$$\Delta_T = \frac{1}{|T|} \sum_{(x,y) \in T} |\log \pi_{\theta_t}(y \mid x) - \log \pi_{\theta_0}(y \mid x)|$$

### Expected Result

| | Math (trained) | Code | History | Biology | Grammar |
|---|---|---|---|---|---|
| Full FT | ↑↑↑ | ↑↑ | ↑ | ↑ | ↑ |
| LoRA | ↑↑↑ | ↑ | ↑ | ~ | ~ |
| FURA | ↑↑↑ | ↑ | ~ | ~ | ~ |

FURA's influence is concentrated on the trained topic and semantically close topics (Code), with minimal leakage to unrelated topics.

### Key Plot

Heatmap or grouped bar chart showing $\Delta_T$ across topics, one panel per method. Makes an excellent intro/motivation figure.

---

## Implementation Notes

### Generating Probing Responses

Follow Ren & Sutherland's protocol:
- **Rephrases ($y_\text{gpt}^+$):** Use GPT-4 with prompt: "Rephrase the following response while keeping the same meaning: [response]". Generate 2 versions: style-rephrase and full-rephrase.
- **Cross-question ($y_{j \neq u}^+$):** Randomly sample another training example's target response.
- **Random sentence ($y_\text{rnd}$):** Sample from a held-out English corpus (e.g., BookCorpus), matched in token count.
- **Permuted ($y_{u,\text{perm}}^+$):** Randomly permute the word order of $y_u^+$.

### Computational Budget

| Experiment | Extra compute over standard training |
|-----------|--------------------------------------|
| Exp 1 (influence tracking) | ~20% (periodic forward pass on 500 examples × 6 response types) |
| Exp 2 (CKA) | ~5% (forward pass on 1000 examples per checkpoint, no backward) |
| Exp 3 (forgetting) | ~10% (periodic eval on held-out benchmarks) |
| Exp 4 (spectral analysis) | Negligible (post-hoc SVD of $\Delta W$ per layer, one-time) |
| Exp 5 (topic isolation) | Cheap (small model, 200 training examples, fast) |

### Suggested Prioritization

1. **Experiment 5** — cheapest, produces the intro-figure, establishes the intuition.
2. **Experiment 1** — the centerpiece; directly connects to Ren & Sutherland's framework.
3. **Experiment 4** — the mechanistic bridge; shows *why* FURA helps (no off-subspace component).
4. **Experiment 3** — practical consequence; forgetting reduction is the payoff reviewers care about.
5. **Experiment 2** — supporting evidence; CKA drift confirms the mechanism at representation level.

---

## Narrative Arc for the Paper Section

> **§ Analysis: Why Spectral Preconditioning Reduces Interference**
>
> Paragraph 1: Introduce the learning dynamics framework (cite Ren & Sutherland). The per-step decomposition shows cross-example interference is governed by $\mathcal{K}_t$, the empirical NTK.
>
> Paragraph 2: State the hypothesis—FURA's projection $\mathbf{U}_k \mathbf{U}_k^\top$ stabilizes $\mathcal{K}_t$ by preventing off-manifold feature creation.
>
> Paragraph 3: Present Experiment 5 (topic isolation) as a motivating example. "When fine-tuning on Math, FURA's influence on unrelated topics (History, Grammar) is 3× smaller than Full FT's."
>
> Paragraph 4: Present Experiment 1 (influence tracking) as the main result. "The cross-contamination peak $\max_t \bar{L}_t(y_{j \neq u}^+)$ is X% lower under FURA than Full FT."
>
> Paragraph 5: Present Experiment 4 (spectral analysis) as the explanation. "Full FT places 25–35% of its update energy outside the pretrained singular subspace; this off-subspace fraction correlates with cross-contamination (Pearson $r = 0.73$). FURA eliminates this component by construction."
>
> Paragraph 6: Present Experiment 3 (forgetting) as the practical implication. "On held-out commonsense tasks, FURA retains 97% of pretrained accuracy while Full FT retains only 91%."
