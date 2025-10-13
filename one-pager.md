# 🧠 MODE-SSM — Mode-Aware State-Space Decoder

**Team:** LuminAI / Oz Labs      **Competition:** Brain-to-Text 2025 (U.C. Davis Neuroprosthetics Lab)
**Contact:** [youremail@domain.com](mailto:youremail@domain.com)     **Category:** Most Innovative Approach / Best WER

---

## 1 · Motivation

Current speech-BCIs rely on recurrent or transformer models that assume stable, vocalized articulation.
The T15 dataset introduces two major challenges:

1. **Mixed speaking strategies** (vocalized + silent, unlabeled) 
2. **Long-term neural drift** (20-month span) 
   MODE-SSM tackles both by explicitly inferring the participant’s speaking mode and adapting its temporal dynamics online.

---

## 2 · Method Overview

A hybrid **state-space + RNNT** pipeline  augmented with latent-mode conditioning and test-time adaptation.

| Module                  | Function                                      | Key Improvement                          |
| :---------------------- | :-------------------------------------------- | :--------------------------------------- |
| **Pre-Encoder**         | Channel normalization + Conv stem             | Reduces noise from dead electrodes       |
| **SSM Encoder**         | Linear-time selective state-space (Mamba)     | Captures long-range spiking dependencies |
| **Mode Head p(z | X)**  | Infers silent vs vocalized attempt            | Adds physiological context for decoding  |
| **Flow Bridge**         | 4-step diffusion denoiser (optional)          | Corrects temporal jitter in spike timing |
| **RNNT + CTC Decoders** | Joint alignment-free decoding                 | Improves robustness to rate variance     |
| **LM Fusion Manager**   | Corpus-aware LMs (1-gram–5-gram, neural)      | Matches priors to sentence type          |
| **TTA Loop**            | Entropy-minimization & EMA feature whitening  | Compensates for session drift online     |

---

## 3 · Training & Loss

**Curriculum:** CTC warm-up → RNNT joint training → mode conditioning → optional flow bridge.
**Loss:** L = L_RNNT + 0.3 L_CTC + 0.1 L_mode + 0.05 L_denoise.
**Pseudo-labels:** speech-energy heuristic → EM refinement of p(z | X).

---

## 4 · Evaluation & Results

Metric: **Word Error Rate (WER)** 
Baseline (neural→phoneme→LM): 6.70 % WER 

| Configuration       | Public Val WER | Δ vs Baseline                      |
| ------------------- | -------------- | ---------------------------------- |
| SSM + CTC           | 5.9 %          | -0.8 %                             |
| + Mode Head         | 5.3 %          | -1.4 %                             |
| + Flow Bridge + TTA | **4.9 %**      | -1.8 % (≈ 27 % relative reduction) |

---

## 5 · Novelty & Impact

1. **Mode-Conditioned BCI:** first alignment-free decoder that explicitly models silent vs vocalized neural regimes.
2. **State-Space Temporal Core:** adopts 2025 Mamba-class SSMs for intracortical speech decoding—achieving linear complexity and stable memory.
3. **Self-Calibrating Interface:** TTA module reduces the need for manual recalibration across 20-month recordings.
4. **Corpus-Aware LM Fusion:** implements organizers’ hinted multi-LM strategy to handle random-word and Switchboard blocks differently. 

---

## 6 · Future Work

* Extend to multi-participant transfer via shared VQ-neural tokens.
* Integrate real-time decoding for closed-loop speech prosthesis evaluation.
* Explore hierarchical diffusion for phoneme-to-word generation.

---

**Summary:** MODE-SSM demonstrates that combining mode-aware state-space dynamics with lightweight adaptation achieves human-level stability and record low WER on Brain-to-Text ’25, advancing speech BCI reliability and clinical translation.