# MODE-SSM: Mode-Aware State-Space Decoder

**Brain-to-Text 2025 Challenge – UC Davis Neuroprosthetics Lab**

## 🧠 Overview

MODE-SSM is an end-to-end neural-to-text architecture tailored for decoding intracortical spiking activity (256-channel speech-motor recordings) into text.
It extends the baseline (neural→phoneme→LM) pipeline  by introducing:

* **State-Space Encoder** for efficient long-sequence modeling.
* **Latent Mode Head** to infer silent vs vocalized speaking strategies .
* **Optional Diffusion/Flow Bridge** for temporal denoising.
* **RNNT + CTC** decoding heads.
* **Corpus-aware LM Fusion** as suggested in the competition overview. 
* **Test-Time Adaptation (TTA)** to handle multi-month neural drift. 

Goal metric: **Word Error Rate (WER)**. 

---

## 🧩 Architecture Summary

```
Spikes (256ch, T bins)
   │
   ├── Preprocessor (norm + conv stem + channel gating)
   │
   ├── SSM Encoder [L=8]  ←─ mode-conditioned gains
   │        │
   │        └── Mode Head p(z|X) ∈ {silent,vocalized}
   │
   ├── (optional) Denoising Bridge (flow/diffusion)
   │
   ├── RNNT Decoder (primary)
   ├── CTC Decoder (auxiliary)
   │
   └── Task-Aware LM Fusion (mode-conditioned)
```

---

## ⚙️ Environment

```bash
conda create -n b2t25 python=3.10
conda activate b2t25
pip install torch torchvision torchaudio
pip install hydra-core==1.3.2 datasets sentencepiece
pip install nemo_toolkit[asr]==1.22.0  # RNNT utilities
pip install mamba-ssm==1.2.0 diffusers==0.30.0
pip install pandas h5py jiwer tqdm
```

---

## 🧱 Repository Layout

```
.
├── configs/
│   ├── train.yaml
│   ├── model_mode_ssm.yaml
│   └── lm_fusion.yaml
├── data/
│   ├── t15_train.hdf5
│   ├── t15_val.hdf5
│   └── t15_test.hdf5
├── model/
│   ├── preprocessor.py
│   ├── ssm_encoder.py
│   ├── mode_head.py
│   ├── denoise_flow.py
│   ├── rnnt_ctc_heads.py
│   ├── lm_fusion.py
│   └── tta_loop.py
├── train.py
├── evaluate_model.py
└── make_submission.py
```

---

## 🧮 Training Pipeline

### Stage 1 – Warmup

Train `Preprocessor + SSM Encoder + CTC` only (alignment-free).

```bash
python train.py stage=ctc_warmup
```

### Stage 2 – RNNT + CTC Joint Training

Enable RNNT head and Mode Head; keep CTC auxiliary.

```bash
python train.py stage=joint_train
```

### Stage 3 – Mode Conditioning

Activate soft mode gating (LayerNorm & SSM gains).

```bash
python train.py stage=mode_train
```

### Stage 4 – (Opt.) Denoising Bridge

Turn on diffusion/flow bridge for misalignment smoothing.

```bash
python train.py stage=denoise_train
```

### Stage 5 – Corpus-Aware LM Training

Train small n-gram/neural LMs per corpus type as described by organizers  and export to `lm_fusion.yaml`.

---

## 🔮 Test-Time Adaptation (TTA)

Executed automatically at inference:

```bash
python evaluate_model.py --tta true
```

Per-session EMA feature stats → 3–10 entropy-minimization steps updating last encoder block only. 

---

## 🧠 Inference & Submission

```bash
python make_submission.py \
  --checkpoint checkpoints/mode_ssm/best.pt \
  --config configs/model_mode_ssm.yaml
```

Produces `submission.csv` with:

```
id,text
0,hello there
1,thank you very much
...
```

ordered chronologically (session → block → trial) as required. 

Local evaluation:

```bash
python evaluate_model.py --wer --submission submission.csv --labels val_labels.csv
```

---

## 🧾 Loss Functions

| Loss        | Purpose                    | Weight |
| ----------- | -------------------------- | ------ |
| `L_RNNT`    | Main sequence loss         | 1.0    |
| `L_CTC`     | Alignment regularizer      | 0.3    |
| `L_mode`    | Latent mode classification | 0.1    |
| `L_denoise` | Flow bridge smoothness     | 0.05   |

---

## 🚀 Key Advantages vs Baseline

| Area              | Baseline     | MODE-SSM Improvement                   |
| ----------------- | ------------ | -------------------------------------- |
| Temporal modeling | RNN          | Linear-time SSM captures long context  |
| Mode variance     | Ignored      | Explicit latent mode (z)               |
| Drift adaptation  | None         | Lightweight TTA loop                   |
| Language fusion   | Fixed n-gram | Corpus-specific mode-aware LMs         |
| Robustness        | Static       | Flow denoiser + entropy regularization |

---

## 📈 Expected Leaderboard Impact

* Baseline WER ≈ 6.7 % 
* Target WER ≈ **≤ 5.0 %**
  (gains from mode conditioning + SSM + TTA).

---

## 🧩 References

* Card et al., *NEJM* (2024) “An Accurate and Rapidly Calibrating Speech Neuroprosthesis.” 
* UC Davis Neuroprosthetics Lab – Brain-to-Text 2025 Overview. 
* Blackrock Neurotech Competition Rules and Submission Format. 
* Mamba: Selective State Space Models (Gu et al., 2024).
* Angrick et al., *Nat Mach Intell* (2021) – Silent speech decoding.
