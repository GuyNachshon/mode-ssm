# 🧠 MODE-SSM (UV Dual-GPU Edition)

**Project:** Brain-to-Text ’25 Speech BCI
**Hardware:** 2× A100 40 GB (or ≥ RTX 4090 24 GB)
**Environment:** [uv](https://docs.astral.sh/uv/) for environment & dependency management
**Framework:** PyTorch 2.2 + DDP + AMP
**Goal:** ≤ 5 % WER (private LB)
**Duration:** 6 weeks

---

## ⚙️ 1 · Environment Setup (with UV)

```bash
# initialize project
uv init mode-ssm
cd mode-ssm

# create reproducible environment
uv venv --python 3.10
source .venv/bin/activate

# install dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install mamba-ssm diffusers hydra-core jiwer nemo_toolkit[asr] h5py pandas tqdm rich
uv pip install deepspeed accelerate tensorboard
```

**Advantages of UV**

* 10× faster dependency installs
* Locked `uv.lock` ensures reproducibility across GPUs
* Integrates seamlessly with `torchrun` for DDP

✅ Deliverables

* `pyproject.toml` + `uv.lock`
* `.venv` with identical repro on both GPUs

---

## 🧩 2 · Data Preparation

* Download `t15_train.hdf5`, `t15_val.hdf5`, `t15_test.hdf5`. 
* Split train set into 8 shards (`train_00.h5` … `train_07.h5`) for two-GPU DDP sampling.
* Implement `datasets/brain2text.py` with `DistributedSampler`.
* Apply temporal masking + time-warp augmentation. 

✅ Deliverable → `data/processed/` with normalized bins and stats cache.

---

## 🧱 3 · Core Model Implementation (Week 2–3)

| Module       | File                      | Notes                                 |
| :----------- | :------------------------ | :------------------------------------ |
| Preprocessor | `model/preprocessor.py`   | Conv1D stem + channel gates           |
| SSM Encoder  | `model/ssm_encoder.py`    | 8 Mamba blocks (d=512, state=64)      |
| Mode Head    | `model/mode_head.py`      | Infers silent/vocalized latent z      |
| Flow Bridge  | `model/denoise_flow.py`   | 4-step latent diffusion denoiser      |
| Decoders     | `model/rnnt_ctc_heads.py` | RNNT (primary) + CTC (aux)            |
| LM Fusion    | `model/lm_fusion.py`      | Corpus-aware LMs (1–5 gram + neural)  |

**Dual-GPU Split**

* GPU-0 → Encoder + Mode Head
* GPU-1 → Decoders + Flow Bridge + LM fusion
* Synchronized via DDP (NCCL)

✅ Deliverable → `MODE_SSM(nn.Module)` returns phoneme logits + `p(z | X)`.

---

## 🚀 4 · Distributed Training (Week 3–4)

### Launch

```bash
torchrun --nproc_per_node=2 train.py \
  config=configs/train.yaml \
  --ddp true --amp true
```

### Curriculum

1. CTC warm-up (10 epochs)
2. Joint RNNT training (15 epochs)
3. Mode conditioning (10 epochs)
4. Flow bridge fine-tune (5 epochs)

**Optimizers**

```python
AdamW(lr=2e-4, betas=(0.9,0.98), weight_decay=0.05)
scheduler = cosine_with_warmup(5k)
grad_accum = 4
precision = "bf16"
```

Training time ≈ 30 hours wall (2× A100).

✅ Checkpoint → `checkpoints/mode_ssm/best.pt` (val WER ≤ 5.3 %).

---

## 🧠 5 · Test-Time Adaptation (TTA) (Week 5)

```bash
torchrun --nproc_per_node=2 evaluate_model.py \
  --checkpoint checkpoints/mode_ssm/best.pt --tta true
```

Algorithm

1. Compute feature stats (μ, Σ) per session.
2. Run 3–10 steps of entropy minimization on RNNT outputs.
3. Update LayerNorm & final encoder only. 

✅ WER gain ≈ 0.3–0.5 pp.

---

## 🧮 6 · Inference & Submission (Week 6)

```bash
torchrun --nproc_per_node=2 make_submission.py \
  --checkpoint checkpoints/mode_ssm/best.pt \
  --tta true --output submission.csv
```

Output → `id,text` chronologically ordered per rules. 
Local eval: `python evaluate_model.py --wer`.

✅ Deliverable → final submission + one-page summary (upload by Dec 31 2025).

---

## 📊 7 · UV-Driven Reproducibility

| Command                             | Purpose                              |
| :---------------------------------- | :----------------------------------- |
| `uv pip freeze > requirements.txt`  | Lock snapshot for Kaggle upload      |
| `uv run train.py`                   | Run inside locked environment        |
| `uv export --format json > uv.lock` | Exact dependency versions            |
| `uv sync`                           | Instant replication on other machine |

✅ One-command repro across both GPUs / research nodes.

---

## 📅 8 · Timeline Summary

| Week | GPU Usage    | Focus                  | Deliverable            |
| :--- | :----------- | :--------------------- | :--------------------- |
| 1    | Single GPU   | Env + Data Prep        | Loader + uv.lock ready |
| 2–3  | Dual GPU DDP | Model implementation   | MODE_SSM functional    |
| 3–4  | Dual GPU     | Training + tuning      | WER ≤ 5.5 %            |
| 5    | Dual GPU     | TTA + Bridge tests     | WER ≤ 5.0 %            |
| 6    | Dual GPU     | Inference + Submission | Final CSV + summary    |

---

## 💡 Final Tips

* Run `uv cache clean` weekly to avoid disk bloat.
* Enable `torch.compile()` for Mamba blocks on A100 – ≈ 15 % speed boost.
* Set `NCCL_P2P_DISABLE=1` if using mixed A100/4090 pair.
* Mirror checkpoints to `weights/` with `uv run torch.save` so versions are tied to `uv.lock`.
