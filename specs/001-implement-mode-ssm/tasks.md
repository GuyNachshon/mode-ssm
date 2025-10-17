# Tasks: MODE-SSM Brain-to-Text Neural Decoder

**Input**: Design documents from `/specs/001-implement-mode-ssm/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are MANDATORY per constitution (Test-Driven Development NON-NEGOTIABLE). All tests must be written FIRST and FAIL before implementation begins.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `mode_ssm/`, `datasets/`, `configs/`, `tests/`, `scripts/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure for all user stories

- [x] T001 Create project directory structure per implementation plan
- [x] T002 Initialize Python 3.10 virtual environment and install dependencies (PyTorch 2.2, Mamba-SSM 1.2.0, NeMo Toolkit 1.22.0, Hydra-core 1.3.2, h5py, jiwer)
- [x] T003 [P] Configure Git repository with .gitignore for neural data and checkpoints
- [x] T004 [P] Setup basic logging configuration in `configs/logging.yaml`
- [x] T005 [P] Create synthetic neural data generation fixtures in `tests/fixtures/synthetic_neural.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T006 Create base configuration files: `configs/train.yaml`, `configs/model_mode_ssm.yaml`, `configs/tta.yaml`, `configs/lm_fusion.yaml`
- [ ] T007 [P] Implement HDF5 data loading infrastructure in `datasets/brain2text.py` with T15 dataset format validation
- [ ] T008 [P] Create phoneme vocabulary mapping (40-class LOGIT_TO_PHONEME) in `datasets/phoneme_vocab.py`
- [ ] T009 [P] Implement distributed training setup with PyTorch DDP in `scripts/distributed_utils.py`
- [ ] T010 [P] Create checkpoint management system in `mode_ssm/checkpoint_manager.py`
- [ ] T011 Setup WER evaluation metrics using jiwer in `mode_ssm/evaluation_metrics.py`
- [ ] T012 [P] Create training stage management (CTC→Joint→Mode→Denoise curriculum) in `mode_ssm/training_stages.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Model Training and Validation (Priority: P1) 🎯 MVP

**Goal**: Train MODE-SSM neural decoder on T15 dataset achieving validation WER ≤ 6%

**Independent Test**: Can verify by training on subset of T15 data and measuring WER on validation set

### Tests for User Story 1 (MANDATORY - TDD) ⚠️

**NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T013 [P] [US1] Unit test for preprocessor normalization in `tests/unit/test_preprocessor.py`
- [ ] T014 [P] [US1] Unit test for Mamba SSM encoder forward pass in `tests/unit/test_ssm_encoder.py`
- [ ] T015 [P] [US1] Unit test for mode head classification in `tests/unit/test_mode_head.py`
- [ ] T016 [P] [US1] Unit test for RNNT decoder joint network in `tests/unit/test_decoders.py`
- [ ] T017 [P] [US1] Unit test for CTC decoder forward pass in `tests/unit/test_decoders.py`
- [ ] T018 [P] [US1] Integration test for end-to-end training pipeline in `tests/integration/test_end_to_end.py`
- [ ] T019 [P] [US1] Integration test for multi-stage curriculum in `tests/integration/test_curriculum.py`

### Implementation for User Story 1

- [ ] T020 [P] [US1] Implement neural signal preprocessor in `mode_ssm/models/preprocessor.py` (z-score normalization, channel gating, temporal augmentation)
- [ ] T021 [P] [US1] Implement bidirectional Mamba SSM encoder in `mode_ssm/models/ssm_encoder.py` (d_model=512, d_state=64, n_layers=8)
- [ ] T022 [P] [US1] Implement speaking mode classification head in `mode_ssm/models/mode_head.py` (binary silent/vocalized classification)
- [ ] T023 [US1] Implement RNNT decoder with joint network in `mode_ssm/models/rnnt_ctc_heads.py` (depends on T020, T021, T022)
- [ ] T024 [US1] Add CTC decoder for alignment supervision in `mode_ssm/models/rnnt_ctc_heads.py` (depends on T023)
- [ ] T025 [US1] Create main MODE-SSM model class integrating all components in `mode_ssm/models/mode_ssm_model.py`
- [ ] T026 [P] [US1] Implement data augmentation transforms (temporal masking, time warping) in `datasets/transforms.py`
- [ ] T027 [P] [US1] Create distributed data sampler for multi-GPU training in `datasets/distributed_sampler.py`
- [ ] T028 [US1] Implement multi-stage training script in `scripts/train.py` with automatic checkpoint resume
- [ ] T029 [US1] Add comprehensive training metrics logging (loss curves, WER, GPU memory) in `scripts/train.py`
- [ ] T030 [US1] Create model evaluation script in `scripts/evaluate_model.py`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently - can train neural decoder and measure WER

---

## Phase 4: User Story 2 - Competition Submission Generation (Priority: P2)

**Goal**: Generate formatted CSV submissions from trained models for Brain-to-Text 2025 challenge

**Independent Test**: Can test by loading trained checkpoint and processing test data to generate proper CSV format

### Tests for User Story 2 (MANDATORY - TDD) ⚠️

- [ ] T031 [P] [US2] Unit test for CSV submission formatting in `tests/unit/test_submission_format.py`
- [ ] T032 [P] [US2] Unit test for chronological ordering validation in `tests/unit/test_submission_format.py`
- [ ] T033 [P] [US2] Integration test for complete submission generation pipeline in `tests/integration/test_submission_generation.py`

### Implementation for User Story 2

- [ ] T034 [P] [US2] Implement beam search decoding for RNNT in `mode_ssm/models/rnnt_ctc_heads.py`
- [ ] T035 [P] [US2] Create submission formatter ensuring chronological ordering (session→block→trial) in `mode_ssm/submission_formatter.py`
- [ ] T036 [US2] Implement competition submission generation script in `scripts/make_submission.py`
- [ ] T037 [US2] Add submission validation (CSV format, prediction count) in `scripts/validate_submission.py`
- [ ] T038 [US2] Integrate with trained model loading and batch prediction processing

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - can train model and generate valid competition submissions

---

## Phase 5: User Story 3 - Model Performance Optimization (Priority: P3)

**Goal**: Optimize performance through test-time adaptation for neural drift, achieving ≥0.3pp WER improvement

**Independent Test**: Can test by comparing WER before and after TTA on held-out session data

### Tests for User Story 3 (MANDATORY - TDD) ⚠️

- [ ] T039 [P] [US3] Unit test for session statistics adaptation in `tests/unit/test_tta_loop.py`
- [ ] T040 [P] [US3] Unit test for entropy minimization steps in `tests/unit/test_tta_loop.py`
- [ ] T041 [P] [US3] Integration test for multi-session TTA workflow in `tests/integration/test_tta_adaptation.py`

### Implementation for User Story 3

- [ ] T042 [P] [US3] Implement session-level feature statistics adaptation in `mode_ssm/models/tta_loop.py`
- [ ] T043 [P] [US3] Implement entropy minimization for RNNT outputs in `mode_ssm/models/tta_loop.py`
- [ ] T044 [US3] Add TTA integration to evaluation and submission scripts (depends on T042, T043)
- [ ] T045 [P] [US3] Create neural drift analysis script in `scripts/analyze_neural_drift.py`
- [ ] T046 [US3] Implement TTA performance comparison and reporting

**Checkpoint**: All user stories should now be independently functional with TTA optimization

---

## Phase 6: Advanced Features (Optional Components)

**Purpose**: Optional flow bridge denoising component from research

- [ ] T047 [P] [US1] Implement optional diffusion/flow bridge denoiser in `mode_ssm/models/denoise_flow.py`
- [ ] T048 [P] [US1] Add corpus-aware language model fusion in `mode_ssm/models/lm_fusion.py`
- [ ] T049 [US1] Integrate flow bridge into training curriculum (denoise_train stage)

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T050 [P] Add comprehensive error handling and recovery across all scripts
- [ ] T051 [P] Create performance profiling and optimization scripts in `scripts/profile_training.py`
- [ ] T052 [P] Add memory optimization for 512-channel sequences (gradient checkpointing, memory pooling)
- [ ] T053 [P] Create data quality analysis script in `scripts/analyze_data_quality.py`
- [ ] T054 [P] Add hardware-specific optimizations (torch.compile for A100, NCCL configuration)
- [ ] T055 [P] Create comprehensive documentation and usage examples
- [ ] T056 Run quickstart.md validation and update based on implementation
- [ ] T057 [P] Add reproducibility verification across different hardware configurations

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Requires trained model from US1 for testing but implementation is independent
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Requires model architecture from US1 but can develop TTA independently

### Within Each User Story

- **Tests MUST be written and FAIL before implementation** (TDD requirement)
- Models before integration scripts
- Core components before advanced features
- Individual components before full pipeline integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Unit test for preprocessor normalization in tests/unit/test_preprocessor.py"
Task: "Unit test for Mamba SSM encoder forward pass in tests/unit/test_ssm_encoder.py"
Task: "Unit test for mode head classification in tests/unit/test_mode_head.py"

# Launch all model components for User Story 1 together:
Task: "Implement neural signal preprocessor in mode_ssm/models/preprocessor.py"
Task: "Implement bidirectional Mamba SSM encoder in mode_ssm/models/ssm_encoder.py"
Task: "Implement speaking mode classification head in mode_ssm/models/mode_head.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (tests FIRST, then implementation)
4. **STOP and VALIDATE**: Test User Story 1 independently - can train on T15 data and achieve validation WER ≤ 6%
5. Ready for research/competition use

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → **MVP: Can train neural decoder!**
3. Add User Story 2 → Test independently → **Can generate competition submissions!**
4. Add User Story 3 → Test independently → **Can handle neural drift with TTA!**
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (core training pipeline)
   - Developer B: User Story 2 (submission generation)
   - Developer C: User Story 3 (test-time adaptation)
3. Stories complete and integrate independently

---

## Notes

- **TDD Mandatory**: All tests must be written FIRST per constitution
- [P] tasks = different files, no dependencies within same user story
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing (red-green-refactor cycle)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- **Hardware fallback**: All implementations must support single-GPU fallback
- **Reproducibility**: Fixed random seeds and deterministic operations required
- **Memory management**: Implement gradient checkpointing and memory optimization for 512 channels