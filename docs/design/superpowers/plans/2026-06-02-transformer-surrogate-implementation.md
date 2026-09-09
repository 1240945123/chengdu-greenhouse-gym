# Transformer Surrogate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Transformer dynamics surrogate that predicts the next GreenLight state from historical state/action/weather sequences.

**Architecture:** Reuse the PINN data-generation and normalization style. Add sequence-window construction, a PyTorch Transformer encoder model, and train/evaluate scripts that save model artifacts and JSON metrics.

**Tech Stack:** Python, PyTorch, NumPy, unittest, GreenLight-Gym2 environment utilities.

---

### Task 1: Add Failing Transformer Tests

**Files:**
- Create: `tests/transformer_surrogate.py`

- [ ] Write tests for Transformer output shape, sequence dataset construction, normalizer save/load, and training loss path.
- [ ] Run `python -m unittest tests.transformer_surrogate` and verify it fails because the transformer package does not exist yet.

### Task 2: Add Transformer Model

**Files:**
- Create: `glassgym/models/transformer/__init__.py`
- Create: `glassgym/models/transformer/surrogate.py`

- [ ] Implement `TransformerDynamicsSurrogate`.
- [ ] Implement `train_one_epoch`.
- [ ] Run `python -m unittest tests.transformer_surrogate` and verify model tests progress.

### Task 3: Add Sequence Dataset Builder

**Files:**
- Create: `glassgym/models/transformer/sequence_dataset.py`

- [ ] Implement `build_sequence_dataset`.
- [ ] Implement `generate_mixed_sequences`.
- [ ] Run `python -m unittest tests.transformer_surrogate` and verify all unit tests pass.

### Task 4: Add Train And Evaluate Scripts

**Files:**
- Create: `configs/agents/transformer.yml`
- Create: `experiments/train_transformer.py`
- Create: `experiments/evaluate_transformer.py`

- [ ] Train from mixed sequence data and save `transformer_model.pt` plus `normalizer.npz`.
- [ ] Evaluate one-step and multi-step errors and save `transformer_metrics.json`.

### Task 5: Smoke Train And Verify

**Files:**
- Generated: `train_data/transformer/transformer_model.pt`
- Generated: `train_data/transformer/normalizer.npz`
- Generated: `results/transformer/transformer_metrics.json`

- [ ] Run a short smoke training job.
- [ ] Run evaluation.
- [ ] Run unit tests again.
