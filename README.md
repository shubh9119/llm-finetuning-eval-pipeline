# End-to-End LLM SFT / LoRA Fine-Tuning & Automated CI/CD Evaluation Gate

[![CI Pipeline](https://img.shields.io/badge/CI-Passing-brightgreen)](#)
[![Fine-Tuning](https://img.shields.io/badge/PEFT-LoRA%20%7C%204--Bit%20QLoRA-blue)](#)
[![Evaluation](https://img.shields.io/badge/Quality%20Gate-P01--P12%20Diagnostics-purple)](#)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-informational)](#)

Production **LLM Post-Training (SFT / LoRA / 4-bit QLoRA)** and **Automated CI/CD Quality Gate** pipeline. Combines ShareGPT chat-template standardization, LoRA adapter training across attention & MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), P01–P12 RAG/LLM failure pattern diagnostics, and automated deployment blocking when hallucination or latency thresholds regress.

---

## Post-Training & CI/CD Quality Gate Architecture

```mermaid
flowchart LR
    subgraph Data["Pre/Post-Training Data Curation"]
        RAW["Instruction & Preference Corpus"]
        STD["ShareGPT Standardization & Dedup"]
        TPL["Llama-3 Chat Template Tokenizer"]
    end

    subgraph Train["4-Bit QLoRA SFT Training"]
        BASE["Base LLM (4-bit Quantized)"]
        LORA["LoRA Adapters (r=16, Attention + MLP)"]
        SFT["TRL SFTTrainer (BF16 / FP16 Auto-Detect)"]
    end

    subgraph Gate["Automated CI/CD Evaluation Gate"]
        DIAG["P01-P12 Failure Pattern Diagnostics"]
        METRICS["Faithfulness / Hallucination / P95 Eval"]
        DECIDE{"Passes Thresholds?"}
    end

    RAW --> STD --> TPL --> SFT
    BASE --> LORA --> SFT
    SFT --> DIAG & METRICS --> DECIDE
    DECIDE -->|"Pass (<5% Hallucination)"| DEPLOY(["Promote to Production Registry"])
    DECIDE -->|"Fail"| BLOCK(["Auto-Block CI/CD Deployment"])
```

---

## Evaluation & Triage Sequence Workflow

```mermaid
sequenceDiagram
    autonumber
    participant D as Developer / CI Trigger
    participant T as LoRA SFT Trainer
    participant G as Quality Gate (quality_gate.py)
    participant C as Diagnostics Clinic (P01-P12)

    D->>T: Launch LoRA SFT fine-tuning job
    T->>T: Configure pad_token & hardware-aware BF16/FP16 flags
    T-->>G: Export fine-tuned adapter checkpoint
    G->>G: Evaluate Faithfulness (>=0.90), Hallucination (<=0.05), P95 (<=500ms)
    alt Threshold Violation Detected
        G->>C: Run P01-P12 root-cause failure pattern triage
        C-->>D: Emit structured JSON failure report & block release
    else All Thresholds Passed
        G-->>D: Approve model promotion
    end
```

---

## Key Bug Fixes & Engineering Improvements

1. **Hardware-Safe BF16/FP16 Detection (`finetune_sft_lora.py`):** Guarded `torch.cuda.is_bf16_supported()` with `torch.cuda.is_available()` so importing or validating `SFTConfig` on CPU-only CI runners does not crash.
2. **Tokenizer Pad Token Initialization:** Explicitly assigned `tokenizer.pad_token = tokenizer.eos_token` when `pad_token is None`, resolving sequence padding errors during batched `DataCollatorForSeq2Seq` collation.
3. **Automated CI/CD Deployment Blocker (`quality_gate.py`):** Added deterministic threshold validation (`evaluate_ci_quality_gate`) and P01–P12 taxonomy verification (`validate_diagnosis_pattern_id`) with unit tests executed on every push via GitHub Actions.

---

## Quickstart

```bash
git clone https://github.com/shubh9119/llm-finetuning-eval-pipeline.git
cd llm-finetuning-eval-pipeline
pip install -r requirements.txt
pytest -v
```
