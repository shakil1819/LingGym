"""
LingGym Qwen2.5-7B Inference on Kaggle GPU

This script runs metalinguistic reasoning inference on the LingGym benchmark
using the Qwen2.5-7B-Instruct model with 4-bit quantization for T4/P100 GPU compatibility.

Expected runtime: ~2-3 hours for Fwe (147 questions)
GPU: NVIDIA T4/P100 (16 GB VRAM)
"""

# Install dependencies
import subprocess
import sys

try:
    print("[SETUP] Installing transformers...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
    print("[SETUP] transformers installed successfully")
except Exception as e:
    print(f"[WARNING] transformers installation failed: {e}")

import os
import json
import re
from pathlib import Path
from datetime import datetime
import gc

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import pandas as pd
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "main"

# Paths (Kaggle environment)
# NOTE: Kaggle only persists /kaggle/working as retrievable output
DATA_DIR = Path("/kaggle/input/linggym-benchmark/Benchmark_multiple_choice/Fwe")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Debug: print full prompt+response for the first N questions to stdout (retrievable in log)
DEBUG_FIRST_N = 3

# Inference configuration
BATCH_SIZE = 1  # Sequential inference
MAX_NEW_TOKENS = 512  # Match paper's setting for full reasoning
DO_SAMPLE = False  # Greedy decoding for reproducibility
TEMPERATURE = 1.0  # Ignored with greedy decoding
TOP_P = 1.0  # Ignored with greedy decoding

# Checkpoint configuration
CHECKPOINT_INTERVAL = 50  # Save progress every N questions
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ============================================================================
# Utility Functions
# ============================================================================


def log_message(msg: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def check_gpu_memory():
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log_message(
            f"GPU Memory: {allocated:.2f}GB / {reserved:.2f}GB reserved / {total:.2f}GB total"
        )


def clear_gpu_memory():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def extract_letter_answer(text: str) -> str:
    """Extract A-D letter from model output (comprehensive logic)."""
    text_upper = text.upper()

    # Patterns in priority order
    patterns = [
        # Direct answers: "The answer is A" or "Answer: A"
        r"(?:THE\s+)?(?:ANSWER|CHOICE|OPTION)\s+(?:IS\s+)?[\s:]*([A-D])(?:\s|$|[.,:)])",
        # Bold answers: **A**
        r"\*\*([A-D])\*\*",
        # Single letter on its own line
        r"^\s*([A-D])\s*$",
        # Letter with suffix: A) A. A]
        r"([A-D])\s*[)\].]",
        # Letter at start of sentence
        r"^([A-D])\b",
        # Any letter A-D
        r"[A-D]",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_upper, re.MULTILINE)
        if match:
            return match.group(1)

    return "?"


def save_checkpoint(results: list, checkpoint_num: int):
    """Save intermediate results."""
    df = pd.DataFrame(results)
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{checkpoint_num:03d}.csv"
    df.to_csv(checkpoint_path, index=False)
    log_message(f"Checkpoint saved: {checkpoint_path} ({len(results)} questions)")


# ============================================================================
# Model Loading
# ============================================================================


def load_model_and_tokenizer():
    """Load Qwen2.5-7B optimized for T4 x2 GPUs."""
    log_message("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    log_message(f"Loading model: {MODEL_NAME}")
    log_message(f"GPUs available: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            log_message(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Optimize for T4 x2: Use FP16 with device_map="auto" for multi-GPU
    try:
        log_message("Loading on T4 GPU(s) with FP16 dtype")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        log_message("✓ Model loaded with multi-GPU support (FP16)")
    except Exception as e:
        log_message(f"Multi-GPU load failed: {e}", level="WARNING")
        log_message("Falling back to single GPU or quantization...", level="WARNING")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            log_message("✓ Model loaded with FP16")
        except Exception as e2:
            log_message(f"FP16 failed, trying 8-bit quantization...", level="WARNING")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
            log_message("✓ Model loaded with 8-bit quantization")

    model.eval()
    check_gpu_memory()
    log_message("Model ready for inference")

    return model, tokenizer


# ============================================================================
# Data Loading
# ============================================================================


def load_benchmark_questions() -> list:
    """Load all question files from benchmark directory."""
    global DATA_DIR

    log_message(f"Loading questions from: {DATA_DIR}")

    if not DATA_DIR.exists():
        log_message(f"WARNING: Data directory not found: {DATA_DIR}", level="WARNING")
        log_message("Searching for benchmark data...", level="WARNING")

        # List what's in /kaggle/input
        input_dir = Path("/kaggle/input")
        if input_dir.exists():
            log_message(f"Contents of /kaggle/input: {[p.name for p in input_dir.iterdir()]}", level="WARNING")

        # Search common Kaggle dataset paths
        search_paths = [
            Path("/kaggle/input/linggym-benchmark/Benchmark_multiple_choice/Fwe"),
            Path("/kaggle/input/datasets/linggym-benchmark/Benchmark_multiple_choice/Fwe"),
            Path("/kaggle/input/datasets/shakil19/linggym-benchmark/Benchmark_multiple_choice/Fwe"),
            Path("/kaggle/input/datasets/Benchmark_multiple_choice/Fwe"),
        ]

        # Recursively search for any *questions.txt files
        if input_dir.exists():
            for item in input_dir.rglob("*_questions.txt"):
                data_dir = item.parent
                if data_dir.name == "Fwe":
                    search_paths.append(data_dir)
                    log_message(f"Found questions file at: {item}", level="INFO")

        for path in search_paths:
            if path.exists():
                log_message(f"Found data at: {path}", level="INFO")
                DATA_DIR = path
                break

        if not DATA_DIR.exists():
            log_message(f"ERROR: Could not find benchmark data in any location", level="ERROR")
            log_message(f"Searched {len(search_paths)} paths", level="ERROR")
            sys.exit(1)

    questions = []
    for question_file in sorted(DATA_DIR.glob("*_questions.txt")):
        with open(question_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        questions.append({"file": question_file.name, "content": content})

        log_message(
            f"  Loaded: {question_file.name} ({content.count('Question ')} questions)"
        )

    total_questions = sum(q["content"].count("Question ") for q in questions)
    log_message(f"Total questions loaded: {total_questions}")

    return questions


# ============================================================================
# Inference
# ============================================================================


def run_inference(model, tokenizer, questions: list) -> list:
    """Run inference on all questions."""
    log_message("Starting inference...")
    results = []
    checkpoint_num = 0

    for file_data in questions:
        content = file_data["content"]
        file_name = file_data["file"]

        # Split into question blocks
        blocks = re.split(r"\n(?=Question \d+:)", content)
        log_message(f"Processing {file_name} ({len(blocks)} blocks)")

        for block_idx, block in enumerate(tqdm(blocks, desc=file_name)):
            if not block.strip().startswith("Question"):
                continue

            lines = block.split("\n")
            if len(lines) < 12:
                continue

            try:
                # Extract question components
                question_num = lines[0]  # "Question X:"

                # Use the exact 10-line prompt format from reference implementation
                # This matches lines 1-10 which includes the instruction line
                prompt = "\n".join(lines[1:11])

                # Correct answer is on line 11 ("Correct Answer: X").
                # IMPORTANT: match the letter AFTER the colon, not the "C" in "Correct".
                correct_line = lines[11] if len(lines) > 11 else ""
                m = re.search(r"Correct Answer:\s*([A-D])", correct_line)
                if not m:
                    # Fallback: last standalone A-D on the line
                    m = re.search(r"([A-D])\s*$", correct_line.strip())
                correct_answer = m.group(1) if m else "?"

                # ---- Generate prediction ----
                messages = [{"role": "user", "content": prompt}]

                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                # Generate with greedy decoding (no sampling for reproducibility)
                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        do_sample=False,
                        max_new_tokens=512,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # Decode response
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]
                response = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                # Extract letter from response
                prediction = extract_letter_answer(response)

                # DEBUG: dump full prompt + response for first few questions
                if len(results) < DEBUG_FIRST_N:
                    log_message("=" * 60, level="DEBUG")
                    log_message(f"DEBUG {file_name} {question_num}", level="DEBUG")
                    log_message(f"--- PROMPT SENT ---\n{prompt}", level="DEBUG")
                    log_message(f"--- RAW RESPONSE ---\n{response}", level="DEBUG")
                    log_message(
                        f"--- EXTRACTED: '{prediction}' | CORRECT: '{correct_answer}' ---",
                        level="DEBUG",
                    )
                    log_message("=" * 60, level="DEBUG")

                # Record result
                result = {
                    "file": file_name,
                    "question_id": question_num,
                    "prediction": prediction,
                    "correct_answer": correct_answer,
                    "match": prediction == correct_answer,
                    "response_preview": response[:100],  # First 100 chars
                }

                results.append(result)

                # Checkpoint every N questions
                if len(results) % CHECKPOINT_INTERVAL == 0:
                    checkpoint_num += 1
                    save_checkpoint(results, checkpoint_num)
                    clear_gpu_memory()

            except Exception as e:
                log_message(f"Error processing question {question_num}: {str(e)}", level="ERROR")
                results.append(
                    {
                        "file": file_name,
                        "question_id": question_num,
                        "prediction": "ERROR",
                        "correct_answer": "?",
                        "match": False,
                        "response_preview": f"ERROR: {str(e)[:30]}",
                    }
                )

    log_message(f"✓ Inference complete. Processed {len(results)} questions.")
    return results


# ============================================================================
# Results Processing & Analysis
# ============================================================================


def process_results(results: list) -> dict:
    """Calculate accuracy and generate summary."""
    log_message("Processing results...")

    df = pd.DataFrame(results)

    # Overall accuracy
    total = len(df)
    correct = df["match"].sum()
    overall_accuracy = correct / total if total > 0 else 0

    log_message(f"Overall Accuracy: {overall_accuracy:.4f} ({correct}/{total})")

    # Per-file accuracy
    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "dataset": "LingGym Fwe",
            "total_questions": total,
            "correct_predictions": int(correct),
            "accuracy": float(overall_accuracy),
        },
        "per_file": {},
    }

    for file_name in sorted(df["file"].unique()):
        file_df = df[df["file"] == file_name]
        file_correct = file_df["match"].sum()
        file_total = len(file_df)
        file_accuracy = file_correct / file_total if file_total > 0 else 0

        summary["per_file"][file_name] = {
            "total": int(file_total),
            "correct": int(file_correct),
            "accuracy": float(file_accuracy),
        }

        log_message(f"  {file_name}: {file_accuracy:.4f} ({file_correct}/{file_total})")

    return df, summary


def save_results(df: pd.DataFrame, summary: dict):
    """Save results to files."""
    log_message("Saving results...")

    # Save full predictions CSV
    predictions_path = OUTPUT_DIR / "predictions.csv"
    df.to_csv(predictions_path, index=False)
    log_message(f"  Predictions saved: {predictions_path}")

    # Save summary JSON
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_message(f"  Summary saved: {summary_path}")

    # Save detailed report
    report_path = OUTPUT_DIR / "detailed_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("LingGym Qwen2.5-7B Inference Results\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Timestamp: {summary['metadata']['timestamp']}\n")
        f.write(f"Model: {summary['metadata']['model']}\n")
        f.write(f"Dataset: {summary['metadata']['dataset']}\n\n")

        f.write("OVERALL RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Questions: {summary['metadata']['total_questions']}\n")
        f.write(f"Correct Predictions: {summary['metadata']['correct_predictions']}\n")
        f.write(f"Accuracy: {summary['metadata']['accuracy']:.4f}\n\n")

        f.write("PER-FILE RESULTS\n")
        f.write("-" * 70 + "\n")
        for file_name, metrics in summary["per_file"].items():
            f.write(f"{file_name}\n")
            f.write(f"  Total: {metrics['total']}\n")
            f.write(f"  Correct: {metrics['correct']}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")

    log_message(f"  Detailed report saved: {report_path}")

    log_message("✓ All results saved to /kaggle/output/")


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main execution flow."""
    log_message("=" * 70)
    log_message("LingGym Qwen2.5-7B Inference on Kaggle GPU")
    log_message("=" * 70)

    # Check GPU
    log_message(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_message(f"GPU Name: {torch.cuda.get_device_name(0)}")
        check_gpu_memory()

    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()

        # Load questions
        questions = load_benchmark_questions()

        # Run inference
        results = run_inference(model, tokenizer, questions)

        # Process results
        df, summary = process_results(results)

        # Save results
        save_results(df, summary)

        log_message("=" * 70)
        log_message("✓ Pipeline completed successfully")
        log_message("=" * 70)

    except Exception as e:
        log_message(f"FATAL ERROR: {str(e)}", level="ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
