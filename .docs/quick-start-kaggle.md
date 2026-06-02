# Quick Start: Run LingGym on Kaggle GPU (5 Steps)

**Target**: Run Qwen2.5-7B inference on LingGym benchmark (Fwe, 147 questions)  
**Environment**: Kaggle GPU (T4, 16GB VRAM)  
**Local IDE**: VS Code  
**Time to completion**: 1-2 hours setup + 1 hour execution

---

## Prerequisites

- ✅ Kaggle account (shakil19)
- ✅ Kaggle API configured (`~/.kaggle/kaggle.json`)
- ✅ Kaggle CLI installed (`kaggle --version` returns 2.2.0+)
- ✅ Files ready in `e:/misc/LingGym/kaggle-linggym/`

**Verify setup**:
```bash
kaggle --version
# Output: Kaggle CLI 2.2.0
```

---

## Step 1: Upload Dataset to Kaggle (10 minutes)

### Option A: Via Web UI (Recommended if CLI has issues)

1. Go to: https://www.kaggle.com/settings/account
2. Click "Create" → "Dataset"
3. Name: `linggym-benchmark`
4. Upload the folder: `e:/misc/LingGym/kaggle-linggym/linggym-dataset/`
5. Set to **Private**
6. Click "Create"
7. Get the dataset ID from the URL: `https://www.kaggle.com/datasets/shakil19/linggym-benchmark`

### Option B: Via Kaggle CLI

```bash
cd e:/misc/LingGym/kaggle-linggym/linggym-dataset
kaggle datasets create -p . --dir-mode zip
```

**Expected output**:
```
Dataset created successfully: shakil19/linggym-benchmark
```

---

## Step 2: Configure Kernel (5 minutes)

Edit `e:/misc/LingGym/kaggle-linggym/kernel-metadata.json`:

```json
{
  "id": "shakil19/linggym-kernel",
  "title": "LingGym Qwen2.5-7B Inference",
  "code_file": "linggym_inference.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "shakil19/linggym-benchmark"   // <-- Update if dataset name is different
  ],
  "tags": ["linguistics", "nlm", "llm", "benchmark", "metalinguistic-reasoning"]
}
```

**Key settings**:
- `"enable_gpu": true` → Use T4 GPU
- `"dataset_sources"` → Must match your uploaded dataset name

---

## Step 3: Push Kernel to Kaggle (2 minutes)

In VS Code terminal:

```bash
cd e:/misc/LingGym/kaggle-linggym
kaggle kernels push -p .
```

**Expected output**:
```
Kernel version 1 successfully created.
https://www.kaggle.com/code/shakil19/linggym-kernel
```

---

## Step 4: Monitor Execution (1-2 hours)

### Real-time Monitoring from Terminal

```bash
# Check status
kaggle kernels status shakil19/linggym-kernel

# Continuous monitor (every 10 sec)
for i in {1..720}; do
  echo "[$(date)] Status:"
  kaggle kernels status shakil19/linggym-kernel
  sleep 10
done
```

### Web Dashboard

Visit: https://www.kaggle.com/code/shakil19/linggym-kernel

Watch for status changes:
- `queued` → Waiting for GPU (1-5 min)
- `running` → Executing (50-70 min typical)
- `complete` → Done ✓

### Expected Timeline

```
Time 0:00    Kernel queued (waiting for GPU)
Time 0:05    GPU allocated, dependencies installing
Time 0:10    Model downloading from HuggingFace (2-3 min)
Time 0:15    Inference starting
Time 1:00    Inference complete
Time 1:05    Results saved to /kaggle/output/
```

---

## Step 5: Retrieve Results (5 minutes)

Once status is `complete`:

```bash
# Download results
kaggle kernels output shakil19/linggym-kernel -p ./kaggle-results/

# View results
cat kaggle-results/summary.json

# View predictions
cat kaggle-results/predictions.csv | head -20

# View detailed report
cat kaggle-results/detailed_report.txt
```

---

## Expected Output

### summary.json

```json
{
  "timestamp": "2026-06-02T21:55:00.000000",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "dataset": "LingGym Fwe",
  "total_questions": 147,
  "correct_predictions": 104,
  "accuracy": 0.7074,
  "per_file": {
    "min_knowledge_points_4_questions.txt": {
      "total": 43,
      "correct": 31,
      "accuracy": 0.7209
    },
    ...
  }
}
```

### detailed_report.txt

```
======================================================================
LingGym Qwen2.5-7B Inference Results
======================================================================

Timestamp: 2026-06-02T21:55:00.000000
Model: Qwen/Qwen2.5-7B-Instruct
Dataset: LingGym Fwe

OVERALL RESULTS
----------------------------------------------------------------------
Total Questions: 147
Correct Predictions: 104
Accuracy: 0.7074

PER-FILE RESULTS
----------------------------------------------------------------------
min_knowledge_points_4_questions.txt
  Total: 43
  Correct: 31
  Accuracy: 0.7209
...
```

---

## Troubleshooting

### "Dataset not found in /kaggle/input/"

**Solution**: Dataset not properly linked to kernel

1. Verify dataset exists: `kaggle datasets list | grep linggym`
2. Update `kernel-metadata.json` with correct dataset name
3. Re-push: `kaggle kernels push -p .`

### "CUDA out of memory"

**Solution**: Model not fitting with quantization

Edit `linggym_inference.py`:
```python
# Change line ~115:
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Changed from float16
    bnb_4bit_quant_type="nf4",
)
```

Then re-push:
```bash
kaggle kernels push -p .
```

### "Kernel timeout after 12 hours"

**Solution**: Normal limit on Kaggle (12-hour session max)

- For 147 questions (Fwe): Should complete in ~1 hour ✓
- For all 19,620 questions: Would need multiple kernels or weekly splits

---

## What's Running on Kaggle

### Model Configuration

```
Model:              Qwen2.5-7B-Instruct
Quantization:       4-bit (NF4)
VRAM Usage:         6-8 GB
GPU:                NVIDIA T4 (16 GB total)
Decoding:           Greedy (deterministic)
Max Output Tokens:  32 (for single letter answer)
```

### Code Structure

**Three execution options** (pick one):

#### Option 1: Python Script (Default)
```bash
# Runs: linggym_inference.py (13.5 KB)
# Input: /kaggle/input/linggym-benchmark/Benchmark_multiple_choice/Fwe/
# Output: /kaggle/output/{summary.json, predictions.csv, detailed_report.txt}
```

#### Option 2: Jupyter Notebook
```bash
# Runs: linggym_inference.ipynb (12.7 KB)
# 7 cells with full documentation
# Same inputs/outputs as Python script
```

#### Option 3: Manual Run in Kaggle Web Editor
1. Go to kernel URL
2. Edit code directly
3. Click "Run All"
4. View output in browser

---

## Monitoring from VS Code

### Terminal Commands

```bash
# Watch logs in real-time (if available)
kaggle kernels pull shakil19/linggym-kernel -p /tmp/kernel
tail -f /tmp/kernel/linggym_inference.py

# Get kernel output
kaggle kernels output shakil19/linggym-kernel

# List all your kernels
kaggle kernels list -u shakil19
```

### Python Monitoring Script

Create `monitor.py`:

```python
#!/usr/bin/env python3
import subprocess
import time
from datetime import datetime

def check_status():
    result = subprocess.run(
        ['kaggle', 'kernels', 'status', 'shakil19/linggym-kernel'],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

print("Monitoring kernel execution...")
while True:
    status = check_status()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status}")
    
    if 'complete' in status.lower():
        print("✓ Kernel execution complete!")
        break
    elif 'error' in status.lower():
        print("✗ Kernel failed!")
        break
    
    time.sleep(30)
```

Run with:
```bash
python monitor.py
```

---

## After Getting Results

### Save to Repository

```bash
# Copy results
cp -r kaggle-results/ .docs/kaggle-results-2026-06-02/

# Create summary
cat > .docs/kaggle-results-summary.md << EOF
# Results Summary

- Model: Qwen2.5-7B-Instruct
- Dataset: LingGym Fwe
- Accuracy: $(cat kaggle-results/summary.json | grep accuracy)
- Date: $(date)

See kaggle-results-2026-06-02/ for full details.
EOF

# Commit to git
git add .docs/kaggle-results*
git commit -m "Add Kaggle GPU inference results"
```

### Analyze Results

```bash
# Generate analysis
python << EOF
import json
import pandas as pd

# Load results
with open('kaggle-results/summary.json') as f:
    summary = json.load(f)

df = pd.read_csv('kaggle-results/predictions.csv')

print(f"Overall Accuracy: {summary['accuracy']:.4f}")
print(f"\nPaper Baseline: 0.7109")
print(f"Difference: {(summary['accuracy'] - 0.7109):.4f}")

print("\nPer-file breakdown:")
for file_name, metrics in summary['per_file'].items():
    print(f"  {file_name}: {metrics['accuracy']:.4f}")
EOF
```

---

## Success Checklist

After completion, you should have:

- [x] Kernel created on Kaggle
- [x] Execution completed (status: `complete`)
- [x] Results downloaded to local machine
- [x] `summary.json` shows accuracy
- [x] `predictions.csv` has 147 rows
- [x] Results saved to `.docs/` folder
- [x] Changes committed to git

---

## Next Steps

### Scale to Full Benchmark

To run all 18 languages (19,620 questions):

1. Upload full benchmark: `cp -r Benchmark_multiple_choice/* linggym-dataset/`
2. Create new dataset: `kaggle datasets create -p linggym-dataset-full`
3. Create weekly kernels:
   - Week 1: Languages A-E
   - Week 2: Languages F-I
   - Week 3: Languages J-N
   - Week 4: Languages O-R

4. Aggregate results:
   ```bash
   cat results-week{1,2,3,4}/predictions.csv > all-results.csv
   ```

---

## Reference

| Item | Value |
|------|-------|
| Dataset | shakil19/linggym-benchmark |
| Kernel | shakil19/linggym-kernel |
| Model | Qwen/Qwen2.5-7B-Instruct |
| Questions (smoke test) | 147 (Fwe) |
| Questions (full) | 19,620 (all languages) |
| Expected Accuracy | 71.09% (Qwen2.5-7B baseline) |
| Runtime (smoke test) | ~60 minutes |
| Runtime (full) | ~40-80 hours |
| Monthly Kaggle GPU quota | 120+ hours |

---

**Created**: June 2, 2026  
**Status**: Ready to execute  
**Questions?**: See implementation-log-option-a.md for detailed guide
