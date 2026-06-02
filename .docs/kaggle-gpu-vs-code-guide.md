# Running LingGym on Kaggle GPU from VS Code

**Status**: ✅ Kaggle CLI verified (v2.2.0) | API working | Credentials configured  
**User**: shakil19  
**Date**: June 2, 2026

---

## Quick Start (5 minutes)

```bash
# 1. Verify Kaggle is configured
kaggle --version

# 2. Create a Kaggle notebook
kaggle kernels init -l python

# 3. Push code to Kaggle
kaggle kernels push -p linggym-kernel

# 4. Monitor execution
kaggle kernels status shakil19/linggym-kernel

# 5. Pull results
kaggle kernels output shakil19/linggym-kernel
```

---

## Table of Contents

1. [Prerequisites & Verification](#prerequisites--verification)
2. [Option A: VS Code Remote Kernel Connection](#option-a-vs-code-remote-kernel-connection)
3. [Option B: Kaggle Web IDE + Local Development](#option-b-kaggle-web-ide--local-development)
4. [Setup: Upload LingGym Data to Kaggle](#setup-upload-linggym-data-to-kaggle)
5. [Running LingGym Inference](#running-linggym-inference)
6. [Code Template for Kaggle](#code-template-for-kaggle)
7. [Monitoring & Debugging](#monitoring--debugging)
8. [Saving & Retrieving Results](#saving--retrieving-results)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites & Verification

### ✅ Already Configured

```
Kaggle CLI Version: 2.2.0
Username: shakil19
Auth Method: LEGACY_API_KEY
Config Location: C:\Users\Expert\.kaggle\kaggle.json
API Status: ✓ Working (verified)
```

### Verify Setup

```bash
# Check CLI installation
kaggle --version

# Check credentials
kaggle config view

# Test API connection
kaggle datasets list | head -5
```

**Expected Output**: List of Kaggle datasets (confirms API access is working)

---

## Option A: VS Code Remote Kernel Connection

### Why This Approach

- Edit code locally in VS Code
- Execute on Kaggle GPU (T4 16GB)
- Real-time output streaming
- Persistent storage integration

### Prerequisites

```bash
# In VS Code terminal
pip install kaggle ipython jupyter notebook

# Verify
which kaggle
```

### Step 1: Create Kaggle Notebook Locally

Create `linggym-kernel.ipynb` structure:

```bash
# Navigate to your project
cd e:/misc/LingGym

# Initialize a Kaggle kernel project
kaggle kernels init -l python
```

This creates:
```
kernel-metadata.json
linggym-kernel.ipynb
```

### Step 2: Configure Kernel Metadata

Edit `kernel-metadata.json`:

```json
{
  "id": "shakil19/linggym-kernel",
  "title": "LingGym Qwen2.5-7B Inference",
  "code_file": "linggym-kernel.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["shakil19/linggym-benchmark"],
  "dependency_sources": []
}
```

**Key Settings**:
- `"enable_gpu": true` – T4 GPU allocation
- `"enable_internet": true` – Download Qwen model from HuggingFace
- `"is_private": true` – Private notebook (results not public)

### Step 3: Create Main Notebook

Create `linggym-kernel.ipynb` with cells:

#### Cell 1: Setup & Dependencies

```python
# Install GPU-accelerated dependencies
!pip install -q transformers torch accelerate bitsandbytes
!pip install -q huggingface-hub

# Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

#### Cell 2: Load Model (with 4-bit Quantization)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("Loading Qwen2.5-7B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("✓ Model loaded successfully")
```

#### Cell 3: Load Benchmark Data

```python
import json
from pathlib import Path

# Read from /kaggle/input/linggym-benchmark/
data_dir = Path("/kaggle/input/linggym-benchmark/Benchmark_multiple_choice/Fwe")

questions = []
for file in sorted(data_dir.glob("*_questions.txt")):
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    questions.append({
        'file': file.name,
        'content': content
    })

print(f"Loaded {len(questions)} question files")
print(f"Files: {[q['file'] for q in questions]}")
```

#### Cell 4: Inference Loop

```python
import re
from tqdm import tqdm

results = []

for file_data in questions:
    content = file_data['content']
    blocks = re.split(r"\n(?=Question \d+:)", content)
    
    for block in tqdm(blocks, desc=f"Processing {file_data['file']}"):
        if not block.strip().startswith("Question"):
            continue
        
        lines = block.split('\n')
        if len(lines) < 12:
            continue
        
        question_num = lines[0]
        # Extract prompt (lines 1-10 are the question block)
        prompt = '\n'.join(lines[1:11])
        correct_answer = lines[11].split(":")[-1].strip()[0] if len(lines) > 11 else "?"
        
        # Generate answer
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                do_sample=False,  # Greedy decoding
                max_new_tokens=32,  # Short output
            )
        
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract letter from response
        pred_letter = re.search(r"[A-D]", response.upper())
        pred = pred_letter.group(0) if pred_letter else "?"
        
        results.append({
            'file': file_data['file'],
            'question': question_num,
            'prediction': pred,
            'correct': correct_answer,
            'match': pred == correct_answer
        })

print(f"Completed {len(results)} predictions")
```

#### Cell 5: Calculate Accuracy

```python
import pandas as pd

# Create results dataframe
df = pd.DataFrame(results)

# Calculate accuracy
overall_acc = df['match'].sum() / len(df)
print(f"\nOverall Accuracy: {overall_acc:.4f} ({df['match'].sum()}/{len(df)})")

# Per-file accuracy
print("\nPer-file Results:")
for file in df['file'].unique():
    file_df = df[df['file'] == file]
    acc = file_df['match'].sum() / len(file_df)
    print(f"  {file}: {acc:.4f} ({file_df['match'].sum()}/{len(file_df)})")
```

#### Cell 6: Save Results

```python
# Save to Kaggle output directory
output_dir = Path("/kaggle/output")
output_dir.mkdir(exist_ok=True)

# Save as CSV
df.to_csv(output_dir / "predictions.csv", index=False)

# Save summary
summary = {
    'total_questions': len(df),
    'correct': int(df['match'].sum()),
    'accuracy': float(overall_acc),
    'per_file': {}
}

for file in df['file'].unique():
    file_df = df[df['file'] == file]
    summary['per_file'][file] = {
        'total': len(file_df),
        'correct': int(file_df['match'].sum()),
        'accuracy': float(file_df['match'].sum() / len(file_df))
    }

import json
with open(output_dir / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("✓ Results saved to /kaggle/output/")
```

### Step 4: Upload Data to Kaggle

Before running the kernel, upload benchmark data:

```bash
# Create dataset structure
mkdir -p linggym-benchmark/Benchmark_multiple_choice/Fwe

# Copy Fwe questions (smoke test)
cp e:/misc/LingGym/Benchmark_multiple_choice/Fwe/*.txt linggym-benchmark/Benchmark_multiple_choice/Fwe/

# Create dataset on Kaggle
kaggle datasets create -p linggym-benchmark

# Output shows: Successfully created dataset shakil19/linggym-benchmark
```

### Step 5: Push Notebook to Kaggle

```bash
# From your project directory with kernel-metadata.json
kaggle kernels push -p .

# Output:
# Kernel version 1 successfully created.
# https://www.kaggle.com/shakil19/linggym-kernel
```

### Step 6: Monitor Execution from VS Code

```bash
# Option 1: Watch status
kaggle kernels status shakil19/linggym-kernel

# Option 2: Continuous monitoring
while true; do kaggle kernels status shakil19/linggym-kernel; sleep 10; done

# Option 3: Get logs (when complete)
kaggle kernels output shakil19/linggym-kernel -p output/
```

### Step 7: Pull Results to Local

```bash
# Download results
kaggle kernels output shakil19/linggym-kernel -p ./kaggle-results/

# View results locally
cat kaggle-results/summary.json
```

---

## Option B: Kaggle Web IDE + Local Development

### Workflow

1. **Edit locally** in VS Code
2. **Push updates** to Kaggle
3. **Run in Kaggle web IDE**
4. **Pull results** back to local

### Commands

```bash
# Push updated notebook
kaggle kernels push -p .

# Switch to Kaggle web IDE to run
# https://www.kaggle.com/code/shakil19/linggym-kernel

# Pull results after execution
kaggle kernels output shakil19/linggym-kernel -p ./results/
```

---

## Setup: Upload LingGym Data to Kaggle

### Method 1: Using Kaggle CLI (Recommended for first time)

```bash
# 1. Create local directory structure
mkdir linggym-dataset
mkdir linggym-dataset/Benchmark_multiple_choice

# 2. Copy all language folders (or just Fwe for smoke test)
cp -r e:/misc/LingGym/Benchmark_multiple_choice/Fwe \
      linggym-dataset/Benchmark_multiple_choice/

# 3. Create dataset metadata
cat > linggym-dataset/dataset-metadata.json << EOF
{
  "id": "shakil19/linggym-benchmark",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ],
  "keywords": [
    "linguistics",
    "llm",
    "benchmark",
    "metalinguistic-reasoning"
  ],
  "title": "LingGym Benchmark Dataset",
  "ref": "shakil19/linggym-benchmark"
}
EOF

# 4. Upload dataset to Kaggle
kaggle datasets create -p linggym-dataset

# Output: Successfully created dataset shakil19/linggym-benchmark
```

### Method 2: Upload All 18 Languages (for full benchmark)

```bash
# Same as above, but copy all language folders
cp -r e:/misc/LingGym/Benchmark_multiple_choice/* \
      linggym-dataset/Benchmark_multiple_choice/

# Then create dataset
kaggle datasets create -p linggym-dataset
```

### Verify Upload

```bash
# List files in dataset
kaggle datasets files shakil19/linggym-benchmark

# Should show Fwe or all language folders
```

---

## Running LingGym Inference

### Full Inference (All 18 Languages)

**Modify kernel metadata**:

```json
{
  "dataset_sources": ["shakil19/linggym-benchmark-full"],
  "enable_gpu": true,
  "enable_internet": true
}
```

**Modify Cell 3 to load all languages**:

```python
import os
from pathlib import Path

data_dir = Path("/kaggle/input/linggym-benchmark-full/Benchmark_multiple_choice")

questions = []
for lang_dir in sorted(data_dir.iterdir()):
    if lang_dir.is_dir():
        for file in sorted(lang_dir.glob("*_questions.txt")):
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            questions.append({
                'language': lang_dir.name,
                'file': file.name,
                'content': content
            })

print(f"Loaded {len(questions)} files across languages")
```

**Add language tracking to Cell 4**:

```python
results.append({
    'language': file_data['language'],  # Add this
    'file': file_data['file'],
    'question': question_num,
    'prediction': pred,
    'correct': correct_answer,
    'match': pred == correct_answer
})
```

**Update summary by language (Cell 5)**:

```python
# Per-language results
print("\nPer-Language Accuracy:")
for lang in df['language'].unique():
    lang_df = df[df['language'] == lang]
    acc = lang_df['match'].sum() / len(lang_df)
    print(f"  {lang}: {acc:.4f} ({lang_df['match'].sum()}/{len(lang_df)})")
```

---

## Code Template for Kaggle

### Complete Minimal Template

```python
# Cell 1: Setup
!pip install -q transformers torch accelerate bitsandbytes

# Cell 2: Configuration
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from pathlib import Path
import re
import json
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = Path("/kaggle/input/linggym-benchmark/Benchmark_multiple_choice/Fwe")
OUTPUT_DIR = Path("/kaggle/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cell 3: Load Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# Cell 4: Inference
results = []

for question_file in sorted(DATA_DIR.glob("*_questions.txt")):
    with open(question_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    blocks = re.split(r"\n(?=Question \d+:)", content)
    
    for block in tqdm(blocks, desc=question_file.name):
        if not block.strip():
            continue
        
        lines = block.split('\n')
        if len(lines) < 12:
            continue
        
        prompt = '\n'.join(lines[1:11])
        correct = lines[11].split(":")[-1].strip()[0]
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=32
            )
        
        output = tokenizer.batch_decode(
            [output_ids[0][len(model_inputs.input_ids[0]):]], 
            skip_special_tokens=True
        )[0]
        
        pred = re.search(r"[A-D]", output.upper())
        pred = pred.group(0) if pred else "?"
        
        results.append({
            'file': question_file.name,
            'prediction': pred,
            'correct': correct,
            'match': pred == correct
        })

# Cell 5: Summary & Save
import pandas as pd

df = pd.DataFrame(results)
accuracy = df['match'].sum() / len(df)

print(f"\nAccuracy: {accuracy:.4f} ({df['match'].sum()}/{len(df)})")

df.to_csv(OUTPUT_DIR / "results.csv", index=False)
with open(OUTPUT_DIR / "summary.json", 'w') as f:
    json.dump({
        'accuracy': float(accuracy),
        'total': len(df),
        'correct': int(df['match'].sum())
    }, f, indent=2)

print("✓ Results saved")
```

---

## Monitoring & Debugging

### Monitor from VS Code Terminal

```bash
# Check notebook status
kaggle kernels status shakil19/linggym-kernel

# Output examples:
# Status: running
# Status: complete
# Status: error
```

### Common Status Codes

| Status | Meaning | Action |
|--------|---------|--------|
| `queued` | Waiting for GPU | Wait 1-5 min |
| `running` | Currently executing | Monitor progress |
| `complete` | Finished successfully | Pull results |
| `error` | Failed with error | Check logs |
| `timedout` | Exceeded time limit | Reduce scope |

### View Error Logs

```bash
# After error, get output
kaggle kernels output shakil19/linggym-kernel -p ./debug/

# Check error file (if exists)
cat debug/stderr.txt
```

### Debug GPU Memory Issues

Add to notebook:

```python
import torch
import gc

def check_memory():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB used")
    print(f"GPU Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")

check_memory()

# Clear cache between batches
gc.collect()
torch.cuda.empty_cache()
check_memory()
```

---

## Saving & Retrieving Results

### Save During Execution

```python
# Save incrementally (every 100 questions)
if i % 100 == 0:
    temp_df = pd.DataFrame(results)
    temp_df.to_csv(f"/kaggle/output/checkpoint_{i}.csv")
    print(f"✓ Checkpoint saved at question {i}")
```

### Pull Results from Kaggle

```bash
# Download all outputs
kaggle kernels output shakil19/linggym-kernel -p ./results/

# View downloaded files
ls -la results/

# Open results locally
cat results/summary.json
cat results/predictions.csv  # View in Excel/VS Code
```

### Archive Results

```bash
# Create archive
tar -czf linggym-results-$(date +%Y%m%d).tar.gz results/

# Back up to cloud (optional)
# Upload to Google Drive, GitHub, etc.
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution**: Add more aggressive quantization

```python
# Use bfloat16 instead of float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Change this
    bnb_4bit_quant_type="nf4",
)
```

### Issue: "Model takes too long to load"

**Solution**: Model is downloading from HuggingFace for first time

```
Expected: 3-5 minutes on first run (14GB download)
Subsequent runs: 10-30 seconds
✓ This is normal
```

### Issue: "Dataset not found in /kaggle/input/"

**Solution**: Re-upload dataset or check name

```bash
# Verify dataset exists on Kaggle
kaggle datasets list -u shakil19

# Should show: shakil19/linggym-benchmark

# If missing, re-upload
kaggle datasets create -p linggym-dataset
```

### Issue: "Kernel timed out after X hours"

**Solution**: Kaggle has 12-hour session limit

```python
# Reduce batch size or number of questions
# Process in multiple runs (checkpoint after each)
```

### Issue: "Import errors for transformers"

**Solution**: Pip packages sometimes conflict

```bash
# Pin versions in notebook
!pip install -q transformers==4.46.0 torch==2.1.2 accelerate==0.31.0 bitsandbytes
```

---

## Performance Tips

### Optimize Inference Speed

```python
# Batch multiple questions
batch_size = 4
for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i+batch_size]
    model_inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
    # Process batch
```

### Reduce VRAM Usage

```python
# Use 8-bit instead of 4-bit if OOM still occurs
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Less aggressive quantization
)
```

### Speed Up Data Loading

```python
# Cache question files in memory
questions_cache = {}
for file in DATA_DIR.glob("*.txt"):
    with open(file, 'r') as f:
        questions_cache[file.name] = f.read()
```

---

## Complete Workflow

### Day 1: Setup (30 minutes)

```bash
# 1. Upload benchmark data
cd e:/misc/LingGym
mkdir linggym-dataset/Benchmark_multiple_choice
cp -r Benchmark_multiple_choice/Fwe linggym-dataset/Benchmark_multiple_choice/
kaggle datasets create -p linggym-dataset

# 2. Create kernel project
mkdir kaggle-linggym
cd kaggle-linggym
kaggle kernels init -l python
# Copy linggym-kernel.ipynb template
# Edit kernel-metadata.json
```

### Day 2-3: Run Smoke Test (4 hours)

```bash
# 1. Push notebook
cd kaggle-linggym
kaggle kernels push -p .

# 2. Monitor
kaggle kernels status shakil19/linggym-kernel

# 3. Wait ~1-2 hours for Fwe inference
# 4. Pull results
kaggle kernels output shakil19/linggym-kernel -p ./results/
```

### Day 4-7: Full Benchmark (if using all 18 languages)

```bash
# 1. Re-upload full dataset
cp -r e:/misc/LingGym/Benchmark_multiple_choice/* linggym-dataset/
kaggle datasets create -p linggym-dataset-full

# 2. Update notebook metadata
# 3. Push and run (20-40 hours total)
```

---

## References & Resources

- [Kaggle Kernels Documentation](https://www.kaggle.com/docs/kernels)
- [Kaggle CLI Reference](https://github.com/Kaggle/kaggle-api)
- [Transformers Library Quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)

---

## Next Steps

1. **Verify setup**: Run the quick start commands above
2. **Upload data**: `kaggle datasets create -p linggym-dataset`
3. **Create kernel**: Push initial notebook
4. **Test inference**: Run on Fwe first (147 questions, ~2-3 hours)
5. **Scale up**: Extend to all languages after verification

---

**Document Version**: 1.0  
**Last Updated**: June 2, 2026  
**Status**: Ready for implementation
