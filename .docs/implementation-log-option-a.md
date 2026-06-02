# LingGym on Kaggle GPU - Implementation Log & Guide

**Date**: June 2, 2026  
**Status**: Implementation in Progress  
**Approach**: Option A - VS Code + Remote Kernel (Recommended)

---

## Executive Summary

This document logs the implementation of LingGym Qwen2.5-7B inference on Kaggle GPU using Option A (VS Code + Remote Kernel). The setup allows:

- **Local Development**: Edit and manage code in VS Code
- **Remote Execution**: Run on Kaggle GPU (T4, 16GB VRAM)
- **Real-time Monitoring**: Track execution from VS Code terminal
- **Result Streaming**: Pull results back to local machine

---

## Implementation Status

### ✅ Completed

- [x] Directory structure created: `kaggle-linggym/`
- [x] Benchmark data prepared: Fwe (147 questions)
- [x] Kernel metadata configured: `kernel-metadata.json`
- [x] Main inference script written: `linggym_inference.py`
  - 4-bit quantization enabled (6-8GB VRAM usage)
  - Greedy decoding (reproducible results)
  - Checkpoint system (save every 50 questions)
  - Comprehensive logging
- [x] Jupyter notebook created: `linggym_inference.ipynb`
  - 7-cell notebook with all steps
  - Compatible with Kaggle notebooks
  - Documentation included in each cell
- [x] Dataset metadata prepared: `dataset-metadata.json`

### ⏳ In Progress

- [ ] Dataset upload to Kaggle (CLI auth issue being resolved)
- [ ] Kernel push to Kaggle
- [ ] Execution on Kaggle GPU
- [ ] Results retrieval

### 📋 Project Structure

```
e:/misc/LingGym/
├── kaggle-linggym/                    # Main Kaggle kernel project
│   ├── kernel-metadata.json           # Kernel configuration
│   ├── linggym_inference.py           # Main Python script (13.5 KB)
│   ├── linggym_inference.ipynb        # Jupyter notebook (12.7 KB)
│   └── linggym-dataset/               # Dataset to upload
│       ├── dataset-metadata.json
│       └── Benchmark_multiple_choice/
│           └── Fwe/                   # 147 questions (7 files)
│               ├── min_knowledge_points_4_questions.txt
│               ├── min_knowledge_points_5_questions.txt
│               ├── min_knowledge_points_6_questions.txt
│               ├── min_knowledge_points_8_questions.txt
│               ├── min_knowledge_points_10_questions.txt
│               ├── min_knowledge_points_12_questions.txt
│               └── min_knowledge_points_13_questions.txt
│
├── .docs/                             # Documentation
│   ├── kaggle-gpu-vs-code-guide.md
│   ├── free-cloud-gpu-feasibility-report.md
│   ├── setup-verification.md
│   └── implementation-log-option-a.md (this file)
│
├── .venv/                             # Python virtual environment
├── Benchmark_multiple_choice/         # Full benchmark (all 18 languages)
├── CVS-format/                        # CSV files
├── IGT-format/                        # IGT files
└── shuffled_multiple/                 # Smoke test folder (Fwe)
```

---

## Configuration Details

### Kernel Metadata (`kernel-metadata.json`)

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
    "shakil19/linggym-benchmark"
  ],
  "tags": [
    "linguistics",
    "nlp",
    "llm",
    "benchmark",
    "metalinguistic-reasoning"
  ]
}
```

**Key Settings**:
- `"enable_gpu": true` → T4 GPU allocation (16GB VRAM)
- `"enable_internet": true` → Download Qwen model from HuggingFace
- `"dataset_sources"` → Reference to benchmark dataset
- `"is_private": true` → Results not publicly visible

### Inference Script (`linggym_inference.py`)

**Features**:
- **Model**: Qwen2.5-7B-Instruct with 4-bit quantization
- **VRAM Usage**: ~6-8 GB (fits on T4)
- **Decoding**: Greedy (do_sample=False) for reproducibility
- **Max Tokens**: 32 (sufficient for single letter answer)
- **Checkpointing**: Save every 50 questions
- **Logging**: Comprehensive timestamps and status

**Functions**:
- `load_model_and_tokenizer()` → Load Qwen with BitsAndBytes quantization
- `load_benchmark_questions()` → Read question files from Kaggle dataset
- `run_inference()` → Process all questions with error handling
- `process_results()` → Calculate accuracy metrics
- `save_results()` → Export CSV, JSON, and report

**Outputs**:
- `predictions.csv` - All predictions with metadata
- `summary.json` - Accuracy summary by file and overall
- `detailed_report.txt` - Human-readable results
- `checkpoint_*.csv` - Intermediate checkpoints

### Jupyter Notebook (`linggym_inference.ipynb`)

**7 Cells**:
1. Install dependencies
2. Verify GPU and setup
3. Load model with 4-bit quantization
4. Load questions from dataset
5. Run inference loop
6. Calculate accuracy
7. Save results

---

## Execution Workflow

### Step 1: Upload Dataset to Kaggle

**Via Kaggle Web UI (if CLI has auth issues)**:

1. Go to https://www.kaggle.com/datasets?fileType=zip
2. Click "Create New Dataset"
3. Name: `linggym-benchmark`
4. Upload zip file with:
   ```
   Benchmark_multiple_choice/
   └── Fwe/
       └── *.txt files (7 files, 202 KB total)
   ```
5. Set to **Private**
6. Note the dataset URL/ID

**Via CLI (Recommended)**:

```bash
cd e:/misc/LingGym/kaggle-linggym/linggym-dataset
kaggle datasets create -p . --dir-mode zip
```

Expected output:
```
Starting dataset creation...
Dataset created successfully: shakil19/linggym-benchmark
```

### Step 2: Update Kernel Metadata

If dataset name is different, update `kernel-metadata.json`:

```json
"dataset_sources": [
  "shakil19/linggym-benchmark"  // Update if different
]
```

### Step 3: Push Kernel to Kaggle

From VS Code terminal in `kaggle-linggym/` directory:

```bash
cd e:/misc/LingGym/kaggle-linggym
kaggle kernels push -p .
```

Expected output:
```
Kernel version 1 successfully created.
https://www.kaggle.com/code/shakil19/linggym-kernel
```

### Step 4: Monitor Execution

**Option A: Real-time Status**:

```bash
# Check status
kaggle kernels status shakil19/linggym-kernel

# Continuous monitoring (every 10 seconds)
while true; do
  kaggle kernels status shakil19/linggym-kernel
  echo "---"
  sleep 10
done
```

**Option B: Web Dashboard**:

Visit: `https://www.kaggle.com/code/shakil19/linggym-kernel`

Status indicators:
- `queued` → Waiting for GPU (1-5 minutes typical)
- `running` → In progress
- `complete` → Finished, results ready
- `error` → Failed (check logs)

**Expected Timeline**:

```
Queued:     1-5 minutes
Loading:    2-3 minutes (model download)
Inference:  ~45-60 minutes (147 questions × ~20-30 sec each)
Total:      ~50-70 minutes
```

### Step 5: Retrieve Results

Once execution is complete:

```bash
# Download all outputs
kaggle kernels output shakil19/linggym-kernel -p ./kaggle-results/

# View results
cat kaggle-results/summary.json
cat kaggle-results/predictions.csv | head -10
```

**Output Files**:

```
kaggle-results/
├── summary.json              # Accuracy metrics (JSON)
├── predictions.csv           # All predictions (CSV)
├── detailed_report.txt       # Human-readable report
├── checkpoint_001.csv        # Checkpoint at 50 questions
├── checkpoint_002.csv        # Checkpoint at 100 questions
└── ... (more checkpoints)
```

---

## Expected Results

### Paper Baseline (Qwen2.5-7B on S+G+KP+T)

```
Overall Accuracy: 71.09%
```

### What to Expect from Kaggle Run

```
Expected Accuracy: 70-71%
(Minimal difference from paper due to quantization)
```

### Example Output

```
overall_accuracy=0.7109 (147/147)

Fwe Results by File:
- min_knowledge_points_4_questions.txt:  0.7200 (31/43)
- min_knowledge_points_5_questions.txt:  0.7188 (23/32)
- min_knowledge_points_6_questions.txt:  0.6923 (9/13)
- min_knowledge_points_8_questions.txt:  0.7143 (15/21)
- min_knowledge_points_10_questions.txt: 1.0000 (1/1)
- min_knowledge_points_12_questions.txt: 0.7500 (3/4)
- min_knowledge_points_13_questions.txt: 0.6970 (23/33)
```

---

## Troubleshooting

### Issue: Dataset Not Found in /kaggle/input/

**Cause**: Dataset not uploaded or linked

**Solution**:
1. Upload dataset to Kaggle (see Step 1)
2. Get dataset ID from Kaggle
3. Update `kernel-metadata.json`
4. Re-push kernel

```bash
kaggle datasets list  # Find your dataset
# Update kernel-metadata.json with correct name
kaggle kernels push -p .
```

### Issue: "CUDA out of memory"

**Cause**: 4-bit quantization not working properly

**Solution**: Modify inference script to use more aggressive quantization:

```python
# In linggym_inference.py, change:
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Change from float16
    bnb_4bit_quant_type="nf4",
)
```

Then re-push:

```bash
kaggle kernels push -p .
```

### Issue: Kernel Timeout (> 12 hours)

**Cause**: Kaggle has 12-hour session limit

**Solution**: Use checkpoint system (already implemented)
- Automatically saves every 50 questions
- Run multiple kernels if processing all 19,620 questions

### Issue: API Authentication Error

**Cause**: Kaggle credentials not recognized

**Solution**:

```bash
# Re-authenticate
kaggle auth login

# Or manually set credentials
export KAGGLE_USERNAME=shakil19
export KAGGLE_KEY=your_api_key_here
```

Then retry:

```bash
kaggle kernels push -p .
```

---

## Advanced Usage

### Running All 18 Languages

Modify the setup to run full benchmark:

1. **Upload full dataset**:
   ```bash
   # Copy all languages
   cp -r Benchmark_multiple_choice/* linggym-dataset/Benchmark_multiple_choice/
   kaggle datasets create -p linggym-dataset --dir-mode zip
   ```

2. **Update kernel metadata**:
   ```json
   "dataset_sources": ["shakil19/linggym-benchmark-full"]
   ```

3. **Modify inference script** - Load all languages:
   ```python
   DATA_DIR = Path("/kaggle/input/linggym-benchmark-full/Benchmark_multiple_choice")
   for lang_dir in sorted(DATA_DIR.iterdir()):
       if lang_dir.is_dir():
           # Process each language
   ```

4. **Expected runtime**: 40-80 hours total (fits within monthly Kaggle quota)

### Batch Processing Multiple Kernels

For faster execution across multiple languages:

```bash
# Create separate kernels for each language
for lang in Fwe Gyeli Ik Japhug Kagayanen ...; do
  # Copy specific language
  cp Benchmark_multiple_choice/$lang ...
  # Update kernel metadata
  # Push kernel
  kaggle kernels push -p linggym-kernel-$lang
done
```

---

## Version Control & Documentation

### Save Implementation to Git

```bash
cd e:/misc/LingGym
git add kaggle-linggym/
git add .docs/implementation-log-option-a.md
git commit -m "Add Kaggle GPU inference setup (Option A: VS Code + Remote Kernel)"
git push
```

### Document Results

After execution, create a results document:

```bash
# Copy results to repo
cp -r kaggle-results/ .docs/kaggle-results-<DATE>/

# Create summary markdown
cat > .docs/kaggle-results-summary.md << 'EOF'
# Kaggle GPU Inference Results

**Date**: $(date)
**Model**: Qwen2.5-7B-Instruct
**Dataset**: LingGym Fwe
**Accuracy**: (from summary.json)

... include summary and analysis ...
EOF
```

---

## Next Steps

1. **Resolve dataset upload** (CLI auth workaround)
2. **Push kernel** to Kaggle
3. **Monitor execution** (~60 minutes)
4. **Retrieve and analyze results**
5. **Document findings** in .docs folder
6. **Commit to git** with results

---

## Files Created

| File | Size | Purpose |
|------|------|---------|
| `kernel-metadata.json` | 512 B | Kaggle kernel configuration |
| `linggym_inference.py` | 13.5 KB | Main inference script (Python) |
| `linggym_inference.ipynb` | 12.7 KB | Jupyter notebook version |
| `dataset-metadata.json` | 680 B | Kaggle dataset configuration |
| `Benchmark_multiple_choice/Fwe/*.txt` | 202 KB | Question data (7 files) |

**Total size for upload**: ~250 KB (very fast)

---

## Resources

- **Kaggle Kernel Docs**: https://www.kaggle.com/docs/kernels
- **Kaggle CLI Reference**: https://github.com/Kaggle/kaggle-api
- **Qwen Documentation**: https://qwen.readthedocs.io/
- **BitsAndBytes**: https://github.com/TimDettmers/bitsandbytes

---

## Sign-off

**Implemented by**: Claude Code  
**Status**: Ready for Kaggle push  
**Last Updated**: June 2, 2026 21:45 UTC  
**Next Action**: Resolve dataset upload and push kernel to Kaggle

---

## Appendix: Quick Command Reference

```bash
# Setup
cd e:/misc/LingGym/kaggle-linggym

# 1. Upload dataset
cd linggym-dataset
kaggle datasets create -p . --dir-mode zip
cd ..

# 2. Push kernel
kaggle kernels push -p .

# 3. Monitor
kaggle kernels status shakil19/linggym-kernel

# 4. Get results
kaggle kernels output shakil19/linggym-kernel -p ./results/

# 5. View summary
cat results/summary.json
```

---
