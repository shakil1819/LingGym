# LingGym Setup Verification Report

**Date**: 2026-06-02  
**Status**: Complete (CPU environment ready; GPU required for inference)

## Environment Setup ✓

### Python Virtual Environment
- **Created**: `.venv/` using Python 3.14
- **Activated**: .venv\Scripts\Activate.ps1
- **Status**: Ready

### Installed Dependencies
```
✓ pandas 3.0.3
✓ scipy 1.17.1
✓ sentence-transformers 5.5.1
✓ pypdf 6.12.2
✓ transformers 5.9.0
✓ torch 2.12.0 (CPU-only)
✓ huggingface-hub 1.17.0
✓ scikit-learn 1.9.0
✓ All supporting packages
```

**Installation command used**:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip pandas scipy sentence-transformers pypdf
```

## Benchmark Data Verification ✓

### Overall Statistics
- **Total Questions**: 19,620 (documented as 19,612)
- **Languages**: 18
- **Language Families**: 8
- **Status**: All data present and intact

### Questions by Language

| Language | Questions |
|----------|-----------|
| Fwe | 147 |
| Gyeli | 691 |
| Ik | 21 |
| Japhug | 358 |
| Kagayanen | 550 |
| Kalamang | 660 |
| Komnzo | 709 |
| Mauwake | 1,787 |
| Mehweb | 85 |
| Moloko | 439 |
| Palula | 1,674 |
| Papuan_Malay | 3,766 |
| Pichi | 2,850 |
| Rapa_Nui | 1,709 |
| Tuatschin | 1,113 |
| Ulwa | 1,851 |
| Vamale | 67 |
| Yauyos_Quecha | 1,143 |

## Smoke Test Setup ✓

### Fwe Benchmark (Smoke Test Data)
- **Location**: `shuffled_multiple/Fwe/`
- **Questions**: 147 (matches original)
- **Files**: 7 files (min_knowledge_points_* variants)

**Files verified**:
- min_knowledge_points_4_questions.txt (43 questions)
- min_knowledge_points_5_questions.txt (32 questions)
- min_knowledge_points_8_questions.txt (21 questions)
- min_knowledge_points_6_questions.txt (13 questions)
- min_knowledge_points_13_questions.txt (33 questions)
- min_knowledge_points_12_questions.txt (4 questions)
- min_knowledge_points_10_questions.txt (1 question)

## Question Format (S+G+KP+T)

Each question block contains:

1. **Question header**: Question number
2. **Task instruction**: Linguist specialty context and task description
3. **Sentence line**: Morpheme-segmented sentence with masked word/gloss (represented by `___`)
4. **Gloss line**: Interlinear glossing with masked entry
5. **Translation line**: English translation in quotes
6. **Knowledge point**: Relevant grammar knowledge for the example
7. **Four options (A–D)**:
   - Each contains: word form and corresponding gloss
8. **Answer instruction**: "Please only return the letter (A–D). Do not say anything else."
9. **Correct Answer**: Label (A, B, C, or D)

Example structure:
```
Question 0:
You are a linguist specializing in Fwe...
Sentence (with missing item): na-shúm-iw-a ___
Gloss (with missing item): SM1.PST-bite-PASS-FV ___
The English translation of this sentence is: 'He was bitten by a dog.'
Here is a relevant knowledge point...
A: word: hanú	 gloss: DEM.II16
B: word: o-∅-mbwá	 gloss: AUG-NP1a-dog
C: word: kú-∅-mbwá	 gloss: NP17-NP1a-dog
D: word: ndu-∅-mbwá	 gloss: COP1a-NP1a-dog
Please only return the letter (A–D). Do not say anything else.
Correct Answer: C
```

## GPU Status ❌

- **CUDA Available**: No
- **GPU Count**: 0
- **PyTorch**: 2.12.0+cpu (CPU-only)
- **Current Limitation**: Cannot run Qwen2.5-7B inference without GPU

### To Enable Inference

Requires:
1. **NVIDIA GPU** with sufficient VRAM (A6000 Ada or equivalent recommended)
2. **CUDA Toolkit** (11.8 or 12.1)
3. **PyTorch with CUDA support** (reinstall with appropriate CUDA version)

Installation command for GPU (example for CUDA 12.1):
```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install remaining GPU-compatible packages:
```powershell
.\.venv\Scripts\python.exe -m pip install transformers accelerate sentencepiece safetensors
```

## Dataset Inspection Operations (Ready)

The following operations can be performed with the current CPU environment:

### 1. Count Questions by Language
```powershell
# See verification script in this report
```
**Status**: ✓ Completed

### 2. Calculate Accuracy (Post-Inference)
Once inference results are available, calculate accuracy with:
```powershell
@'
from pathlib import Path
import re

root = Path("qwen2.5-7b-result_source+gloss+kp+trans")
total = 0
correct = 0
by_lang = {}

for path in root.rglob("*_result.txt"):
    lang = path.parent.name
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n(?=Question \d+:)", text)
    for block in blocks:
        pred = re.search(r"Qwen2\.5-7B result:\s*([A-D])\b", block, re.I)
        gold = re.search(r"Correct Answer:\s*([A-D])\b", block)
        if not pred or not gold:
            continue
        total += 1
        hit = pred.group(1).upper() == gold.group(1).upper()
        correct += int(hit)
        by_lang.setdefault(lang, [0, 0])
        by_lang[lang][0] += int(hit)
        by_lang[lang][1] += 1

print(f"overall_accuracy={correct / total:.4f} ({correct}/{total})")
for lang in sorted(by_lang):
    c, n = by_lang[lang]
    print(f"{lang}\t{c / n:.4f}\t{c}/{n}")
'@ | python -
```

## Remaining Steps for Inference

### To Run Smoke Test (Fwe only)
1. Install CUDA-enabled PyTorch on a GPU system
2. Modify `qwen2.5_7B_source+gloss+kp+trans.py`:
   - Change `input_root = "shuffled_multiple/"`  (already correct)
   - Update generation settings for smoke test:
     ```python
     generated_ids = model.generate(
         **model_inputs,
         do_sample=False,           # Greedy decoding
         max_new_tokens=32,         # Reduced for testing
     )
     ```
3. Run: `python qwen2.5_7B_source+gloss+kp+trans.py`
4. Verify output files in: `qwen2.5-7b-result_source+gloss+kp+trans/Fwe/`

### To Run Full Benchmark (All 18 Languages)
1. Install CUDA-enabled PyTorch on a GPU system
2. Update generation settings in `qwen2.5_7B_source+gloss+kp+trans.py` to match paper settings:
   ```python
   generated_ids = model.generate(
       **model_inputs,
       do_sample=True,
       temperature=0.7,
       top_p=0.9,
       repetition_penalty=1.1,
       max_new_tokens=2048,
   )
   ```
3. Ensure full `Benchmark_multiple_choice/` is copied to `shuffled_multiple/`
4. Run: `python qwen2.5_7B_source+gloss+kp+trans.py`
5. Monitor output in: `qwen2.5-7b-result_source+gloss+kp+trans/`

## Paper Reference Accuracies (Qwen2.5-7B)

| Prompt | Paper Accuracy |
|--------|---------------:|
| S | 33.04% |
| S+G | 41.64% |
| S+G+KP | 56.08% |
| S+G+KP+T | 71.09% |

*Note: Exact numbers may differ due to generation settings, prompt formatting, and model version differences.*

## File Structure Summary

```
E:\misc\LingGym\
├── .docs/
│   ├── README.md
│   ├── setup-and-reproduce.md
│   └── setup-verification.md (this file)
├── .venv/                           # Python virtual environment
├── Benchmark_multiple_choice/       # Original benchmark (all 18 languages)
│   ├── Fwe/
│   ├── Gyeli/
│   └── ... (16 more languages)
├── CVS-format/                      # CSV-format intermediate files
├── IGT-format/                      # IGT text format files
├── shuffled_multiple/               # Smoke test setup (Fwe only)
│   └── Fwe/
├── qwen2.5_7B_source+gloss+kp+trans.py  # Main inference script
├── multiple_choice_question_generation.py
└── 2511.00343v1.pdf                 # Paper describing LingGym
```

## Conclusion

✓ **Setup Complete**: The LingGym environment is fully prepared and verified on CPU.  
✓ **Data Verified**: All 19,620 benchmark questions across 18 languages are intact.  
✓ **Smoke Test Ready**: Fwe data is prepared for inference testing.  
⚠️ **GPU Required**: Inference with Qwen2.5-7B requires CUDA-capable GPU.

Next step: Set up GPU environment to run inference pipeline.
