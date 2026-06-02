# LingGym Setup and Reproduction Guide

This guide is based on the current repository contents and the local paper
`2511.00343v1.pdf`.

## What This Repository Contains

- `Benchmark_multiple_choice/`: released multiple-choice benchmark files.
- `CVS-format/`: processed CSV files with knowledge points and aligned IGT
  content. The folder name is `CVS-format` in this repo.
- `IGT-format/`: cleaned IGT text files.
- `multiple_choice_question_generation.py`: helper script for generating
  multiple-choice questions from one language's CSV folder.
- `qwen2.5_7B_source+gloss+kp+trans.py`: Qwen2.5-7B inference script for the
  richest prompt setting, equivalent to sentence + gloss + knowledge point +
  translation (`S+G+KP+T`).
- `2511.00343v1.pdf`: paper describing LingGym.

The paper reports 18 languages, 8 language families, and 19,612 aligned
knowledge-point/IGT examples. The checked-in benchmark files match that total:

```powershell
Get-ChildItem Benchmark_multiple_choice -Recurse -File -Filter '*_questions.txt' |
  ForEach-Object {
    Select-String -Path $_.FullName -Pattern '^Question ' -AllMatches |
      Measure-Object |
      Select-Object -ExpandProperty Count
  } |
  Measure-Object -Sum
```

Expected sum: `19612`.

## Paper Task Summary

LingGym evaluates metalinguistic reasoning over low-resource language reference
grammar material. The core task is word-gloss inference:

1. An IGT sentence is shown with one word/gloss pair masked.
2. The model receives some combination of sentence, gloss line, knowledge point,
   and English translation.
3. The model chooses one answer from `A`, `B`, `C`, or `D`.
4. Accuracy is the primary metric.

The paper's main prompt settings are:

- `S`: source/morpheme-segmented sentence only.
- `S+G`: sentence plus gloss line.
- `S+G+KP`: sentence, gloss line, and relevant grammar knowledge point.
- `S+G+KP+T`: sentence, gloss line, knowledge point, and English translation.

The local Qwen script only targets `S+G+KP+T`.

## Environment Setup

Use Python 3.10 or 3.11. The repo does not include a `requirements.txt`, so
dependencies must be installed from imports.

### CPU-only Dataset Inspection

For inspecting data, regenerating questions, and calculating accuracy:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas scipy sentence-transformers pypdf
```

### GPU Inference Setup

For Qwen2.5-7B inference, install PyTorch for your CUDA version from the
official PyTorch selector, then install the remaining packages:

```powershell
python -m pip install transformers accelerate sentencepiece safetensors
```

The checked-in inference script loads `Qwen/Qwen2.5-7B-Instruct` in `float16`
with `device_map="auto"`. A CUDA GPU with sufficient VRAM is expected. The paper
used A6000 Ada GPUs. CPU inference is technically possible only after code
changes, but it is not a practical reproduction path for the benchmark.

If the Hugging Face model download is gated or rate-limited, authenticate first:

```powershell
huggingface-cli login
```

## Dataset Inventory

Count questions by language:

```powershell
Get-ChildItem Benchmark_multiple_choice -Directory |
  Sort-Object Name |
  ForEach-Object {
    $lang = $_.Name
    $q = (
      Get-ChildItem $_.FullName -File -Filter '*_questions.txt' |
        ForEach-Object {
          (Select-String -Path $_.FullName -Pattern '^Question ' -AllMatches |
            Measure-Object).Count
        } |
        Measure-Object -Sum
    ).Sum
    [PSCustomObject]@{Language=$lang; Questions=[int]$q}
  } |
  Format-Table -AutoSize
```

Expected totals:

| Language | Questions |
| --- | ---: |
| Fwe | 147 |
| Gyeli | 691 |
| Ik | 21 |
| Japhug | 358 |
| Kagayanen | 550 |
| Kalamang | 656 |
| Komnzo | 709 |
| Mauwake | 1787 |
| Mehweb | 85 |
| Moloko | 439 |
| Palula | 1674 |
| Papuan_Malay | 3766 |
| Pichi | 2846 |
| Rapa_Nui | 1709 |
| Tuatschin | 1113 |
| Ulwa | 1851 |
| Vamale | 67 |
| Yauyos_Quecha | 1143 |

## Reproduce From Released Benchmark

This is the most direct local reproduction path because the released
multiple-choice data already exists in `Benchmark_multiple_choice/`.

### 1. Prepare Input Folder Expected By Script

The Qwen script expects `input_root = "shuffled_multiple/"`, while this repo's
released benchmark folder is `Benchmark_multiple_choice/`.

Either copy the benchmark:

```powershell
Copy-Item -Recurse Benchmark_multiple_choice shuffled_multiple
```

Or edit `qwen2.5_7B_source+gloss+kp+trans.py`:

```python
input_root = "Benchmark_multiple_choice/"
```

The output folder defaults to:

```python
output_root = "qwen2.5-7b-result_source+gloss+kp+trans/"
```

### 2. Run Qwen2.5-7B Inference

```powershell
python "qwen2.5_7B_source+gloss+kp+trans.py"
```

The script:

- walks every `.txt` file under the input root;
- identifies language from the folder name;
- reconstructs each prompt block from fixed line offsets;
- writes model output under `qwen2.5-7b-result_source+gloss+kp+trans/<language>/`;
- copies the local `Correct Answer` line into the result file.

### 3. Calculate Accuracy

After inference, calculate exact-letter accuracy:

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

Paper reference for Qwen2.5-7B on all languages:

| Prompt | Paper accuracy |
| --- | ---: |
| `S` | 33.04 |
| `S+G` | 41.64 |
| `S+G+KP` | 56.08 |
| `S+G+KP+T` | 71.09 |

Expect exact numbers to differ unless generation settings and prompt formatting
are aligned with the paper.

## Match Paper Generation Settings

Appendix A of the paper reports:

| Parameter | Value |
| --- | --- |
| Temperature | `0.7` |
| Top-p | `0.9` |
| Max tokens | `2048` |
| Repetition penalty | `1.1` |
| Decoding | sampling-based |

The local Qwen script currently calls:

```python
generated_ids = model.generate(**model_inputs, max_new_tokens=512)
```

For closer paper reproduction, update it to:

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

For deterministic local smoke tests, use greedy decoding instead:

```python
generated_ids = model.generate(**model_inputs, do_sample=False, max_new_tokens=32)
```

Do not compare deterministic smoke-test accuracy directly to the paper table.

## Regenerate Multiple-Choice Questions From CSV

`multiple_choice_question_generation.py` can generate question files from CSV
data, but it is not a complete final-benchmark reproduction script as checked
in.

Important limitations:

- `input_folder` and `output_txt_folder` are hard-coded to a Google Drive Fwe
  path.
- The prompt text hard-codes `Fwe`.
- It writes the correct option as `A` and does not perform the paper's final
  answer-position randomization.
- It does not write a `Correct Answer:` line.
- It generates only the full prompt shape, not all `S`, `S+G`, `S+G+KP`, and
  `S+G+KP+T` variants.

To run it locally for Fwe:

1. Edit the bottom of `multiple_choice_question_generation.py`:

```python
input_folder = "CVS-format/Fwe"
output_txt_folder = "generated_multiple/Fwe"
generate_mcq_txt_per_csv(input_folder, output_txt_folder)
```

2. Run:

```powershell
python multiple_choice_question_generation.py
```

For another language, also update the prompt line that says:

```python
"You are a linguist specializing in Fwe..."
```

Use the generated files as intermediate artifacts, not as the final randomized
benchmark.

## Full Paper Reproduction Scope

The paper evaluates Qwen2.5, Gemma 3, DeepSeek-R1, LLaMA3, and GPT-4/o4-mini
variants across multiple prompt settings. This repository does not include a
complete orchestration layer for all models, prompt variants, CoT/no-CoT runs,
vLLM serving, answer shuffling, or aggregation tables.

To reproduce the full paper from this repo, add these missing pieces:

1. Prompt-variant builder for `S`, `S+G`, `S+G+KP`, `S+G+KP+T`, and optional
   CoT variants.
2. Model runner abstraction for local `transformers`, `vLLM`, and API-backed
   models.
3. Sampling configuration matching Appendix A.
4. Output parser that extracts the first valid `A`-`D` answer.
5. Aggregation scripts by language, language family, prompt setting, model, and
   linguistic subfield.
6. Regeneration script that shuffles answer positions and writes `Correct
   Answer:` consistently.

## Practical Smoke Test

Before launching full inference, test one or two files:

1. Temporarily copy one language folder:

```powershell
New-Item -ItemType Directory -Force shuffled_multiple | Out-Null
Copy-Item -Recurse Benchmark_multiple_choice\Fwe shuffled_multiple\Fwe
```

2. Optionally reduce generation length in the Qwen script:

```python
generated_ids = model.generate(**model_inputs, do_sample=False, max_new_tokens=32)
```

3. Run inference:

```powershell
python "qwen2.5_7B_source+gloss+kp+trans.py"
```

4. Confirm result files are created under:

```text
qwen2.5-7b-result_source+gloss+kp+trans/Fwe/
```

Then restore paper-style sampling before any paper-comparable run.

## Known Reproduction Caveats

- The local benchmark data is present and count-verifiable.
- Raw LaTeX source extraction from Language Science Press grammars is described
  in the paper but not fully implemented in the checked-in code.
- `multiple_choice_question_generation.py` appears to be a helper or earlier
  generation script, not the final released benchmark generator.
- The inference script assumes fixed line positions in each question block.
  Changes to benchmark formatting can break prompt reconstruction.
- The script imports `BitsAndBytesConfig` but does not use it.
- The paper used both `vLLM` and `transformers`; the local script uses
  `transformers` only.
- Paper-comparable accuracy requires matching decoding settings, prompt
  variants, model revisions, hardware precision, and answer parsing.

