# Tutorial: Reproducing LingGym Fwe Accuracy on Kaggle

This tutorial explains the Kaggle code path that ran Qwen2.5-7B-Instruct on the
LingGym Fwe benchmark and produced `63.27%` accuracy (`93/147`). It connects the
implementation to the research paper, defines the main concepts, and explains
why this run is a close reproduction variant rather than an exact paper run.

## 1. Goal

Run a focused LingGym reproduction on Kaggle:

- Language subset: `Fwe`
- Question count: `147`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Prompt setting: `S+G+KP+T`
- Platform: Kaggle GPU
- Observed result from `C:\tmp\Downloads\linggym-kernel (11).log`: `0.6327`
  accuracy, or `93/147`

The relevant implementation is:

```text
kaggle-linggym/
+-- kernel-metadata.json
+-- linggym_inference.py
+-- linggym-dataset/
    +-- Benchmark_multiple_choice/
        +-- Fwe/
```

## 2. Definitions

### LingGym

LingGym is a benchmark for testing whether large language models can reason
about low-resource language grammar using structured linguistic evidence from
reference grammars.

### Fwe

Fwe is one of the 18 languages in the LingGym benchmark. In the paper's released
dataset table, Fwe has `147` examples. The Kaggle run used exactly this Fwe
subset.

### IGT

IGT means Interlinear Glossed Text. It is a linguistic representation where a
sentence is broken into aligned lines:

- `gsrc`: source/orthographic sentence
- `gll`: morpheme-segmented line
- `gls`: gloss line explaining grammatical meaning
- `glt`: free English translation

Example structure:

```text
\gsrc yende
\gll end-e
\gls go-PFV.SBJV
\glt 'Go!'
```

### Knowledge Point

A knowledge point, or `KP`, is a grammar explanation associated with one or more
IGT examples. It gives the rule or linguistic context needed to solve the
question.

### Word-Gloss Inference

Word-gloss inference is the paper's multiple-choice cloze task:

1. One word and its gloss are masked.
2. The model sees the remaining context.
3. The model chooses one of `A`, `B`, `C`, or `D`.

### Prompt Components

The paper defines prompt settings using shorthand:

| Symbol | Meaning |
| --- | --- |
| `S` | Source or morpheme-segmented sentence |
| `G` | Gloss line |
| `KP` | Knowledge point from the grammar |
| `T` | English translation |

The Kaggle run used `S+G+KP+T`, the richest prompt setting.

## 3. The Core Formula

Accuracy is the primary metric:

```text
accuracy = number_of_correct_predictions / total_number_of_questions
```

For this Kaggle run:

```text
accuracy = 93 / 147 = 0.632653...
accuracy_percent = 0.632653... * 100 = 63.27%
```

Per-file accuracy uses the same formula:

```text
accuracy_file_i = correct_file_i / total_file_i
```

The log reports:

| File | Accuracy | Correct / Total |
| --- | ---: | ---: |
| `min_knowledge_points_10_questions.txt` | `0.0000` | `0/1` |
| `min_knowledge_points_12_questions.txt` | `0.5000` | `2/4` |
| `min_knowledge_points_13_questions.txt` | `0.4848` | `16/33` |
| `min_knowledge_points_4_questions.txt` | `0.6279` | `27/43` |
| `min_knowledge_points_5_questions.txt` | `0.6562` | `21/32` |
| `min_knowledge_points_6_questions.txt` | `0.6154` | `8/13` |
| `min_knowledge_points_8_questions.txt` | `0.9048` | `19/21` |
| Overall | `0.6327` | `93/147` |

## 4. Research Paper Alignment

### Dataset Alignment

The paper reports Fwe has `147` benchmark examples. The Kaggle log confirms the
same count:

```text
Total questions loaded: 147
```

The loaded files were:

```text
min_knowledge_points_10_questions.txt: 1
min_knowledge_points_12_questions.txt: 4
min_knowledge_points_13_questions.txt: 33
min_knowledge_points_4_questions.txt: 43
min_knowledge_points_5_questions.txt: 32
min_knowledge_points_6_questions.txt: 13
min_knowledge_points_8_questions.txt: 21
```

These sum to:

```text
1 + 4 + 33 + 43 + 32 + 13 + 21 = 147
```

### Prompt Alignment

The prompt sent in the log contains all four paper components:

```text
Sentence (with missing item): ...
Gloss (with missing item): ...
The English translation of this sentence is: ...
Here is a relevant knowledge point for this example...
A: word: ... gloss: ...
B: word: ... gloss: ...
C: word: ... gloss: ...
D: word: ... gloss: ...
```

That is `S+G+KP+T`.

### Model Alignment

The Kaggle script uses:

```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
```

This aligns with the paper's Qwen2.5-7B instruction-tuned model family.

### Metric Alignment

The paper reports standard multiple-choice accuracy. The Kaggle script computes:

```python
overall_accuracy = correct / total if total > 0 else 0
```

This matches the paper's metric.

### Result Alignment

The paper's Fwe row for `Qwen2.5-7B` reports:

| Prompt | Paper Fwe accuracy |
| --- | ---: |
| `S` | `35.37` |
| `S+G` | `44.22` |
| `S+G+KP` | `59.18` |
| `S+G+KP+T` | `65.31` |

The Kaggle run produced:

```text
63.27
```

This is close to the paper's `65.31` for `Qwen2.5-7B` on Fwe with `S+G+KP+T`.
The gap is:

```text
65.31 - 63.27 = 2.04 percentage points
```

That is a reasonable reproduction delta because the Kaggle run intentionally
uses deterministic greedy decoding and different generation settings from the
paper.

## 5. Why The Result Is Similar But Not Exact

The implementation matches the paper in the most important parts:

- Same Fwe question count: `147`
- Same model family and model size: `Qwen2.5-7B-Instruct`
- Same task: multiple-choice word-gloss inference
- Same rich prompt setting: `S+G+KP+T`
- Same metric: exact letter accuracy

The implementation differs from the paper in generation settings:

| Setting | Paper | Kaggle run |
| --- | --- | --- |
| Decoding | sampling-based | greedy |
| Temperature | `0.7` | not used because `do_sample=False` |
| Top-p | `0.9` | not used because `do_sample=False` |
| Max tokens | `2048` | `512` |
| Runtime backend | `vLLM` and `transformers` | `transformers` |

The greedy setting was a pragmatic choice for Kaggle:

- It is reproducible: same prompt usually gives the same answer.
- It avoids stochastic variance on a small `147`-question subset.
- It is faster and simpler to debug.
- The task asks for a single letter, so `512` generated tokens are already more
  than enough when the prompt says "Please only return the letter".

Because decoding changed, the result should be described as:

```text
Kaggle deterministic reproduction variant: 63.27% Fwe S+G+KP+T accuracy.
```

It should not be described as an exact replication of the paper's sampled run.

## 6. Code Walkthrough

### 6.1 Kernel Metadata

`kaggle-linggym/kernel-metadata.json` configures Kaggle execution:

```json
{
  "code_file": "linggym_inference.py",
  "kernel_type": "script",
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["shakil19/linggym-benchmark"]
}
```

Justification:

- `enable_gpu=true`: Qwen2.5-7B inference is not practical on CPU.
- `enable_internet=true`: the model is downloaded from Hugging Face.
- `dataset_sources`: attaches the uploaded Fwe benchmark.

### 6.2 Model Configuration

The script sets:

```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 512
DO_SAMPLE = False
```

The model is loaded as:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
```

Justification:

- `device_map="auto"` lets `transformers` place layers across available GPUs.
- `torch.float16` reduces memory usage enough for Kaggle T4 GPUs.
- `trust_remote_code=True` is commonly used for model families that require
  custom architecture code.

The successful log confirms:

```text
GPU Name: Tesla T4
GPUs available: 2
Model loaded with multi-GPU support (FP16)
```

### 6.3 Dataset Discovery

The script first tries:

```python
DATA_DIR = Path("/kaggle/input/linggym-benchmark/Benchmark_multiple_choice/Fwe")
```

In the successful log, that exact path was not present. The script then searched
under `/kaggle/input` and found:

```text
/kaggle/input/datasets/shakil19/linggym-benchmark/Fwe
```

Justification:

- Kaggle sometimes mounts uploaded datasets under paths that differ from the
  expected local folder structure.
- Recursive fallback discovery makes the kernel robust to Kaggle mount layout
  differences.

### 6.4 Prompt Construction

Each question block in the benchmark has fixed lines:

```text
Question 0:
instruction
Sentence ...
Gloss ...
Translation ...
Knowledge point ...
A ...
B ...
C ...
D ...
Please only return ...
Correct Answer: X
```

The script uses:

```python
prompt = "\n".join(lines[1:11])
```

This sends the model the instruction, input context, options, and output
constraint, but not the gold answer.

Then it extracts the gold answer with:

```python
m = re.search(r"Correct Answer:\s*([A-D])", correct_line)
```

Justification:

- The model must not see `Correct Answer`.
- The parser must match the answer letter after the colon, not the `C` in the
  word `Correct`.

### 6.5 Chat Formatting

The prompt is wrapped with the tokenizer's chat template:

```python
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

Justification:

- Qwen2.5-Instruct is chat-tuned.
- Using the model's native chat template is closer to intended inference than
  passing raw text directly.

### 6.6 Generation

The successful run used:

```python
generated_ids = model.generate(
    **model_inputs,
    do_sample=False,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id,
)
```

Conceptually, greedy decoding selects the highest probability next token at each
step:

```text
y_t = argmax_v P(v | x, y_<t)
```

Where:

- `x` is the prompt
- `y_<t` is the already generated prefix
- `v` ranges over vocabulary tokens

Because the desired output is a letter, the ideal generation is one token such
as `A`, `B`, `C`, or `D`.

### 6.7 Answer Extraction

The model output is normalized by `extract_letter_answer`.

The function tries patterns in priority order:

- `Answer: A`
- `The answer is A`
- `**A**`
- a single standalone line `A`
- `A)` or `A.`
- fallback to any `A`-`D`

Justification:

- Instruction-tuned models sometimes return more than one character despite the
  instruction.
- Robust answer extraction prevents formatting noise from being counted as an
  automatic failure.

### 6.8 Checkpointing

The script saves a checkpoint every 50 questions:

```python
CHECKPOINT_INTERVAL = 50
```

The log confirms:

```text
checkpoint_001.csv (50 questions)
checkpoint_002.csv (100 questions)
```

Justification:

- Kaggle sessions can stop unexpectedly.
- Checkpoints preserve partial predictions.
- This is useful when extending from Fwe to all 19,612 benchmark questions.

### 6.9 Result Saving

The script writes:

```text
/kaggle/working/predictions.csv
/kaggle/working/summary.json
/kaggle/working/detailed_report.txt
```

Definitions:

- `predictions.csv`: one row per question with prediction, gold answer, match,
  and response preview.
- `summary.json`: machine-readable overall and per-file metrics.
- `detailed_report.txt`: human-readable run report.

## 7. Reproduction Steps

### Step 1: Verify Local Fwe Dataset

From repository root:

```powershell
Get-ChildItem Benchmark_multiple_choice\Fwe -File -Filter '*_questions.txt' |
  Select-Object Name,Length
```

Expected files:

```text
min_knowledge_points_4_questions.txt
min_knowledge_points_5_questions.txt
min_knowledge_points_6_questions.txt
min_knowledge_points_8_questions.txt
min_knowledge_points_10_questions.txt
min_knowledge_points_12_questions.txt
min_knowledge_points_13_questions.txt
```

Verify count:

```powershell
Get-ChildItem Benchmark_multiple_choice\Fwe -File -Filter '*_questions.txt' |
  ForEach-Object {
    (Select-String -Path $_.FullName -Pattern '^Question ' -AllMatches |
      Measure-Object).Count
  } |
  Measure-Object -Sum
```

Expected sum: `147`.

### Step 2: Prepare Kaggle Dataset

The project already contains:

```text
kaggle-linggym/linggym-dataset/
```

Upload it:

```powershell
cd E:\misc\LingGym\kaggle-linggym\linggym-dataset
kaggle datasets create -p . --dir-mode zip
```

If the dataset already exists, create a new version:

```powershell
kaggle datasets version -p . -m "Fwe benchmark for LingGym reproduction"
```

### Step 3: Push Kernel

```powershell
cd E:\misc\LingGym\kaggle-linggym
kaggle kernels push -p .
```

### Step 4: Monitor Kernel

```powershell
kaggle kernels status shakil19/linggym-kernel
```

Or use the Kaggle web UI and open the kernel logs.

Expected successful milestones:

```text
GPU Available: True
Loading model: Qwen/Qwen2.5-7B-Instruct
Total questions loaded: 147
Inference complete. Processed 147 questions.
Overall Accuracy: ...
```

### Step 5: Download Results

```powershell
kaggle kernels output shakil19/linggym-kernel -p .\kaggle-results
```

Inspect:

```powershell
Get-Content .\kaggle-results\summary.json
Import-Csv .\kaggle-results\predictions.csv | Select-Object -First 5
```

## 8. How To Read The Successful Log

The log confirms setup:

```text
GPU Name: Tesla T4
GPUs available: 2
Loading on T4 GPU(s) with FP16 dtype
Model loaded with multi-GPU support (FP16)
```

The log confirms dataset discovery:

```text
Found data at: /kaggle/input/datasets/shakil19/linggym-benchmark/Fwe
Total questions loaded: 147
```

The log confirms result:

```text
Inference complete. Processed 147 questions.
Overall Accuracy: 0.6327 (93/147)
```

The log confirms saved artifacts:

```text
Predictions saved: /kaggle/working/predictions.csv
Summary saved: /kaggle/working/summary.json
Detailed report saved: /kaggle/working/detailed_report.txt
```

## 9. Why This Implementation Is Technically Defensible

### It Uses The Released Benchmark Directly

The run does not regenerate questions. It consumes the already released Fwe MCQ
files, avoiding risk from imperfect local regeneration scripts.

### It Preserves The Paper's Richest Prompt Condition

Each prompt includes sentence, gloss, knowledge point, and translation. This is
the condition where the paper expects the strongest Qwen2.5-7B performance on
Fwe.

### It Uses Exact Gold Labels

The benchmark files include `Correct Answer: X`; the script parses this field
after generation. The gold label is excluded from the prompt.

### It Uses Deterministic Decoding For Debuggability

Greedy decoding is not the same as paper sampling, but it is appropriate for a
Kaggle reproduction tutorial because it makes reruns easier to compare.

### It Saves Enough Evidence

The debug output prints the first prompts, raw responses, extracted answers, and
gold labels. The CSV and JSON outputs preserve the full result table.

## 10. Common Failure Modes

### Dataset Path Not Found

Symptom:

```text
WARNING: Data directory not found
```

Resolution:

- Keep recursive dataset search enabled.
- Confirm Kaggle dataset contains the Fwe question files.
- Confirm `dataset_sources` includes `shakil19/linggym-benchmark`.

### Model Download Rate Limit

Symptom:

```text
Warning: You are sending unauthenticated requests to the HF Hub
```

Resolution:

- Add a Hugging Face token as a Kaggle secret if repeated downloads fail.

### Out Of Memory

Symptom:

```text
CUDA out of memory
```

Resolution:

- Keep `BATCH_SIZE = 1`.
- Keep `torch_dtype=torch.float16`.
- Use `device_map="auto"`.
- Reduce `max_new_tokens` for smoke tests.

### Incorrect Accuracy Because Gold Label Is Misparsed

Symptom:

- Many predictions appear wrong even when the model returns the correct letter.

Resolution:

- Use `Correct Answer:\s*([A-D])`, not a generic first-letter regex on the
  whole line.

## 11. Extending To Full Paper Reproduction

To move from the Fwe tutorial run to full paper reproduction, add:

1. A loop over all 18 language folders.
2. Prompt builders for `S`, `S+G`, `S+G+KP`, and `S+G+KP+T`.
3. Paper sampling settings:

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

4. Model runners for Qwen2.5, Gemma 3, DeepSeek-R1, LLaMA3, and API-backed
   models if needed.
5. Aggregation by model, language, language family, prompt setting, and
   linguistic subfield.

## 12. Recommended Citation Language For This Run

Use this wording:

```text
We ran a deterministic Kaggle reproduction variant of LingGym on the Fwe subset
using Qwen2.5-7B-Instruct and the S+G+KP+T prompt condition. The run processed
all 147 Fwe questions and achieved 63.27% accuracy (93/147), close to the
paper's reported 65.31% Fwe accuracy for Qwen2.5-7B under S+G+KP+T. The
difference is expected because the Kaggle run used greedy decoding
(do_sample=False, max_new_tokens=512), whereas the paper reports
sampling-based decoding with temperature 0.7, top-p 0.9, repetition penalty
1.1, and max tokens 2048.
```
