# ============================================================
# Cell 1: Imports & Environment Detection
# ============================================================
import os
import re
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from collections import Counter
from typing import List, Optional

import torch
import pandas as pd

IS_KAGGLE = Path('/kaggle/input').exists()
print(f"Environment : {'Kaggle' if IS_KAGGLE else 'Local (mock mode)'}")
print(f"Python      : {sys.version.split()[0]}")
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA        : {torch.cuda.is_available()} "
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")

# ============================================================
# Cell 2: Configuration
# ============================================================

# --- Model (Kaggle Dataset slug) ---
# Add the dataset via kernel-metadata.json dataset_sources.
# Recommended: deepseek-ai/deepseek-r1-distill-qwen-7b (7B, fits P100 16 GB in INT4)
MODEL_DATASET   = 'deepseek-r1-distill-qwen-7b'          # dataset slug
MODEL_PATH      = f'/kaggle/input/{MODEL_DATASET}'        # full path on Kaggle
LOAD_IN_4BIT    = True    # INT4 via bitsandbytes; set False for FP16 (needs >14 GB VRAM for 7B)

# --- Context ---
MAX_MODEL_LEN   = 8192    # hard cap for prompt+generation (P100 memory budget)

# --- Sampling ---
NUM_SAMPLES     = 8 if IS_KAGGLE else 4    # maj@N
TEMPERATURE     = 0.6
TOP_P           = 0.95
MAX_NEW_TOKENS  = 4096    # per sample (keep lower to fit memory + 9 h time budget)

# --- TIR self-correction ---
MAX_TIR_ROUNDS  = 3
CODE_TIMEOUT    = 30      # seconds per code block

# --- I/O ---
if IS_KAGGLE:
    TEST_CSV = '/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv'
else:
    TEST_CSV = 'dummy_test.csv'
OUTPUT_CSV = 'submission.csv'

print('Configuration:')
for k, v in {
    'MODEL_PATH': MODEL_PATH, 'LOAD_IN_4BIT': LOAD_IN_4BIT,
    'NUM_SAMPLES': NUM_SAMPLES, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS,
    'MAX_TIR_ROUNDS': MAX_TIR_ROUNDS, 'CODE_TIMEOUT': CODE_TIMEOUT,
}.items():
    print(f'  {k:20s}: {v}')

# ============================================================
# Cell 3: Python Sandbox
# ============================================================

def execute_python(code: str, timeout: int = CODE_TIMEOUT) -> str:
    """Run code in isolated subprocess; return stdout+stderr (max 2000 chars)."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, encoding='utf-8'
    ) as f:
        f.write(code)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        output = proc.stdout + proc.stderr or '(no output)'
    except subprocess.TimeoutExpired:
        output = f'TimeoutError: exceeded {timeout}s'
    except Exception as exc:
        output = f'ExecutionError: {exc}'
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    if len(output) > 2000:
        output = output[:1500] + '\n...[truncated]...\n' + output[-500:]
    return output.rstrip()


_test = execute_python('print(sum(range(1, 101)))')
assert _test.strip() == '5050', f'Sandbox failed: {_test!r}'
print(f'Sandbox OK -> {_test}')

# ============================================================
# Cell 4: Prompt Templates & Regex Helpers
# ============================================================

SYSTEM_PROMPT = (
    'You are an expert mathematician solving competition math problems.\n'
    'Reason step by step. You may write Python code to assist with calculations.\n'
    '\n'
    'Wrap Python code in triple backticks:\n'
    '```python\n'
    '# your code here\n'
    '```\n'
    'The code output will be shown to you. Use code for arithmetic, modular arithmetic,\n'
    'combinatorics, exhaustive search, or verification.\n'
    '\n'
    'After reasoning, state your final answer EXACTLY as:\n'
    'The answer is \\boxed{<integer>}'
)


def build_initial_prompt(problem: str) -> str:
    return f'{SYSTEM_PROMPT}\n\nProblem: {problem}\n\nSolution:'


_CODE_RE  = re.compile(r'```python\s*(.*?)```', re.DOTALL)
_BOXED_RE = re.compile(r'\\boxed\{([^}]+)\}')


def extract_code_blocks(text: str) -> List[str]:
    return [b.strip() for b in _CODE_RE.findall(text)]


def extract_final_answer(text: str) -> Optional[str]:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


assert extract_final_answer('The answer is \\boxed{42}') == '42'
assert extract_code_blocks('```python\nprint(1)\n```') == ['print(1)']
print('Prompt helpers OK')

# ============================================================
# Cell 5: Model Loading
#   Kaggle : transformers + bitsandbytes (pre-installed, no internet)
#   Local  : MockLLM
# ============================================================

# --- Shared output classes (same interface on both paths) ---
class _CompletionOutput:
    def __init__(self, text: str): self.text = text

class _RequestOutput:
    def __init__(self, texts: List[str]):
        self.outputs = [_CompletionOutput(t) for t in texts]

class SamplingParams:
    def __init__(self, temperature=0.6, top_p=0.95, max_tokens=4096, n=1):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.n = n


if IS_KAGGLE:
    # ---- Real model via HuggingFace transformers ----
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f'Model not found: {MODEL_PATH}\n'
            f'Add dataset "{MODEL_DATASET}" to kernel-metadata.json dataset_sources.'
        )

    print(f'Loading tokenizer from {MODEL_PATH} ...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f'Loading model (load_in_4bit={LOAD_IN_4BIT}) ...')
    model_kwargs = dict(device_map='auto', trust_remote_code=True)
    if LOAD_IN_4BIT:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    else:
        model_kwargs['torch_dtype'] = torch.float16

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    hf_model.eval()
    print('Model loaded.')
    print(f'Device map: {hf_model.hf_device_map}')

    class TransformersLLM:
        """Thin wrapper around hf_model with the same .generate() interface used below."""

        def generate(self, prompts: List[str], sampling_params: SamplingParams) -> List[_RequestOutput]:
            results = []
            for prompt in prompts:
                enc = tokenizer(
                    prompt, return_tensors='pt', truncation=True,
                    max_length=MAX_MODEL_LEN - sampling_params.max_tokens,
                ).to(hf_model.device)

                with torch.no_grad():
                    out_ids = hf_model.generate(
                        **enc,
                        max_new_tokens=sampling_params.max_tokens,
                        do_sample=(sampling_params.temperature > 0),
                        temperature=max(sampling_params.temperature, 1e-6),
                        top_p=sampling_params.top_p,
                        num_return_sequences=sampling_params.n,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                input_len = enc.input_ids.shape[1]
                texts = [
                    tokenizer.decode(ids[input_len:], skip_special_tokens=True)
                    for ids in out_ids
                ]
                results.append(_RequestOutput(texts))
            return results

    llm = TransformersLLM()

else:
    # ---- MockLLM for local testing ----
    print('Local mode: using MockLLM')

    _MOCK_RESPONSES = [
        (
            'Let me compute.\n'
            '```python\n'
            'print(sum(range(1, 101)))\n'
            '```\n'
            'The answer is \\boxed{5050}'
        ),
        (
            'Writing code:\n'
            '```python\n'
            'print(100 * 101 // 2)\n'
            '```'
        ),
        'Using formula: 5050. The answer is \\boxed{5050}',
        'The output confirms it. The answer is \\boxed{5050}',
    ]

    class MockLLM:
        _cnt = 0
        def generate(self, prompts: List[str], sampling_params: SamplingParams) -> List[_RequestOutput]:
            results = []
            for _ in prompts:
                texts = [
                    _MOCK_RESPONSES[(MockLLM._cnt + i) % len(_MOCK_RESPONSES)]
                    for i in range(sampling_params.n)
                ]
                MockLLM._cnt += sampling_params.n
                results.append(_RequestOutput(texts))
            return results

    llm = MockLLM()
    print('MockLLM ready.')

# ============================================================
# Cell 6: TIR Inference with Self-Correction
# ============================================================

def _run_tir_continuation(
    conversation: str,
    llm_instance,
    max_rounds: int = MAX_TIR_ROUNDS,
) -> str:
    """Execute code, feed back output, regenerate until \\boxed{} appears or rounds exhausted."""
    sp = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS, n=1)
    latest = conversation

    for _ in range(max_rounds):
        blocks = extract_code_blocks(latest)
        if not blocks:
            break
        code_out = execute_python(blocks[-1])
        conversation += (
            f'\n\nExecution output:\n```\n{code_out}\n```\n\n'
            'Continue your solution:'
        )
        out = llm_instance.generate([conversation], sp)
        latest = out[0].outputs[0].text
        conversation += latest
        ans = extract_final_answer(latest)
        if ans is not None:
            return ans

    return extract_final_answer(conversation) or ''


def tir_batch(
    problem: str,
    llm_instance,
    num_samples: int = NUM_SAMPLES,
) -> List[str]:
    """Generate num_samples TIR solutions; return list of answer strings."""
    prompt = build_initial_prompt(problem)
    sp = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS, n=num_samples)
    batch = llm_instance.generate([prompt], sp)

    answers: List[str] = []
    for sample in batch[0].outputs:
        text = sample.text
        ans = extract_final_answer(text)
        if ans is not None:
            answers.append(ans)
            continue
        if extract_code_blocks(text):
            ans = _run_tir_continuation(prompt + text, llm_instance)
        else:
            nudge = prompt + text + '\n\nFinal integer answer: \\boxed{'
            sp1 = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64, n=1)
            cont = llm_instance.generate([nudge], sp1)
            raw = '\\boxed{' + cont[0].outputs[0].text
            ans = extract_final_answer(raw) or ''
        answers.append(ans)
    return answers

# ============================================================
# Cell 7: Self-Consistency (Majority Voting)
# ============================================================

def majority_vote(answers: List[str]) -> str:
    valid = [a for a in answers if a.strip()]
    return Counter(valid).most_common(1)[0][0] if valid else '0'


def solve_problem(problem: str, llm_instance, num_samples: int = NUM_SAMPLES) -> dict:
    all_ans = tir_batch(problem, llm_instance, num_samples)
    final   = majority_vote(all_ans)
    return {
        'answer': final,
        'all_answers': all_ans,
        'vote_distribution': dict(Counter(a for a in all_ans if a)),
        'num_valid': sum(1 for a in all_ans if a),
    }


print('Voting helpers OK')

# ============================================================
# Cell 8: Main Pipeline
# ============================================================

def run_pipeline(
    llm_instance,
    test_csv: str = TEST_CSV,
    output_csv: str = OUTPUT_CSV,
    num_samples: int = NUM_SAMPLES,
) -> pd.DataFrame:
    df = pd.read_csv(test_csv)
    assert {'id', 'problem'}.issubset(df.columns), f'Bad columns: {list(df.columns)}'

    rows, total, t0 = [], len(df), time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        print(f"\n{'='*60}\n[{i+1}/{total}] id={row['id']}")
        print(f"Problem : {str(row['problem'])[:120]}")

        ts = time.time()
        res = solve_problem(row['problem'], llm_instance, num_samples)
        print(f"Answer  : {res['answer']}  Votes: {res['vote_distribution']}  "
              f"Valid: {res['num_valid']}/{num_samples}  Time: {time.time()-ts:.1f}s")

        rows.append({'id': row['id'], 'answer': res['answer']})

    sub = pd.DataFrame(rows)
    sub['answer'] = pd.to_numeric(sub['answer'], errors='coerce').fillna(0).astype(int)
    sub.to_csv(output_csv, index=False)
    print(f"\nDone: {total} problems in {(time.time()-t0)/60:.1f} min  ->  {output_csv}")
    print(sub.to_string(index=False))
    return sub

# ============================================================
# Cell 9: Local Dummy Test  (skipped on Kaggle)
# ============================================================

if not IS_KAGGLE:
    print('>>> Local pipeline test <<<')
    dummy = [
        {'id': 1, 'problem': 'What is the sum of integers from 1 to 100?'},
        {'id': 2, 'problem': 'What is 2^10 mod 7?'},
        {'id': 3, 'problem': 'How many primes are less than 20?'},
    ]
    pd.DataFrame(dummy).to_csv(TEST_CSV, index=False)

    sub = run_pipeline(llm, test_csv=TEST_CSV, output_csv=OUTPUT_CSV)

    res = pd.read_csv(OUTPUT_CSV)
    assert list(res.columns) == ['id', 'answer']
    assert len(res) == 3
    assert pd.api.types.is_integer_dtype(res['answer'])
    print('\n>>> All assertions passed. <<<')

# ============================================================
# Cell 10: Run on Kaggle
# ============================================================

if IS_KAGGLE:
    submission = run_pipeline(llm, test_csv=TEST_CSV, output_csv=OUTPUT_CSV)
