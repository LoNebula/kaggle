# AIMO Progress Prize 3 — 戦略メモ

作成日: 2026-04-01

---

## 1. 最新SOTAアプローチ

### コンペ概要
- AIMO3は2026年4月終了予定。問題難易度はIMOレベル（110問、国内・国際オリンピックレベルの代数・組合せ・幾何・整数論）
- AIMO2のトップスコアは34/50（NemoSkills）。商用モデル（OpenAI）は47/50を達成しており、その差を縮めることがAIMO3の命題
- H100 GPU (80GB) が提供され、AIMO2比で約2倍の計算資源

### ベースモデルの選択

| モデル | 備考 |
|----|----|
| **DeepSeek-R1-Distill-Qwen-14B** (AWQ 4bit) | AIMO2で最も人気。vLLM + AWQで単一H100に収まる |
| **Qwen2.5-14B / 32B** | AIMO2 1位チームのベース。FP8で32BもH100に搭載可能 |
| **DeepSeek-R1-0528** | 最新版。AIME2025で87.5%の精度（旧版70%から大幅改善） |
| **Qwen3-Next (将来候補)** | H100の拡張メモリを活かした大規模モデル |

### ファインチューニング戦略（AIMO2 1位 NemoSkillsの知見）
1. **SFT (CoT)**: OpenMathReasoning / OpenR1-Math などの540K問題 × 3.2Mソリューションで学習
2. **TIR (Tool-Integrated Reasoning)**: Pythonコード実行をLLMの推論に統合。Flaskサンドボックスで並列実行。1.7M TIRソリューションで追加SFT
3. **モデルマージ**: CoTチェックポイントとTIRチェックポイントを **線形結合（mergekit）** でマージ → 精度向上＋生成長短縮の相乗効果
4. **難問絞り込みSFT**: pass@32 > 0.3 の易問を除外し、トークン数 < 5000 のソリューションを除外した難問サブセットで追加学習

### Test-Time Compute の工夫

#### Majority Voting (SC-TIR)
- AIMO1優勝のProject Numinaが確立した手法
- 同一問題をM個のプロンプトに複製 → M個の候補を並列サンプリング → Python実行 → 多数決で最終答案を決定
- `maj@64`（64サンプルの多数決）がNemoSkillsの評価基準

#### GenSelect（多数決を超える選択）
- AIMO2 1位NemoSkillsが開発した独自手法
- 推論モデルが複数候補を生成 → Qwen2.5-32B-Instructで各解の詳細サマリーを生成 → QwQ-32Bが最良解を選択
- 16候補を1グループとして`majority@8 of GenSelect`を適用
- 32候補を超えるとプロンプトが不安定になるため上限注意
- Kaggle提出には**使用しなかった**（推論速度の制約のため）

#### MCTSr (Monte Carlo Tree Self-Refine)
- LLMをMCTSに統合：ノード = 解候補パス、スコア = ヒューリスティック自己批判による品質評価
- オリンピックレベルの問題で成功率を大幅改善
- 実装コストが高いため、推論時間制約があるKaggleでは注意が必要

---

## 2. Kaggle Notebook環境でのベストプラクティス

### リソース制約
- **GPU**: H100 × 1枚（VRAM 80GB, HBM3）
- **実行時間**: 9時間（提出用カーネル）
- **インターネット**: 提出カーネルはオフライン（モデルはKaggle Datasetから事前ロード必須）

### vLLMによるバッチ推論

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/kaggle/input/your-model-dataset/model",
    gpu_memory_utilization=0.92,   # H100 80GBのうち約73GBをKVキャッシュに使用
    max_model_len=32768,           # 長い思考連鎖に対応
    dtype="bfloat16",              # H100はbf16が最適
    # quantization="awq",          # AWQ量子化モデルの場合
    enforce_eager=False,           # CUDAグラフで高速化
)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=32768,
    n=16,                          # 1問あたり16サンプル（maj@16）
)

# 全問題をバッチ推論（vLLMは内部で並列処理）
outputs = llm.generate(prompts, sampling_params)
```

### 時間予算の目安（H100 × 1枚、Qwen2.5-14B bf16）

| 推論設定 | 速度感 | 100問での合計目安 |
|----|----|----|
| 14B bf16, 1サンプル | ~5-10 tok/s (long CoT) | ~1-2時間 |
| 14B bf16, 16サンプル | 並列込みで~3-4倍 | ~3-5時間 |
| 14B AWQ 4bit, 32サンプル | より高速 | ~4-6時間 |
| 32B AWQ 4bit, 16サンプル | メモリ注意 | ~6-8時間 |

→ **14B AWQ 4bit + 16~32サンプル** が9時間制約下でのバランス点

### Majority Votingの実装

```python
from collections import Counter
import re

def extract_answer(text: str) -> str | None:
    """最終答案を抽出（\\boxed{} 形式）"""
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    return matches[-1] if matches else None

def majority_vote(outputs_for_problem: list) -> str:
    """maj@N: 最多票の答案を返す"""
    answers = [extract_answer(o.outputs[0].text) for o in outputs_for_problem]
    answers = [a for a in answers if a is not None]
    if not answers:
        return "-1"
    return Counter(answers).most_common(1)[0][0]
```

### TIR（Pythonコード実行統合）の実装のポイント
- Python実行用のサンドボックスを別プロセスで起動（セキュリティ境界）
- コードブロック（```python ... ```）を正規表現で抽出 → `subprocess`で実行 → 出力をLLMに返す
- タイムアウト設定が必須（1コードブロックあたり10-30秒程度）
- 無限ループ・大量メモリ消費を防ぐプロセス制限

```python
import subprocess, signal

def execute_python(code: str, timeout: int = 20) -> str:
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True, timeout=timeout,
            # メモリ制限は ulimit で（Linux環境）
        )
        return result.stdout[-3000:] + result.stderr[-1000:]  # トークン節約
    except subprocess.TimeoutExpired:
        return "TimeoutError: code took too long"
    except Exception as e:
        return f"Error: {e}"
```

---

## 3. オフライン環境でのモデル構成案

### 推奨モデル構成（単一H100 80GB）

#### 優先度A: 検証済みの鉄板構成
```
DeepSeek-R1-Distill-Qwen-14B-AWQ (4bit)
  - VRAM使用量: ~8-9GB (weights) + KVキャッシュ
  - Kaggle Dataset: casperhansen/deepseek-r1-distill-qwen-14b-awq
  - 特徴: AIMO2で最多採用、推論速度と精度のバランス最良
```

#### 優先度B: 精度重視構成
```
Qwen2.5-32B-Instruct-AWQ (4bit)
  - VRAM使用量: ~16-18GB (weights) + KVキャッシュ
  - 特徴: 14Bより精度高いが推論が遅い → サンプル数を16以下に抑える
```

#### 優先度C: 最高精度（時間に余裕がある場合）
```
DeepSeek-R1-0528 (FP8 or AWQ)
  - VRAM使用量: ~38GB (FP8 671Bは不可、7B/14B蒸留版を使用)
  - DeepSeek-R1-Distill-Qwen-14B ベースの改善版
```

### Kaggle Datasetへのモデルの追加手順

```bash
# 1. HuggingFace からモデルをダウンロード（ローカルPCまたはKaggleノートブックで）
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='casperhansen/deepseek-r1-distill-qwen-14b-awq',
    local_dir='./model-weights'
)
"

# 2. Kaggle Dataset として登録
kaggle datasets create -p ./model-weights \
  --title "deepseek-r1-distill-qwen-14b-awq" \
  --license "other"

# 3. Notebook から参照
# /kaggle/input/deepseek-r1-distill-qwen-14b-awq/
```

### 推奨Kaggle Datasets（AIMO3対応）

| Dataset名 | 内容 | 優先度 |
|----|----|:----:|
| `casperhansen/deepseek-r1-distill-qwen-14b-awq` | DeepSeek-R1蒸留14B AWQ 4bit | ★★★ |
| `Qwen/Qwen2.5-14B-Instruct-AWQ` (HF) | Qwen2.5-14B AWQ | ★★★ |
| `Qwen/Qwen2.5-32B-Instruct-AWQ` (HF) | Qwen2.5-32B AWQ（遅いが高精度）| ★★ |
| `nvidia/OpenMathReasoning` (HF) | NemoSkillsのファインチューニング済みモデル | ★★★ |

### ファインチューニングを行う場合の準備（H100 × 128枚アクセス申請）

- Fields Model Initiative 経由でH100 × 128枚の申請可能
- ファインチューニングデータセット: `nvidia/OpenMathReasoning`（540K問題、3.2Mソリューション、MITライセンス）
- 最低限の学習セット: 難問SFT（pass@32 < 0.3 かつ解長 > 5000トークン）

---

## 実装優先順位

1. **ベースライン構築**: DeepSeek-R1-Distill-Qwen-14B-AWQ + vLLM + maj@16
2. **TIR追加**: Pythonサンドボックスを組み込み、コード実行型推論を有効化
3. **サンプル数チューニング**: 9時間制約を計測しながら最大化（目標: maj@32）
4. **GenSelect導入**: Qwen2.5-32BをVerifierとして使い、候補から最良解を選択
5. **難問戦略**: 残り時間を難問（多数決が割れる問題）に集中投資

---

## 参考リンク

- [AIMO3 Kaggle Competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [NemoSkills 論文 (arXiv:2504.16891)](https://arxiv.org/abs/2504.16891)
- [OpenMathReasoning Dataset (HuggingFace)](https://huggingface.co/datasets/nvidia/OpenMathReasoning)
- [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1)
- [Qwen2.5-Math HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Math-72B-Instruct)
- [vLLM 公式ドキュメント](https://docs.vllm.ai)
- [AIMO2 Winning Solution Notebook (yekenot)](https://www.kaggle.com/code/yekenot/aimo-2-deepseek-r1-distill-qwen-7b-awq)
