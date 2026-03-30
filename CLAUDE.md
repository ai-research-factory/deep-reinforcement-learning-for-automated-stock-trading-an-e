# Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy

## Project ID
proj_5e275486

## Taxonomy
ReinforcementLearning

## Current Cycle
2

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
This paper addresses the inherent instability and lack of robustness commonly observed in single-agent Deep Reinforcement Learning (DRL) systems for automated stock trading. The core problem is that a single DRL algorithm can be sensitive to market conditions and hyperparameters, leading to unreliable performance. To mitigate this, the paper proposes an ensemble strategy that combines the decisions of multiple, diverse DRL agents—specifically Proximal Policy Optimization (PPO), Advantage Actor-Critic (A2C), and Deep Deterministic Policy Gradient (DDPG). The hypothesis is that by aggregating the outputs of these different algorithms, the ensemble can achieve more stable and profitable trading performance, effectively smoothing out the idiosyncratic errors of individual agents.

### Datasets
Dow Jones Industrial Average (DJIA) 30 component stocks, sourced via the 'yfinance' library. The period will be from 2009-01-01 to 2020-12-31 for training and initial testing, with 2021-01-01 onwards reserved for out-of-time validation.

### Targets
The primary objective is to train the DRL agents to learn a trading policy that maximizes the risk-adjusted return of the portfolio, as measured by the Sharpe Ratio. The agents' output is a discrete action (buy, sell, or hold) for each asset in the portfolio at each time step.

### Model
The core of the proposed model is an ensemble of three distinct DRL agents: PPO, A2C, and DDPG. Each agent is trained independently on the same financial data to learn its own trading policy. The state representation for each agent consists of a lookback window of historical market data and technical indicators. The final trading decision is made by an ensemble module that aggregates the actions proposed by the three individual agents using a majority voting rule. This approach leverages algorithmic diversity to enhance decision-making robustness.

### Training
The system is trained and evaluated using a walk-forward validation methodology to simulate a realistic trading timeline and prevent look-ahead bias. The historical data is divided into multiple overlapping windows, each with a training and a subsequent testing period. In each window, the three DRL agents (PPO, A2C, DDPG) are trained from scratch on the training data. The reward function for training is based on the portfolio's change in value or its differential Sharpe ratio. The trained ensemble is then evaluated on the unseen test data for that window.

### Evaluation
The primary evaluation metric is the out-of-sample Sharpe Ratio, calculated across all walk-forward test periods. Other key metrics include Annualized Return, Annualized Volatility, and Maximum Drawdown. The performance of the ensemble strategy is benchmarked against: 1) each of the individual DRL agents (PPO, A2C, DDPG) operating alone, and 2) a passive 'buy-and-hold' strategy on a market index ETF (e.g., DIA for the DJIA).


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## Preflight チェック（実装開始前に必ず実施）

**Phase の実装コードを書く前に**、以下のチェックを実施し結果を `reports/cycle_2/preflight.md` に保存すること。

### 1. データ境界表
以下の表を埋めて、未来データ混入がないことを確認:

```markdown
| 項目 | 値 |
|---|---|
| データ取得終了日 | YYYY-MM-DD (今日以前であること) |
| Train期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| Validation期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| Test期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| 重複なし確認 | Yes / No |
| 未来日付なし確認 | Yes / No |
```

### 2. Feature timestamp 契約
- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → Yes / No
- Scaler / Imputer は train データのみで fit しているか？ → Yes / No
- Centered rolling window を使用していないか？ → Yes / No (使用していたら修正)

### 3. Paper spec 差分表
論文の主要パラメータと現在の実装を比較:

```markdown
| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | (論文の記述) | (実装の値) | Yes/No |
| ルックバック期間 | (論文の記述) | (実装の値) | Yes/No |
| リバランス頻度 | (論文の記述) | (実装の値) | Yes/No |
| 特徴量 | (論文の記述) | (実装の値) | Yes/No |
| コストモデル | (論文の記述) | (実装の値) | Yes/No |
```

**preflight.md が作成されるまで、Phase の実装コードに進まないこと。**

## ★ 今回のタスク (Cycle 2)


### Phase 2: データパイプラインと特徴量エンジニアリング [Track ]

**Track**:  (A=論文再現 / B=近傍改善 / C=独自探索)
**ゴール**: DJIA 30銘柄のデータをyfinanceから取得し、論文で想定される技術指標を計算して保存する。

**具体的な作業指示**:
1. `src/data/downloader.py`に`download_djia_data`関数を実装する。2009-01-01から2022-12-31までのDJIA 30銘柄の日足OHLCVデータをyfinanceから取得し、`data/raw/djia_data.csv`に保存する。2. `src/features/build_features.py`に`add_technical_indicators`関数を実装する。この関数はDataFrameを入力とし、RSI(14), MACD(12,26,9), Bollinger Bands(20,2)などの標準的な技術指標を計算して列として追加する。3. `notebooks/01_data_exploration.ipynb`を作成し、ダウンロードしたデータを読み込み、欠損値の確認、基本統計量の表示、および数銘柄の価格チャートと技術指標を可視化する。4. 処理済みのデータを`data/processed/djia_processed.feather`として保存するスクリプトを作成する。

**期待される出力ファイル**:
- data/processed/djia_processed.feather
- notebooks/01_data_exploration.ipynb

**受入基準 (これを全て満たすまで完了としない)**:
- djia_processed.featherファイルが生成され、技術指標の列が含まれている。
- 探索ノートブックでデータの欠損値が処理されていることが確認できる。




## データ問題でスタックした場合の脱出ルール

レビューで3サイクル連続「データ関連の問題」が指摘されている場合:
1. **データの完全性を追求しすぎない** — 利用可能なデータでモデル実装に進む
2. **合成データでのプロトタイプを許可** — 実データが不足する部分は合成データで代替し、モデルの基本動作を確認
3. **データの制約を open_questions.md に記録して先に進む**
4. 目標は「論文の手法が動くこと」であり、「論文と同じデータを揃えること」ではない


## スコア推移
Cycle 1: 60%



## 前回の結果
# Technical Findings — Cycle 2: Data Pipeline and Feature Engineering

## Implementation Summary

### Data Pipeline (`src/data/downloader.py`)
- Implemented `download_djia_data()` function that fetches OHLCV data from the ARF Data API
- Downloads 17 available DJIA component stocks (out of 30 total) for the period 2009-01-01 to 2022-12-31
- Rate-limited requests (0.5s between tickers) to avoid API overload
- Output: `data/raw/djia_data.csv` with 59,908 rows (17 tickers x 3,524 trading days)

### Feature Engineering (`src/features/build_features.py`)
- Implemented `add_technical_indicators()` function with the following indicators:
  - **RSI(14)**: Relative Strength Index using EWM (Wilder's smoothing)
  - **MACD(12, 26, 9)**: MACD line, signal line, and histogram
  - **Bollinger Bands(20, 2)**: Upper, middle, and lower bands
  - **Daily return**: Close-to-close percentage change
  - **Volume change**: Volume percentage change
  - **Close-open ratio**: Intraday price movement indicator
  - **High-low spread**: Normalized daily price range
- All indicators computed per-ticker with proper grouping
- Warmup rows (33 per ticker, 561 total) dropped to eliminate NaN from indicator initialization
- Output: `data/processed/djia_processed.feather` with 59,347 rows and 18 columns

### Data Integrity
- Zero NaN values in the processed dataset
- No future data leakage confirmed (all rolling windows use `center=False`, all EWMs are backward-looking)
- No duplicate ticker-date combinations
- RSI values bounded within [0, 100]
- Bollinger Band ordering verified: upper >= middle >= lower
- 13/13 tests passing

## Key Observations

1. **Universe Coverage**: 17/30 DJIA stocks available via ARF API (56.7%). Missing stocks documented in `docs/open_questions.md`. This is a known limitation that may reduce diversification benefits in the ensemble strategy.

2. **Data Quality**: All 17 tickers have identical date coverage (3,524 trading days from 2009-01-02 to 2022-12-30), making panel alignment straightforward.

3. **Indicator Parameters**: All match the paper specification — RSI(14), MACD(12,26,9), Bollinger Bands(20,2).

4. **Trading Metrics**: All trading-related metrics (Sharpe, returns, etc.) are set to 0.0 in metrics.json as this phase covers data pipeline only. These will be populated in subsequent phases.

## Files Created/Modified
- `src/data/__init__.py` — Package init
- `src/data/downloader.py` — ARF API data downloader
- `src/features/__init__.py` — Package init
- `src/features/build_features.py` — Technical indicator computation
- `tests/test_data_integrity.py` — 13 tests for data integrity and leakage prevention
- `notebooks/01_data_exploration.ipynb` — Data exploration and visualization notebook
- `reports/cycle_2/preflight.md` — Preflight checks
- `reports/cycle_2/metrics.json` — Metrics (data phase, trading metrics N/A)
- `reports/cycle_2/technical_findings.md` — This file
- `docs/open_questions.md` — Open questions and limitations




## レビューからのフィードバック
### レビュー改善指示
1. 【最重要】ユニバース再現性の向上計画の策定：次サイクルでモデル実装に入る前に、データソース問題を解決する。`src/data/downloader.py`を修正し、論文同様に`yfinance`ライブラリをフォールバックとして使用し、ARF APIで取得できない残り13銘柄を補完する機能を実装する。`DJIA_30_FULL`リストを活用し、30銘柄全てを揃えることを目指す。これにより`universeFidelity`が向上し、再現実験の信頼性が高まる。
2. 【重要】特徴量セットの明確化と分離：`src/features/build_features.py`の`add_technical_indicators`関数に、論文準拠の特徴量のみを生成する`strict=True`のようなフラグを追加する。デフォルトでは論文通りの特徴量セットとし、追加特徴量はオプションで生成できるようにする。これにより、純粋な再現実験と、特徴量を追加した拡張実験を明確に区別できるようになる。
3. 【推奨】ベースライン戦略データセットの準備：次のサイクルでRLモデルと比較するために、単純なベースライン戦略（例：Buy & Hold）の評価に必要なデータを準備する。例えば、`reports/cycle_2/metrics.json`に`baseline_1n_sharpe`等の項目があるが、これを計算するためのポートフォリオレベルの価格系列（例：均等加重ポートフォリオのインデックス）を`data/processed/baseline_data.feather`として生成するスクリプトを`src/features`に追加する。
### マネージャー指示 (次のアクション)
1. 【最優先】`src/data/get_raw_data.py`を修正し、ARF APIで取得できないDJIA 13銘柄を補完するため、`yfinance`ライブラリを代替データソースとして利用するロジックを追加する。取得したデータは既存のパイプラインと互換性のある形式に整形し、`data/raw`ディレクトリに保存すること。
2. 【重要】`tests/test_data_integrity.py`に、`data/processed`に保存された最終的なfeatherファイルがDJIA 30銘柄すべてのデータを含んでいることを検証するテスト `test_universe_completeness` を追加する。このテストは `len(df['tic'].unique()) == 30` をアサーションすること。
3. 【推奨】`src/features/build_features.py`に、論文で指定された特徴量（RSI, MACD, BB）のみを生成するモードを追加する。具体的には、`config.yml`に`features.use_paper_only: true`のような設定項目を設け、このフラグに応じて追加特徴量の計算をスキップするロジックを実装する。これにより、論文再現と独自探索の切り替えを容易にする。


## 全体Phase計画 (参考)

✓ Phase 1: シングルエージェント(PPO)と取引環境の実装 — 単一銘柄を対象に、PPOエージェントが学習・取引できる基本的なGym環境を実装する。
→ Phase 2: データパイプラインと特徴量エンジニアリング — DJIA 30銘柄のデータをyfinanceから取得し、論文で想定される技術指標を計算して保存する。
  Phase 3: マルチエージェントとアンサンブル戦略の実装 — A2CとDDPGエージェントを実装し、3つのエージェントの決定を統合する多数決アンサンブル戦略を実装する。
  Phase 4: バックテストエンジンとコストモデルの実装 — 取引コストを考慮したベクトル化バックテストエンジンを実装し、アンサンブル戦略のパフォーマンスを評価する。
  Phase 5: ウォークフォワード検証フレームワークの構築 — 論文の再現性を担保するため、厳密なウォークフォワード検証を実装し、複数期間での性能を評価する。
  Phase 6: ハイパーパラメータ最適化 (Optuna) — 論文のデフォルト値近傍で各DRLエージェントの主要ハイパーパラメータを最適化し、性能向上を図る。
  Phase 7: ロバスト性検証：取引コスト感度分析 — 戦略の収益性が取引コストに対してどの程度頑健であるかを評価する。
  Phase 8: アンサンブル手法の比較分析 — 多数決以外のアンサンブル手法を試し、パフォーマンスを比較検討する。
  Phase 9: 代替特徴量セットの実験 — 論文で標準的に使われるものとは異なる特徴量セット（例：ボラティリティやモメンタム指標中心）の影響を調査する。
  Phase 10: 代替モデル(SAC)の導入実験 — 論文で言及されていない新しいDRLアルゴリズム（SAC）をアンサンブルに導入し、性能への影響を評価する。
  Phase 11: 最終レポートと可視化 — 全フェーズの結果を統合し、論文再現の結論と改善提案をまとめた包括的なテクニカルレポートを生成する。
  Phase 12: コードのクリーンアップとドキュメント整備 — プロジェクトの再現性と保守性を高めるため、コードのリファクタリング、ドキュメントの拡充、ユニットテストの追加を行う。


## ベースライン比較（必須）

戦略の評価には、以下のベースラインとの比較が**必須**。metrics.json の `customMetrics` にベースライン結果を含めること。

| ベースライン | 実装方法 | 意味 |
|---|---|---|
| **1/N (Equal Weight)** | 全資産に均等配分、月次リバランス | 最低限のベンチマーク |
| **Vol-Targeted 1/N** | 1/N にボラティリティターゲティング (σ_target=10%) を適用 | リスク調整後の公平な比較 |
| **Simple Momentum** | 12ヶ月リターン上位50%にロング | モメンタム系論文の場合の自然な比較対象 |

```python
# metrics.json に含めるベースライン比較
"customMetrics": {
  "baseline_1n_sharpe": 0.5,
  "baseline_1n_return": 0.05,
  "baseline_1n_drawdown": -0.15,
  "baseline_voltarget_sharpe": 0.6,
  "baseline_momentum_sharpe": 0.4,
  "strategy_vs_1n_sharpe_diff": 0.1,
  "strategy_vs_1n_return_diff": 0.02,
  "strategy_vs_1n_drawdown_diff": -0.05,
  "strategy_vs_1n_turnover_ratio": 3.2,
  "strategy_vs_1n_cost_sensitivity": "論文戦略はコスト10bpsで1/Nに劣後"
}
```

「敗北」の場合、**どの指標で負けたか** (return / sharpe / drawdown / turnover / cost) を technical_findings.md に明記すること。

## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 再現モードのルール（論文忠実度の維持）

このプロジェクトは**論文再現**が目的。パフォーマンス改善より論文忠実度を優先すること。

### パラメータ探索の制約
- **論文で既定されたパラメータをまず実装し、そのまま評価すること**
- パラメータ最適化を行う場合、**論文既定パラメータの近傍のみ**を探索（例: 論文が12ヶ月なら [6, 9, 12, 15, 18] ヶ月）
- 論文と大きく異なるパラメータ（例: 月次論文に対して日次10営業日）で良い結果が出ても、それは「論文再現」ではなく「独自探索」
- 独自探索で得た結果は `customMetrics` に `label: "implementation-improvement"` として記録し、論文再現結果と明確に分離

### データ条件の忠実度
- 論文のデータ頻度（日次/月次/tick）にできるだけ合わせる
- ユニバース規模が論文より大幅に小さい場合、その制約を `docs/open_questions.md` に明記
- リバランス頻度・加重方法も論文に合わせる



## 禁止事項

### データ・特徴量の禁止パターン（具体的）
- `scaler.fit(full_data)` してから split → **禁止**。`scaler.fit(train_data)` のみ
- `df.rolling(window=N, center=True)` → **禁止**。`center=False` (デフォルト) を使用
- データの `end_date` が今日以降 → **禁止**。`end_date` を明示的に過去に設定
- `merge` で未来のタイムスタンプを持つ行が特徴量に混入 → **禁止**
- ラベル生成後に特徴量を合わせる（ラベルの存在を前提に特徴量を選択）→ **禁止**

### 評価・報告の禁止パターン
- コストなしのgross PnLだけで判断しない
- テストセットでハイパーパラメータを調整しない
- 時系列データにランダムなtrain/test splitを使わない
- README に metrics.json と異なる数値を手書きしない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**
- **論文が既定するパラメータから大幅に逸脱した探索を「再現」として報告しない**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_2/preflight.md` — Preflight チェック結果（必須、実装前に作成）
- `reports/cycle_2/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_2/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ（Single Source of Truth）
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。

### レポート生成ルール（重要: 数値の一貫性）
- **`metrics.json` が全ての数値の唯一のソース (Single Source of Truth)**
- README や technical_findings に書く数値は **必ず metrics.json から引用** すること
- **手打ちの数値は禁止**。metrics.json に含まれない数値を README に書かない
- technical_findings.md で数値に言及する場合も metrics.json の値を参照
- README.md の Results セクションは metrics.json を読み込んで生成すること

### テスト必須
- `tests/test_data_integrity.py` のテストを実装状況に応じて有効化すること
- 新しいデータ処理や特徴量生成を追加したら、対応する leakage テストも追加
- `pytest tests/` が全パスしない場合、サイクルを完了としない

### その他の出力
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
