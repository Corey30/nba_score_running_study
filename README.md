# NBA Scoring Run Prediction

基于 NBA Play-by-Play（逐回合）数据的得分潮预测项目。项目从比赛事件重建比分和比赛状态，构造多时间窗口的滚动特征，并使用 XGBoost Focal Loss、LightGBM 和 CatBoost 完成时间序列交叉验证与对比实验。

## 项目特点

- 严格按赛季划分训练、验证和测试集，避免时间泄漏。
- 使用未来 180 秒内净得分至少 10 分作为正样本标签。
- 提供从上下文、事件计数、效率/节奏、动量、空间、连续性到球队赛季先验的分层特征集。
- 支持基线交叉验证、网格搜索、特征消融、模型对比、交叉熵与 Focal Loss 对比，以及 SHAP 特征选择。

## 目录结构

```text
.
├── server_pipeline.py       # 命令行入口
├── config.py                # 特征分组、交叉验证和模型参数
├── data_loader.py           # Kaggle 数据下载
├── preprocessing.py         # 数据读取、比分重建和标签生成
├── feature_engineering.py   # 特征工程
├── split.py                 # 按赛季切分数据
├── models/                  # XGBoost、LightGBM、CatBoost 训练器
├── experiments/             # 交叉验证及各类实验脚本
├── nba_pbp_data/            # 本地数据目录（CSV 不提交到 Git）
└── figures/                 # 生成的图表
```

## 环境安装

建议使用 Python 3.10 或更高版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

XGBoost 默认启用 GPU（`config.py` 中的 `USE_GPU = True`）。没有 CUDA 环境时，将其改为 `False`。LightGBM 和 CatBoost 当前使用各自训练器的默认设备配置。

## 数据准备

运行入口会尝试通过 Kaggle CLI 下载数据。请先配置 Kaggle API 凭据（通常为 `~/.kaggle/kaggle.json`）：

```bash
python server_pipeline.py --data-dir ./nba_pbp_data
```

也可以手动将 `pbp*.csv` 放入 `nba_pbp_data/`，然后跳过下载：

```bash
python server_pipeline.py --skip-download --data-dir ./nba_pbp_data
```

数据来源：Kaggle 数据集 [`szymonjwiak/nba-play-by-play-data-1997-2023`](https://www.kaggle.com/datasets/szymonjwiak/nba-play-by-play-data-1997-2023)。请遵守数据集和 NBA 数据的适用许可。原始 CSV 文件体积较大，已通过 `.gitignore` 排除。

## 常用命令

默认运行 5 折时间序列交叉验证：

```bash
python server_pipeline.py --skip-download
```

默认命令会依次评估 legacy 和 stage1–stage7 共 8 组特征；`--ablation` 使用同一交叉验证流程。

运行特征阶段消融和网格搜索：

```bash
python server_pipeline.py --skip-download --ablation
python server_pipeline.py --skip-download --grid-search --grid-mode quick --n-trials 20
```

运行模型对比、损失函数对比或 SHAP 特征选择：

```bash
python server_pipeline.py --skip-download --model-compare --n-trials 20
python server_pipeline.py --skip-download --ce-vs-fl
python server_pipeline.py --skip-download --shap-selection
```

`--ce-vs-fl` 在相同树参数搜索空间内分别调优 CE 与 Focal Loss，无需预先生成参数文件。两者分别搜索 `scale_pos_weight` 和 `gamma_fl` / `alpha_fl`，每种损失默认搜索 50 次，可通过 `--n-trials` 调整。该实验使用固定搜索空间，不受 `--grid-mode` 控制；`--n-estimators` 和 `--grid-early-stop` 控制训练轮数上限和早停耐心值。

统一输出 `ce_vs_fl_results.csv`、`ce_vs_fl_best_params.json` 和用于 PR 曲线的 `ce_vs_fl_predictions.npz`；设置 `--experiment-tag NAME` 后文件名增加 `exp_NAME_` 前缀，绘图时使用相同标签。旧的独立调参入口已合并为 `--ce-vs-fl`。

`--shap-selection` 需要先生成 `output_result/grid_best_params_stage7.json`，可先执行 `--grid-search --tune-on stage7` 或 `--model-compare`。使用 `--experiment-tag` 时，前后命令应使用相同标签。

输出默认写入 `output_result/`。实验结果可使用以下命令绘图：

```bash
python experiments/plot_figures.py --output-dir output_result --fig-dir figures
```

## 复现说明

默认赛季范围为 2020–2026，交叉验证配置位于 `config.py` 的 `CV_FOLDS`。数据预处理、特征构造和模型训练都可能占用较多内存与计算时间；完整网格搜索建议使用 GPU，并先用 `--grid-mode quick` 验证流程。

请确认本地数据包含交叉验证所需的全部赛季；下载源不保证提供默认配置中的所有文件。仅修改 `--seasons` 不会改变 `CV_FOLDS`。依赖文件声明的是版本下限，并非已验证的完整版本锁定环境。

当前评估流程存在研究上的限制：F1 最优阈值在测试集上选取，SHAP 特征选择会汇总测试折的重要性。因此 F1 和特征选择后的结果应视为探索性结果，严格评估时应在训练/验证集完成阈值与特征选择。绘图中的随机基线为固定历史样本正例比例，数据改变后需要重新计算。

## 代码检查

提交前可运行：

```bash
python -m compileall -q .
```

## 许可

如需公开发布，请根据你的研究成果、代码依赖和所使用数据集的许可补充合适的 LICENSE 文件。
