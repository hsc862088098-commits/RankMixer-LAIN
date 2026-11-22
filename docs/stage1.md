# 阶段一：AutoDL 准备与 Smoke Test

当前代码只实现阶段一；LAIN 及正式实验尚未启动。原始 `model_zoo/RankMixer.py`、`config/model_config.yaml` 和 `config/dataset_config.yaml` 保留。以下命令供获授权后在 **AutoDL Linux** 上执行，本地 Codex 未连接该服务器。

## 1. 环境与代码

代码放在 `/root/autodl-tmp/RankMixer-LAIN`，独立配置在 `config/project`。先检查机器资源：

```bash
nvidia-smi
free -h
df -h /root/autodl-tmp
```

安装依赖需单独授权；确认驱动兼容后，参考仓库环境安装：

```bash
conda create -n rankmixer python=3.10
conda activate rankmixer
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_environment.py
python -m unittest discover -s tests -t . -v
```

测试用标准库 unittest，无需安装 pytest。没有 pyarrow 时真实 Parquet 测试显式跳过；缺少完整模型依赖时端到端测试显式跳过。源码独立组件测试的通过不代表完整依赖、真实数据或 CUDA 已验收。

## 2. 固定版本数据与审计

下载需单独授权；先预览，再将固定版本放入以下目录。不要使用 Hugging Face 自动转换的 viewer Parquet 替代发布的三表分块。

```bash
hf download salmon1802/Taobao --repo-type dataset \
  --revision fd01fb8edbfa58bb4b0d6556f1d5838074b0dbe5 --dry-run

# 获授权后才执行实际下载
hf download salmon1802/Taobao --repo-type dataset \
  --revision fd01fb8edbfa58bb4b0d6556f1d5838074b0dbe5 \
  --local-dir /root/autodl-tmp/datasets/Taobao_Action
```

目录需要 `meta_data.json` 和每个 split 的 `data/user_info/item_info/part-*.parquet`。三类文件编号须完全一致。运行审计：

```bash
python scripts/audit_taobao.py \
  --data /root/autodl-tmp/datasets/Taobao_Action \
  --output /root/autodl-tmp/reports/taobao-audit.json
```

审计扫描全部数据，核对元数据行数、标签和词表、唯一索引、历史与目标对齐、曝光时间顺序及重复目标，并保存输入文件哈希、同时间戳过滤计数、长度统计和训练四分位数。它不测量真实反馈泄漏比例。

审计报告拒绝覆盖。重新审计需使用新路径，同时更新独立项目配置中的 `data_audit_report`。训练会重新核对报告覆盖的文件集合和哈希；运行快照中仍将公开 revision 标记为 expected，而不是伪称已通过远端校验。

## 3. 生成并审计独立子集

```bash
python scripts/prepare_taobao_smoke.py \
  --source /root/autodl-tmp/datasets/Taobao_Action \
  --destination /root/autodl-tmp/datasets/Taobao_Action_Smoke

python scripts/audit_taobao.py \
  --data /root/autodl-tmp/datasets/Taobao_Action_Smoke \
  --output /root/autodl-tmp/reports/taobao-smoke-audit.json
```

子集从编号最小的分块开始，以固定种子的随机排列选择数据行；若某任务单类，按同一排列扩充，再按需进入下一分块。配套侧表完整复制，索引不重编；原数据不修改。初始 8192/4096/4096 行，因此正常约 64 个训练批次；扩充后的真实数量写入 `smoke_manifest.json`。这是为流程测试而按标签检查的子集，不用于正式比较。

目标目录必须不存在；失败时保留带 `PREPARATION_INCOMPLETE` 标记的部分输出，供排查，不能用于训练。

## 4. Smoke Test（获训练授权后）

```bash
python run_expid.py --config ./config/project \
  --expid RankMixer_Taobao_Action_Smoke --gpu 0

# CPU 排查时明确关闭 BF16；此路径不代替 GPU 验收
python run_expid.py --config ./config/project \
  --expid RankMixer_Taobao_Action_Smoke --gpu -1 --enable_bf16 false
```

不要直接运行 `run_all.sh`。Smoke 配置仍保留四任务、原尺寸模型、max_len=100，关闭 compile，workers=0。输出自动放入 `experiments/<UTC时间_实验名_种子_随机后缀>/`，包括：

- `run.json`：提交、工作区状态、参数、环境、实际命令及期望数据版本。
- `config/` 和 `data_audit.json`：配置与审计快照。
- 数据集子目录中的 `.log` 和 `.model`：日志和保留的权重。
- `metrics.json`：验证／测试指标、样本数、参数量、包含验证的 fit 耗时和吞吐、显存峰值及重载验证。

`run.json` 的 prepared 状态仅表示已创建运行，只有 `metrics.json` complete 才表示完成；训练异常会生成 `failure.json`。显存数据不代替固定 batch 下的正式延迟基准。手动指定 `--run_id` 时，已有目录会报错而不是覆盖。

## 5. 验收后才运行基线

两个正式配置均要求干净且有提交的 Git 工作区：

```bash
# 上游协议参照：保留原 action、四任务平均 AUC、原尾批行为
python run_expid.py --config ./config/project \
  --expid RankMixer_Taobao_Action_Reference --gpu 0

# 研究基线：仅曝光、完整评测、Click AUC 选模、batch 512
python run_expid.py --config ./config/project \
  --expid RankMixer_Taobao_Action_Exposure --gpu 0
```

原始参照存在已知历史反馈可观测性问题，仅用于工程复现。四卡参考日志的 batch 8192 是每卡 batch；单卡参照有效批量和优化步数不同。若参照显存不足，先提交调整方案；不悄悄更改配置继续声称原设置复现。

正式训练前，先提交 Smoke 实测与总运行次数、时间及费用估算，再取得授权。真实 Smoke 和两套基线未验收前，不进入 LAIN。全项目计划见 [PLAN.md](PLAN.md)，时间语义见 [审查记录](taobao_observability.md)。
