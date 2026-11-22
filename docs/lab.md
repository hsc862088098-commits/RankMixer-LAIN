# 实验室主机运行说明

## 已确认的机器与环境

- SSH 别名：`4090`；远程用户 `b220`。
- 项目：`/mnt/nas/volume2/hsc/UniRank`。
- Conda 环境：`/home/b220/miniconda3/envs/hsc`，Python 3.11。
- RTX 4090 24GB；PyTorch 2.8.0+cu126；NVIDIA 驱动 570.133.07。
- `volume2` 是 NAS 挂载 `/mnt/nas/volume2`，不是根目录 `/volume2`。

旧 hsc 环境按用户授权清除；删除前的环境清单保存在 `/mnt/nas/volume2/hsc/environment-backups/`。其他 Conda 环境不修改。

上游 README 的 Python 3.10 建议不能满足当前依赖组合：HeavyBall 3.2.0 要求 NumPy>=2.3.4，而该 NumPy 要求 Python>=3.11。实验室使用 Python 3.11，并通过 `requirements-lab.txt` 锁定实际安装版本。校园 PyPI 镜像的 HeavyBall 链接曾返回 404，因此安装命令显式选择官方 PyPI，不修改全局镜像。

```bash
source /home/b220/miniconda3/etc/profile.d/conda.sh
conda activate hsc
cd /mnt/nas/volume2/hsc/UniRank
python scripts/check_environment.py
python -m pip check
```

新建环境时使用 `conda create -n hsc python=3.11 pip`，然后执行 `python -m pip install --index-url https://pypi.org/simple -r requirements-lab.txt`。不要在已有环境上直接执行删除命令。

## 路径与测试

实验室配置使用 `--config ./config/lab`。数据放在 `/mnt/nas/volume2/hsc/datasets`，审计放在 `/mnt/nas/volume2/hsc/reports`，运行产物放在 `/mnt/nas/volume2/hsc/experiments`。`config/project` 的 AutoDL 路径保持原样。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python -m unittest discover -s tests -t . -v
```

## 固定版本数据与 Smoke Test

```bash
hf download salmon1802/Taobao --repo-type dataset \
  --revision fd01fb8edbfa58bb4b0d6556f1d5838074b0dbe5 \
  --local-dir /mnt/nas/volume2/hsc/datasets/Taobao_Action

python scripts/audit_taobao.py \
  --data /mnt/nas/volume2/hsc/datasets/Taobao_Action \
  --output /mnt/nas/volume2/hsc/reports/taobao-audit.json

python scripts/prepare_taobao_smoke.py \
  --source /mnt/nas/volume2/hsc/datasets/Taobao_Action \
  --destination /mnt/nas/volume2/hsc/datasets/Taobao_Action_Smoke

python scripts/audit_taobao.py \
  --data /mnt/nas/volume2/hsc/datasets/Taobao_Action_Smoke \
  --output /mnt/nas/volume2/hsc/reports/taobao-smoke-audit.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python run_expid.py --config ./config/lab \
  --expid RankMixer_Taobao_Action_Smoke --gpu 0
```

本轮授权包括依赖安装、数据下载、审计和一次完整 Smoke 流程，不包含正式全量基线训练或 LAIN。Smoke 指标只验证流程，不作为正式效果结果。

NAS 的 UID 映射需要 Git 精确目录信任；该用户已配置 `safe.directory=/mnt/nas/volume2/hsc/UniRank`。不添加通配符信任。
