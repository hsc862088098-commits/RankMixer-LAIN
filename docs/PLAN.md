# LAIN＋RankMixer CTR 精排项目计划

## 目标与执行边界

完成可追溯的 RankMixer 基线、LAIN 适配、长度分组实验和效率分析，形成可复现的实习求职项目。主实验采用**仅曝光历史**，四任务标签保留 `is_click/cart/fav/buy`；上游原始历史协议只作复现参照，两套结果分表。

当前仅实施阶段一。依赖安装、公开数据下载和正式 GPU 训练需分别授权；不在基线验收前实现 LAIN。单卡与原参考日志四卡的有效批量不同，不能声称严格复现原论文指标。

环境修正：实际依赖验证确认 HeavyBall 3.2.0 要求 NumPy>=2.3.4，后者要求 Python>=3.11。因此使用 Python 3.11，不再采用原 README 的 Python 3.10 建议。实验室配置和授权范围见 [lab.md](lab.md)。

固定 UniRank 起始提交 `90bb99a60724e975ec565836266fd86dcad0039d`；Taobao revision `fd01fb8edbfa58bb4b0d6556f1d5838074b0dbe5`。发布文件清单约 0.77 GiB。真实下载内容还须审计并保存哈希，不用“期望 revision”冒充已验证版本。

## 阶段一：数据协议、工程准备与基线

- `history_feedback_policy=original` 保持兼容默认行为；`exposure_only` 不读取历史反馈序列，有效历史 action 映射为元数据的 `exposure`，Padding 和当前候选 action 为 0。
- 根据曝光时间先排除目标、未来及同时间戳曝光，再保留最近 `max_len` 条历史。保留过滤前长度、过滤后截断前 `L_prefix`、实际 `L_input` 和被排除的同时间戳条数；这些字段不进入 Embedding。
- 审计三表分块集合、唯一索引、词表、标签顺序、时间顺序、目标对齐、重复目标身份、样本数量及长度分布。审计报告保存各输入文件 SHA256，运行前重新核验。
- 原始 RankMixer 与原始配置保留。独立项目配置、唯一运行目录、配置快照、完整参数、提交与工作区状态、环境、数据审计及测试指标均可追踪；项目配置保留 checkpoint。
- 修正 BF16 参数、任务名前缀选模键及缺失键校验。研究验证／测试不丢尾批，实际样本数必须匹配。
- Smoke 配置：batch 128、max_len 100、embedding 16、token_dim 256、4 Token、3 Mixer 层、1 epoch、无累积；workers 0、cache 1、compile 与 gradient checkpointing 关闭。子集初始大小 8192/4096/4096，按固定规则扩充至四任务均有正负样本。
- 先完成合成数据与 CPU 测试，再在获授权的 AutoDL 环境运行真实数据 Smoke；之后依次运行上游协议参照与仅曝光研究基线。

阶段一验收：真实批次加载、四任务有限损失及参数更新、完整评测、正确 Click 选模、权重保存和重载一致。未完成真实数据与 GPU 验收时不标记基线复现完成。

## 阶段二：LAIN 适配（待基线验收）

新增独立 `RankMixerLAIN`，原主干、Tokenizer、四预测塔和输出键不变。接口：target `[B,112]`、history `[B,T,112]`、mask `[B,T]`、lengths `[B]`，输出 interest `[B,112]`。

依次实现温度调制、SLE＋完整 LMA、LCP。默认 Fourier 维度 32、长度表示 512、4 个 112 维 Prompt、256 维单头 Attention、LAIN dropout 0.1。温度采用正值参数化，长度中心初始化为训练样本平均 `L_input`；记录与官方默认值的差异。长度在插入 Prompt 前计算；Prompt 不进入行为长度分组。无 Prompt 的空历史输出有限零向量，有 Prompt 时允许学习上下文表示。

## 阶段三：实验、归因与效率（待阶段二）

| 编号 | 设置 |
|---|---|
| R0 | 原始 Target Attention＋RankMixer，仅曝光历史 |
| R1 | R0＋归一化 log1p(L_input)，投影为 16 维普通长度特征 |
| R2 | 完整 LAIN＋RankMixer |
| R3 | R2 的 SLE 换普通长度 MLP，输出仍为 512 维 |
| R4 | R2 去掉 LCP |
| R5 | R2 禁用 LMA，保留 SLE、LCP |
| R6 | R2 的动态 Prompt 换静态可学习 Prompt |

优先 R0/R1/R2，再做消融。正式种子 20262027、20262028、20262029；固定曝光历史、划分、四任务损失和 Click 选模规则，报告均值、标准差及配对差值，不把三种子直接解释为统计显著。研究默认单卡 batch 512、1 epoch、无累积、关闭 compile；显存不足时统一减半并重跑受影响对照。

长度边界使用训练样本 `L_prefix` 的四分位数，重复边界合并，之后固定。每组报告样本数、去重用户数、点击率、AUC、LogLoss，单类 AUC 记不可计算。GAUC 以同时有正负样本的有效用户为组、曝光数加权，报告覆盖率。

同 GPU、精度与编译设置下，batch 1 及共同训练 batch 各预热 50 次、计时 200 次；同步 CUDA，排除读取数据。报告 P50/P95 延迟、训练吞吐、Embedding/Dense 参数和峰值显存。

## 限制与交付

历史 action 的未来反馈泄漏已有源码反事实证据，见 [Taobao 时间语义审查](taobao_observability.md)。未读取真实公开 Parquet，实际受影响比例未知。仅曝光协议不解决辅助标签的类别／品牌归因、窗口右删失、用户过滤或词表边界问题，也不代表完整线上回放。

第一版不加入门控、DCNv2 或 EB-NeRD。交付代码、测试、独立配置、数据审计、运行清单、真实实验表、效率报告、README 和基于真实结果的简历描述。AutoDL 操作入口见 [阶段一运行说明](stage1.md)。
