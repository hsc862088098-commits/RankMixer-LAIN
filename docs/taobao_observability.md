# Taobao 历史反馈可观测性审查

审查基于 UniRank 提交 `90bb99a60724e975ec565836266fd86dcad0039d` 及发布版元数据 revision `fd01fb8edbfa58bb4b0d6556f1d5838074b0dbe5`。

## 已确认的链路

1. `attach_behavior_labels_by_forward_window()` 找出曝光时刻之后 86400 秒内、同用户／类别／品牌的 Cart/Favorite/Buy 行为。
2. `process_partition()` 将上述最终标签和 Click 一起编码为 `action`，写入 `full_action_seq`。
3. 原始加载器按 `seq_len` 取曝光前缀，同时取最终 action 前缀，没有检查反馈发生时间。
4. RankMixer 将 action 作为商品／行为字段参与 Embedding，并送入 Target Attention。因此，即使目标 action 为 0，历史仍可能携带预测时刻之后的信息。

源码反事实测试：历史曝光在 10:00，当前预测在 10:30，购物发生在 12:00。删除 12:00 的购买事件，10:00 的历史输入由 `buy` 变为 `exposure`。历史输入依赖预测时刻之后的事件，构成泄漏路径。

[发布元数据](https://huggingface.co/datasets/salmon1802/Taobao/resolve/fd01fb8edbfa58bb4b0d6556f1d5838074b0dbe5/meta_data.json) 确认使用曝光后窗口；`full_timestamp_seq` 记录的是曝光时间，没有反馈发生时间。不能仅由这些字段恢复真实反馈可见状态，也不能直接统计真正泄漏的比例。尚未下载或检查真实 Parquet 内容。

## 采用的研究协议

`exposure_only` 不读取 `full_action_seq`；每条有效历史曝光使用元数据的 `exposure` ID。先以 `timestamp < target_timestamp` 排除同时间戳事件，再截取最近的历史。不得用反馈标签筛选序列。目标的四任务标签保留，只参与监督，不进入目标或历史输入。

同时间戳事件的内部先后顺序不可知，故保守排除；这不等于排除所有同商品历史。此前发生的同商品曝光可以保留。

批次附带的审计字段：

| 字段 | 含义 |
|---|---|
| `seq_len` | 发布数据的原始目标位置，仍用于索引；不覆盖 |
| `history_length_before_filter` | 时间过滤前的曝光前缀长度 |
| `history_length_prefix` | 时间过滤后、截断前的 L_prefix |
| `history_length_input` | 实际输入的 L_input |
| `history_ties_removed` | 从前缀排除的同时间戳事件数 |

上述附加字段被 BaseModel 显式排除于 Embedding。`original` 模式保留原历史反馈和截断行为；两类协议分开报告。

## 尚未解决的限制

- 购物标签为用户／类别／品牌匹配，不是广告商品的直接转化归因。
- 观察窗口末尾的购物标签可能右删失；辅助监督窗口也可能跨越划分边界。
- Click 没有独立的发生时间，不能保证曝光后的任意下一时刻已收到该点击。
- 用户过滤使用整个发布窗口，词表构建也不等同于线上训练时可用状态。
- 曝光历史长度仅代表公开观察窗口内的历史，不是用户终身历史或严格冷启动群体。

因此，本项目修复的是历史输入的反馈可观测性，不宣称实现全部线上时间协议。
