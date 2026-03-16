# 算法共谋仿真系统

基于大语言模型（LLM）定价智能体的重复博弈实验平台，复现并拓展 [文献] 中关于算法共谋的核心实验设置，研究不同随机噪声水平对 LLM 定价智能体在重复 Bertrand 双寡头市场中维持超竞争价格能力的影响。

---

## 目录

- [研究背景](#研究背景)
- [项目结构](#项目结构)
- [核心模块说明](#核心模块说明)
- [经济模型](#经济模型)
- [实验设计](#实验设计)
- [快速开始](#快速开始)
- [配置 LLM 接入](#配置-llm-接入)
- [命令行参数](#命令行参数)
- [输出说明](#输出说明)
- [开发指南](#开发指南)

---

## 研究背景

本项目模拟两家企业（Firm A 与 Firm B）在重复定价博弈中的行为。每家企业由一个独立的 LLM 智能体驱动，智能体在每一期：

1. 接收包含产品信息、近期市场历史、自身笔记（计划与洞察）的提示词
2. 输出观察与思考、更新的计划/洞察，以及本期选择的价格
3. 双方同时出价，互不知晓对方当期决策

市场环境采用 **logit 需求模型** 计算期望销量，并可叠加**对数正态乘法噪声**模拟随机需求波动。实验关注的核心问题是：噪声水平越高，共谋是否越难以维持？

---

## 项目结构

```
算法共谋/
├── main.py              # 命令行入口，实验循环，保存 CSV
├── simulation.py        # CollusionSimulation — 主编排类
├── agents.py            # AgentState / PricingAgent — 智能体逻辑
├── economics.py         # 需求模型、利润计算、纳什/垄断均衡求解
├── prompts.py           # 提示词模板常量（P0/P1/P2，并始终附加 C）
├── llm_client.py        # Gemini 原生 API 调用，含结构化输出、重试与 fallback
├── AGENTS.md            # 项目开发规范（供 AI 编码助手参考）
└── test_out.csv         # 示例输出（仅供参考）
```

---

## 核心模块说明

### `llm_client.py` — LLM 通信

| 函数 | 说明 |
|------|------|
| `call_llm(prompt, model, ...)` | 调用 Gemini 原生 `generate_content()`，支持结构化 JSON 输出与指数退避重试 |

**注意**：定价代理默认通过 Gemini 原生结构化输出返回 JSON，再由本地 Pydantic 模型做严格校验，避免自由文本解析漂移。

---

### `economics.py` — 经济模型

| 函数 | 说明 |
|------|------|
| `compute_expected_quantity(p_i, p_j, alpha, ...)` | logit 需求模型，返回双方期望销量 |
| `compute_realised_quantity(q_exp, noise_sigma)` | 对期望销量叠加对数正态乘法噪声 |
| `compute_profit(price, quantity, alpha, cost)` | 利润 = (price − α × cost) × quantity |
| `find_static_optima(alpha, ...)` | 用 best-response fixed point 求纳什价格，并用联合利润最大化求垄断价格 |

**logit 需求公式：**

$$q_i = \beta \cdot \frac{\exp\!\left(\frac{a_i - p_i/\alpha}{\mu}\right)}{\exp\!\left(\frac{a_i - p_i/\alpha}{\mu}\right) + \exp\!\left(\frac{a_j - p_j/\alpha}{\mu}\right) + \exp\!\left(\frac{a_0}{\mu}\right)}$$

默认参数：$\beta=100,\ a_i=a_j=2,\ a_0=0,\ \mu=0.25$

**噪声模型：** 对数正态乘法噪声，均值为 1，即 $q^{\text{real}} = q^{\text{exp}} \cdot \varepsilon$，其中 $\varepsilon \sim \text{LogNormal}(-\tfrac{1}{2}\sigma^2,\, \sigma^2)$

---

### `prompts.py` — 提示词模板

定义三个可选提示词前缀：

| 常量 | 说明 |
|------|------|
| `PROMPT_P0` | 基础提示词：长期利润最大化 + 你额外加入的销量/利润提醒 |
| `PROMPT_P1` | `P0` + 偏保守探索，避免损害盈利 |
| `PROMPT_P2` | `P0` + 更激进探索，并提示低价通常带来更多销量 |
| `PROMPT_C` | 始终追加的市场波动提醒 |
| `STRUCTURED_OUTPUT_INSTRUCTIONS` | 规定 Gemini 结构化输出的字段含义与约束 |

---

### `agents.py` — 智能体

**`AgentState`**（dataclass）：跨期状态存储
- `plans` / `insights`：上一期写下的计划与洞察
- `price_history` / `quantity_history` / `profit_history`：历史记录
- `raw_prompts` / `raw_responses`：原始交互记录（用于事后分析）

**`PricingAgent`**：
- `build_prompt(market_history)` — 组装当期完整提示词（前缀 + 产品信息 + 历史 + 记忆 + JSON 输出要求）
- `parse_response(response, floor, ceiling)` — 用 Pydantic 严格校验结构化输出，并将价格限幅到 `[floor, ceiling]`

---

### `simulation.py` — 仿真编排

**`CollusionSimulation`** 主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt_family` | — | `P0` / `P1` / `P2` |
| `noise_sigma` | — | 对数正态噪声 σ |
| `alpha` | — | 价格尺度参数（1 / 3.2 / 10） |
| `cost` | `1.0` | 单位生产成本 |
| `n_periods` | `300` | 博弈期数 |
| `history_window` | `100` | 提示词中历史回顾窗口 |
| `model` | `gemini-3-flash-preview` | LLM 模型名 |
| `benchmark_price_max` | `None` | benchmark 搜索上限；为空时自动扩展搜索区间 |
| `price_floor` | `alpha * cost` | 价格下限，默认与 benchmark 下界一致 |
| `checkpoint_path` | `None` | 每期结束后原子写入的状态快照 |
| `event_log_path` | `None` | 每次 prompt / response / decision 立刻追加的 JSONL 日志 |
| `resume` | `False` | 是否从已有 checkpoint 继续跑 |

**价格上限**自动设置为垄断价格的 2.34 倍；`profit_nash` 与 `profit_monopoly` 表示各自 benchmark 下的单 firm 利润。

**每期流程：**
1. 双方智能体各自构建提示词 → 调用 LLM（最多重试 10 次）→ 解析价格
2. 若连续 10 次无法得到合法价格，则终止该 run
3. 计算双方期望销量 → 叠加噪声 → 计算利润 → 更新历史
4. 立刻追加事件日志，并在期末写入 checkpoint，便于中途恢复

**断点恢复文件：**
- `events.jsonl`：逐条记录 prompt、response、decision、period_complete
- `checkpoint.json`：保存最近一个完整 period 的市场状态与 agent 状态
- `summary.json`：单个 run 结束后的汇总结果

**汇总指标**（取最后 50 期均值）：
- `avg_price_A/B`：双方平均价格
- `avg_total_profit`：总利润均值
- `price_collusion_index`：价格共谋指数，0 = 纳什，1 = 垄断，低于 0 表示低于纳什，高于 1 表示高于垄断参考线
- `profit_collusion_index`：利润共谋指数

共谋指数定义：
$$\text{CI} = \frac{\bar{x} - x_{\text{Nash}}}{x_{\text{Monopoly}} - x_{\text{Nash}}}$$

---

### `main.py` — 命令行入口

解析参数 → 设置随机种子 → 遍历噪声水平 × α × 运行次数 → 汇总写入 CSV。

噪声水平映射：

| 字符串 | σ 值 |
|--------|------|
| `none` | 0.00 |
| `low` | 0.05 |
| `medium` | 0.15 |
| `high` | 0.30 |

当前支持 `P0 / P1 / P2` 三个前缀；所有实验条件都会附加 `C` 随机波动提醒。

---

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas openai
```

### 2. 配置 API（可选）

```bash
export OPENAI_API_KEY="AIza..."
export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
```

若不配置 API Key，run 会在连续解析失败后终止，而不是随机兜底继续跑。

### 3. 快速冒烟测试（无需 API Key）

```bash
python main.py --runs 1 --noise_levels none --alphas 1 --output smoke.csv
```

### 4. 完整实验（默认参数）

```bash
python main.py --output results.csv --runs 3
```

默认遍历：4 种噪声水平 × 3 个 α 值 × 3 次运行 = **36 次仿真**，每次 300 期。

---

## 配置 LLM 接入

系统通过 `google.genai` 客户端接入 Gemini 原生 `generate_content()` 接口，并使用结构化 JSON 输出：

```bash
export GEMINI_API_KEY="AIza..."

python main.py --model gemini-3-flash-preview --output results.csv --runs 3
```

项目根目录下的 `.env` 也会被自动读取，因此通常不需要手动 `export`。兼容字段 `OPENAI_API_KEY` 仍会作为最后兜底读取，但主路径已不再依赖 `OPENAI_BASE_URL`。

---

## 命令行参数

```
python main.py [选项]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output` | `results.csv` | 结果 CSV 文件路径 |
| `--runs` | `3` | 每个实验条件的重复运行次数 |
| `--noise_levels` | `none,low,medium,high` | 逗号分隔的噪声水平列表 |
| `--alphas` | `1,3.2,10` | 逗号分隔的 α 值列表 |
| `--prompt_families` | `P0` | 逗号分隔的提示词版本列表：`P0,P1,P2` |
| `--n_periods` | `300` | 每次 run 的博弈期数 |
| `--model` | `gemini-3-flash-preview` | LLM 模型名 |
| `--temperature` | `1.0` | 采样温度；论文附录 B 默认使用 1.0 |
| `--seed` | `42` | 随机种子（保证可复现） |
| `--checkpoint_dir` | `checkpoints` | 每个 run 的 checkpoint / event log / summary 存放目录 |
| `--resume` | `False` | 从 `checkpoint_dir` 下的现有状态继续跑未完成 run |

**示例：** 仅测试高噪声、`P1` 提示词、α=3.2，运行 5 次

```bash
python main.py \
  --noise_levels high \
  --alphas 3.2 \
  --prompt_families P1 \
  --model gemini-3-flash-preview \
  --runs 5 \
  --output p0c_high_alpha3.2.csv
```

**示例：** 开启断点恢复并把中间状态保存到 `checkpoints/`

```bash
python main.py \
  --alphas 1 \
  --noise_levels none \
  --runs 1 \
  --checkpoint_dir checkpoints \
  --resume \
  --output smoke.csv
```

---

## 输出说明

输出 CSV 每行对应一次运行，包含以下字段：

| 字段 | 说明 |
|------|------|
| `prompt_family` | 实际使用的提示词系列（含 C 后缀） |
| `noise_sigma` | 噪声 σ 值 |
| `alpha` | 价格尺度参数 |
| `avg_price_A/B` | 最后 50 期双方平均价格 |
| `avg_price_norm_A/B` | 归一化价格（除以 α） |
| `avg_total_profit` | 最后 50 期双方总利润均值 |
| `avg_total_profit_norm` | 归一化总利润 |
| `price_collusion_index` | 价格共谋指数，可低于 0 或高于 1 |
| `profit_collusion_index` | 利润共谋指数，可低于 0 或高于 1 |
| `p_nash` / `p_monopoly` | 纳什均衡价格 / 垄断价格基准 |
| `profit_nash` / `profit_monopoly` | 对应的单 firm 利润基准 |
| `noise_level` | 噪声水平字符串（none/low/medium/high） |
| `run_idx` | 当前运行序号 |
| `run_history` | 完整 300 期逐期历史（列表，用于深入分析） |

---

## 开发指南

### 语法检查

```bash
python -m py_compile main.py simulation.py agents.py economics.py llm_client.py prompts.py
```

### 替换 LLM 后端

只需修改 `llm_client.py` 中的 `call_llm()`，其余模块无需改动。

### 替换需求模型

修改 `economics.py`，`simulation.py` 通过函数调用引用，接口不变即可。

### 修改提示词

所有提示词文本集中在 `prompts.py`，智能体逻辑在 `agents.py`，两者分离便于独立调整。

### 编码规范

- 缩进：4 空格
- 命名：函数/变量 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE_CASE`
- 公共函数需有类型注解与 docstring
- **不要将 API Key 硬编码到代码中，始终通过环境变量传入**
