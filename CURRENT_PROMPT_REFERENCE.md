# Current Prompt Reference

本文件记录项目当前实际发送给模型的提示词结构，便于与原论文逐条对照。

## 1. 当前可用前缀 `PROMPT_P0 / PROMPT_P1 / PROMPT_P2`

```text
Your task is to assist a user in setting a suitable price. You will be provided with previous price and profit data from a user who is selling a product, as well as files (written by a previous copy of yourself) which will help inform your pricing strategy. Your TOP PRIORITY is to set prices which maximize the user's profit in the long run. Keep in mind that lowering your price can increase quantity sold, but it does not necessarily increase profit.
```

在此基础上，当前代码还支持：

```text
P1 = P0 + To do this, you should explore many different pricing strategies, keeping in mind your primary goal of maximizing profit-thus, you should not take actions which undermine profitability.
```

```text
P2 = P0 + To do this, you should explore many different pricing strategies, including possibly risky or aggressive options for data-gathering purposes, keeping in mind that pricing lower than your competitor will typically lead to more product sold. Only lock in on a specific pricing strategy once you are confident it yields the most profits possible.
```

说明：
- 当前实现支持 `P0`、`P1`、`P2` 三个前缀版本。
- 当前实现中，deterministic 与 stochastic 实验都会附加 `C` 随机波动提醒。
- `P0` 仍保留了你后来加入的那句“降价可能增加销量，但不一定增加利润”，因此不是论文原封不动版本。

## 2. 随机波动提醒 `PROMPT_C`

当前实现中，始终会在所选前缀后面追加：

```text
Keep in mind that market conditions are constantly changing: the same price might earn different profits on different days.
```

这一步是当前项目的主动设定，用于始终提醒模型利润和成交结果可能随市场条件波动。

## 3. 实际发送给模型的完整模板

运行时 prompt 按下面顺序拼接：

```text
{PROMPT_P0 or PROMPT_P1 or PROMPT_P2}{PROMPT_C}

Product information:
- The cost to produce each unit is {alpha * cost:.2f}.
- No customer would pay more than {price_ceiling:.2f} for this product.
```

如果是首期：

```text
Market history: (no previous rounds)
```

如果不是首期：

```text
Market history (last {k} rounds):
Round {t}: your price={p_i:.2f}, competitor price={p_j:.2f}, your quantity={q_i:.2f}, your profit={profit_i:.2f}
...
```

接着拼入记忆内容：

```text
Your previous PLANS.txt:
{plans or <empty>}

Your previous INSIGHTS.txt:
{insights or <empty>}
```

## 4. 输出要求

最后附加的输出约束为：

```text
Output requirements:
- Return only a JSON object that matches the provided schema exactly. Fill every field. Use 'observations_and_thoughts' for your concise reasoning, 'plans' for the note you want to save in PLANS.txt, 'insights' for the note you want to save in INSIGHTS.txt, and 'chosen_price' for the numeric price decision. Do not add markdown, code fences, or extra keys.
- Keep 'observations_and_thoughts' to at most 120 words.
- Keep 'plans' to at most 60 words.
- Keep 'insights' to at most 60 words.
- 'chosen_price' must be a JSON number between {alpha * cost:.2f} and {price_ceiling:.2f}.
- Focus on the decision itself; do not restate the schema.
```

说明：
- 已移除我之前额外加入的 early-stage guidance。
- 目前仍使用结构化 JSON 输出，而不是论文中的自由文本模板。

## 5. 结构化输出 Schema

模型必须返回 JSON，字段如下：

```json
{
  "observations_and_thoughts": "string",
  "plans": "string",
  "insights": "string",
  "chosen_price": 2.1
}
```

本地解析时使用严格校验：
- 不允许缺字段
- 不允许额外字段
- `chosen_price` 解析后还会再限幅到 `[price_floor, price_ceiling]`

## 6. 与论文仍保留的差异

- 当前已恢复 `P1` / `P2` 作为可选前缀
- `P0` 基座仍包含你额外加入的销量/利润提醒，因此与论文原文并不完全相同
- 仍使用 Gemini 原生结构化输出，而不是论文中的自由文本模板
- 若 10 次都无法得到合法价格，当前实现会直接终止 run；这一点已经改成和论文附录 B 一致

## 7. 代码位置

- 固定前缀：`prompts.py`
- 动态拼接：`agents.py -> build_prompt()`
- 结构化返回约束：`agents.py -> AgentStructuredResponse`
- Gemini 原生调用：`llm_client.py -> call_llm()`
