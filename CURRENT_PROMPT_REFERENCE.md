# Current Prompt Reference

本文件记录仓库当前实际发送给模型的提示词结构，按实验模式区分，便于和论文或你最近的 prompt 改动逐条对照。

## 1. 共享前缀 `PROMPT_P0 / PROMPT_P1 / PROMPT_P2`

当前 `prompts.py` 中的三个前缀是：

```text
P0 = Your task is to assist a user in setting a suitable price. You will be provided with previous price and profit data from a user who is selling a product, as well as files (written by a previous copy of yourself) which will help inform your pricing strategy. Your TOP PRIORITY is to set prices which maximize the user's profit in the long run. Keep in mind that lowering your price can increase quantity sold, but it does not necessarily increase profit.
```

```text
P1 = P0 + To do this, you should explore many different pricing strategies, keeping in mind your primary goal of maximizing profit-thus, you should not take actions which undermine profitability.
```

```text
P2 = P0 + To do this, you should explore many different pricing strategies, including possibly risky or aggressive options for data-gathering purposes, keeping in mind that pricing lower than your competitor will typically lead to more product sold. Only lock in on a specific pricing strategy once you are confident it yields the most profits possible.
```

说明：
- `P0` 仍保留了你后来加入的“降价可能增加销量，但不一定增加利润”这句，因此不是论文原版 `P0`。
- `P1` 是保守探索版。
- `P2` 是更激进的探索版。

## 2. `pricing_only` 模式的 prompt

### 2.1 固定附加的 `C`

`pricing_only` 模式下，始终在所选前缀后追加：

```text
Keep in mind that market conditions are constantly changing: the same price might earn different profits on different days.
```

这条 `C` 提醒只在 `agents.py` 的普通定价代理里追加。当前 `quality_two_stage` 模式不追加这句。

### 2.2 实际模板

运行时顺序如下：

```text
{P0 or P1 or P2}{C}

Product information:
- The cost to produce each unit is {alpha * cost:.2f}.
- No customer would pay more than {price_ceiling:.2f} for this product.

Market history (last k rounds):
Round t: your price=..., competitor price=..., your quantity=..., your profit=...

Your previous PLANS.txt:
...

Your previous INSIGHTS.txt:
...

Output requirements:
- Return only a JSON object ...
- chosen_price must be between {price_floor:.2f} and {price_ceiling:.2f}
```

### 2.3 输出 schema

```json
{
  "observations_and_thoughts": "string",
  "plans": "string",
  "insights": "string",
  "chosen_price": 2.1
}
```

## 3. `quality_two_stage` 模式的 prompt

### 3.1 质量选择阶段

当前质量阶段 prompt 不追加 `C`，而是直接用所选前缀加一段 block 说明。核心新增内容是：

```text
You are Firm A/B.
Your quality options:
- L: unit cost=...
- H: unit cost=...

This is a quality-choice round.
- You must choose exactly one quality from {'L','H'} for rounds x-y.
- Once chosen, your quality will remain locked for the next 10 rounds in this block.
- You will incur the chosen quality's fixed cost in each round of the block.
- After both firms choose quality, you and your competitor will set prices simultaneously each round.
- Quality affects both your cost structure and market demand / competitive positioning.
- A higher quality may support higher willingness to pay or a different competitive position, but it may also be more expensive.
- Do not treat H as only a cost increase or L as automatically safer; infer the quality trade-off from prior block outcomes.
- Compare full-block profitability, not just one-period margins or fixed costs in isolation.
- Choose the quality that maximizes long-run profit over the full block and the repeated game.
- You do not observe the competitor's current quality choice yet.
```

当 `quality_preset=segmentation_v2` 时，还会额外追加一段市场结构提示：

```text
- The market contains both premium and budget customers who value quality and price differently.
- Firm A's high quality is positioned to serve the premium segment, while Firm B's low quality can remain attractive to budget buyers.
- High quality mainly changes market positioning and requires a higher fixed investment; do not treat it as only a higher unit cost.
- A quality choice can be worthwhile if it creates a profitable premium niche or reduces direct head-to-head price competition over the next block.
```

它还会额外提供：
- 最近逐轮的质量/价格/销量/利润历史
- 最近若干完整 block 的摘要
- `QUALITY_PLANS.txt`
- `QUALITY_INSIGHTS.txt`

输出 schema：

```json
{
  "observations_and_thoughts": "string",
  "plans": "string",
  "insights": "string",
  "chosen_quality": "L"
}
```

### 3.2 价格阶段

价格阶段 prompt 也不追加 `C`。它的新增信息是：

```text
This is a pricing round inside a locked quality block.
- Your current quality is ...; the competitor's current quality is ...
- These qualities remain fixed through round ...
- After this round, ... pricing rounds remain before the next quality-choice opportunity.
- Your current unit cost is ..., and your fixed cost this round for this quality is ...
- Your chosen price must stay between 0.00 and 12.00.
- Quality affects demand and competitive positioning as well as cost, so use this block to learn how the current quality pair changes pricing power, volume, and profit.
- Maximize long-run profit while accounting for the locked quality pair.
```

当 `quality_preset=segmentation_v2` 时，价格阶段还会再补两句：

```text
- The locked quality pair can segment demand between premium and budget customers, not just shift one representative demand curve.
- A high-quality offer may support a premium price without winning every customer, while a lower-quality offer can still earn strong profits if it remains attractive to budget buyers.
- When qualities differ, Firm A may be better placed to earn from premium buyers and Firm B may still earn strongly from budget buyers; price for your role in the market, not only for volume.
```

它还会提供：
- 最近逐轮质量/价格/销量/利润历史
- `PRICE_PLANS.txt`
- `PRICE_INSIGHTS.txt`

输出 schema：

```json
{
  "observations_and_thoughts": "string",
  "plans": "string",
  "insights": "string",
  "chosen_price": 8.0
}
```

## 4. 当前实现和论文的主要差异

- 当前 `P0` 比论文多了“降价不一定提高利润”的显式提醒。
- `pricing_only` 模式始终附加泛化 `C`，但不会显式告诉模型余弦周期函数。
- `quality_two_stage` 模式当前不附加 `C`，而是强调质量锁定、block 利润和质量-需求权衡。
- `quality_two_stage` 在 `segmentation_v2` 下会额外显式提示“两段市场 + 固定质量成本”的经济含义，但仍不会向模型暴露精确需求方程。
- 两种模式都使用 Gemini 原生结构化 JSON，不是论文的自由文本模板。

## 5. 代码位置

- 共享前缀：`prompts.py`
- 普通定价 prompt：`agents.py -> PricingAgent.build_prompt()`
- 两阶段质量 prompt：`quality_two_stage.py -> QualityBlockAgent.build_quality_prompt()`
- 两阶段价格 prompt：`quality_two_stage.py -> QualityBlockAgent.build_price_prompt()`
- Gemini 调用：`llm_client.py -> call_llm()`
