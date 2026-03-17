# Experiment Run Guide

本文件给出当前项目的完整实验运行方案，对应四类实验：

1. 周期主实验
2. `alpha` 稳健性实验
3. 周期与销量随机性的交互实验
4. 提示词对照实验

这些命令都基于当前本地最新版实现，默认使用 Gemini、structured output、checkpoint 和 session 时间标签。

另外，`quality_two_stage` 是一条独立实验线，不包含在下面四类周期实验里。当前推荐的质量预设是 `--quality_preset segmentation_v2`，它使用“两段市场 + 固定质量成本”，目标结构是 `LL` 为质量纳什、`HL` 为联合最优。

## 0. 准备

先确保环境可用：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas google-genai
```

配置 API：

```bash
export GEMINI_API_KEY="你的_key"
```

如果项目根目录已有 `.env`，也可以直接使用，不必重复 `export`。

## 1. 共同约定

除非特别说明，下面四类实验都建议使用这组共同设置：

- `n_periods = 300`
- `temperature = 1`
- `cycle_period = 150`
- `cycle_baseline = 1.0`
- `cycle_effect_shares = 0,0.1,0.2,0.3,0.4`

这表示：

- 总共运行 `300` 期
- 每 `150` 期走完一个完整 `2π` 周期
- `0.1/0.2/0.3/0.4` 分别对应峰谷差是均值的 `10%/20%/30%/40%`
- 当 `cycle_effect_share = 0` 时，就是“无周期”基线组

当前代码中的 `noise_levels` 表示的是“个体 realised quantity 的对数正态噪声”，不是共同市场冲击。也就是说，第 3 类实验研究的是“周期 + 个体销量随机性”的交互，不是“周期 + 共同随机市场冲击”。

## 2. 先做短测

正式跑 300 期前，先用 20 或 30 期短测链路是否正常：

```bash
python main.py \
  --prompt_families P2 \
  --alphas 3.2 \
  --noise_levels none \
  --cycle_effect_shares 0.2 \
  --cycle_period 20 \
  --n_periods 40 \
  --temperature 1 \
  --runs 1 \
  --output smoke_cycle.csv
```

## 3. 实验一：周期主实验

目标：研究公共经济周期强度是否改变高价协调。

推荐先固定：

- `prompt = P2`
- `alpha = 3.2`
- `noise = none`

运行命令：

```bash
python main.py \
  --prompt_families P2 \
  --alphas 3.2 \
  --noise_levels none \
  --cycle_effect_shares 0,0.1,0.2,0.3,0.4 \
  --cycle_period 150 \
  --cycle_baseline 1.0 \
  --n_periods 300 \
  --temperature 1 \
  --runs 5 \
  --output exp1_cycle_main.csv
```

重点看：

- `avg_price_A`, `avg_price_B`
- `avg_total_profit`
- `price_collusion_index`
- `profit_collusion_index`
- `high_phase_avg_price`, `low_phase_avg_price`
- `high_phase_avg_total_profit`, `low_phase_avg_total_profit`

## 4. 实验二：alpha 稳健性

目标：检查周期效应是否在不同 `alpha` 下仍成立。

固定：

- `prompt = P2`
- `noise = none`

变化：

- `alpha = 1,3.2,10`

运行命令：

```bash
python main.py \
  --prompt_families P2 \
  --alphas 1,3.2,10 \
  --noise_levels none \
  --cycle_effect_shares 0,0.1,0.2,0.3,0.4 \
  --cycle_period 150 \
  --cycle_baseline 1.0 \
  --n_periods 300 \
  --temperature 1 \
  --runs 3 \
  --output exp2_cycle_alpha_robustness.csv
```

建议先重点读 `alpha=3.2`，再比较 `1` 和 `10` 是否有同方向结论。

## 5. 实验三：周期与销量随机性的交互

目标：研究公共经济周期和个体销量随机性是否共同影响共谋。

固定：

- `prompt = P2`
- `alpha = 3.2`

变化：

- `noise = none,low,medium,high`
- `cycle_effect_share = 0,0.1,0.2,0.3,0.4`

运行命令：

```bash
python main.py \
  --prompt_families P2 \
  --alphas 3.2 \
  --noise_levels none,low,medium,high \
  --cycle_effect_shares 0,0.1,0.2,0.3,0.4 \
  --cycle_period 150 \
  --cycle_baseline 1.0 \
  --n_periods 300 \
  --temperature 1 \
  --runs 3 \
  --output exp3_cycle_noise_interaction.csv
```

噪声含义：

- `none = 0.00`
- `low = 0.05`
- `medium = 0.15`
- `high = 0.30`

这一组最适合回答：

- 周期本身是否促进或削弱高价协调
- 当 realised quantity 更随机时，这种周期效应会不会变弱

## 6. 实验四：提示词对照

目标：研究周期效应在不同提示词人格下是否一致。

固定：

- `alpha = 3.2`
- `noise = none`

变化：

- `prompt = P0,P1,P2`

运行命令：

```bash
python main.py \
  --prompt_families P0,P1,P2 \
  --alphas 3.2 \
  --noise_levels none \
  --cycle_effect_shares 0,0.1,0.2,0.3,0.4 \
  --cycle_period 150 \
  --cycle_baseline 1.0 \
  --n_periods 300 \
  --temperature 1 \
  --runs 3 \
  --output exp4_prompt_cycle_compare.csv
```

解释建议：

- `P0` 更中性
- `P1` 更保守
- `P2` 更主动探索

如果三组都表现出相似的周期响应，结论更稳；如果只有 `P2` 对周期敏感，那更像是 prompt-driven effect。

## 7. 真正“全部跑完”的总命令

如果你想把上面四组逻辑一次性压成全组合，也可以直接跑：

```bash
python main.py \
  --prompt_families P0,P1,P2 \
  --alphas 1,3.2,10 \
  --noise_levels none,low,medium,high \
  --cycle_effect_shares 0,0.1,0.2,0.3,0.4 \
  --cycle_period 150 \
  --cycle_baseline 1.0 \
  --n_periods 300 \
  --temperature 1 \
  --runs 3 \
  --output full_matrix.csv
```

这个矩阵规模是：

- `3 prompt × 3 alpha × 4 noise × 5 cycle × 3 runs = 540 runs`

很慢，也比较贵，不建议第一次就这样跑。

## 8. 建议的执行顺序

推荐按下面顺序做：

1. 先跑短测，确认模型和 checkpoint 正常
2. 跑实验一，先看周期主效应
3. 跑实验二，确认 `alpha` 稳健性
4. 跑实验三，研究周期与个体销量随机性的交互
5. 最后跑实验四，确认 prompt 效应是不是主导因素

## 9. 输出和 checkpoint

每次 run 都会生成：

- `events.jsonl`
- `checkpoint.json`
- `summary.json`

目录格式：

```text
checkpoints/prompt_{prompt}/noise_{noise}/cycle_{share}/period_{period}/baseline_{baseline}/alpha_{alpha}/run_{idx}/session_{timestamp}/
```

如果中途中断，可直接加 `--resume` 继续；现在同样支持把较短 run 接着扩展到更长的 `n_periods`：

```bash
python main.py \
  --prompt_families P2 \
  --alphas 3.2 \
  --noise_levels none \
  --cycle_effect_shares 0.2 \
  --cycle_period 150 \
  --cycle_baseline 1.0 \
  --n_periods 300 \
  --temperature 1 \
  --runs 1 \
  --resume \
  --output resumed.csv
```

## 10. 最推荐的一套主实验

如果你现在只想先回答一个最核心的问题：

“市场经济周期强度会不会影响算法定价共谋？”

那就先跑这一套：

```bash
python main.py \
  --prompt_families P2 \
  --alphas 3.2 \
  --noise_levels none \
  --cycle_effect_shares 0,0.1,0.2,0.3,0.4 \
  --cycle_period 150 \
  --cycle_baseline 1.0 \
  --n_periods 300 \
  --temperature 1 \
  --runs 5 \
  --output core_cycle_study.csv
```

这是当前最干净、最适合先出结论的一组。

## 11. 两阶段质量实验（方案二 / segmentation_v2）

这条实验线和上面的四类周期实验分开跑。当前版本的共同设置是：

- `experiment_mode = quality_two_stage`
- `quality_preset = segmentation_v2`
- `quality_block_length = 10`
- `n_periods = 500`
- deterministic only：`noise_levels` 必须是 `none`，`cycle_effect_shares` 必须是 `0`

### 11.1 先做 20 期短测

```bash
python main.py \
  --experiment_mode quality_two_stage \
  --prompt_families P2 \
  --quality_preset segmentation_v2 \
  --quality_block_length 10 \
  --n_periods 20 \
  --runs 1 \
  --resume \
  --output quality_v2_smoke_20.csv
```

### 11.2 正式 500 期单组实验

```bash
python main.py \
  --experiment_mode quality_two_stage \
  --prompt_families P2 \
  --quality_preset segmentation_v2 \
  --quality_block_length 10 \
  --n_periods 500 \
  --runs 1 \
  --resume \
  --output quality_v2_500.csv
```

### 11.3 做 `P0/P1/P2` 对照

```bash
python main.py \
  --experiment_mode quality_two_stage \
  --prompt_families P0,P1,P2 \
  --quality_preset segmentation_v2 \
  --quality_block_length 10 \
  --n_periods 500 \
  --runs 3 \
  --resume \
  --output quality_v2_prompt_compare.csv
```

### 11.4 推荐重点读的字段

- `quality_nash_pair`
- `joint_optimum_pair`
- `quality_pair_share_LL/LH/HL/HH`
- `last_window_quality_pair_share_LL/LH/HL/HH`
- `avg_price_A`, `avg_price_B`
- `avg_total_profit`
- `profit_coordination_index`

如果你要研究“是否从保守状态逐步走向 A 高质量 / B 低质量的分工协调”，最关键的是：

- 后 100 期或后若干 block 的 `HL` 占比
- `HL` 条件下 A 的价格是否持续高于 B
- 总利润提高到底来自“更多进入 HL”，还是来自“进入 HL 后又进一步加价”
