---
type: anki-deck-index
deck: Sutton-Barto
total_cards: ~250
format: cloze (Obsidian Spaced Repetition plugin)
created: 2026-05-24
---

# Sutton & Barto Anki Cloze 卡片集

> **使用方式**：装 Obsidian 的 [Spaced Repetition](https://github.com/st3v3nmw/obsidian-spaced-repetition) 插件，每天 review 20 分钟。
>
> **格式**：planetbanatt 风格 single-term cloze——遮掉一个关键术语 / 数字 / 公式片段，**不遮整公式**（不可 grade）。
>
> **覆盖原则**：只覆盖 interview-load-bearing 章节。Ch 4/7/8/10/14/15/16/17 故意跳过。

## Decks

| 文件 | 章节 | 卡数 | 主题 |
|---|---|---|---|
| [[ch03-mdp-cards]] | Ch 3 MDP | ~35 | Bellman 方程 / return / policy |
| [[ch05-mc-cards]] | Ch 5 Monte Carlo | ~35 | first/every-visit / IS / weighted vs ordinary |
| [[ch06-td-cards]] | Ch 6 TD | ~38 | TD(0) / SARSA / Q-learning / Expected SARSA / max bias / Double Q |
| [[ch09-fa-cards]] | Ch 9 函数近似预测 | ~35 | VE / semi-gradient / linear FA / tile coding / LSTD |
| [[ch11-deadlytriad-cards]] | Ch 11 致命三元组 | ~35 | Baird / MSBE / MSPBE / GTD / Emphatic-TD |
| [[ch12-gae-cards]] | Ch 12 ET + GAE | ~35 | λ-return / backward TD(λ) / True Online / GAE |
| [[ch13-pg-cards]] | Ch 13 Policy Gradient | ~42 | PG 定理 / REINFORCE / A2C / TRPO / **PPO clip** / **GRPO** / **DPO** / RLVR |

**总计**：~255 张

## Anki 卡片质量原则（planetbanatt）

1. **单 term 遮挡**：遮掉一个关键术语 / 数字 / 符号——而不是整段公式（整公式无法 grade）
2. **公式 + 直觉同框**：让卡片同时考公式片段 + 直觉解释
3. **不要 abstract 概念卡**："什么是 RL" 这种太宽，不会被 review
4. **可逆性**：好的卡片往往可双向——A→B 也 B→A
5. **interview gold**：优先收手撕公式 + 高频面试题答案

## Review Schedule（建议）

| 日 | 学习 |
|---|---|
| 周一 | review 上周已学 + 学 Ch 3 + Ch 5 |
| 周二 | review + 学 Ch 6 |
| 周三 | review + 学 Ch 9 |
| 周四 | review + 学 Ch 11 |
| 周五 | review + 学 Ch 12 |
| 周末 | 学 Ch 13（最重）+ 全 deck mock review |

第二周开始稳定 review，**目标在 mock interview 前 hit 全 deck 至少 3 轮**。

## Anti-pattern（不要这样写卡）

- ❌ "Q-learning 的更新公式是什么？" → 答案太长，写不完
- ✅ "Q-learning update: Q(s,a) ← Q(s,a) + α[R + γ {{c1::max_a' Q(s', a')}} - Q(s,a)]" → 遮一个 term
- ❌ "TD 是什么" → 太宽
- ✅ "TD 跟 MC 比，bias 更 {{c1::高}}，variance 更 {{c1::低}}"
- ❌ 遮整公式
- ✅ 遮关键 term / 数字 / 符号

## 链回

- [[_overview]] — RL 学习总览
- [[_chapter-status]] — 各章 v1/v2/v3 状态
- [[_reading-guide]] — 8 周阅读计划
