---
title: Sutton & Barto 2e — RL vault 深读总览
created: 2026-05-24
tags: [area/rl, type/book-overview, status/v1, book/sutton-barto-2e]
authors: ["Richard S. Sutton", "Andrew G. Barto"]
edition: "2nd Edition (2018)"
pages: 548
---

# Sutton & Barto 2e — 深读总览

**入门方法**：先看 [[_reading-guide]] 制定节奏 → 按 [[_chapter-status]] 周勾选 → 单章深度笔记看 `chXX-*.md`。

## 📖 17 章详表

| Ch | 标题 | 页 | 难度 | 预算 | RLHF 优先级 | first-pass | 深读 |
|---|---|---|---|---|---|---|---|
| 1 | Introduction | 1-22 | easy | 1 天 | nice | [[../../../brain/Areas/rl-books/sutton-barto-2e/ch01-introduction]] | [[ch01-introduction]] |
| 2 | Multi-armed Bandits | 25-44 | easy | 2 天 | nice | — | [[ch02-multi-armed-bandits]] ⭐ |
| 3 | **Finite MDPs** | 47-72 | hard | 3-4 天 | **must** | — | [[ch03-finite-mdps]] ⭐⭐⭐ |
| 4 | Dynamic Programming | 73-92 | medium | 2-3 天 | nice | — | （待写）|
| 5 | Monte Carlo Methods | 91-115 | medium | 3 天 | **must** | — | （待写，注意 §5.5）|
| 6 | **Temporal-Difference** | 117-138 | hard | 3-4 天 | **must** | — | （待写）⭐⭐⭐ |
| 7 | n-step Bootstrapping | 141-156 | medium | 2 天 | nice | — | （待写）|
| 8 | Planning and Learning | 159-186 | hard | skim | skip | — | （待写）|
| 9 | On-policy Prediction w/ FA | 197-218 | medium | 3 天 | **must** | — | （待写）|
| 10 | On-policy Control w/ FA | 243-262 | medium | 2 天 | nice | — | （待写）|
| 11 | Off-policy Methods w/ FA | 257-283 | **hardest** | **4-6 天** | nice | — | （待写）⚠️ deadly triad |
| 12 | Eligibility Traces | 287-321 | medium | 3 天 | must（§12.1-12.3）| — | （待写）|
| 13 | **Policy Gradient** | 321-343 | hard | **5-7 天** | **must** | — | （待写）⭐⭐⭐ |
| 14 | Psychology | 357-378 | easy | skim | skip | — | （待写）|
| 15 | Neuroscience | 381-415 | easy | skim | skip | — | （待写）|
| 16 | Applications and Case Studies | 421-448 | easy | 2 天 | nice | — | （待写，看 AlphaGo）|
| 17 | Frontiers | 451-468 | easy | 1 天 | nice | — | （待写）|

⭐ = 当前已有深度笔记（本次会话只写 Ch 2 + Ch 3，其余等周推进）

## 🎯 三种用户路径

**完整学**（8 周）：按 [[_reading-guide]] 顺序读全 17 章
**RLHF 速通**（4 周）：Ch 3 → 5 → 6 → 9 → §12 → 13，跳其它
**面试突击**（2 周）：Ch 3 + Ch 13 + 配 Silver Lec 2 / 7

## 🔗 资源（详见 _reading-guide）

- Silver UCL 2015 视频
- ShangtongZhang 代码仓库
- planetbanatt / strikingloo / lcalem 学习笔记
- Stanford CS234 / Berkeley CS285

## 🌉 跟其它书的桥

- Ch 6 TD/Q-learning ↔ Lapan Ch 5-6（实战代码）
- Ch 13 Policy Gradient ↔ Lapan Ch 11-12 ↔ RLHF Book Ch 6
- Ch 5 §5.5 importance sampling ↔ PPO ratio ↔ RLHF Book Ch 6
- Ch 5/6 → Bellemare（distributional 扩展）
