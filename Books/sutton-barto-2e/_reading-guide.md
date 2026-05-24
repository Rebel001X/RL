---
title: Sutton & Barto 2e — 8 周深读指南
created: 2026-05-24
tags: [area/rl, type/reading-guide, status/v1, book/sutton-barto-2e]
---

# Sutton & Barto 2e — 8 周深读指南

社区共识整合（Stanford CS234 / Silver UCL 2015 / planetbanatt / strikingloo / lcalem / Reddit r/reinforcementlearning / 知乎）。

---

## 总览

- **总投入**：8 周 × 1.5 h/day ≈ **84 小时**
- **目标读者**：DL 基础 + 想做 RLHF/LLM 后训练
- **顺序**：**线性 + 一处重排**（Ch 13 提前到 Ch 9-10 后；按 Silver UCL 顺序）

---

## 📅 8 周日程

| 周 | 章节 | 难度 | 重点 | Silver Lec |
|---|---|---|---|---|
| **W1** | Ch 2 (skim) + **Ch 3 (deep)** | easy + hard | MDP 形式化、Bellman 方程 | Lec 2 |
| **W2** | Ch 4 + Ch 5 | medium | DP / MC、importance sampling §5.5 ⚠️ | Lec 3, 4 |
| **W3** | **Ch 6 (deep)** + Ch 7 (skim) | hard | TD / SARSA / Q-learning ⭐ | Lec 4, 5 |
| **W4** | Ch 9 + Ch 10 | medium | 函数近似、semi-gradient | Lec 6 |
| **W5** | **Ch 11**（最难） + Ch 8 (skim) | **hardest** | **deadly triad**（off-policy + FA + bootstrap） | Lec 6 |
| **W6** | Ch 12 + Schulman GAE 论文 | medium | eligibility traces → GAE | — |
| **W7** | **Ch 13 (deep)** | **hard** | PG 定理（**RLHF 基石**）⭐⭐⭐ | Lec 7 |
| **W8** | Ch 14-17 + TRPO/PPO/InstructGPT/DPO | easy | 收口 + 桥到 RLHF Book | Lec 8-10 |

**可选 W9-10**：自己跑 PPO-on-GPT2 toy RLHF demo（TRL / nano-rlhf）。

---

## ⏱️ 每天 1.5h session 节奏（60/30/0 简化版）

| 阶段 | 时长 | 做什么 |
|---|---|---|
| 阅读 | 60 min | 手边带笔，每个公式自己推一遍 |
| 笔记 | 20 min | 章末 summary box → cloze flashcards（Anki / Mochi） |
| 代码 | 10 min | 跑/diff 一段 ShangtongZhang 对应 chapter 代码 |

**周末 2h 整合**：本周章节总结写入 `RL/Books/sutton-barto-2e/chXX-*.md`（参考 Ch 2/3 样板格式）。

---

## 🎯 难点章节预警

| 章 | 为什么难 | 应对 |
|---|---|---|
| **Ch 11** | deadly triad 是 RL 著名难题 | **§11.3 重读 3 遍**；planetbanatt 段落必看 |
| **Ch 13** | PG 定理推导（含 score function trick） | lcalem [博客](https://lcalem.github.io/blog/2019/03/21/sutton-chap13)；Silver Lec 7 反复看 |
| **Ch 5 §5.5** | importance sampling（PPO ratio 的根） | 推一遍自己证明 unbiasedness |
| **Ch 9 §9.4** | projection operator（Ch 11 前置） | 不懂 §9.4 就读不懂 §11.3 |

---

## 📚 RLHF 最短路径（如果只为面试）

**Ch 3 + 5 + 6 + 9 + 13 + §12 部分** ≈ **30 小时**够看 PPO/GRPO。
- Ch 3：MDP 形式化
- Ch 5 §5.5：importance sampling（PPO ratio）
- Ch 6：Q-learning（理解 value-based 才能对照 policy-based）
- Ch 9：函数近似（理解为什么需要 NN）
- §12.1-12.3：λ-return（GAE 前置）
- **Ch 13：Policy Gradient 全章**

之后直接跳：Schulman GAE → TRPO → PPO → Christiano 2017 → InstructGPT → DPO。

---

## 🧪 自测三层

| 层 | 方法 |
|---|---|
| **L1 概念** | 闭眼 1 分钟讲清章 thesis（Feynman） |
| **L2 公式** | 闭眼推导章末 summary box 里的公式 |
| **L3 实现** | 不看 Zhang 仓库，从零写一个核心算法 |

**面试用**：能在白板上画出 **"RL 算法家族树"**（model-based ↔ model-free / on-policy ↔ off-policy / value ↔ policy ↔ actor-critic）+ 把 Ch 4-13 每个算法摆位置。

---

## 🔗 资源链接

| 资源 | 用途 |
|---|---|
| [Silver UCL 2015 playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) | **章对章配套视频**，必看 |
| [planetbanatt 学习心得](https://planetbanatt.net/articles/sutton.html) | cloze + 后续路径（HN/Reddit 最高赞） |
| [strikingloo 章节读法](https://strikingloo.github.io/wiki/reinforcement-learning-sutton) | "deep/careful/skim" 表，r/reinforcementlearning 钉子帖 |
| [lcalem Ch 13 神文](https://lcalem.github.io/blog/2019/03/21/sutton-chap13) | PG 定理最佳解读 |
| [ShangtongZhang 代码](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction) | **14.5k stars，章对章 figure 复现** |
| [Scott Jeen 习题 PDF](https://enjeeneer.io/sutton_and_barto/rl_exercises.pdf) | 习题答案参考 |
| [Stanford CS234](https://web.stanford.edu/class/cs234/) | Emma Brunskill RL 课，看 lecture slides |
| [CS285 (Sergey Levine)](https://rail.eecs.berkeley.edu/deeprlcourse/) | **学完 S&B 之后**的 deep RL 进阶 |
| [HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course) | 并行实操，含 RLHF bonus 单元 |
| [知乎读法汇总 p/663058215](https://zhuanlan.zhihu.com/p/663058215) | 中文社区综合 |

---

## ⚠️ 常见踩坑

1. **跳过 Ch 11 deadly triad**：以为后面用不上 → Ch 13 函数近似 + off-policy PG 立刻撞上
2. **不做 Ch 13 PG 定理推导**：背公式不够，面试必问推导
3. **跳 Ch 5 §5.5 importance sampling**：直接看 PPO 论文看不懂 ratio 是什么
4. **只看视频不看书**：Silver 课讲得快，细节略；书的公式严格度高
5. **不写 cloze 卡片**：1 个月后忘 80%

---

## 🗺️ 跟 brain vault 的关系

- **brain vault** `Areas/rl-books/sutton-barto-2e/chXX-*.md` = first-pass 概览（每章 1 页）
- **RL vault** `Books/sutton-barto-2e/chXX-*.md` = 深读（每章 8-15 KB，含代码 + 习题 + RLHF 桥）

两者每章互相 `[[link]]`。Obsidian 里能跳。

## Related

- [[_overview]]
- [[_chapter-status]]
