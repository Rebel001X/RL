---
title: "Sutton & Barto Ch 17. Frontiers"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 17
difficulty: easy
budget_days: 1
priority_for_rlhf: nice
silver_lecture: 10
---

# Ch 17. Frontiers

## 0. 阅读元信息

- **难度**：easy（综述性，无算法）
- **预算**：1 天 / 含跟 2026 现状对比
- **必读理由**：理解 2018 年 RL 看到的前沿（很多今天已成主流）；对比 2026 现状能看出 RL 的实际演进路径
- **配套**：Silver UCL Lec 10
- **跟 RLHF 的桥**：本章末段提到 "RL + supervised learning for AI assistants"——RLHF 的远见

## 1. 一句话

2018 年 Sutton & Barto 看到的 RL 前沿：general value functions, options/temporal abstraction, intrinsic motivation, model-based RL with learned model, hierarchical RL, multi-agent, transfer learning, lifelong learning, safe RL, interpretability——**8 年后看 (2026)，这些 frontier 多数已 mainstream 或被 LLM 颠覆**。

## 2. 关键概念地图

| 2018 frontier | 2026 现状 |
|---|---|
| **General value functions (GVFs)** | 仍研究中，未 mainstream |
| **Options / Temporal Abstraction** | 部分进 HRL；LLM agent 的 tool use 类似 |
| **Intrinsic motivation (curiosity)** | 仍是 sparse-reward 研究热点 |
| **Model-based with learned model** | **大成功**：MuZero, Dreamer 系列, world models |
| **Hierarchical RL** | 部分突破；LLM agent 多层规划是变体 |
| **Multi-agent RL** | 工业落地（金融、调度）+ 学术活跃 |
| **Transfer / Meta-learning** | 部分被 LLM 大模型预训练吸收 |
| **Lifelong / Continual learning** | catastrophic forgetting 仍未解 |
| **Safe RL** | constrained RL 持续发展；RLHF 部分回答 |
| **Interpretability** | 仍 RL 痛点 |
| **AGI / AI safety** | **RLHF + alignment 是当前主流 path** |

## 3. 关键公式与算法

无新公式。本章讨论方向，不给算法。

## 4. 8 年回顾：哪些 frontier 已被攻克 / 改写

**已主流**：
- Deep RL（end-to-end NN + RL）
- Model-based with NN model（MuZero / Dreamer）
- Self-play（AlphaZero 范式）
- RLHF on LLMs（书写于 2018，2022 才爆发）

**仍前沿**：
- Sample efficiency（人类几个 trial 学会 task）
- Lifelong learning（不忘记旧 task）
- Interpretability of learned policies
- Provably safe RL

**被 LLM 颠覆**：
- Transfer learning：大模型预训练几乎覆盖
- Common sense reasoning：LLM-based agent 主流
- Few-shot learning：in-context learning

## 5. 习题精选

无算法习题。**思考题**：

**Q：2018 年看不到 RLHF 大爆发——这是为什么？**
- 提示：LLM scale 不够（GPT-3 2020）；人类标注成本 + RM 训练流水线 2020 才打通

**Q：MuZero 实现了 2018 frontier 哪一项？**
- 提示：model-based with learned model——把 dynamics 也学进 NN

## 6. 代码伴侣

无对应代码。推荐：
- Survey papers on intrinsic motivation, HRL, model-based RL
- DeepMind world models 系列：Dreamer-V1/V2/V3
- OpenAI Spinning Up（RL 教学集合）

## 7. 反直觉点 / 易错点

- **反直觉**：2018 Sutton 没预测到 RLHF 会成为 2022 后的主流——技术 timeline 不是直线
- **反直觉**：很多 frontier 的"突破"来自 scale + compute，不是新算法（DQN→AlphaGo→AlphaZero 多是工程优化）
- **反直觉**：safe RL 在 2018 是 niche，2024-2026 因 alignment 焦虑成 mainstream

## 8. 跟 RLHF 的桥

- 本章 §17.6 "Application to General AI" 提到 "tutorial systems"——RLHF for AI assistants 的雏形
- safe RL 思想 → constitutional AI / red-teaming
- intrinsic motivation → reasoning model 的 self-reflection 奖励
- transfer learning → 大模型 pre-training 的 RL fine-tuning

**本章是 RL 的"未来视角"——读完后你能感受到 RL 不只是 PPO/DQN，而是更广的研究方向**。

## 9. 苏格拉底 Q&A

- **Q：2018 frontier 哪些"完全意外"地被 LLM 解决？**
  - 提示：transfer / common sense / few-shot——通过大模型预训练而非 RL 算法创新
- **Q：哪些 frontier 仍是开放问题？**
  - 提示：sample efficiency / lifelong learning / interpretability / provably safe RL
- **Q：作为 2026 读者，应该把本章当历史看还是当 roadmap？**
  - 提示：既是历史（2018 视角）也是 roadmap（多数仍 open）；建议 + 读 2024-2026 综述对比

## 10. 自测清单（闭卷）

- [ ] 列出 5 个 2018 frontier
- [ ] 指出哪些已主流，哪些仍开放
- [ ] 解释 LLM 如何颠覆部分 frontier
- [ ] 答 Q1-Q3 至少 2 个

## Related

- **上一章**：[[ch16-applications-case-studies]]
- **第一遍 brain**：[[../../../brain/Areas/rl-books/sutton-barto-2e/_overview]]
- **进度**：[[_chapter-status]] W8（收口）

## Sources

- 原书 Ch 17 pp. 451-468
- [Silver UCL Lec 10](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- 后续阅读：2024-2026 Deep RL surveys
- OpenAI Spinning Up
- DeepMind world models 系列论文

## 完结里程碑

读完 Ch 17 你应该完成：
- [ ] 17 章全部章节笔记完成（看 [[_chapter-status]]）
- [ ] 闭眼推 Bellman 方程 + PG 定理
- [ ] 跑通 ShangtongZhang 至少 5 个 demo
- [ ] 自己实现 Q-learning + REINFORCE + PPO
- [ ] 读完 TRPO/PPO/InstructGPT/DPO 论文
- [ ] **画"RL 算法家族树"**——面试用

**下一步路径**：
1. CS285（Sergey Levine）—— 接 deep RL 进阶
2. RLHF Book 完整读完
3. HuggingFace Deep RL Course 实操
4. 自跑 toy RLHF on GPT-2 small（TRL）

## 附：8 年看 RL 的"实际演进"vs"预期演进"

**Sutton 2018 预期**：
- 通用 AI 需要 RL 跟其它 ML 协同（特别 supervised）
- model-based 应该崛起
- safe RL 会重要

**2026 实际**：
- LLM + RLHF 走在前面——RL 跟 SL 协同实现了，但形式是后训练而非传统 model-based RL
- world models（Dreamer 系列）兑现 model-based 预言
- safe RL 通过 alignment 部分实现（RLHF, constitutional AI）
- **未预测到**：reasoning model (o1/R1) 让 RLVR 兴起；inference-time scaling 成新轴

**启示**：技术演进非直线，**应用场景驱动**比"算法演进"更决定方向。Sutton 看不到 ChatGPT 会驱动 RL 复兴；今天的我们也未必能预测下 5 年。

## 附：面试用 RL 算法家族树（建议手画）

```
RL 算法
├─ Tabular (Ch 4-7)
│  ├─ DP: VI / PI
│  ├─ MC: First-visit / Every-visit
│  └─ TD: SARSA / Q-learning / Expected SARSA
│
├─ Function Approximation (Ch 9-13)
│  ├─ Value-based
│  │  ├─ Semi-grad TD / SARSA
│  │  ├─ DQN / Double DQN / Rainbow
│  │  └─ Distributional: C51 / QR-DQN / IQN
│  ├─ Policy-based
│  │  ├─ REINFORCE
│  │  └─ TRPO / PPO / GRPO
│  └─ Actor-Critic
│     ├─ A2C / A3C
│     ├─ SAC (max entropy)
│     └─ DDPG / TD3 (continuous)
│
├─ Model-based
│  ├─ Dyna / Prioritized sweeping
│  ├─ MuZero / Dreamer
│  └─ AlphaGo / AlphaZero (search-based)
│
└─ Beyond pure RL
   ├─ RLHF / DPO / RLAIF (LLM alignment)
   ├─ RLVR (verifiable rewards)
   ├─ Multi-agent (MARL)
   └─ Inverse RL / Imitation learning
```

能在白板画出这棵树 + 每个 node 摆位 = 面试主动权。