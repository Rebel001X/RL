---
title: "Sutton & Barto Ch 16. Applications and Case Studies"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 16
difficulty: easy
budget_days: 2
priority_for_rlhf: nice
silver_lecture: 10
---

# Ch 16. Applications and Case Studies

## 0. 阅读元信息

- **难度**：easy（叙述性，无新算法）
- **预算**：2 天 / 必读 AlphaGo 段
- **必读理由**：看 RL 真实落地——TD-Gammon, Samuel checkers, AlphaGo, Watson, Atari DQN 等里程碑；**AlphaGo 段是面试常聊**
- **配套**：Silver UCL Lec 10
- **跟 RLHF 的桥**：弱直接，但 AlphaGo/MuZero 的 MCTS + value/policy 思路启发了 reasoning model

## 1. 一句话

本章是 RL 应用集锦——从 1990s TD-Gammon 到 2016 AlphaGo，覆盖游戏、控制、推荐、调度等领域；**每个 case 都展示 RL 框架的某个独特强项**（self-play / function approximation / model-based / multi-agent 等）。

## 2. 关键概念地图

| 案例 | 年份 | 核心创新 |
|---|---|---|
| **TD-Gammon**（Tesauro）| 1992 | TD(λ) + NN + self-play → 超人类水平 backgammon |
| **Samuel's Checkers** | 1959 | 早期 RL 雏形——learning + self-play |
| **Watson on Jeopardy!** | 2011 | RL for wagering（决定 bet 多少） |
| **Atari DQN**（DeepMind）| 2013 | end-to-end raw pixel → action，DQN 经典 |
| **AlphaGo / AlphaZero**（DeepMind）| 2016/2017 | MCTS + value/policy net + self-play → 围棋世界冠军 |
| **Personalized Web Services** | — | bandit + RL for recommendation |
| **Memory Control（CPU cache）** | — | RL for cache replacement policy |

## 3. 关键公式与算法

本章无新公式，但介绍各 case 用的 RL 配方：

### TD-Gammon 配方
- TD(λ) on hand-crafted features
- Self-play 数百万局
- λ ≈ 0.7

### AlphaGo 配方
- **Policy net P(a|s)**：监督 + RL 微调
- **Value net V(s)**：评估 board position
- **MCTS**：search 时用 P 引导 + V 评估 leaf
- **Self-play**：通过 RL 不断强化

### AlphaZero（简化）
- 一个 NN 同时输出 P 和 V
- 完全自学（无人类数据）
- 通用框架——围棋 / 象棋 / 将棋 / Shogi

### Atari DQN 配方（突破性元素）
- Convolutional NN on raw pixels
- Experience replay
- Target network
- Reward clipping
- 一个 architecture 跨 49 games

## 4. 关键算法（伪代码）

**AlphaGo MCTS** 单 simulation：
```
def simulate(state):
    if state in tree:
        # Selection: PUCT formula
        a = argmax_a [Q(s, a) + c * P(s, a) * sqrt(N(s)) / (1 + N(s, a))]
        next_state = step(state, a)
        v = simulate(next_state)
    else:
        # Expansion + Evaluation
        tree.add(state)
        P, v = network(state)        # policy + value
    # Backprop
    update Q, N along path
    return v
```

## 5. 习题精选

无算法习题。**思考题**：

**Q：TD-Gammon 为什么 backgammon 上超人类但跳棋不行？**
- 提示：backgammon 自带 stochasticity（骰子），exploration 自然；deterministic 游戏（chess）需要更复杂的 search

**Q：AlphaGo 比 DeepBlue 强在哪？**
- 提示：DeepBlue 纯 search + 手工 eval function；AlphaGo learned eval function（NN）+ MCTS——general framework

## 6. 代码伴侣

- ShangtongZhang 无对应 case study 代码（叙述章）
- 推荐阅读：
  - Tesauro 1995 "Temporal Difference Learning and TD-Gammon"
  - Mnih et al. 2013/2015 "Playing Atari with Deep Reinforcement Learning" / Nature paper
  - Silver et al. 2016 Nature AlphaGo
  - Silver et al. 2017 Nature AlphaGo Zero
  - Schrittwieser et al. 2020 Nature MuZero

## 7. 反直觉点 / 易错点

- **反直觉**：TD-Gammon 1992 用 NN 比 backgammon expert 还强——证明 RL + NN 早就能 work，只是 compute 不够
- **反直觉**：AlphaZero 从零自学超过 AlphaGo（用人类棋谱）——人类数据反成桎梏
- **反直觉**：MuZero 把 model 也学出来（latent dynamics）——model-based 重生
- **反直觉**：DQN 单一架构跨 49 个 Atari games——end-to-end + 大量 compute 的胜利

## 8. 跟 RLHF 的桥

- **AlphaGo MCTS 范式** ↔ **reasoning model search**：o1/R1 推理时的"思考"虽不是显式 MCTS，但精神类似
- **AlphaZero self-play** ↔ **iterative RLHF**：模型自己产生数据再学
- **DQN end-to-end pixel → action** ↔ **LLM end-to-end token → token**
- **MuZero learned model** ↔ **world model in agent**（如 Genie、Dreamer 系列）

## 9. 苏格拉底 Q&A

- **Q：AlphaGo Zero 比 AlphaGo 强多少？**
  - 提示：100-0 对战胜率（完全无人类数据 + 更深 NN + 单 net）
- **Q：为什么 DQN 在 49 games 表现差异巨大？**
  - 提示：reward density / state observability / horizon 差异大——DQN 不是 universal solver
- **Q：RLHF 跟 AlphaGo 是同一种"self-play"吗？**
  - 提示：RLHF 是 model + RM + RL 闭环；AlphaGo 是 model 互对战——结构不同但都"自学"
- **Q：MCTS 在 LLM agent 上的应用前景？**
  - 提示：Tree of Thoughts、ReAct 等是浅 MCTS 的 prompt 实现；深度集成尚在研究

## 10. 自测清单（闭卷）

- [ ] 列出 5 个本章 case 及其核心创新
- [ ] 解释 TD-Gammon 怎么 work
- [ ] 解释 AlphaGo 三件套（P, V, MCTS）
- [ ] 比较 AlphaGo / AlphaZero / MuZero 的演进
- [ ] 答 Q1-Q4 至少 2 个

## Related

- **上一章**：[[ch15-neuroscience]]
- **下一章**：[[ch17-frontiers]]
- **MCTS 理论**：[[ch08-planning-and-learning]]
- **进度**：[[_chapter-status]] W8

## Sources

- 原书 Ch 16 pp. 421-448
- TD-Gammon paper（Tesauro 1995）
- DQN papers（Mnih 2013/2015）
- AlphaGo / AlphaZero / MuZero papers
- [Silver UCL Lec 10](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

## 附：AlphaGo 详细 walkthrough（面试常考）

1. **Supervised learning of policy net**：从 16M 人类业余高手棋谱训 SL policy P_σ（57% 预测下一步）
2. **Self-play RL of policy net**：P_σ → P_ρ 通过 self-play + REINFORCE 微调，胜率提升
3. **Value net training**：从 self-play games 学 V_θ，预测 game outcome
4. **MCTS at decision time**：用 PUCT formula，每个 leaf 用 V_θ + rollout 估值

**关键创新（2016 当时）**：
- NN 替代手工 eval function（DeepBlue 风格）
- 两个 NN（P 和 V）协同
- self-play 闭环

**AlphaZero（2017）简化**：
- 单一 NN 同时输出 P 和 V
- 完全无人类数据（从随机 self-play 起）
- 通用化（围棋 / 象棋 / 将棋）

**面试问法**：
- "AlphaGo 用 RL 还是 SL？" → 两者都用，SL 初始化 + RL self-play 强化
- "MCTS 为什么不直接用 random rollout？" → 太慢；用 P 引导 + V 评估 leaf 才高效
- "AlphaZero 比 AlphaGo 强在哪？" → 100-0；更深 NN + 单 net + 无人类数据
