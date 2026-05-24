---
title: "Sutton & Barto Ch 3. Finite MDPs"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 3
difficulty: hard
budget_days: 3-4
priority_for_rlhf: must
silver_lecture: 2
---

# Ch 3. Finite MDPs ⭐⭐⭐

## 0. 阅读元信息

- **难度**：hard（**最 load-bearing 的一章**——后续所有内容都假设你会本章）
- **预算**：3-4 天 / ~6-8 小时
- **必读理由**：MDP 是 RL 的"形式化语言"，Bellman 方程是 RL 的"基本物理定律"；不懂这章看 PPO/Q-learning/DPO 都是天书
- **配套**：Silver UCL Lec 2 + Lec 3（Planning by DP）
- **跟 RLHF 的桥**：LLM 生成一段 response = 一个 trajectory；KL penalty = 修改 reward 的"shaped MDP"；RLHF 的 RL 部分就是用 MDP 形式化 LLM 后训练

## 1. 一句话

MDP 是 RL 的**形式化框架**：用五元组 (S, A, P, R, γ) 描述 agent-environment 交互，**Bellman 方程**把"长期价值"递归地拆成"当前 reward + 折扣未来价值"——所有 RL 算法都是这个方程的不同近似解法。

## 2. 关键概念地图

| 概念 | 一句话 |
|---|---|
| **MDP 五元组** | (S, A, P, R, γ)：状态集 / 动作集 / 转移概率 / reward 函数 / 折扣因子 |
| **Markov property** | P(s', r \| s, a) 只依赖最近 (s, a)——历史可压缩成 state |
| **Trajectory** | (S_0, A_0, R_1, S_1, A_1, R_2, ...) |
| **Return G_t** | $G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + \ldots = \sum_{k=0}^∞ γ^k R_{t+k+1}$ |
| **Policy π(a\|s)** | 在状态 s 下选动作 a 的概率分布 |
| **State value v_π(s)** | $E_π[G_t \| S_t = s]$——在 π 下从 s 出发的期望累积 reward |
| **Action value q_π(s, a)** | $E_π[G_t \| S_t = s, A_t = a]$ |
| **Bellman expectation** | v_π 和 q_π 各自满足的递归方程（在 π 下成立） |
| **Optimal value v\*, q\*** | 所有 π 中最好的 value |
| **Bellman optimality** | v\* 和 q\* 各自满足的递归方程（含 max） |
| **Optimal policy π\*** | 至少有一个 π\*，可从 q\* 直接读出：π\*(s) = argmax_a q\*(s, a) |

## 3. 必背公式与推导

### Return 折扣

$$G_t \doteq R_{t+1} + \gamma G_{t+1}$$

**推导**：直接从定义展开。这个递归是 Bellman 方程的"种子"。

### Bellman expectation equation for v_π

$$v_\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

**白板推导步骤**：
1. 从定义出发：$v_\pi(s) = E_\pi[G_t | S_t = s]$
2. 展开 $G_t = R_{t+1} + \gamma G_{t+1}$
3. 期望按 a, s', r 分解：$\sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma E_\pi[G_{t+1} | S_{t+1} = s']]$
4. 内层期望 = $v_\pi(s')$ —— 递归出现

**易错**：$p(s', r | s, a)$ 是 **joint** 转移概率，不是条件相乘。

### v ↔ q 的 4 个 conversion（必背）

$$v_\pi(s) = \sum_a \pi(a|s) q_\pi(s, a)$$
$$q_\pi(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

**两个的组合可以推出 Bellman**：把 v_π 代进 q_π 或反之。

### Bellman optimality equation for v\*

$$v_*(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_*(s')]$$

注意把 `Σ_a π(a|s)` 换成 `max_a`——因为最优 policy 是 deterministic 选最佳 action。

### Bellman optimality for q\*

$$q_*(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} q_*(s', a')]$$

**这是 Q-learning 的目标**：Q-learning 就是 sample-based 求解这个方程。

## 4. 关键算法（伪代码）

本章主要是定义，没有可运行算法。但 Bellman optimality equation 直接引出 Ch 4 的 Value Iteration：

```
init V(s) = 0 for all s
repeat:
    Δ = 0
    for each s:
        v = V(s)
        V(s) = max_a Σ_{s', r} p(s', r | s, a) [r + γ V(s')]
        Δ = max(Δ, |v - V(s)|)
until Δ < θ
```

## 5. 习题精选（cherry-picked）

cherry-picked from ~30 exercises。**这些必做**：

**3.5（episode 与折扣）**：如果环境是 episodic，γ=1 时 return 是否仍有限？**关键答案**：episodic 任务 reward 总数有限，所以 G_t 必然有限；continuing 任务必须 γ<1 才保证 G_t 有限。

**3.13（v 和 q 转换）**：用 q_π 表达 v_π。**答案**：$v_\pi(s) = \sum_a \pi(a|s) q_\pi(s, a)$。

**3.17（Bellman for q_π）**：写 q_π 的 Bellman 方程。**答案**：
$$q_\pi(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \sum_{a'} \pi(a'|s') q_\pi(s', a')]$$

**3.19（policy improvement 直觉）**：贪心改进的 policy 一定不更差——这是 Ch 4 policy iteration 的核心，先在 Ch 3 体会。

**3.22（v 和 q 都最优时 π 怎么读出）**：**π\*(s) = argmax_a q\*(s, a)**。重点：**q\* 比 v\* 更方便决策**——v\* 决策需要知道 p（model-based），q\* 不需要（model-free 友好）。

**3.24（gridworld 中 v\* 数值）**：手算或检查 figure 3.5 数字，验证理解。

**3.29（state-only Bellman）**：把 p(s', r | s, a) 拆成 p(s' | s, a) 和 r(s, a, s') 两个分量；推导以 r(s, a) 为基础的 Bellman 方程。证明跟原始等价但更简洁——很多论文用这个版本。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter03/grid_world.py` —— 复现 figure 3.2, 3.5（gridworld value iteration）
- **关键观察**：figure 3.5 的 v\*(s) 数值能自己跑出来——验证你理解 Bellman optimality
- **建议**：自己写一个 5x5 gridworld 的 value iteration，10 行 numpy 就够

## 7. 反直觉点 / 易错点

- **反直觉**：折扣 γ 不只是"数值稳定 trick"——它**编码了对未来不确定性的偏好**。γ → 0 = 短视，γ → 1 = 长远；不同任务该用不同 γ
- **反直觉**：Markov property 不是"所有信息都在 state 里"——是"**给定 state 后，未来跟过去独立**"。这是个数学性质，不是物理性质
- **反直觉**：可能有**多个 optimal policy**（共享同一个 v\*），但 v\* 唯一
- **易错**：$p(s', r | s, a)$ 是 4 维 distribution（next state + reward），不是 next state distribution 跟 reward 函数分开
- **易错**：episodic 任务里 T 是随机变量（不固定）；公式里把 T 当随机处理才严格
- **易错**：v_π 和 q_π 都是 **on-policy**（依赖 π），换 π 后值变；v\* 和 q\* 不依赖具体 π

## 8. 跟 RLHF 的桥

| MDP 概念 | LLM 后训练对应 |
|---|---|
| State s | (prompt + 已生成 tokens) 的完整文本 |
| Action a | 下一个 token（vocab 大小的离散 action） |
| Trajectory | 一个完整 response 的 token 序列 |
| Reward R | **稀疏 episodic**——只在 response 末尾给一次 RM 分；其余 R=0 |
| γ | 通常 = 1（chat 场景）或 0.99 |
| Policy π(a\|s) | LLM 的 next-token distribution |
| v_π / q_π | RLHF 一般不学（PPO 用 critic 学 v；GRPO 干脆不学） |
| KL penalty | shaped reward：r' = r_RM - β KL(π \| π_ref) |
| Bellman | 隐式存在——advantage 估计（GAE）就是基于 TD error 的 Bellman 一致性 |

**关键认知**：RLHF 的 MDP 是个"退化版"——只有最后一步有非零 reward，state 是文本 prefix。但理论框架完全继承自本章。

## 9. 苏格拉底 Q&A

- **Q：为什么 RL 要 Markov 假设？现实世界不 Markov 怎么办？**
  - 提示：Markov 让算法有限步可计算；非 Markov 用 RNN / frame stacking / history features 近似成 Markov
- **Q：v_π 和 q_π 哪个更"有用"？**
  - 提示：q_π 更有用——决策时不需要 model（直接 argmax_a q），这是 model-free RL 的根
- **Q：为什么 RL 不直接搜索（minimax/MCTS）？**
  - 提示：state 空间太大（围棋 10^170，文本 vocab^seq_len）；必须用 v/q 函数近似 + 泛化
- **Q：γ=1 的 continuing 任务为什么 G_t 可能发散？**
  - 提示：无限步 reward 求和；除非 reward 自然衰减或有 absorbing state，否则数学不严格
- **Q：Bellman expectation vs optimality 的区别是什么？**
  - 提示：expectation 在固定 π 下成立（用 Σ_a π(a\|s)）；optimality 涉及最优 π（用 max_a）—— 后者是非线性方程
- **Q：optimal policy 一定 deterministic 吗？**
  - 提示：finite MDP 至少有一个 deterministic 最优；但可能多个最优 policy 同时存在（含 stochastic 的）
- **Q：如果两个 state 的 v_π 相同，能否合并？**
  - 提示：value 同不代表 transition 同——不能合并；但可作"状态抽象"的启发

## 10. 自测清单（闭卷）

- [ ] 不看书 1 分钟讲清 MDP 五元组给非技术朋友
- [ ] **闭眼推导 v_π 的 Bellman 方程**（白板）
- [ ] **闭眼写出 v↔q 的 4 个 conversion 等式**
- [ ] 写出 v\* 和 q\* 的 Bellman optimality 方程，标出跟 expectation 版的关键差异（max_a vs Σ_a π）
- [ ] 解释 q\* 比 v\* 在决策上"更方便"——为什么
- [ ] 答 Q1-Q7 中至少 4 个
- [ ] 完成习题 3.5 / 3.13 / 3.17 / 3.22 / 3.29 至少 3 题
- [ ] 跑通 ShangtongZhang `chapter03/grid_world.py`，理解 figure 3.5 数字
- [ ] **能解释 RLHF 是个怎样的"退化 MDP"**

## Related

- **first-pass**：（无）
- **上一章**：[[ch02-multi-armed-bandits]]
- **下一章**：[[ch04-dynamic-programming]]
- **进度**：[[_chapter-status]] W1
- **RLHF 应用**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning]]
- **brain Slipbox 关联**：[[../../../brain/Slipbox/kv-cache-memory-formula]]（LLM rollout 是 MDP trajectory）

## Sources

- 原书 Ch 3 pp. 47-72
- [Silver UCL Lec 2: MDPs](https://www.youtube.com/watch?v=lfHX2hHRMVQ)
- [Silver UCL Lec 3: Planning by DP](https://www.youtube.com/watch?v=Nd1-UUMVfz4)
- ShangtongZhang `chapter03/grid_world.py`
- [strikingloo § Ch 3](https://strikingloo.github.io/wiki/reinforcement-learning-sutton)
- [planetbanatt § Ch 3](https://planetbanatt.net/articles/sutton.html)
