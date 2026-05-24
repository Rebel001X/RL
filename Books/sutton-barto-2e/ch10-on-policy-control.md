---
title: "Sutton & Barto Ch 10. On-policy Control with Approximation"
created: 2026-05-24
tags: [area/rl, type/note, status/v3, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 10
difficulty: medium
budget_days: 2
priority_for_rlhf: nice
silver_lecture: 6
---

# Ch 10. On-policy Control with Approximation ⭐⭐

---

## 📋 本章讲了什么（TL;DR）

1. **Ch 9 prediction → control 升级**：v̂(s, w) → q̂(s, a, w) + ε-greedy = NN 前时代最实用 RL 控制框架
2. **Average reward formulation**：continuing task 不用折扣 γ，改用 differential value——为长 trajectory 任务提供数学框架
3. **Mountain Car + tile coding** 是经典试金石——能跑通这个算 RL 入门毕业

**RLHF 相关性**：弱直接，但 **multi-turn agent 长 chat 没自然终态** → average reward 概念有借鉴价值

## 🎯 Top 5 关键洞察

| # | 洞察 | 为什么关键 |
|---|---|---|
| 1 | **Semi-gradient SARSA** 直接把 Ch 6 SARSA 升级为 FA | 工程上几乎 plug-in 替换 |
| 2 | **Average reward** 替代 discounted reward | continuing task 无终态时 γ=1 数学不严格 |
| 3 | **§10.4 反对 discounted continuing** | "discounting 在 FA + continuing 下几乎无意义" |
| 4 | **Mountain Car 用 SARSA + tile coding** 比 DQN 早 20 年 | 简单方法在简单任务上够用 |
| 5 | **本章方法已"博物馆化"** | DQN 后被 deep value-based / actor-critic 替代 |

## 🧭 在 Obsidian graph 中的位置

- **上游**：[[ch09-on-policy-prediction]]（prediction）+ [[ch06-temporal-difference-learning]]（tabular SARSA）
- **下游**：[[ch11-off-policy-methods]]（off-policy + FA = deadly triad）· [[ch13-policy-gradient]]（policy-based 替代）
- **现代**：DQN（[[../../../brain/Areas/rl-books/deep-rl-hands-on-lapan-3e/_overview|Lapan Ch 6]]）几乎完全取代本章方法

---

## 0. 阅读元信息

- **难度**：medium
- **预算**：2 天
- **必读理由**：average reward 是 multi-turn agent 数学借鉴；Mountain Car 是 RL 入门毕业试金石
- **配套**：Silver UCL Lec 6

## 1. 一句话 + 在 RL 体系中的位置

**Ch 9 prediction → control 升级**——v̂(s, w) → q̂(s, a, w)，加 ε-greedy 选 action = NN 前时代最实用 RL 控制框架。

```
RL Value-based + FA
├── On-policy Prediction (Ch 9)
├── On-policy Control (Ch 10) ⭐
│   ├── Semi-gradient SARSA
│   ├── Average reward formulation
│   └── Differential value
└── Off-policy Control (Ch 11) ⚠️ deadly triad
```

## 2. 概念地图（§3 详解目录）

| # | 概念 | §3.x | 重要度 |
|---|---|---|---|
| 1 | **Episodic Semi-gradient SARSA** | §3.1 | ⭐⭐⭐ 主算法 |
| 2 | **Semi-gradient n-step SARSA** | §3.2 | ⭐⭐ 推广 |
| 3 | **Mountain Car + Tile Coding 经典实战** | §3.3 | ⭐⭐⭐ 试金石 |
| 4 | **Average Reward Formulation** | §3.4 | ⭐⭐⭐ continuing task 数学 |
| 5 | **Differential Value Function** | §3.5 | ⭐⭐⭐ |
| 6 | **Differential Semi-gradient SARSA** | §3.6 | ⭐⭐ |
| 7 | **Discounting in Continuing Tasks（反对论）**| §3.7 | ⭐⭐ |

---

## 3. 核心概念详解（每个从 0-1 讲清）

### 3.1 Episodic Semi-gradient SARSA ⭐⭐⭐

#### 是什么
Ch 6 SARSA 的 FA 版——q̂(s, a, w) 替代 tabular Q，用 semi-gradient 更新。

#### 数学
$$w ← w + α [R + γ q̂(S', A', w) - q̂(S, A, w)] · ∇q̂(S, A, w)$$

注意：
- $A'$ 按当前 ε-greedy policy 采（on-policy）
- target $γ q̂(S', A', w)$ 不对 w 求梯度（semi-gradient）

#### 伪代码
```python
def episodic_semi_grad_SARSA(env, q_hat, w, alpha=0.1, gamma=0.99, eps=0.1, n_episodes=500):
    for ep in range(n_episodes):
        s = env.reset()
        a = epsilon_greedy([q_hat(s, a, w) for a in range(env.n_actions)], eps)
        done = False
        while not done:
            s_next, r, done = env.step(a)
            if done:
                w += alpha * (r - q_hat(s, a, w)) * q_hat.grad(s, a, w)
                break
            a_next = epsilon_greedy([q_hat(s_next, a, w) for a in range(env.n_actions)], eps)
            target = r + gamma * q_hat(s_next, a_next, w)
            w += alpha * (target - q_hat(s, a, w)) * q_hat.grad(s, a, w)
            s, a = s_next, a_next
    return w
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| α | 0.1 - 0.5 |
| γ | 0.99 |
| ε | 0.1 |
| Q(terminal) | 0 |

#### 跟 tabular SARSA 对比
| | Tabular SARSA | Episodic Semi-grad SARSA |
|---|---|---|
| Q 表示 | dict | parametric q̂(s, a, w) |
| 收敛 | ✅ | ✅（on-policy + linear FA）|
| 适用 | 小 state | 中等（tile coding）|

#### 高频面试问法
**Q：Episodic Semi-grad SARSA 跟 tabular SARSA 关键差异？**
**A**：把 Q(s, a) 从 dict 改成 parametric q̂(s, a, w)，update 用 semi-gradient（不对 target 取 w 梯度）。on-policy + linear FA 收敛保证仍成立。

---

### 3.2 Semi-gradient n-step SARSA

#### 是什么
n-step return + SARSA + FA。

#### 数学
$$G_{t:t+n} = R_{t+1} + ... + γ^{n-1} R_{t+n} + γ^n q̂(S_{t+n}, A_{t+n}, w)$$
$$w ← w + α [G_{t:t+n} - q̂(S_t, A_t, w)] · ∇q̂(S_t, A_t, w)$$

#### 跟 1-step 对比
- n=1 退化 §3.1
- n>1 信用分配跨步——学得快但 buffer 多步

---

### 3.3 Mountain Car + Tile Coding 经典实战 ⭐⭐⭐

#### 是什么
**RL 入门毕业试金石**——Mountain Car 任务用 SARSA + tile coding，是 1990s 经典 baseline，至今仍跑。

#### 任务
- State：2D（pos, vel）
- Action：3 个（left / no-op / right）
- Reward：每步 -1（鼓励快速到达终点）
- 终点：pos >= 0.5
- **难点**：sparse reward + 需要"前后摇摆"积累动能

#### 配方
1. **Tile coding** feature：8 个 tiling × 8×8 grid = 512 feature
2. **Linear q̂(s, a, w) = w^T x(s, a)**
3. **Episodic semi-gradient SARSA** 训练
4. ε-greedy 探索

#### 实战参数（Sutton Example 10.1）
| 超参 | 值 |
|---|---|
| n_tilings | 8 |
| tiles_per_dim | 8 |
| Total features | 8 × 8 × 8 × 3 actions = 1536 |
| α / n_tilings | 0.5 / 8 |
| γ | 1.0（episodic）|
| ε | 0 (no exploration) 或 0.1 |
| n_episodes | 500 |

#### 实证表现
- 200-step time limit 下能 reach goal（reward > -200）
- 500 episodes 内收敛

#### 跟 DQN 对比
| | SARSA + Tile Coding | DQN |
|---|---|---|
| 提出年代 | 1996 | 2013 |
| 在 Mountain Car 上学习速度 | **更快**（小任务）| 慢 |
| 在 Atari 上 | 不工作（state space 大）| SOTA |

#### 反直觉点
- **Mountain Car SARSA 比 DQN 快 5x**——小任务上简单方法更优
- **大任务上 DQN 完胜**——NN feature learning 是关键

#### 高频面试问法
**Q：Mountain Car 用 SARSA + Tile Coding 比 DQN 更快——为什么？**
**A**：(1) state space 小（2D），tile coding 手工 feature 已足；(2) DQN 要从零学 feature（CNN），需要更多 data；(3) **简单任务上手工 prior 胜过 end-to-end learning**。**Mountain Car 任务规模选错算法的经典例子**。

---

### 3.4 Average Reward Formulation ⭐⭐⭐

#### 是什么
**continuing task 不用 γ 折扣，改用 average reward**：
$$r̄(π) \doteq \lim_{h→∞} \frac{1}{h} \sum_{t=1}^h E[R_t | A_{0:t-1} \sim π]$$

—— 每步平均 reward 当 policy 性能指标。

#### 为什么需要它
- continuing task（无终态）γ=1 时 G_t 可能发散
- γ<1 时 expected return ≈ $\frac{r̄}{1-γ}$（一个常数）——γ 信号失效
- → 用 r̄ 直接当指标

#### 数学

**Differential return** G_t（替代 discounted return）：
$$G_t = R_{t+1} - r̄ + R_{t+2} - r̄ + R_{t+3} - r̄ + ...$$

**关键**：每步 reward 减去平均 reward——relative to baseline 的剩余 reward。

**Differential value**:
$$v_π(s) = E_π[G_t | S_t = s]$$
$$q_π(s, a) = E_π[G_t | S_t = s, A_t = a]$$

**Differential Bellman 方程**：
$$v_π(s) = \sum_a π(a|s) \sum_{s', r} p(s', r | s, a)[r - r̄(π) + v_π(s')]$$

**注意**：没有 γ，但有 r̄ 修正。

#### 跟 discounted formulation 对比

| | Discounted | Average Reward |
|---|---|---|
| Performance | $v_π(s_0) = E[\sum γ^t R]$ | $r̄(π) = \lim \frac{1}{h} \sum R$ |
| γ | <1 | 1（无折扣）|
| Continuing | 数学不严格 | **正确** |
| Episodic | 主流 | 罕用 |

#### 反直觉点
- **Average reward 在 OR 有 80 年传统**——RL 借鉴比 OR 晚 30 年
- **现代 deep RL 在 continuing task 仍用 discounted**——trick 不是理论支持

#### 高频面试问法
**Q：为什么 continuing task 不能用 γ<1？**
**A**：γ<1 时 expected discounted return $\frac{r̄}{1-γ}$ 是 r̄ 的常数倍——γ 信号被 r̄ 吃掉，不再区分 policy。**γ=1 数学发散，γ<1 信号失效**——continuing task 数学上必须用 average reward formulation。

---

### 3.5 Differential Value Function

#### 是什么
**relative-to-average 的 value**——比 discounted V 更适合 continuing task。

#### 数学
$$v_π(s) = E_π\left[\sum_{k=0}^∞ (R_{t+k+1} - r̄(π)) | S_t = s\right]$$

**直觉**：
- $R - r̄$ 是"高于平均水平的剩余 reward"
- v(s) = 从 s 出发的"剩余 reward 累加"
- 不需要 γ 因为 reward 已减 r̄ → 不会发散

#### 收敛性
- $E[R - r̄] = 0$ at steady state
- 所以 $\sum_{k=0}^∞ E[R_{t+k+1} - r̄]$ 是 0 + transient → 有限

#### 反直觉点
- **v_π(s) 可以为负**——s 是"低于平均"的 state
- **v_π 只到加常数等价**——加常数不改 policy 改进方向

#### 高频面试问法
**Q：Differential value 跟 discounted value 关系？**
**A**：differential v 是"相对平均的剩余 reward"，没有 γ。在 continuing task 上数学严格（discounted 时 γ=1 发散，γ<1 信号失效）。**v_π 只到 additive constant 等价——不影响 policy 优化**。

---

### 3.6 Differential Semi-gradient SARSA

#### 是什么
Average reward 版的 SARSA + FA。

#### 数学
$$δ_t = R_{t+1} - \bar{R}_t + q̂(S_{t+1}, A_{t+1}, w) - q̂(S_t, A_t, w)$$
$$w ← w + α δ_t · ∇q̂(S_t, A_t, w)$$
$$\bar{R}_{t+1} ← \bar{R}_t + β δ_t$$

—— 多一个 step size β 学 r̄。

#### 伪代码
```python
def differential_semi_grad_SARSA(env, q_hat, w, alpha=0.1, beta=0.01, eps=0.1, n_steps=int(1e5)):
    r_bar = 0.0
    s = env.reset()
    a = epsilon_greedy([q_hat(s, a_, w) for a_ in range(env.n_actions)], eps)
    for step in range(n_steps):
        s_next, r, done = env.step(a)
        a_next = epsilon_greedy([q_hat(s_next, a_, w) for a_ in range(env.n_actions)], eps)
        delta = r - r_bar + q_hat(s_next, a_next, w) - q_hat(s, a, w)
        w += alpha * delta * q_hat.grad(s, a, w)
        r_bar += beta * delta
        s, a = s_next, a_next
    return w, r_bar
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| α | 0.1 |
| **β** | **0.01**（r̄ 更新比 w 慢）|
| ε | 0.1 |

#### 反直觉点
- **β << α**——r̄ 是 "auxiliary"，跟得上 w 即可
- **两个 step size 让训练 robust**

---

### 3.7 Discounting in Continuing Tasks（§10.4 反对论）

#### 是什么
**Sutton §10.4 重要论点**：在 FA + continuing 下用 discounted return "几乎无意义"。

#### 数学论证

设 continuing task，γ<1：
$$v_π^{disc}(s) = E_π[\sum_{t=0}^∞ γ^t R_{t+1}]$$

**Steady state 下**：
$$E[R_t | π] → r̄(π) \text{ (constant)}$$

所以：
$$v_π^{disc}(s) → \frac{r̄(π)}{1-γ}$$

—— **v_π(s) 在大多数 state 上趋近一个常数** $\frac{r̄}{1-γ}$，**几乎不区分 state**！

**含义**：discount γ 在 continuing task + FA 上失去意义——所有 state value 几乎相等，γ 信号失效。

#### 反对论点
- 用 average reward formulation
- discounted formulation 仅适合 episodic

#### 实践影响
- 现代 deep RL 在 continuing task（如 SAC on robotics）**仍用 discounted**——是 trick 不是理论支持
- **Average reward formulation 没成主流**——工程惯性

#### 高频面试问法
**Q：§10.4 反对 discounted continuing 的核心论证？**
**A**：γ<1 时 steady-state $v_π^{disc}(s) → \frac{r̄(π)}{1-γ}$（常数）——v 在大多数 state 上几乎相等 → γ 信号被 r̄ 吃掉 → discount 失去区分 policy 的功能。**continuing task 数学上必须 average reward formulation**。

---

## 4. 跨概念对比表

### 4.1 本章算法 family tree
```
On-policy Control + FA
├── Episodic SARSA + FA
│   ├── 1-step
│   └── n-step
├── Average Reward Formulation ⭐
│   ├── Differential Value
│   └── Differential SARSA + FA
└── 现代延伸
    └── 几乎被 PG（PPO/SAC）取代
```

### 4.2 选型表

| 场景 | 推荐 |
|---|---|
| Tabular small | tabular SARSA |
| Linear FA (Mountain Car) | **SARSA + Tile Coding** |
| Atari / 大 state | **DQN**（不是本章）|
| Continuous control | **SAC**（不是本章）|
| LLM RLHF | **PPO**（不是本章）|

**关键洞察**：本章方法在现代 deep RL 基本被取代——但 average reward 概念对 continuing task 仍有理论价值。

---

## 5. 习题精选

**Exercise 10.1（Mountain Car）**：tile coding + SARSA 解 Mountain Car，扫 α。
**Exercise 10.7（differential SARSA）**：实现 average reward 版在 access-control queuing。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter10/mountain_car.py`（Example 10.1）、`access_control.py`（Example 10.2 average reward）
- **必跑 Mountain Car**——RL 入门毕业试金石

## 7. 跟 RLHF 的桥

| RLHF 概念 | 来自本章 §3.x |
|---|---|
| **Multi-turn agent / chatbot 长 chat** | §3.4 average reward formulation |
| Differential value 在 RLHF | 弱直接，但思想适用长 trajectory |
| **PPO/SAC 取代 SARSA + FA** | 本章方法已"博物馆化" |

## 8. 苏格拉底 Q&A

详见每个 §3.x。综合：

- **Q：为什么 §10 在现代 deep RL 不主流？**
  - 答：DQN（Ch 11 后）取代 value-based 主流；PPO/SAC 取代 control 主流。本章方法在 tabular / linear FA 经典任务（Mountain Car）仍有价值。
- **Q：multi-turn agent 长 chat 怎么应用本章思想？**
  - 答：average reward formulation——把每 turn reward 减平均 → 适用无终态 continuing chat

## 9. 自测清单（闭卷）

- [ ] **写 episodic semi-gradient SARSA 算法**
- [ ] **写 differential semi-gradient SARSA 算法**（含 r̄ 更新）
- [ ] **解释 average reward vs discounted 的根本差异**
- [ ] **跑通 Mountain Car SARSA**
- [ ] **解释 §10.4 反对 discounted continuing 的论证**
- [ ] **解释为什么本章方法"博物馆化"**

---

## 💡 拓展知识点（书外 trivia）

### A. Mountain Car 是 RL 的"Hello World"之一
1990s Singh & Sutton 提出——小车需"前后摇摆"积累动能爬山。sparse reward → 早期 RL 算法常学不会。**SARSA + tile coding 能学会 = FA control 标志性实验**。

### B. Average Reward 是 OR 的老传统
- continuing task average reward formulation 在运筹学（OR）有 80 年传统
- **Howard 1960** *Dynamic Programming and Markov Processes* 系统化
- RL 借鉴比 OR 晚 30 年

### C. §10.4 反对 discounted continuing 的影响
**实践仍用 discounted**——γ<1 在 SAC / robotics 经验上 work（虽然 §10.4 说不严格）。**RL 工程不总跟随理论**——trick 文化。

### D. Multi-turn agent / chatbot RL 设计
- 长 chat 无自然终态 ≈ continuing task
- 现代实践：**人工切 "episode"**（每 N turn 重启 episode）——不严格但 work
- **未来方向**：differential value-style RLHF for agents

### E. 本章方法 vs DQN 时间线
| 年 | 算法 | 状态 |
|---|---|---|
| 1996 | SARSA + tile coding on Mountain Car | 当时 SOTA |
| 2013 | DQN on Atari | 几乎完全取代 SARSA + tile coding |
| 2017 | PPO | 取代 value-based 当 RL 默认 |
| 2022 | RLHF | LLM 后训练默认 |

### F. 一个反直觉数字：Mountain Car SARSA 比 DQN 快收敛
- SARSA + tile coding：~100 episodes 收敛
- DQN：~500-1000 episodes
- **原因**：tile coding 是人工 prior，DQN 要从零学 feature——简单任务 deep learning 反而吃亏

### G. CS285 vs Sutton 在本章的态度
- Sutton：semi-gradient SARSA 是"主流 control"
- Sergey Levine CS285：基本跳过，直接讲 DQN / SAC / PPO
- 反映了**学界从 value-based 转向 policy-based 的潮流**

---

## 🔗 更多 wikilinks（深挖）

| 主题 | 跳转 |
|---|---|
| DQN 取代了本章方法 | [[../../../brain/Areas/rl-books/deep-rl-hands-on-lapan-3e/_overview\|Lapan Ch 6]] |
| Off-policy + FA = deadly triad | [[ch11-off-policy-methods]] |
| 现代主流：policy-based | [[ch13-policy-gradient]] |
| Multi-turn agent RL | [[../../../brain/Areas/rl-books/rlhf-lambert/ch13-tool-use-function-calling]] |

## Related

- **上一章**：[[ch09-on-policy-prediction]]
- **下一章**：[[ch11-off-policy-methods]]
- **进度**：[[_chapter-status]] W4

## Sources

- 原书 Ch 10 pp. 243-262
- [Silver UCL Lec 6: Value Function Approximation](https://www.youtube.com/watch?v=UoPei5o4fps)
- ShangtongZhang `chapter10/`
- **核心论文**：
  - Singh & Sutton 1996 *Reinforcement Learning with Replacing Eligibility Traces*
  - Howard 1960 *Dynamic Programming and Markov Processes*（average reward OR 经典）
