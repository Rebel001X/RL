---
title: "Sutton & Barto Ch 13. Policy Gradient Methods"
created: 2026-05-24
tags: [area/rl, type/note, status/v3, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 13
difficulty: hard
budget_days: 5-7
priority_for_rlhf: must
silver_lecture: 7
---

# Ch 13. Policy Gradient Methods ⭐⭐⭐

> 🚀 **高阶深读版 jupyter notebook**：[[../../Code/sutton-barto-2e/ch13-policy-gradient/ch13-policy-gradient.ipynb|ch13.ipynb]]

---

## 📋 本章讲了什么（TL;DR）

1. **从 value-based 转向 policy-based**：不再学 Q(s,a) 再 derive policy，而是直接参数化 π_θ(a|s) 并沿 J(θ) 梯度上升
2. **Policy Gradient Theorem**：$∇J = E_π[∇\log π · Q^π]$——state distribution μ 不出现 → sample-based 估计成为可能
3. **REINFORCE → AC → TRPO → PPO → GRPO/DPO**：所有现代 LLM 后训练算法都从本章公式派生

**为什么这章是 RL 圣经第一章**：S&B 17 章里只有 Ch 3（MDP）和 Ch 13（PG）是真正"绕不开"的——前者是语言，后者是 LLM 时代的 RL 入口。

---

## 🎯 Top 5 关键洞察

| # | 洞察 | 为什么关键 |
|---|---|---|
| 1 | **Score function trick**：$∇π = π · ∇\log π$ | 把"对 π 求导"变成"对 log π 求导 + 当 expectation" |
| 2 | **State distribution μ_π 不显式出现** | Rollout 样本天然按 μ 分布——Sutton 1999 革命 |
| 3 | **Baseline 不变 unbiasedness** | 任意 b(s) 都能减，降方差不要钱 |
| 4 | **PG 绕开 max → 绕开 deadly triad** | 对 NN 更友好 |
| 5 | **PPO clip 是 trust region 的工程化** | RLHF 工业标准 |

## 🧭 在 Obsidian graph 中的位置

- **上游**：[[ch03-finite-mdps]] · [[ch05-monte-carlo-methods]] · [[ch09-on-policy-prediction]] · [[ch12-eligibility-traces]]
- **横向**：[[ch11-off-policy-methods]]
- **下游**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning|RLHF Ch 6]] · [[../../../brain/Areas/rl-books/rlhf-lambert/ch08-direct-alignment-algorithms|RLHF Ch 8]]

---

## 0. 阅读元信息

- **难度**：hard（PG 定理推导是 RL 经典数学难关）
- **预算**：5-7 天 / **手推 PG 定理 ≥ 2 次**
- **必读理由**：**RLHF 的全部数学基础**——PPO/GRPO/DPO 都从 PG 定理派生
- **配套**：Silver UCL Lec 7、lcalem Ch 13 神文

## 1. 一句话 + 在 RL 体系中的位置

Policy Gradient = **直接对 π_θ(a|s) 参数化 + 沿 J(θ) 梯度上升**——跟 value-based（Q-learning 等学 Q 再 derive policy）并列的 RL 另一支主流。

**在 RL 算法家族树里**：
```
RL
├─ Value-based: Q-learning / DQN / Rainbow / C51（学 Q 或 V）
├─ Policy-based: REINFORCE / TRPO / PPO / GRPO（直接学 π）⭐ 本章
├─ Actor-Critic: A2C / SAC / DDPG（V 和 π 同时学）
├─ Model-based: Dyna / MuZero / Dreamer（学环境 model）
└─ Beyond pure RL: DPO / RLAIF / RLVR（绕开传统 RL）
```

## 2. 概念地图（§3 详解目录）

下面 14 个核心概念**每个**在 §3 都有完整 0-1 小节。这里只是索引：

| # | 概念 | §3.x | 重要度 |
|---|---|---|---|
| 1 | **Policy Gradient Theorem** | §3.1 | ⭐⭐⭐ 数学根 |
| 2 | **REINFORCE** | §3.2 | ⭐⭐ 入门算法 |
| 3 | **REINFORCE + Baseline** | §3.3 | ⭐⭐ 降方差 |
| 4 | **Actor-Critic** | §3.4 | ⭐⭐⭐ 主流框架 |
| 5 | **A2C / A3C** | §3.5 | ⭐⭐ 同步异步实现 |
| 6 | **GAE** | §3.6 | ⭐⭐⭐ PPO 用的 advantage |
| 7 | **TRPO** | §3.7 | ⭐⭐ PPO 的前身 |
| 8 | **PPO** | §3.8 | ⭐⭐⭐⭐ **RLHF 工业标准** |
| 9 | **GRPO** | §3.9 | ⭐⭐⭐⭐ **DeepSeek R1** |
| 10 | **DPO** | §3.10 | ⭐⭐⭐⭐ **绕开 RL 的 alignment** |
| 11 | **IPO / KTO / SimPO / ORPO** | §3.11 | ⭐⭐ DPO 家族 |
| 12 | **SAC** | §3.12 | ⭐⭐ off-policy AC |
| 13 | **DDPG / TD3** | §3.13 | ⭐ continuous deterministic |
| 14 | **RLVR** | §3.14 | ⭐⭐⭐ reasoning model 关键 |

---

## 3. 核心概念详解（每个从 0-1 讲清）

### 3.1 Policy Gradient Theorem（PG 定理）

#### 是什么
**RL 最重要的数学定理之一**：对于参数化策略 π_θ(a|s)，目标 J(θ) = v_{π_θ}(s_0) 的梯度是

$$\boxed{∇J(θ) = E_{s \sim μ_π, a \sim π_θ}\left[ Q^{π_θ}(s, a) · ∇_θ \log π_θ(a|s) \right]}$$

—— **state distribution μ_π 不显式参与梯度计算**。

#### 为什么需要它（动机）
- 想直接优化"长期累积 reward"，但 reward 通过 trajectory 累积，且 state 分布跟 policy 自身有关——看似无法 backprop
- 直接对 J(θ) 写 ∇θ 会出现 μ_π(s) 对 θ 的导数（链式法则），这一项**无法 sample-based 估计**——因为我们不知道环境 dynamics
- **PG Theorem 的惊人之处**：μ_π 那一项**消掉了**，留下的形式只用 (s, a, Q) 三元组就能算

#### 数学推导（6 步白板必会）

**Step 1**：从 Bellman 出发
$$v_π(s) = \sum_a π(a|s) q_π(s, a)$$

**Step 2**：两边对 θ 求导（乘积法则）
$$∇v_π(s) = \sum_a [∇π(a|s) · q_π(s,a) + π(a|s) · ∇q_π(s,a)]$$

**Step 3**：展开 ∇q_π
$$q_π(s, a) = \sum_{s', r} p(s', r | s, a)[r + γv_π(s')]$$
$$∇q_π(s, a) = γ \sum_{s'} p(s'|s, a) ∇v_π(s')$$
（p, r 跟 θ 无关）

**Step 4**：递归代回 → unroll
$$∇v_π(s) = \sum_x \sum_{k=0}^∞ γ^k \Pr(s → x, k, π) \sum_a ∇π(a|x) q_π(x, a)$$

**Step 5**：引入 state-visit distribution
$$\eta(s) = \sum_{k=0}^∞ γ^k \Pr(s_0 → s, k, π), \quad μ_π(s) = η(s) / \sum η$$

$$∇J ∝ \sum_s μ_π(s) \sum_a ∇π(a|s) q_π(s, a)$$

**Step 6**：Score function trick
$$∇π(a|s) = π(a|s) · ∇\log π(a|s)$$
$$\boxed{∇J = E_{s \sim μ, a \sim π}[Q · ∇\log π]}$$

#### 伪代码（一行）
```python
gradient_estimate = (advantage * log_pi.grad).mean()
```

#### 实战参数与超参指南
PG Theorem 本身没有超参——它是定理，不是算法。具体 implementation 见 §3.2 - §3.14。

#### 跟相关算法对比
| 类 | 用了 PG Theorem 吗 |
|---|---|
| REINFORCE / AC / PPO / GRPO / DPO | ✅ 全部基于 |
| Q-learning / DQN | ❌ value-based，不用 PG |
| MCTS | ❌ search-based |

#### 反直觉点
- **μ_π 消失** 是个"会计奇迹"——理论上它该出现，但展开后正好合并进 expectation
- **不依赖 model**：没有 p(s'|s, a) 或 r 函数出现——纯 sample-based

#### 高频面试问法
**Q：手推 PG 定理 6 步**
**A**：v_π Bellman → 求导（乘积）→ 展开 ∇q_π → 递归 unroll → μ_π → score function trick。能写出 6 个公式 = 过关。

---

### 3.2 REINFORCE（基础算法）

#### 是什么
**最古老的 PG 算法**（Williams 1992）：用 sample 的 Monte Carlo return G_t 替代 PG Theorem 中的 Q^π。

#### 为什么需要它（动机）
- PG Theorem 给了 ∇J 形式，但 Q^π 未知——必须从 rollout 估
- 最简单：MC 跑完整 trajectory 后 G_t = Σ γ^k R_{t+k+1} 当 Q^π 估计
- 这就是 REINFORCE

#### 数学推导
从 PG Theorem：
$$∇J = E[Q · ∇\log π]$$

用 G_t 替代 Q^π：
$$∇J ≈ \frac{1}{N} \sum_{episodes} \sum_t γ^t G_t · ∇\log π(A_t|S_t)$$

更新规则：
$$\boxed{θ ← θ + α · γ^t · G_t · ∇\log π(A_t|S_t, θ)}$$

#### 伪代码（完整）
```python
def REINFORCE(env, policy_net, optimizer, n_episodes=1000, gamma=0.99):
    for ep in range(n_episodes):
        # 1. Rollout
        log_probs, rewards = [], []
        s = env.reset()
        while not done:
            probs = policy_net(s)
            a = sample(probs)
            log_probs.append(log(probs[a]))
            s, r, done = env.step(a)
            rewards.append(r)

        # 2. Compute returns G_t (backward)
        G = 0; returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # 3. Policy gradient update
        loss = 0
        for t, (lp, G_t) in enumerate(zip(log_probs, returns)):
            loss -= (gamma**t) * G_t * lp   # 注意 - 因为要最大化
        loss.backward()
        optimizer.step()
```

#### 实战参数
| 超参 | 推荐值 | 说明 |
|---|---|---|
| α (lr) | 1e-3 to 1e-2 | 高方差，lr 不能太大 |
| γ | 0.99 | episodic |
| n_episodes | 1000+ | 收敛慢 |
| baseline | ❌ 无 | 见 §3.3 加 |

#### 跟相关算法对比
| 维度 | REINFORCE | AC | PPO |
|---|---|---|---|
| Critic | ❌ | ✅ | ✅ |
| On-policy | ✅ 严格 | ✅ | ✅ 但多 epoch |
| 方差 | 高 | 中 | 低 |
| Sample 效率 | 低 | 中 | 高 |

#### 反直觉点
- **不需要 critic**：仅 policy 网络
- **必须等 episode 结束** → 无法 online 学
- **方差大**：G_t 把全程随机性都吃进来

#### 高频面试问法
**Q：REINFORCE 跟 PG Theorem 是什么关系？**
**A**：REINFORCE 是 PG Theorem 的"最朴素 sample-based 实现"——直接用 MC return G_t 当 Q^π 估计。所有 PG 算法都从这扩展。

---

### 3.3 REINFORCE + Baseline（降方差）

#### 是什么
在 REINFORCE 基础上**减去 baseline b(s)**——任意只依赖 s 的函数都行。最常用 b(s) = V(s)。

#### 为什么需要它（动机）
- REINFORCE 方差爆炸——G_t 噪音传到所有 step
- 减一个 baseline 不破坏 unbiasedness（数学保证），但**显著降方差**
- 这是 RL 实战必备 trick

#### 数学推导

**Unbiasedness 证明**（习题 13.2）：
$$E_a[b(s) · ∇\log π(a|s)] = b(s) \sum_a π(a|s) ∇\log π(a|s) = b(s) \sum_a ∇π(a|s) = b(s) ∇\sum_a π(a|s) = b(s) ∇1 = 0$$

—— 减任意 b(s) 不改梯度期望，但方差降。

**最优 baseline**（最小化方差）：
$$b^*(s) = \frac{E[(∇\log π)^2 · Q]}{E[(∇\log π)^2]}$$

—— 实战中近似为 V(s)（足够好）。

#### 伪代码（增量改 §3.2）
```python
baseline = 0.0  # running mean of returns
for ep in range(n_episodes):
    ...
    G_t = compute_returns(rewards, gamma)
    baseline = 0.95 * baseline + 0.05 * G_t.mean()  # EMA
    advantages = G_t - baseline
    loss = -sum(adv * lp for adv, lp in zip(advantages, log_probs))
    ...
```

#### 实战参数
| 超参 | 推荐值 |
|---|---|
| baseline | EMA β=0.95-0.99 |
| 或：用单独 V_φ net + MSE loss |

#### 跟相关算法对比
| 算法 | baseline 类型 |
|---|---|
| REINFORCE | 无 |
| REINFORCE + b | running mean / EMA |
| AC | 学到的 V(s) + TD bootstrap |
| PPO | GAE (combines V learning + λ-return) |

#### 反直觉点
- **降方差但不改 bias**——拿走的"奖励信号"被 expectation 抵消
- **常数 baseline 也行**——平均 return 当 baseline 已经显著降方差
- **不学 V 也能用**——running mean 当 baseline 就够小任务用

#### 高频面试问法
**Q：减 baseline 为什么不破坏 unbiasedness？关键步骤是什么？**
**A**：$E_a[b(s)·∇\log π] = b(s)·∇\sum_a π = b·∇1 = 0$。**Σ_a π = 1 是常数，求导为 0**——这是关键。

---

### 3.4 Actor-Critic（AC）

#### 是什么
两个网络合体——**Actor (π_θ)** 学 policy，**Critic (V_φ)** 学价值函数当 baseline + bootstrap。

#### 为什么需要它（动机）
- REINFORCE + baseline 用 MC return → 必须等 episode 结束 + 方差大
- 如果用 V(s) 替代 G_t，能 online + 方差降
- 但 V(s) 需要学 → 引入 critic

#### 数学推导

**TD error 当 advantage**：
$$δ_t = R_{t+1} + γV_φ(S_{t+1}) - V_φ(S_t)$$

注意 δ_t 是 advantage A^π(s, a) 的 1-sample 估计：
$$E[δ_t | S_t = s, A_t = a] = Q^π(s, a) - V^π(s) = A^π(s, a)$$

**Actor 更新**：
$$θ ← θ + α · δ_t · ∇\log π(A_t|S_t)$$

**Critic 更新**（semi-gradient TD）：
$$φ ← φ + β · δ_t · ∇V_φ(S_t)$$

#### 伪代码（一步 AC，online）
```python
def actor_critic_step(s, actor, critic, optimizer_a, optimizer_c, gamma):
    a = sample(actor(s))
    s_next, r, done = env.step(a)
    # TD error
    delta = r + gamma * critic(s_next) * (1 - done) - critic(s)
    # Actor (with detach so critic grad not flow back)
    loss_a = -delta.detach() * log(actor(s)[a])
    # Critic (MSE on TD target)
    loss_c = delta.pow(2)
    loss_a.backward(); optimizer_a.step()
    loss_c.backward(); optimizer_c.step()
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| α (actor lr) | 3e-4 |
| β (critic lr) | 1e-3（critic 学得慢些）|
| γ | 0.99 |
| Critic arch | MLP（跟 actor 同 shape）|

#### 跟相关算法对比
| 算法 | Actor | Critic | bootstrap | on-policy |
|---|---|---|---|---|
| REINFORCE | ✅ | ❌ | ❌ MC | ✅ |
| AC (1-step) | ✅ | ✅ | ✅ TD(0) | ✅ |
| A2C / A3C | ✅ | ✅ | ✅ n-step | ✅ |
| PPO | ✅ | ✅ | ✅ GAE | ✅ 多 epoch |
| SAC | ✅ | ✅ Q | ✅ | ❌ off-policy |

#### 反直觉点
- **Critic 用 TD bootstrap 引入 bias**——但方差降的好处通常更大
- **detach 重要**：actor loss 里 δ 必须 detach，否则 critic 梯度会污染 actor 优化方向
- **两个 lr 通常不同**——critic 通常学得快些（β > α）

#### 高频面试问法
**Q：AC 比 REINFORCE + baseline 强在哪？**
**A**：(1) 用 TD bootstrap → online 更新，不需等 episode 结束；(2) critic 学到的 V(s) 是状态条件的 baseline，比 running mean 更精；(3) 方差进一步降。

---

### 3.5 A2C / A3C（同步与异步 Actor-Critic）

#### 是什么
- **A3C**（Mnih 2016, *Asynchronous Methods for Deep RL*）：多 worker 异步 rollout + 共享参数更新
- **A2C**：A3C 的同步版（OpenAI 2017）——把 N worker 同步收集 batch，再一次更新

#### 为什么需要它（动机）
- 单线程 RL 慢 + 数据相关性高（容易 overfit）
- 多 worker 并行：(1) 提速；(2) 数据多样性破 correlation
- A3C 异步省同步开销；A2C 同步实测更稳，工程上更主流

#### 数学
没有新公式——就是 AC 的 n-step + batch + 多 worker。

n-step return（[[ch07-n-step-bootstrapping]]）：
$$G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V_φ(S_{t+n})$$
$$δ_t = G_{t:t+n} - V_φ(S_t)$$

#### 伪代码（A2C 单步）
```python
def A2C_update(workers, actor, critic, optimizer, n_steps=5):
    # 1. 多 worker 各跑 n_steps
    trajectories = [w.rollout(n_steps) for w in workers]
    # 2. 合并 batch
    states, actions, rewards, log_probs = batch(trajectories)
    # 3. 算 n-step return
    returns = compute_n_step_returns(rewards, critic, gamma, n_steps)
    advantages = returns - critic(states)
    # 4. Actor & Critic loss
    loss_a = -(advantages.detach() * log_probs).mean()
    loss_c = (returns - critic(states)).pow(2).mean()
    loss_entropy = -entropy(actor(states)).mean()  # encourage exploration
    loss = loss_a + 0.5 * loss_c - 0.01 * loss_entropy
    loss.backward(); optimizer.step()
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| n_steps | 5-20 |
| n_workers | 8-32 |
| entropy_coef | 0.01 |
| value_coef | 0.5 |
| max_grad_norm | 0.5 |

#### 跟相关算法对比
| 维度 | A3C | A2C | PPO |
|---|---|---|---|
| 同步性 | 异步 | 同步 | 同步 |
| 多 worker | ✅ | ✅ | ✅ |
| 多 epoch update | ❌ | ❌ | ✅（K 次）|
| Trust region | ❌ | ❌ | ✅ clip |

#### 反直觉点
- **A2C 实测比 A3C 更稳**——同步省了"陈旧 gradient"问题
- **entropy bonus 不可忽略**——少了 exploration 容易陷局部最优

#### 高频面试问法
**Q：A2C vs A3C 区别？为什么 A2C 后来更主流？**
**A**：A3C 异步——worker 各自更新共享参数；A2C 同步——worker 收集完一起更新。A2C 实测更稳（无 stale gradient）+ GPU 友好（同步 batch 可大），所以主流。

---

### 3.6 GAE（Generalized Advantage Estimation）

#### 是什么
**Schulman 2015 提出**（arXiv 1506.02438）：用 TD(λ) 思想做 advantage 估计——λ-加权所有 n-step advantage。

$$A_t^{GAE(γ, λ)} = \sum_{l=0}^∞ (γλ)^l δ_{t+l}$$

#### 为什么需要它（动机）
- **TD(0) advantage**（n=1）：bias 大方差小
- **MC advantage**（n=∞）：无 bias 方差大
- **n-step**：固定 n 不灵活
- **GAE**：用 λ 平滑混合所有 n——bias-variance 连续可调

#### 数学推导

**第一步**：定义 TD error
$$δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)$$

**第二步**：n-step advantage 可写成 δ 的累加
$$A^{(n)}_t = \sum_{l=0}^{n-1} γ^l δ_{t+l}$$

**第三步**：GAE = λ-加权所有 n
$$A^{GAE}_t = (1-λ) \sum_{n=1}^∞ λ^{n-1} A^{(n)}_t = \sum_{l=0}^∞ (γλ)^l δ_{t+l}$$

**极限**：
- λ=0 → A^GAE = δ_t（TD(0)）
- λ=1 → A^GAE = G_t - V(S_t)（MC，无 bootstrap）

#### 伪代码（backward sweep）
```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """逆向计算 GAE（高效 O(T)）"""
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0
    for t in reversed(range(T)):
        next_value = values[t+1] if t < T-1 else 0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns
```

#### 实战参数
| 超参 | 推荐 | 直觉 |
|---|---|---|
| **λ** | **0.95**（PPO 默认）| 接近 MC 但仍 bootstrap |
| γ | 0.99 | 长视野 |
| Normalize advantages | ✅ (减均值除标准差) | 训练稳定 |

#### 跟相关算法对比
| 估计 | 公式 | bias | variance |
|---|---|---|---|
| TD(0) advantage | δ_t | 大 | 小 |
| MC advantage | G_t - V(s_t) | 0 | 大 |
| **GAE λ=0.95** | Σ (γλ)^l δ_{t+l} | 小 | 中（甜点）|

#### 反直觉点
- **λ=1 不是 GAE 论文推荐**——尽管无 bias，方差太大反而劣于 λ=0.95
- **normalize advantages 极其重要**——不 normalize 训练经常崩
- **TD(λ) 跟 GAE 公式一模一样**——只是 GAE 把 V learning 换成 advantage estimation

#### 高频面试问法
**Q：GAE 跟 TD(λ) 关系？**
**A**：数学上完全相同形式 $\sum (γλ)^l δ_{t+l}$。区别：TD(λ) 用 δ 学 V；GAE 用 δ 估 advantage 给 PG 算 actor 梯度。**思想同源，应用不同**。

---

### 3.7 TRPO（Trust Region Policy Optimization）

#### 是什么
**Schulman 2015**（arXiv 1502.05477）：约束 policy update 步长——每次 update 不能让新 π 跟旧 π 在 KL divergence 上偏太多。

#### 为什么需要它（动机）
- 朴素 PG 步长太大 → policy 崩（performance cliff）
- 怎么"安全地大步走"？**Trust region** 思想：在 KL 约束区内 maximize improvement
- TRPO 给出严格数学保证：每步**monotonic improvement**

#### 数学推导

**优化问题**（surrogate objective + KL 约束）：
$$\max_θ E_{s,a \sim π_{old}}\left[ \frac{π_θ(a|s)}{π_{old}(a|s)} A^{π_{old}}(s, a) \right]$$
$$\text{s.t.} \quad E_s[KL(π_{old}(·|s) \| π_θ(·|s))] ≤ δ$$

其中 ratio $r_t = π_θ/π_{old}$ 来自 importance sampling（[[ch05-monte-carlo-methods]]）。

**求解**（用 Lagrangian + 二阶近似）：
1. 一阶近似 surrogate
2. 二阶近似 KL constraint（用 Fisher Information Matrix）
3. 解约束二次规划：$θ_{new} = θ_{old} + α · F^{-1}∇L$
4. 用 conjugate gradient 算 F^{-1}g（避免显式求逆）

#### 伪代码（简化）
```python
def TRPO_step(actor, critic, rollout, delta=0.01):
    advantages = compute_gae(rollout)
    g = compute_pg_gradient(actor, rollout, advantages)  # ∇L
    # 算 Fisher matrix-vector product
    Fvp = lambda v: compute_FVP(actor, rollout, v)
    # CG 求 F^{-1}g
    direction = conjugate_gradient(Fvp, g, n_iters=10)
    # 算步长
    step_size = sqrt(2 * delta / (direction @ Fvp(direction)))
    # 线搜索确保 KL <= δ
    for shrink in [1, 0.5, 0.25, ...]:
        new_theta = theta + shrink * step_size * direction
        if KL(old_pi, new_pi) <= delta and improved:
            return new_theta
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| δ (KL trust radius) | 0.01 |
| CG iters | 10 |
| line search | 必须 |

#### 跟相关算法对比
| 维度 | REINFORCE | TRPO | PPO |
|---|---|---|---|
| 步长控制 | 无 | KL hard constraint | ratio clip 软约束 |
| 实现复杂度 | 简单 | 高（Fisher + CG）| 中 |
| Monotonic improvement | 无 | ✅ 理论保证 | 近似 |
| 工业用 | 罕见 | 较少 | **主流** |

#### 反直觉点
- **TRPO 理论强但实现复杂**——CG + 二阶 Hessian-vector product 是 deep learning 主流框架不太友好
- **2017 PPO 出来后 TRPO 几乎被取代**——一个数学严格性换工程简洁性的经典故事
- **OpenAI 自己从 TRPO 转向 PPO**——他们论文里直接说"PPO is simpler and empirically performs comparably or better"

#### 高频面试问法
**Q：TRPO 跟 PPO 区别？为什么 PPO 替代 TRPO？**
**A**：TRPO 用 KL hard constraint 保证 monotonic improvement，但实现要 Fisher matrix + CG，复杂；PPO 用 clip(ratio, 1-ε, 1+ε) 软约束，实现 5 行 + 性能相当——工程胜数学。

---

### 3.8 PPO（Proximal Policy Optimization）⭐⭐⭐⭐

#### 是什么
**Schulman 2017**（arXiv 1707.06347）：TRPO 的简化版——用 **clip ratio** 替代 KL hard constraint，5 行实现，**RLHF 工业标准**。

#### 为什么需要它（动机）
- TRPO 严格但难——Fisher + CG + line search 太复杂
- 想要"trust region 思想"但实现简洁
- PPO 用 ratio clip 实现"软 trust region"

#### 数学推导

**核心公式 PPO-Clip**：

$$r_t(θ) = \frac{π_θ(a_t|s_t)}{π_{old}(a_t|s_t)}$$

$$\boxed{L^{CLIP}(θ) = E_t\left[ \min\left( r_t · A_t, \text{clip}(r_t, 1-ε, 1+ε) · A_t \right) \right]}$$

**解读**（核心机制）：
- 若 $A_t > 0$（好 action）：rewards 上升 → ratio 涨。clip 限制 ratio ≤ 1+ε，防止过度更新
- 若 $A_t < 0$（坏 action）：rewards 下降 → ratio 跌。clip 限制 ratio ≥ 1-ε，防止过度更新
- 用 min 取较保守的——确保 update 不超出 trust region

**完整 loss**（含 critic + entropy）：
$$L^{total} = L^{CLIP} - c_1 · L^{VF} + c_2 · S[π]$$

- $L^{VF} = (V_φ - V^{target})^2$：critic MSE loss
- $S[π]$：entropy bonus（鼓励 exploration）

#### 伪代码（完整生产级）
```python
def PPO_train(env, actor, critic, n_iters=1000, n_steps=2048, K=10, batch_size=64):
    """K = update epochs per rollout"""
    for it in range(n_iters):
        # === 1. Rollout (collect 2048 steps with current policy) ===
        rollout = collect_rollout(env, actor, n_steps)
        old_log_probs = rollout.log_probs.detach()
        # === 2. Compute GAE advantages ===
        with torch.no_grad():
            values = critic(rollout.states)
            advantages, returns = compute_gae(rollout.rewards, values, gamma=0.99, lam=0.95)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # === 3. K-epoch update (key innovation: reuse data) ===
        for epoch in range(K):
            for batch in minibatches(rollout, batch_size):
                # Re-compute log_probs with current (updated) policy
                new_log_probs = actor.log_prob(batch.states, batch.actions)
                ratio = exp(new_log_probs - batch.old_log_probs)
                # Clip loss
                surr1 = ratio * batch.advantages
                surr2 = clip(ratio, 1-eps, 1+eps) * batch.advantages
                loss_clip = -min(surr1, surr2).mean()
                # Critic loss
                v_pred = critic(batch.states)
                loss_v = ((v_pred - batch.returns) ** 2).mean()
                # Entropy bonus
                entropy = actor.entropy(batch.states).mean()
                # Total
                loss = loss_clip + 0.5 * loss_v - 0.01 * entropy
                optimizer.zero_grad(); loss.backward()
                clip_grad_norm_(parameters, 0.5)
                optimizer.step()
```

#### 实战参数（PPO 默认）
| 超参 | 推荐值 | 调参直觉 |
|---|---|---|
| **ε (clip range)** | **0.2** | trust region 半径；0.1 更保守，0.3 更激进 |
| **K (update epochs)** | **4-10** | 多 epoch 复用数据 → sample 效率高；太多 → off-policy 漂移 |
| n_steps (rollout) | 2048 | per worker，多 worker 时 ×N |
| batch_size | 64 | minibatch 大小 |
| γ | 0.99 | 折扣 |
| **λ (GAE)** | **0.95** | bias-variance 甜点 |
| α (lr) | 3e-4 | Adam |
| value_coef (c1) | 0.5 | critic loss 权重 |
| entropy_coef (c2) | 0.01 | exploration |
| max_grad_norm | 0.5 | 梯度裁剪 |

#### 跟相关算法对比
| 维度 | TRPO | PPO | GRPO |
|---|---|---|---|
| Trust region | KL hard | clip 软 | clip 软 |
| Critic | ✅ | ✅ | ❌ 去掉 |
| Advantage | GAE | GAE | group mean/std |
| 实现复杂度 | 高 | 中 | 低 |
| 显存 | 中 | 中 | 一半 |
| 工业用 | 罕 | **主流** | DeepSeek 路线 |

#### 反直觉点 / 实战陷阱
- **K=1 等价 vanilla PG，没有 PPO 优势**——必须多 epoch（4-10）才有 sample 效率收益
- **K 太大 → 过 off-policy → ratio 跑偏 → 即使 clip 也崩**：监控 approx_kl 在 0.01-0.05
- **Advantage normalization 至关重要**——不 normalize 经常训不起来
- **Reward clipping / scaling**：原始 reward 量级会让 critic 学不准
- **Entropy bonus 0.01 是个魔术数字**——RLHF 里通常调到 0（KL penalty 替代）

#### 高频面试问法

**Q1：手写 PPO loss 公式**
**A**：$L^{CLIP} = E_t[\min(r_t A_t, \text{clip}(r_t, 1-ε, 1+ε) A_t)]$，其中 $r_t = π_θ/π_{old}$，$A_t$ 用 GAE 估计。

**Q2：clip 在数学上是什么意思？**
**A**：clip 是 trust region 的"软"实现。当 ratio 超出 [1-ε, 1+ε] 时，min 选 clipped 项 → 梯度归零 → 该 step 不再更新该 sample。等价 KL constraint 的近似。

**Q3：K (update epochs) 怎么调？**
**A**：太小（K=1）退化为 vanilla PG，无 sample 效率；太大（K=20+）数据过 off-policy，ratio 大量超出 clip 区→无效更新。**经验 K=4-10，监控 approx_kl < 0.05**。

**Q4：PPO 训练时怎么诊断"崩了"？**
**A**：监控 (1) approx_kl > 0.1（policy 漂移过快）；(2) clip_fraction > 0.3（多数 sample 已被 clip）；(3) explained_variance 下降（critic 学不准）。三者任一报警都该 stop。

**Q5：RLHF 用 PPO 跟 Atari PPO 区别？**
**A**：(1) state = (prompt + 已生成 tokens) 文本而非 pixel；(2) action space = vocab (~50k) 而非几个；(3) reward 稀疏（只末尾给一次 RM 分）；(4) 额外 KL penalty 项防止离 SFT 太远；(5) γ 通常 = 1（chat 短 episode 无需折扣）。

---

### 3.9 GRPO（Group Relative Policy Optimization）⭐⭐⭐⭐

#### 是什么
**DeepSeek 2024**（DeepSeekMath 论文 arXiv 2402.03300）：PPO 的简化版——**去掉 critic**，用 group 内归一化算 advantage。

#### 为什么需要它（动机）
- PPO 的 critic 占一半模型参数 + 训练 compute
- 对 LLM 来说 critic 学得不够准（稀疏 reward + 大模型）
- 反正同 prompt 跑多次 rollout 就有 group 信息——用 sample-based baseline 替代 critic

#### 数学推导

**Group advantage**：
设同 prompt 跑 G 个 rollout，得 reward {R_1, ..., R_G}。每个 rollout 的 advantage：

$$\boxed{\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}}$$

**Loss**（跟 PPO 几乎一样）：
$$L^{GRPO} = E\left[ \min(r_i \hat{A}_i, \text{clip}(r_i, 1-ε, 1+ε) \hat{A}_i) - β · KL(π_θ \| π_{ref}) \right]$$

注意：
- 不再有 critic V_φ
- KL penalty 是独立项（不是 reward 修正）
- group size G 取代 critic 提供 baseline

#### 伪代码
```python
def GRPO_step(prompt, actor, ref_actor, group_size=8, eps=0.2, beta=0.01):
    # 1. 同 prompt 跑 group_size 个 rollout
    trajectories = [actor.rollout(prompt) for _ in range(group_size)]
    rewards = [compute_reward(t) for t in trajectories]  # RM or rule-based
    # 2. Group normalize advantage (no critic)
    R = torch.tensor(rewards)
    advantages = (R - R.mean()) / (R.std() + 1e-8)
    # 3. PPO-style update
    for traj, adv in zip(trajectories, advantages):
        old_log_probs = traj.log_probs.detach()
        new_log_probs = actor.log_prob(traj.states, traj.actions)
        ratio = exp(new_log_probs - old_log_probs)
        surr1 = ratio * adv
        surr2 = clip(ratio, 1-eps, 1+eps) * adv
        loss_clip = -min(surr1, surr2).mean()
        # KL to reference (extra term, not in advantage)
        ref_log_probs = ref_actor.log_prob(traj.states, traj.actions)
        kl = (new_log_probs - ref_log_probs).mean()
        loss = loss_clip + beta * kl
        loss.backward()
    optimizer.step()
```

#### 实战参数（DeepSeek R1 训练）
| 超参 | 推荐 |
|---|---|
| **group_size G** | **8 - 64** |
| ε (clip) | 0.2 |
| **β (KL coef)** | **0.001 - 0.04** |
| K (epochs) | 1（GRPO 通常单 epoch）|
| reference policy | SFT model（frozen）|

#### 跟相关算法对比

| 维度 | PPO | GRPO | 区别影响 |
|---|---|---|---|
| Critic | ✅ V_φ | ❌ 无 | GRPO 省一半参数 |
| Advantage | GAE（V 估计）| group mean/std | GRPO 是 MC-style |
| KL | 加进 reward | 独立 loss 项 | 表达力差异微小 |
| Sample 效率 | 高（多 epoch）| 中（通常单 epoch + 多 rollout）| trade-off |
| 适用 | 中小模型 | 大 LLM | 显存关键时 GRPO 赢 |

#### 反直觉点 / 实战陷阱
- **group_size 不能太小**——G=2 时 advantage 几乎全是 ±1，信号弱
- **GRPO 反而对 reward shaping 更敏感**——没 critic 缓冲
- **多 prompt batch 不要混 group**——必须同 prompt 内 normalize
- **β 调小 KL 弱 → reward hacking；β 大 → 学不动**：DeepSeek R1 用 β=0.001 偏激进

#### 高频面试问法

**Q1：GRPO 比 PPO 简化在哪？为什么有效？**
**A**：去 critic，用 group 内 mean/std 算 advantage。有效因为 (1) LLM 上 critic 难学（稀疏 reward + 大模型），不如直接用 sample-based baseline；(2) 省一半显存——对千亿模型至关重要。

**Q2：手写 GRPO advantage 公式**
**A**：$\hat{A}_i = (R_i - \text{mean}(R_{1..G})) / \text{std}(R_{1..G})$，G 是 group size（同 prompt 的 rollout 数）。

**Q3：GRPO 跟 PPO 的 KL 处理差异？**
**A**：PPO 把 KL penalty 加进 reward 一起算 advantage；GRPO 把 KL 当独立 loss 项加到总 loss。**数学略不同但精神同——都是约束 π 不离 ref 太远**。

**Q4：DeepSeek R1 为什么用 GRPO 而非 PPO？**
**A**：(1) 671B MoE 模型，省一半显存（无 critic）巨大；(2) reasoning 任务 reward 极稀疏（仅末尾 0/1），critic bootstrap 几乎无信号；(3) RLVR + GRPO 是天然搭配。

---

### 3.10 DPO（Direct Preference Optimization）⭐⭐⭐⭐

#### 是什么
**Rafailov 2023**（arXiv 2305.18290）：用 closed-form 数学技巧**把 RM 折叠进 policy loss**——不需要单独训 RM，不需要 RL rollout，直接用偏好数据训 SFT。

#### 为什么需要它（动机）
- RLHF pipeline 复杂：SFT → RM → PPO（3 步，3 个模型）
- RM 容易 overfit / reward hacking
- RL rollout 训练慢 + 显存大
- DPO：**1 步**直接训，不需 RM 也不需 RL

#### 数学推导（DPO 的精华）

**第一步**：PPO + KL 约束的最优 policy 有 closed-form
$$L_{PPO+KL} = E_π[r(x, y)] - β · KL(π \| π_{ref})$$
最优解：
$$π^*(y|x) = \frac{1}{Z(x)} π_{ref}(y|x) \exp\left(\frac{r(x, y)}{β}\right)$$

**第二步**：反推 reward
$$r(x, y) = β \log \frac{π^*(y|x)}{π_{ref}(y|x)} + β \log Z(x)$$

**第三步**：代入 Bradley-Terry 偏好模型
$$P(y_w \succ y_l | x) = σ(r(x, y_w) - r(x, y_l))$$

注意 $β \log Z(x)$ 项在 $r(y_w) - r(y_l)$ 中**抵消**——剩下只跟 π 有关：
$$P(y_w \succ y_l) = σ\left( β \log \frac{π(y_w|x)}{π_{ref}(y_w|x)} - β \log \frac{π(y_l|x)}{π_{ref}(y_l|x)} \right)$$

**第四步**：DPO loss = -log likelihood of preferences
$$\boxed{L^{DPO}(θ) = -E_{(x, y_w, y_l) \sim D}\left[ \log σ\left( β \log \frac{π_θ(y_w|x)}{π_{ref}(y_w|x)} - β \log \frac{π_θ(y_l|x)}{π_{ref}(y_l|x)} \right) \right]}$$

**关键**：这是个**纯 supervised loss**——给定偏好对 (x, y_w, y_l)，直接对 π 做 SGD。

#### 伪代码
```python
def DPO_loss(model, ref_model, batch, beta=0.1):
    """batch: list of (prompt, chosen, rejected)"""
    # Log probabilities
    log_p_w = model.log_prob(batch.prompts, batch.chosen)
    log_p_l = model.log_prob(batch.prompts, batch.rejected)
    with torch.no_grad():  # ref 不更新
        log_pref_w = ref_model.log_prob(batch.prompts, batch.chosen)
        log_pref_l = ref_model.log_prob(batch.prompts, batch.rejected)
    # 隐式 reward = β log(π/π_ref)
    chosen_reward = beta * (log_p_w - log_pref_w)
    rejected_reward = beta * (log_p_l - log_pref_l)
    # Bradley-Terry NLL
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss

# Train (looks exactly like supervised learning!)
optimizer = AdamW(model.parameters(), lr=5e-7)
for epoch in range(3):
    for batch in pref_dataset:
        loss = DPO_loss(model, ref_model, batch, beta=0.1)
        loss.backward(); optimizer.step()
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| **β** | **0.1 - 0.5** |
| lr | 5e-7 - 5e-6（**比 SFT 小 100x**！）|
| epochs | 1 - 3 |
| ref model | SFT model（frozen）|

#### 跟相关算法对比

| 维度 | PPO + RM | DPO |
|---|---|---|
| 训练阶段数 | 3 (SFT + RM + RL) | 2 (SFT + DPO) |
| 需要 RM 训练 | ✅ | ❌ |
| 需要 RL rollout | ✅ | ❌ |
| 显存 | 大 (actor + critic + RM + ref) | 中 (actor + ref) |
| 实现复杂度 | 高 | 极低 (≈ SFT) |
| 表现 | 略好 | 接近 |
| 数据要求 | 偏好 + RM data | 偏好 |

#### 反直觉点 / 实战陷阱
- **lr 必须很小**（5e-7）——DPO loss 对 lr 极敏感，比 SFT 小 100x
- **β 影响巨大**：β 小 → 学得快但偏离 ref（reward hacking 类比）；β 大 → 几乎不动
- **DPO 看似简单但调参难**——比 PPO 在小数据集上脆弱
- **不能保证 monotonic improvement**——可能 chosen 跟 rejected 同时降，但 chosen 降得少
- **reference model 必须固定**——动态更新 ref 等于 PPO（破坏 DPO 优雅性）

#### 高频面试问法

**Q1：DPO 跟 PPO 数学上等价吗？**
**A**：**理论上 yes**——同 optimization landscape；**实践上 no**——SGD 找到不同局部最优。

**Q2：手推 DPO loss 从 PPO+KL 出发**
**A**：(1) PPO+KL 最优 π* = π_ref · exp(r/β) / Z；(2) 反推 r = β log(π*/π_ref) + β log Z；(3) 代入 Bradley-Terry P(y_w > y_l) = σ(r_w - r_l)，β log Z 抵消；(4) 取 -log → DPO loss。

**Q3：DPO 为什么不需要 RM？**
**A**：通过数学技巧把 r 表达成 π 的函数 → 偏好 likelihood 直接用 π 写，**RM 隐式在 loss 里**。

**Q4：什么场景用 DPO 而非 PPO？反之？**
**A**：DPO 适合 (1) 数据少；(2) 团队小、不想搭 RL infra；(3) 显存紧。PPO 适合 (1) 数据多；(2) 想用 reward shaping / ensemble RM / online data；(3) 大厂生产。

---

### 3.11 IPO / KTO / SimPO / ORPO（DPO 家族对比）

#### 是什么
DPO 之后的演进版本：

- **IPO**（Identity Preference Optimization, 2023）：去掉 DPO 隐含的 Bradley-Terry 假设
- **KTO**（Kahneman-Tversky Optimization, 2024）：用前景理论替代 BT，**单边数据**（只有"好"或"坏"）也能训
- **SimPO**（Simple Preference Optimization, 2024）：**去掉 reference model**，省一半显存
- **ORPO**（Odds Ratio PO, 2024）：把 SFT 和 alignment **合并为一步**

#### 为什么需要它们（动机）
DPO 不是万能的：
- BT 假设可能不成立（人类偏好不一定 transitive）→ IPO
- 偏好对难收集，但"good/bad"标签易 → KTO
- ref model 占显存 → SimPO
- 两步 pipeline 麻烦 → ORPO

#### 数学（高层对比）

| 算法 | Loss 核心 | 数据要求 |
|---|---|---|
| **DPO** | $-\log σ(β \log \frac{π}{π_{ref}}(y_w) - β \log \frac{π}{π_{ref}}(y_l))$ | (x, y_w, y_l) |
| **IPO** | DPO loss + 二阶正则项 | (x, y_w, y_l) |
| **KTO** | $-\sigma(z_w) - \sigma(-z_l) + ...$（KT-style）| (x, y, good/bad)，**单边**|
| **SimPO** | $-\log σ(\frac{β}{|y_w|}\log π(y_w) - \frac{β}{|y_l|}\log π(y_l))$ | (x, y_w, y_l)，**无 ref** |
| **ORPO** | SFT loss + odds ratio loss | (x, y_w, y_l) |

#### 跟 DPO 对比一句话总结
- **IPO**：DPO 的"理论修正"
- **KTO**：用单边数据 DPO
- **SimPO**：DPO 去 ref
- **ORPO**：DPO + SFT 合一

#### 反直觉点
- **2024-2025 DPO 家族百花齐放**——但**没有谁明显赢过 DPO**
- **大厂 (OpenAI, Anthropic) 仍主要用 PPO + RM**——DPO 系是开源/小团队主流
- **SimPO 长度归一化技巧实际意外有效**——之后多个工作都借鉴

#### 高频面试问法
**Q：DPO 家族这么多，怎么选？**
**A**：默认 DPO；数据是 thumbs up/down 用 KTO；显存极紧用 SimPO；偏好少 + 想合并 SFT 用 ORPO。**没有银弹**——A/B 测试决定。

---

### 3.12 SAC（Soft Actor-Critic）

#### 是什么
**Haarnoja 2018**（arXiv 1801.01290）：**off-policy** Actor-Critic + **max entropy** 框架——continuous control 的 SOTA。

#### 为什么需要它（动机）
- PPO on-policy → sample 效率低
- SAC off-policy + replay buffer → sample 效率高
- max entropy → 自动 exploration + 鲁棒性
- 主要用在机器人 / continuous action 场景

#### 数学（最核心）

**Soft Q + entropy bonus**：
$$L_Q = E[(Q(s, a) - (r + γ(Q(s', a') - α \log π(a'|s'))))^2]$$

**Soft policy improvement**：
$$L_π = E[α \log π(a|s) - Q(s, a)]$$

α 是 entropy 系数（可学习）。

#### 跟 PPO 对比
| 维度 | PPO | SAC |
|---|---|---|
| On/Off-policy | on | **off**（replay）|
| Sample 效率 | 中 | **高** |
| Action space | discrete + continuous | 主要 continuous |
| LLM RLHF 用 | ✅ | ❌ |
| 机器人控制 | 较少 | ✅ 主流 |

#### 反直觉点
- **SAC 在 LLM 上不主流**——LLM action 是 discrete token，SAC 的 max entropy 对 vocab 50k 不直接适用
- **机器人界 SAC > PPO**——sample 效率重要

#### 高频面试问法
**Q：为什么 LLM RLHF 用 PPO 不用 SAC？**
**A**：SAC 设计给 continuous action（机器人），LLM 是 discrete token；离散版 SAC 不主流。PPO 在 discrete + 大 action space + 稀疏 reward 场景最稳。

---

### 3.13 DDPG / TD3（Continuous Deterministic）

#### 是什么
- **DDPG**（Lillicrap 2015）：Deterministic Policy Gradient + DQN tricks for continuous action
- **TD3**（Fujimoto 2018）：DDPG 的"加固版"——双 Q + delayed update + target smoothing

#### 为什么需要它（动机）
- DQN 不能直接处理 continuous action（max_a 没有解析）
- DDPG：deterministic policy + critic 一起学
- TD3 修正 DDPG over-estimation bias

#### 跟 SAC 对比
| 维度 | DDPG | TD3 | SAC |
|---|---|---|---|
| Stochastic policy | ❌ deterministic | ❌ | ✅ |
| Max entropy | ❌ | ❌ | ✅ |
| Sample 效率 | 中 | 中 | 高 |
| 鲁棒性 | 差 | 中 | 高 |

#### 反直觉点
- **TD3 出来后 DDPG 基本被取代**
- **SAC 通常比 TD3 更稳**——max entropy 自动探索

#### 高频面试问法
**Q：continuous control 场景的算法演进？**
**A**：DDPG → TD3 → SAC。一般首选 SAC（max entropy 鲁棒），次选 TD3。

---

### 3.14 RLVR（RL with Verifiable Rewards）⭐⭐⭐

#### 是什么
**2024 新范式**（DeepSeek R1 / o1 用）：reward 不靠学到的 RM，而是**可验证规则**——数学题答案对错 / 代码 unit test 通过 / 格式正确。

#### 为什么需要它（动机）
- 传统 RLHF：RM 容易 reward hacking
- 但 RM 是必要的——因为"helpful/harmless"难规则化
- **观察**：reasoning 任务（数学/代码）**可以规则化验证**——为什么还要 RM？
- → RLVR：跳过 RM，reward 直接规则化

#### 数学（看起来跟 PG 一样，只是 reward 来源不同）

**Reward function 改变**：
$$r(x, y) = \begin{cases} 1 & \text{if verify}(x, y) = \text{correct} \\ 0 & \text{otherwise} \end{cases}$$

再加 format reward / length penalty 等。

**算法**：用 GRPO（或 PPO）训，advantage = group normalize 的 verified reward。

#### 伪代码
```python
def RLVR_reward(prompt, response):
    """For math problems"""
    answer = extract_answer(response)
    correct_answer = solve_problem(prompt)  # ground truth
    return 1.0 if answer == correct_answer else 0.0

def RLVR_train(model, math_dataset, group_size=16):
    for prompt in math_dataset:
        # 1. Generate G rollouts
        responses = [model.generate(prompt) for _ in range(group_size)]
        # 2. Verify each
        rewards = [RLVR_reward(prompt, r) for r in responses]
        # 3. GRPO advantage
        advantages = group_normalize(rewards)
        # 4. PPO-style update
        grpo_update(model, prompt, responses, advantages)
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| group_size | 16 - 64（大组提供 baseline 稳定性）|
| reward design | 0/1 correctness + format bonus + length penalty |
| KL coef | 0.001（弱约束，让模型敢探索 CoT）|

#### 跟传统 RLHF 对比

| 维度 | 传统 RLHF (PPO + RM) | RLVR (GRPO + rule) |
|---|---|---|
| Reward source | 学到的 RM | 规则验证 |
| 适用 | 主观（"helpful"）| 客观（数学/代码）|
| Reward hacking | 严重 | 几乎不可能 |
| 训练成本 | 高（要先训 RM）| 低（无 RM 训练）|
| 代表系统 | InstructGPT, ChatGPT | DeepSeek R1, o1 |

#### 反直觉点
- **RLVR 把 reward 变成了 ground truth 检查器**——彻底消灭 reward hacking
- **只在可验证域 work**——开放写作、风格仍需 RLHF
- **promotes "long CoT"**——模型学会先想再答，因为只末尾 reward 强化"思考+正确答案"组合
- **RL + LLM 反而比 supervised CoT 更有效**——RL 让模型自己探索好的"思考路径"

#### 高频面试问法

**Q1：RLVR 跟 RLHF 本质区别？**
**A**：reward 来源——RLHF 学一个 RM 模仿人类偏好；RLVR 用规则直接验证（数学/代码可对）。RLVR 在可验证域更优，开放任务仍需 RLHF。

**Q2：DeepSeek R1 训练为什么用 GRPO + RLVR 而不是 PPO + RM？**
**A**：(1) reasoning 任务 reward 极稀疏（只末尾给 0/1），critic 学不了 → GRPO 无 critic；(2) 数学题对错可规则验证 → 不需 RM；(3) 671B 模型显存紧 → GRPO 省一半。**三个理由叠加 → GRPO + RLVR**。

**Q3：为什么 RL 训长 CoT 比 supervised CoT 更有效？**
**A**：(1) supervised CoT 受标注数据多样性限制；(2) RL 让模型自己探索"思考路径"——找到 supervised 数据没覆盖的解；(3) 稀疏 reward 强化"思考 → 正确"整体而非记忆具体步骤。

---

## 4. 跨概念对比表（一目了然）

### 4.1 所有算法 family tree

```
PG Theorem (§3.1)
├── REINFORCE (§3.2)
│   └── REINFORCE + Baseline (§3.3)
│       └── Actor-Critic (§3.4)
│           ├── A2C / A3C (§3.5)
│           ├── PPO (§3.8) ← 用 GAE (§3.6)
│           │   └── GRPO (§3.9) ← 去 critic
│           │       └── DeepSeek R1 (用 + RLVR)
│           └── SAC / DDPG / TD3 (§3.12-13) ← off-policy
└── PPO + KL constraint (§3.7 TRPO 简化)
    └── DPO (§3.10) ← closed-form 解 → 绕开 RL
        └── IPO / KTO / SimPO / ORPO (§3.11)
```

### 4.2 RLHF 算法选型表

| 场景 | 推荐 | 原因 |
|---|---|---|
| 中小团队 LLM alignment | DPO | 简单，2 步 pipeline |
| 大厂 LLM 后训练 | PPO + RM | 表现略好，可 reward shaping |
| Reasoning model 训练 | GRPO + RLVR | DeepSeek R1 范式 |
| 仅 thumbs up/down 数据 | KTO | 单边数据友好 |
| 显存极紧 | SimPO 或 GRPO | 无 ref / 无 critic |
| 机器人 continuous control | SAC | 标配 |
| 经典 Atari benchmark | PPO 或 Rainbow DQN | discrete action |

---

## 5. 习题精选

**Exercise 13.1（PG 定理推导）**：写 v_π 关于 θ 的展开。**答案**：见 §3.1 step 2。

**Exercise 13.2（baseline 不变 unbiasedness）**：见 §3.3 数学推导。

**Exercise 13.3（continuous action）**：把 PG 推广到 Gaussian policy。**关键**：log π(a|s) = -log(σ√2π) - (a - μ)² / (2σ²)，对 μ 和 σ 分别求 ∇。

**Exercise 13.5（actor-critic on cliff walking）**：实现 one-step AC on cliff walking，对比 SARSA / Q-learning / AC。

## 6. 代码伴侣

- **ShangtongZhang**：`chapter13/short_corridor.py`、`chapter13/mountain_car.py`、`chapter13/continuous_mountain_car.py`
- **自己实现 REINFORCE on CartPole**：见 [[../../Code/sutton-barto-2e/ch13-policy-gradient/ch13-policy-gradient.ipynb|ch13.ipynb]]
- **CleanRL**：单文件 PPO / SAC / DDPG / TD3 实现，工业级参考
- **TRL** (HuggingFace)：PPO/DPO for LLM 工业实现
- **OpenRLHF**：分布式 RLHF 训练框架
- **verl** (字节)：分布式 RL 框架，支持 PPO/GRPO

## 7. 跟 RLHF 的桥

| RLHF 组件 | 来自本章 §3.x | 工程实现 |
|---|---|---|
| PPO ratio | §3.8 (+ Ch 5 IS) | TRL `PPOTrainer` |
| GAE advantage | §3.6 + §3.4 critic | TRL/CleanRL |
| KL penalty | §3.8 (PPO 工程) | shaped reward = r_RM - β·KL |
| Reference model | §3.10 (DPO) | frozen SFT model |
| RM 训练 | (本章前置)Ch 5 IS | Bradley-Terry NLL |
| GRPO | §3.9 | DeepSeek 自研 + verl 复刻 |
| DPO loss | §3.10 | TRL `DPOTrainer` |
| RLVR | §3.14 | DeepSeek R1 训练栈 |

## 8. 苏格拉底 Q&A

详见每个 §3.x 末尾"高频面试问法"。补充几道综合题：

- **Q：从 REINFORCE 到 PPO 到 GRPO 到 DPO，每一步解决了什么前一步的痛点？**
  - 答：REINFORCE → AC（降方差）；AC → A2C（多 worker + n-step）；A2C → TRPO（trust region 防崩）；TRPO → PPO（简化）；PPO → GRPO（去 critic 省显存）；PPO → DPO（绕开 RL 直接训）
- **Q：手写"RL 算法家族树"**
  - 答：见 §4.1
- **Q：RLHF 三件套（SFT + RM + PPO）每件都跟本章哪节相关？**
  - 答：SFT—无 RL，CE loss；RM—§3.10（实际跟 DPO 数学反推有关）；PPO—§3.8

## 9. 自测清单（闭卷）

- [ ] **手推 PG 定理 6 步**（§3.1）
- [ ] **写出 REINFORCE / AC / PPO / GRPO / DPO 5 个算法的完整伪代码**
- [ ] **手写 PPO clip loss 公式 + 推导直觉**（§3.8）
- [ ] **手写 GRPO advantage 公式**（§3.9）
- [ ] **从 PPO+KL 推导 DPO loss**（§3.10）
- [ ] 实现 REINFORCE on CartPole 跑通
- [ ] 解释 RLVR 跟 RLHF 的本质区别（§3.14）
- [ ] 画 RL 算法家族树（§4.1）

---

## 💡 拓展知识点（书外 trivia，**不再讲核心算法**）

### A. PG 30 年浮沉史
- **1992 REINFORCE**：当年几十引用，无人问津
- **1999 Sutton PG Theorem**：理论奠基，无人意识到重要性
- **2015 TRPO + GAE**：PG 工业化拐点
- **2017 PPO**：工业标准确立
- **2022 ChatGPT**：PG 成 LLM 后训练唯一标准
- **2024 GRPO + R1**：PG 复兴在 reasoning
- **2026**：DPO/RLVR 并行发展

### B. Score function trick 在不同领域的别名
- 统计：Score function
- OR：Likelihood ratio estimator
- RL：REINFORCE estimator
- VAE：discrete latent SF estimator
- NAS：search policy gradient

**同一勾股定理被多次独立发现**。

### C. Sutton 1999 NeurIPS 现场反响
据 RLDM 老人回忆——平淡。Sutton 自己 2017 reddit AMA 说："I didn't realize it would become this important."

### D. PG 跨域应用
| 领域 | 用法 |
|---|---|
| GAN training | reward shaping（部分用 PG）|
| NAS | controller policy（Zoph 2016）|
| AutoML hyperparam | RL for tuning |
| AutoAugment | augment policy via PG |
| 分子设计 | RL for drug discovery |
| TCP congestion | Pantheon |
| LLM tool use | Toolformer |

### E. γ^t in REINFORCE
S&B 算法是 `θ += α · γ^t · G_t · ∇log π`——很多实现忘了 γ^t。理论必须，实践常省。**RLHF γ=1 时无影响；CartPole γ=0.99 时早 step 权重过大**。面试 trap。

### F. PPO 的隐藏痛点：Vanishing variance
训稳后 advantage 趋 0，gradient 弱 → 过早收敛。**解决**：reward shaping / entropy bonus / 周期 reset critic。

### G. RLVR 的"反 RLHF"哲学
传统 RLHF：RM 必要但易 hack。
RLVR：reward = 规则验证，跳过 RM。
2024-2026 RL **reward source 革命**。

### H. PG vs backprop 区别
- backprop = 算梯度算法
- PG = 用 backprop 估 ∇J 的"trick"（score function）

**常见混淆**："PPO = 用 PG 训 LLM"——更准确："PPO 用 backprop 算 ∇log π × advantage"。

### I. 为什么 PPO 在 RLHF 这么成功？
1. discrete + 大 action space（vocab 50k）—— PPO 唯一稳的
2. stochastic policy 必须（多样性）
3. 稀疏 reward（仅末尾）—— GAE λ=0.95 接近 MC 工作好
4. KL penalty 思想（trust region）天然适配"贴 SFT model"需求

---

## 🔗 更多 wikilinks（深挖）

| 主题 | 跳转 |
|---|---|
| RLHF 完整 pipeline | [[../../../brain/Areas/rl-books/rlhf-lambert/ch03-training-overview]] |
| Reward Modeling 数学 | [[../../../brain/Areas/rl-books/rlhf-lambert/ch05-reward-modeling]] |
| Reasoning RL（o1/R1 风格）| [[../../../brain/Areas/rl-books/rlhf-lambert/ch07-reasoning-inference-time-scaling]] |
| DPO / IPO / KTO | [[../../../brain/Areas/rl-books/rlhf-lambert/ch08-direct-alignment-algorithms]] |
| Over-optimization / reward hacking | [[../../../brain/Areas/rl-books/rlhf-lambert/ch16-over-optimization]] |
| RLHF 评测 | [[../../../brain/Areas/rl-books/rlhf-lambert/ch17-evaluation]] |
| AI-Infra: PPO rollout 与推理 | [[../../../brain/Slipbox/inference-metrics-ttft-tpot]] · [[../../../brain/Slipbox/prefill-decode-disaggregation]] |
| AI-Infra: FP8 训练 | [[../../../brain/Slipbox/fp8-training-pipeline]] |
| 系统设计：PPO rollout 服务 | [[../../../brain/Areas/ai-infra/_system-design]] |
| DeepSeek V3 技术报告（GRPO 来源）| [[../../../brain/Papers/arxiv-2412.19437]] |

## Related

- **上一章**：[[ch12-eligibility-traces]]（GAE 前置）
- **下一章**：[[ch14-psychology]]
- **IS 前置**：[[ch05-monte-carlo-methods]] §5.5
- **RLHF 全链**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning]] · [[../../../brain/Areas/rl-books/rlhf-lambert/ch08-direct-alignment-algorithms]]
- **进度**：[[_chapter-status]] W7 ⭐⭐⭐

## Sources

- 原书 Ch 13 pp. 321-343
- [Silver UCL Lec 7: Policy Gradient](https://www.youtube.com/watch?v=KHZVXao4qXs)
- [lcalem Ch 13 神文](https://lcalem.github.io/blog/2019/03/21/sutton-chap13)
- **核心算法论文**：
  - PG Theorem：Sutton et al. 1999 NeurIPS
  - REINFORCE：Williams 1992
  - TRPO：[Schulman 2015 arXiv 1502.05477](https://arxiv.org/abs/1502.05477)
  - GAE：[Schulman 2015 arXiv 1506.02438](https://arxiv.org/abs/1506.02438)
  - PPO：[Schulman 2017 arXiv 1707.06347](https://arxiv.org/abs/1707.06347)
  - DPO：[Rafailov 2023 arXiv 2305.18290](https://arxiv.org/abs/2305.18290)
  - GRPO/DeepSeekMath：[Shao 2024 arXiv 2402.03300](https://arxiv.org/abs/2402.03300)
  - DeepSeek R1：[2024 arXiv 2501.12948](https://arxiv.org/abs/2501.12948)
  - SAC：[Haarnoja 2018 arXiv 1801.01290](https://arxiv.org/abs/1801.01290)
- 实现参考：ShangtongZhang `chapter13/`、CleanRL、TRL、verl、OpenRLHF
