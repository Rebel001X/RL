---
title: "Sutton & Barto Ch 5. Monte Carlo Methods"
created: 2026-05-24
tags: [area/rl, type/note, status/v3, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 5
difficulty: medium
budget_days: 3
priority_for_rlhf: must
silver_lecture: 4
---

# Ch 5. Monte Carlo Methods ⭐⭐⭐

---

## 📋 本章讲了什么（TL;DR）

1. **MC = 跑完整 episode → 算 G_t → 取平均**——第一个 model-free 算法，无需环境转移概率，但必须 episode 结束才更新
2. **§5.5 importance sampling 是 PPO ratio 的根**：`E_π[f] = E_b[ρ·f]`——任何分布的期望可通过另一分布的加权样本估计
3. **Off-policy 学习首次系统化**：用 behavior b 采数据，优化 target π；引出 ordinary vs weighted IS 的方差权衡

**为什么 RLHF 必读 §5.5**：PPO 用旧 policy rollout 多 epoch 更新——本质就是 **off-policy + 单步 IS**。**不懂本章看 PPO 论文必卡**。

## 🎯 Top 5 关键洞察

| # | 洞察 | 为什么关键 |
|---|---|---|
| 1 | **MC 无 bootstrap → 无 bias，但高方差** | 跟 TD 完美 bias-variance 对比 |
| 2 | **IS 数学**：$E_\pi[f] = E_b[\frac{\pi}{b} f]$ | PPO/GRPO 所有 off-policy 算法的根 |
| 3 | **Ordinary IS 方差可能无限大** | Sutton 图 5.4 经典反例——为什么用 weighted IS + clip |
| 4 | **Exploring Starts 保证全 (s,a) 覆盖** | RL 早期 exploration 的最简方案 |
| 5 | **MC 不能用于 continuing task** | 必须 terminal state——RLHF 把 chat 单 turn 当 episode 的原因 |

## 🧭 在 Obsidian graph 中的位置

- **上游**：[[ch03-finite-mdps]]（G_t 定义）· [[ch04-dynamic-programming]]（MC 是 DP 去 model 版）
- **横向**：[[ch06-temporal-difference-learning]]（model-free 双子星）· [[ch07-n-step-bootstrapping]]（MC ↔ TD 桥）
- **下游**：[[ch12-eligibility-traces]]（λ=1 退化为 MC）· **[[ch13-policy-gradient]]**（REINFORCE 用 MC return）
- **RLHF 联动**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning|RLHF Ch 6]]——PPO ratio 直接来自本章

---

## 0. 阅读元信息

- **难度**：medium（§5.5 IS 是难点 + RLHF 必懂）
- **预算**：3 天
- **必读理由**：第一个 model-free 算法 + §5.5 IS 是 PPO ratio 根
- **配套**：Silver UCL Lec 4

## 1. 一句话 + 在 RL 体系中的位置

**MC = 跑完整 episode → 算 G_t → 取平均估 V/Q**；不要 model（无 bootstrap），但必须 episode 结束。§5.5 importance sampling 是 RLHF 的数学源头之一。

**在 RL 算法家族中的位置**：
```
RL Value-based
├── DP (Ch 4)           ← 知 model
├── MC (Ch 5) ⭐         ← model-free, episode-end
│   ├── First-visit MC
│   ├── Every-visit MC
│   ├── MC Control with ES
│   └── Off-policy MC + IS (§5.5)
├── TD (Ch 6)            ← model-free, bootstrap
└── λ / GAE (Ch 12, 13)  ← MC ↔ TD 谱
```

## 2. 概念地图（§3 详解目录）

| # | 概念 | §3.x | 重要度 |
|---|---|---|---|
| 1 | **First-visit MC Prediction** | §3.1 | ⭐⭐⭐ MC 入门 |
| 2 | **Every-visit MC** | §3.2 | ⭐⭐ |
| 3 | **MC Control with Exploring Starts** | §3.3 | ⭐⭐⭐ 第一个 model-free control |
| 4 | **On-policy MC Control (ε-soft)** | §3.4 | ⭐⭐ |
| 5 | **Off-policy MC + IS 理论** | §3.5 | ⭐⭐⭐⭐ **PPO ratio 根** |
| 6 | **Ordinary IS** | §3.6 | ⭐⭐⭐ 方差问题 |
| 7 | **Weighted IS** | §3.7 | ⭐⭐⭐ 实战默认 |
| 8 | **Per-decision IS** | §3.8 | ⭐⭐ 优化 |

---

## 3. 核心概念详解（每个从 0-1 讲清）

### 3.1 First-visit MC Prediction ⭐⭐⭐

#### 是什么
最基础 MC：估计 V_π(s)——跑 N 个 episode，每 episode 里 s **第一次出现**时记录后续 return G_t，取平均。

#### 为什么需要它（动机）
- DP 要 model → 现实少
- 第一个 **model-free** 算法——只需要 sample，不需要 p(s', r | s, a)

#### 数学

**核心思想**：
$$V_π(s) = E_π[G_t | S_t = s]$$

用 sample average 估计：
$$V(s) ≈ \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}$$

其中 $G_t^{(i)}$ 是第 i 次 episode 里 s 第一次出现后的 return。

#### 伪代码
```python
def first_visit_MC_prediction(env, policy, n_episodes=1000, gamma=0.99):
    V = defaultdict(float)
    returns = defaultdict(list)  # state -> list of G_t
    for ep in range(n_episodes):
        # 1. Generate full episode
        trajectory = []
        s = env.reset()
        done = False
        while not done:
            a = policy(s)
            s_next, r, done = env.step(a)
            trajectory.append((s, a, r))
            s = s_next
        # 2. Compute G_t backward, update first-visit
        G = 0
        visited = set()
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t]
            G = r_t + gamma * G
            # First-visit: only update on first occurrence of s_t in this episode
            if s_t not in visited:
                returns[s_t].append(G)
                V[s_t] = np.mean(returns[s_t])
                visited.add(s_t)
    return V
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| n_episodes | 1000+ |
| γ | 0.99 |
| Incremental update | $V += (G - V) / N$（增量平均省内存）|

#### 跟相关算法对比
| | MC | TD(0) | DP |
|---|---|---|---|
| Model | ❌ | ❌ | ✅ |
| Online | ❌ episode-end | ✅ | ✅ |
| Bootstrap | ❌ | ✅ | ✅ |
| Bias | 0 | 有 | 0 |
| Variance | 大 | 小 | 0 |

#### 反直觉点
- **MC 不要 model 但必须 episode 结束**——continuing task 失效
- **First-visit vs every-visit 数学性质不同**，实践相近

#### 高频面试问法
**Q：MC 跟 DP 的本质区别？**
**A**：DP 用 model（p(s', r | s, a)）算 V；MC 用 sample 平均估 V。**model-free 是 MC 的核心优势**——但代价是必须 episodic + 高方差。

---

### 3.2 Every-visit MC

#### 是什么
变种：每次 s 出现都记录 G_t（不只第一次）。

#### 为什么需要它
- First-visit 浪费数据（episode 中 s 出现多次，only 第一次用）
- Every-visit 数据效率高

#### 数学
- First-visit: 无 bias
- Every-visit: **有 bias**（同 episode 内的 G_t 相关），但 N→∞ 收敛到 V_π
- **实践中两者差异小**

#### 伪代码
```python
def every_visit_MC_prediction(env, policy, n_episodes=1000, gamma=0.99):
    # 同 first-visit，但去掉 visited 检查
    for ep in range(n_episodes):
        trajectory = generate_episode(env, policy)
        G = 0
        for t in reversed(range(len(trajectory))):
            s_t, _, r_t = trajectory[t]
            G = r_t + gamma * G
            returns[s_t].append(G)  # 每次都加，不检查 first-visit
            V[s_t] = np.mean(returns[s_t])
```

#### 反直觉点
- **Every-visit 有 bias 但 sample 效率高**——bias-variance trade-off
- **实践常用 every-visit**——bias 小到 N=1000 时无所谓

---

### 3.3 MC Control with Exploring Starts ⭐⭐⭐

#### 是什么
MC 的 control 版本——学 Q(s, a) 而非 V(s)，用 GPI（Generalized Policy Iteration）：交替 evaluation + improvement。

#### 为什么需要它
- 想学最优 policy 不只是 V
- 需要 Q(s, a) 才能决策
- **Exploring Starts** 保证全 (s, a) 覆盖（每个起始 state-action 都有 nonzero 概率）

#### 数学

**Update**:
$$Q(s, a) ← \text{mean of all returns following } (s, a)$$

**Greedy policy**:
$$\pi(s) = \arg\max_a Q(s, a)$$

#### 伪代码
```python
def MC_control_with_ES(env, n_episodes=10000, gamma=0.99):
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    returns = defaultdict(list)  # (s, a) -> list of G_t
    pi = defaultdict(lambda: np.random.randint(env.n_actions))
    for ep in range(n_episodes):
        # Exploring Start: random (s_0, a_0)
        s, a = env.reset_random(), np.random.randint(env.n_actions)
        trajectory = []
        done = False
        while not done:
            s_next, r, done = env.step(a)
            trajectory.append((s, a, r))
            s = s_next
            a = pi[s]  # subsequent: follow current pi
        # Update Q (first-visit)
        G = 0
        visited = set()
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t]
            G = r_t + gamma * G
            if (s_t, a_t) not in visited:
                returns[(s_t, a_t)].append(G)
                Q[s_t][a_t] = np.mean(returns[(s_t, a_t)])
                pi[s_t] = np.argmax(Q[s_t])  # Greedy improvement
                visited.add((s_t, a_t))
    return Q, pi
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| n_episodes | 10000+（control 收敛慢）|
| γ | 0.99 |

#### 跟相关算法对比
| | MC ES | On-policy MC (ε-soft) | Off-policy MC + IS |
|---|---|---|---|
| Exploration | 强制起点 | ε-greedy | behavior policy |
| Coverage | 全 (s,a) | 全 (s,a) via ε | 取决于 b |
| 现实可行 | ❌ 需控制起点 | ✅ | ✅ |

#### 反直觉点
- **Exploring Starts 现实少可行**——大多数环境不能任意设起点
- **GPI 思想**（eval + improve 交替）覆盖所有 RL control 算法

#### 高频面试问法
**Q：Exploring Starts 为什么需要？现实可行吗？**
**A**：保证每个 (s, a) 被无限次访问 → Q 估计收敛到 Q*。但现实多数环境不能任意起点（如真实机器人）→ 不可行 → 引入 ε-soft policy（§3.4）替代。

---

### 3.4 On-policy MC Control (ε-soft policy)

#### 是什么
用 ε-soft policy（每个 action 至少有 ε/|A| 概率）替代 Exploring Starts——天然保证 exploration。

#### 数学
ε-soft policy:
$$\pi(a|s) = \begin{cases} 1 - ε + ε/|A| & \text{if } a = \arg\max_{a'} Q(s, a') \\ ε/|A| & \text{otherwise} \end{cases}$$

—— 就是 ε-greedy。

#### 伪代码
```python
def on_policy_MC_control_eps_soft(env, n_episodes=10000, gamma=0.99, eps=0.1):
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    returns = defaultdict(list)
    for ep in range(n_episodes):
        trajectory = []
        s = env.reset()
        done = False
        while not done:
            # ε-soft policy
            if np.random.random() < eps:
                a = np.random.randint(env.n_actions)
            else:
                a = np.argmax(Q[s])
            s_next, r, done = env.step(a)
            trajectory.append((s, a, r))
            s = s_next
        # Same Q update as before
        G = 0; visited = set()
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t]
            G = r_t + gamma * G
            if (s_t, a_t) not in visited:
                returns[(s_t, a_t)].append(G)
                Q[s_t][a_t] = np.mean(returns[(s_t, a_t)])
                visited.add((s_t, a_t))
    return Q
```

#### 反直觉点
- **学到的 Q 是 ε-soft policy 下的 Q^π（不是 Q\*）**——但 ε 小时接近 Q*
- **不需要 Exploring Starts → 工程现实可行**

#### 高频面试问法
**Q：On-policy MC vs Exploring Starts MC？**
**A**：ES 强制起点保证 coverage 但现实不可行；on-policy ε-soft 用 ε-greedy 探索更现实。**代价**：on-policy 学到的是 Q^π_{ε-soft}，不是 Q*。

---

### 3.5 Off-policy MC + IS 理论 ⭐⭐⭐⭐

#### 是什么
**RLHF 数学根**——用 behavior policy b 采数据，学 target policy π 的 V/Q；用 **Importance Sampling** 修正分布差异。

#### 为什么需要它（动机）
- On-policy：学的就是采数据的 policy → 不能复用旧数据
- 想用 expert demonstration / replay buffer 学 → off-policy
- **数学问题**：sample 是 b 分布的，怎么估 π 分布的 expectation？
- → IS

#### 数学（核心）

**Importance Sampling 基本等式**：
$$E_π[f(τ)] = E_b\left[\frac{π(τ)}{b(τ)} f(τ)\right]$$

—— **任何分布 π 的 expectation 可以用另一分布 b 的加权样本估计**，权重是 likelihood ratio $\frac{π}{b}$。

**证明**：
$$E_b[ρ · f] = \sum_τ b(τ) · \frac{π(τ)}{b(τ)} · f(τ) = \sum_τ π(τ) · f(τ) = E_π[f]$$

**Trajectory IS ratio**：
$$ρ_{0:T-1} = \prod_{t=0}^{T-1} \frac{π(A_t | S_t)}{b(A_t | S_t)}$$

**注意**：环境 dynamics $p(s_{t+1} | s_t, a_t)$ 在 ratio 里**消掉**——因为 π 和 b 用同一个环境，分子分母 p 项相同。

**Off-policy MC V estimate**：
$$V_π(s) ≈ \frac{1}{N(s)} \sum_{i=1}^{N(s)} ρ^{(i)} · G^{(i)}$$

—— b 采的 G，用 ρ 加权。

#### 跟 PPO 的连接

**PPO 单步 IS**：
$$r_t = \frac{π_θ(a_t | s_t)}{π_{old}(a_t | s_t)}$$

—— 是 trajectory IS 的"单步退化"。这就是 PPO ratio 的根。

#### 反直觉点
- **环境 dynamics 不进 ratio**——大多数人第一次见会以为要乘 p_env，错
- **trajectory IS ratio 可能极大极小**——方差爆炸（详 §3.6）

#### 高频面试问法

**Q1：手推 IS 等式**
**A**：$E_b[ρ · f] = \sum_τ b(τ) · \frac{π(τ)}{b(τ)} · f(τ) = E_π[f]$。**关键**：b 必须 cover π 的 support（$b(τ) > 0$ where $π(τ) > 0$）。

**Q2：trajectory IS ratio 为什么不含环境 dynamics p？**
**A**：$ρ = \frac{π(τ)}{b(τ)} = \prod_t \frac{π(a_t|s_t) · p(s_{t+1}|s_t,a_t)}{b(a_t|s_t) · p(s_{t+1}|s_t,a_t)} = \prod_t \frac{π(a_t|s_t)}{b(a_t|s_t)}$。**环境 p 在分子分母同位置 → 消掉**。这就是为什么 IS 在 RL 上 work——不需要环境 model。

**Q3：PPO ratio 跟本章 IS 关系？**
**A**：PPO ratio $r_t = π_θ / π_{old}$ 是 trajectory IS 的**单步退化**——PPO 用 K 个 epoch 复用 rollout，相当于 off-policy + 单步 IS。**这是 RLHF 的数学源头**。

---

### 3.6 Ordinary IS ⭐⭐⭐

#### 是什么
直接 IS：$V_π(s) = \frac{1}{N(s)} \sum_i ρ^{(i)} G^{(i)}$。

#### 为什么会失败
- ratio $ρ = \prod_t π/b$ 在长 trajectory 上可能极大
- 单个 sample 可能 dominate 整体 estimate
- **方差可能无限大**（Sutton §5.5 图 5.4 经典反例）

#### 数学（方差爆炸示例）

设 π 总选 right，b 50-50。trajectory 长 T：
- 长 trajectory 概率 $b(τ) = 0.5^T$，但 $π(τ) = 1$（如果全 right）
- ratio $ρ = (1/0.5)^T = 2^T$
- 单个 sample weight $ρ · G$ 可能指数大

**方差**：
$$Var[ρ · G] = E[ρ^2 G^2] - (E[ρG])^2$$

$E[ρ^2 G^2]$ 可能发散——若 $\sum_τ b(τ) · (π(τ)/b(τ))^2 G(τ)^2 = ∞$。

#### Sutton 图 5.4 反例（必看）

设置：one-state Markov chain，b 50-50 left/right，π 总 right。
- 跑 N 个 episode，每 episode 长度按 0.5^T 衰减但 ratio 按 2^T 增长
- $E[ρG]$ 有限，但 $Var → ∞$
- **estimate 不收敛**（虽然无偏）

#### 跟 weighted IS 对比
| | Ordinary IS | Weighted IS |
|---|---|---|
| Unbiased | ✅ | ❌（有 small bias）|
| Variance | **可能 ∞** | 有限 |
| 实战默认 | ❌ | ✅ |

#### 反直觉点
- **Ordinary IS 无 bias 但方差可能 ∞**——理论好实战 useless
- **Weighted IS 有 bias 但实战 work**——经典 bias-variance trade-off
- **PPO clip 是另一种"控方差"方案**——硬截断 ratio

#### 高频面试问法
**Q：Ordinary IS 方差为什么可能 ∞？**
**A**：trajectory IS ratio $ρ = \prod π/b$ 在长 trajectory 上可指数大。若 b 跟 π 差太多 → 罕见 trajectory 有巨大 weight → variance $E[ρ²G²]$ 可发散。**实战必须用 weighted IS 或 PPO clip 控制**。

---

### 3.7 Weighted IS ⭐⭐⭐

#### 是什么
变种 IS：用 ratio 归一化分母——$V_π(s) = \frac{\sum_i ρ^{(i)} G^{(i)}}{\sum_i ρ^{(i)}}$。

#### 为什么需要它
- Ordinary IS 方差可能 ∞ → 不实用
- Weighted IS 引入 small bias 换 bounded variance → 实战 work

#### 数学

**Weighted IS estimate**：
$$V_π(s) ≈ \frac{\sum_{i=1}^{N(s)} ρ^{(i)} G^{(i)}}{\sum_{i=1}^{N(s)} ρ^{(i)}}$$

**性质**：
- **Biased**：$E[\text{weighted IS}] ≠ V_π$（有 finite-sample bias）
- **Consistent**：N → ∞ 时 bias → 0
- **Variance bounded**：在 G_t 有界时

#### 伪代码
```python
def off_policy_MC_weighted_IS(env, target_pi, behavior_b, n_episodes=10000, gamma=0.99):
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    C = defaultdict(lambda: np.zeros(env.n_actions))  # cumulative weights
    for ep in range(n_episodes):
        # Generate episode under b
        trajectory = []
        s = env.reset(); done = False
        while not done:
            a = sample(behavior_b(s))
            s_next, r, done = env.step(a)
            trajectory.append((s, a, r))
            s = s_next
        # Backward update with weighted IS
        G = 0; W = 1.0  # ratio
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t]
            G = r_t + gamma * G
            C[s_t][a_t] += W
            # Incremental weighted average
            Q[s_t][a_t] += (W / C[s_t][a_t]) * (G - Q[s_t][a_t])
            # Update target policy (greedy)
            if a_t != np.argmax(Q[s_t]):
                break  # ratio = 0
            W *= 1.0 / behavior_b(s_t)[a_t]  # π(a|s)=1 for greedy
    return Q
```

#### 跟 Ordinary IS 对比
| | Ordinary IS | Weighted IS |
|---|---|---|
| Bias | 0 | small（finite-sample）|
| Variance | 可能 ∞ | bounded |
| Consistent | ✅ | ✅ |
| **实战默认** | ❌ | **✅** |

#### 反直觉点
- **Weighted IS 有 bias 但社区默认**——bounded variance 比 unbiased 重要
- **Bias 随 N 衰减**——大数据时无所谓

#### 高频面试问法
**Q：为什么 weighted IS 是实战默认？**
**A**：ordinary IS 方差可能 ∞（不收敛）；weighted IS 用 ratio 归一化分母 → bounded variance（finite）。代价：small bias，但 N→∞ 时消失。**bias-variance 经典 trade-off**。

---

### 3.8 Per-decision IS

#### 是什么
对每步分别用 partial ratio 加权——而非整 trajectory ratio 乘到 reward。

#### 为什么需要它
- 朴素：$ρ_{0:T-1} · G_t$，整 trajectory ratio
- 观察：只有 action 之后的 reward 受 π/b 影响——之前的 reward 不需要 IS 修正
- → per-decision IS

#### 数学

**Trajectory IS**：
$$E_π[G_0] = E_b[ρ_{0:T-1} G_0]$$

**Per-decision IS**：
$$E_π[G_0] = E_b\left[\sum_{t=0}^{T-1} γ^t · ρ_{0:t} · R_{t+1}\right]$$

—— 每个 reward $R_{t+1}$ 只乘到那一步的 ratio $ρ_{0:t}$（而不是全 trajectory ρ）。

**方差降低**：后续 reward 的"未来 ratio"没相乘 → 方差小。

#### 伪代码
```python
def per_decision_IS_return(traj, pi, b, gamma):
    G = 0; rho = 1.0
    for t, (s, a, r) in enumerate(traj):
        rho *= pi(s, a) / b(s, a)
        G += (gamma ** t) * rho * r  # per-decision: ρ_{0:t}, not ρ_{0:T-1}
    return G
```

#### 跟相关算法对比
| | Trajectory IS | Per-decision IS | Weighted IS | Control variate IS |
|---|---|---|---|---|
| Bias | 0 | 0 | 有 | 0 |
| Variance | 大 | **中** | 小 | 最小 |
| 实现 | 简单 | 中 | 中 | 复杂 |

#### 反直觉点
- **Per-decision 不改 unbiasedness**——只改方差
- **跟 weighted 可组合**——bias-variance 双优化

#### 高频面试问法
**Q：per-decision IS 跟 trajectory IS 区别？**
**A**：trajectory IS 把整 trajectory ratio $ρ_{0:T-1}$ 乘到每个 reward → 方差爆；per-decision IS 只乘到那一步的 $ρ_{0:t}$——**无 bias，方差小**。

---

## 4. 跨概念对比表

### 4.1 IS family tree
```
Importance Sampling
├── Ordinary IS (无 bias, 可能 ∞ variance)
├── Weighted IS (small bias, bounded variance)  ← 实战默认
├── Per-decision IS (无 bias, 中 variance)
├── Per-decision weighted (组合)
├── Doubly Robust (control variate)  ← 进阶
└── 单步 IS = PPO ratio  ← RLHF 桥
```

### 4.2 MC 算法选型表

| 场景 | 推荐 |
|---|---|
| 评估固定 policy | First-visit MC（简单）|
| Control（学最优）| MC ES（理论）或 on-policy ε-soft（实战）|
| 用 expert data | Off-policy MC + weighted IS |
| LLM RLHF | PPO（单步 IS clip）或 GRPO（group baseline + MC）|

---

## 5. 习题精选

**Exercise 5.5（first vs every visit）**：在 10 步 episode 上比较两种 estimator。
**Exercise 5.10（weighted IS 增量）**：推导 weighted IS 增量公式。
**Exercise 5.12（Racetrack）**：实现 racetrack control —— 经典 off-policy MC 应用。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter05/blackjack.py`、`infinite_variance.py`（IS 方差爆演示）、`racetrack.py`
- **必跑** `infinite_variance.py`——亲眼看 ordinary IS 方差爆
- 自己实现：Off-policy weighted IS on small MDP

## 7. 跟 RLHF 的桥（⭐ 关键章节）

| RLHF 组件 | 来自本章 §3.x |
|---|---|
| **PPO ratio** | §3.5 trajectory IS 的单步退化 |
| **PPO clip** | §3.6 控 IS variance 的工程版 |
| **GRPO**（group MC baseline）| §3.1 first-visit MC 思想（无 bootstrap）|
| **DPO ref model** | 偏好对的 IS-free 替代 |
| **多 epoch PPO update** | off-policy + 单步 IS（rollout 重用）|

## 8. 苏格拉底 Q&A

详见每个 §3.x。综合题：

- **Q：从 trajectory IS 到 PPO ratio 完整推导链**
  - 答：trajectory IS（§3.5） → ratio 爆炸方差问题（§3.6） → weighted IS（§3.7） → PPO 单步 IS + clip（§3.6 的工程版）
- **Q：为什么 GRPO 抛弃 critic（用纯 MC baseline）work 但 PPO 必须用 critic？**
  - 答：LLM reward 稀疏 → critic 学不准 → GAE 失效 → GRPO 退守 MC（同 prompt 多 rollout 取 mean/std）。**MC 在 LLM 时代靠 GRPO 复活**。

## 9. 自测清单（闭卷）

- [ ] **写 first-visit MC prediction 算法**
- [ ] **手推 IS 等式**（$E_π = E_b[ρ · f]$）
- [ ] **解释 IS ratio 为什么不含环境 dynamics**
- [ ] **解释 ordinary IS 方差可能 ∞**
- [ ] **推导 weighted IS 增量公式**
- [ ] **解释 PPO ratio 跟本章 IS 关系**（RLHF 桥必答）
- [ ] **解释 GRPO 跟 MC 思想关系**

---

## 💡 拓展知识点（书外 trivia）

### A. Monte Carlo 名字的由来
**Stanislaw Ulam 1946** 在 Los Alamos 算原子弹链式反应——闭式解搞不动，提议用随机模拟取平均。他叔叔在 Monte Carlo 赌场赌博，于是起名 "Monte Carlo method"。后来从核物理蔓延到金融、AI——本章是 MC 在 RL 的应用。

### B. Importance Sampling 的统计学根
- **Trotter & Tukey 1956** 在统计 simulation 早提出 IS——比 RL 早 30 年
- **OR 圈子叫 "Likelihood Ratio Method"**（Glynn 1990）
- **金融期权定价**用 IS 减少 rare event 模拟成本
- RL 只是借用别人的工具

### C. 为什么 RLHF 没有"纯 MC for LLM"路线（曾经）
- LLM trajectory = 一个 response（token 序列），结束才有 RM 分——天然 episodic ✓
- 但 trajectory 短 + reward 极稀疏 → MC 估 V 方差爆
- → RLHF 用 PPO + GAE 折中（部分 bootstrap）

**反直觉**：**GRPO 实际上是 group MC**——同 prompt 跑 N 个 rollout，用 group 内归一化估 advantage（无 bootstrap、纯 MC return）。**MC 在 LLM 时代靠 GRPO 复活**。

### D. Doubly Robust Estimation
- IS 方差大；direct method bias 大
- **DR estimator** 结合两者，理论方差更低
- RL 离线评估（OPE）大量用 DR——超出 S&B 范围但 RL 评测必备
- 参考：Jiang & Li 2016 arXiv 1604.00923

### E. 反直觉数字：Blackjack
**Sutton 例 5.5**：first-visit MC 训出的 21 点最优策略，跟 1950s casinos 用人工算出的 "basic strategy" 几乎完全一致——**RL 复现人类几十年优化的策略**。

### F. PPO ratio clip 跟 weighted IS 的精神血缘
- IS ratio $ρ = π/b$ 可能极大 → 方差爆炸
- Ordinary IS：直接用，崩
- Weighted IS：分母也是 ρ 累加，有 bias 但稳
- **PPO clip(ρ, 1-ε, 1+ε)**：硬剪枝大 ρ——跟 weighted IS 异曲同工，**用 bias 换 variance**

### G. 历史上"MC vs TD"之争
1980s-90s 早期 RL 圈分两派：MC 派（无 bias）vs TD 派（高效）。Sutton 写本书时刻意把两者并列章节，宣告**两者互补不替代**。今天看：TD 主流，MC 在 LLM 时代靠 GRPO 复活——历史循环。

### H. Per-decision IS 是 RL OPE（Off-policy Policy Evaluation）的根
**Precup, Sutton, Singh 2000** *Eligibility Traces for Off-Policy Policy Evaluation* —— 把 per-decision IS 跟 eligibility traces 结合，奠定 OPE 现代理论。

### I. IS 在工业 RL 离线评估的角色
推荐系统等场景：
- 真实部署 A/B 测试贵 → 用历史 log（behavior policy）评估新 policy（target）
- 用 weighted IS / DR estimator 做 off-policy evaluation
- 决定是否上线新 policy

**Netflix / Uber 等公司每天用 IS 决策——本章理论 → 商业价值**。

---

## 🔗 更多 wikilinks（深挖）

| 主题 | 跳转 |
|---|---|
| TD 跟 MC 的 bias-variance 对比 | [[ch06-temporal-difference-learning]] §3.7 |
| n-step 是 MC ↔ TD 桥 | [[ch07-n-step-bootstrapping]] |
| PPO 用 MC 思想 | [[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning]] |
| GRPO = group MC | [[../../../brain/Papers/arxiv-2412.19437\|DeepSeek V3 报告]] |
| Off-policy 高级议题 | [[ch11-off-policy-methods]] §3.3 |
| GAE λ=1 退化为 MC | [[ch12-eligibility-traces]] §3.7 |
| AI-Infra: PPO rollout 工程 | [[../../../brain/Slipbox/prefill-decode-disaggregation]] |

## Related

- **上一章**：[[ch04-dynamic-programming]]
- **下一章**：[[ch06-temporal-difference-learning]]
- **横向**：[[ch07-n-step-bootstrapping]] · [[ch12-eligibility-traces]]
- **RLHF 桥**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning]]
- **进度**：[[_chapter-status]] W2

## Sources

- 原书 Ch 5 pp. 91-115
- [Silver UCL Lec 4: Model-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA)
- ShangtongZhang `chapter05/`
- **核心论文**：
  - PPO 论文 [Schulman 2017](https://arxiv.org/abs/1707.06347) —— 引用本章 IS
  - Precup, Sutton, Singh 2000 —— Per-decision IS for OPE
  - Glynn 1990 —— Likelihood Ratio Method
  - Jiang & Li 2016 [arXiv 1604.00923](https://arxiv.org/abs/1604.00923) —— Doubly Robust OPE

---

## 💀 Top 3 Gotchas (v4)

> 最容易 fool 学习者的陷阱——面试 / 实战常踩。

### 💀 Gotcha 1：环境 dynamics 在 IS ratio 里**消掉**了

trajectory IS ratio 推导：
$$
\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k) \cdot p(S_{k+1}|S_k, A_k)}{b(A_k|S_k) \cdot p(S_{k+1}|S_k, A_k)} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
$$

**`p(s'|s,a)` 在分子分母都出现 → 消掉**——所以 IS **不需要 model**。常见错：把 transition prob 也乘进去，纯多余。

### 💀 Gotcha 2：Ordinary IS 方差**可能无限大**

Ordinary IS 估计是 unbiased，但 $\text{Var}(\rho_{t:T-1} \cdot G_t)$ 在 long horizon 下**爆炸**——一两个极端 ratio (~10^4) 就撕方差。

实战只用 **weighted IS**（虽然有 bias 但 variance 有界，N→∞ 也收敛）。教科书定义先讲 ordinary 不代表 practice 用它。

### 💀 Gotcha 3：MC **不能** continuing task

MC 必须等 $G_t$ 才能 update → 必须 episode 终止 → continuing task **死路**。

如果你 task 是 continuing → 一定走 TD（bootstrap 不需要 terminal）。这就是为什么 robotic continuous control（无终止）几乎都用 SAC/PPO 而不是 MC-based PG。

## 链回

- [[_anki/ch05-mc-cards]] — 35 张卡片版（Anki 复习）
- [[../../Code/sutton-barto-2e/_v4_notebooks/]] — 数值 trace notebook
