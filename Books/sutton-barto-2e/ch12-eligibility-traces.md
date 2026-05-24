---
title: "Sutton & Barto Ch 12. Eligibility Traces"
created: 2026-05-24
tags: [area/rl, type/note, status/v3, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 12
difficulty: medium
budget_days: 3
priority_for_rlhf: must
silver_lecture: 4-5
---

# Ch 12. Eligibility Traces ⭐⭐⭐

---

## 📋 本章讲了什么（TL;DR）

1. **TD(λ) 是 TD(0) 和 MC 的连续插值**——λ ∈ [0, 1] 平滑混合所有 n-step return
2. **Backward view 用 eligibility trace** 实现 forward view 的等价 update——单 pass O(1) 无需 buffer
3. **λ-return 是 GAE 的直接前置**——读完本章直接看懂 [[ch13-policy-gradient|PPO]] advantage 估计

**为什么 RLHF 必读**：**GAE = TD(λ) 的 advantage 版**。PPO 默认 λ=0.95——本章是 PPO 工程实现的直接前置。

## 🎯 Top 5 关键洞察

| # | 洞察 | 为什么关键 |
|---|---|---|
| 1 | **λ=0 = TD(0)，λ=1 = MC** | 一个公式覆盖整个 TD-MC 谱 |
| 2 | **Forward / backward view 等价**（offline）| 数学优雅 + 实现可行 |
| 3 | **Eligibility trace z_t** = "最近活跃 feature 的 credit weight" | 单 pass O(1) update 的关键数据结构 |
| 4 | **True online TD(λ)** 修正 backward 的 in-trajectory bias | 等价于 forward——最严格版 |
| 5 | **GAE = λ-加权 TD errors** | 把 TD(λ) 思想用到 advantage estimation，PPO 默认 |

## 🧭 在 Obsidian graph 中的位置

- **上游**：[[ch06-temporal-difference-learning]]（TD(0)）· [[ch07-n-step-bootstrapping]]（固定 n 版本）
- **下游**：**[[ch13-policy-gradient]]**（GAE 在 PG 里）
- **RLHF 联动**：PPO 的 advantage = GAE = TD(λ) 思想
- **横向**：[[ch05-monte-carlo-methods]]（λ=1 退化为 MC）

---

## 0. 阅读元信息

- **难度**：medium（概念优雅但数学密集）
- **预算**：3 天 / §12.1-12.3 必读
- **必读理由**：**GAE 的直接前置**——理解 TD(λ) 才理解 PPO advantage
- **配套**：Silver UCL Lec 4-5

## 1. 一句话 + 在 RL 体系中的位置

**Eligibility traces = TD(0) 和 MC 的连续插值** + 单 pass O(1) 实现的优雅形式——λ ∈ [0,1] 控制 bias-variance；GAE 在 advantage 维度复用这思想。

**在 RL 算法家族中的位置**：
```
信用分配（Credit Assignment）
├── MC（看全程，无 bootstrap）
├── TD(0)（看一步，全 bootstrap）
├── n-step TD（看 n 步，部分 bootstrap）
└── TD(λ) ⭐ —— 平滑混合所有 n
    ├── Forward view（用 λ-return 当 target）
    └── Backward view（用 eligibility trace 单 pass）
```

## 2. 概念地图（§3 详解目录）

| # | 概念 | §3.x | 重要度 |
|---|---|---|---|
| 1 | **λ-return** | §3.1 | ⭐⭐⭐ |
| 2 | **TD(λ) Forward View** | §3.2 | ⭐⭐⭐ |
| 3 | **TD(λ) Backward View + Eligibility Trace** | §3.3 | ⭐⭐⭐⭐ |
| 4 | **True Online TD(λ)** | §3.4 | ⭐⭐ |
| 5 | **SARSA(λ)** | §3.5 | ⭐⭐ control 版本 |
| 6 | **Watkins's Q(λ)** | §3.6 | ⭐ off-policy 版 |
| 7 | **GAE (Generalized Advantage Estimation)** | §3.7 | ⭐⭐⭐⭐ **PPO 关键** |

---

## 3. 核心概念详解（每个从 0-1 讲清）

### 3.1 λ-return ⭐⭐⭐

#### 是什么
**n-step return 的指数加权平均**：
$$G_t^λ \doteq (1-λ) \sum_{n=1}^∞ λ^{n-1} G_{t:t+n}$$

#### 为什么需要它（动机）
- n-step return（[[ch07-n-step-bootstrapping]]）固定 n 不灵活
- 不同 n 的 return 都有信息——能否加权合并？
- λ 控制加权 → 一个公式覆盖整个 TD-MC 谱

#### 数学

**完整公式（含 episode end 截断）**：
$$G_t^λ = (1-λ) \sum_{n=1}^{T-t-1} λ^{n-1} G_{t:t+n} + λ^{T-t-1} G_t$$

- 前项：n=1, 2, ..., T-t-1 的 n-step return 加权
- 后项：完整 G_t 占剩余权重 λ^{T-t-1}

**极限**：
- **λ=0** → 只有 n=1 项 → $G_t^0 = G_{t:t+1} = R + γV(s')$ = **TD(0) target**
- **λ=1** → 后项主导 → $G_t^1 = G_t$ = **MC return**
- **中间 λ** → 平滑混合

#### 为什么权重是 $(1-λ)λ^{n-1}$？

**几何分布权重**：
- 权重总和：$(1-λ) \sum_{n=1}^∞ λ^{n-1} = 1$（标准化）
- n 越大权重越小（指数衰减）
- λ 越大权重分布越"长尾"（更接近 MC）

#### 伪代码（offline λ-return）
```python
def compute_lambda_return(rewards, values, gamma=0.99, lam=0.95):
    """rewards[0..T-1], values[0..T-1] (V(s_T) = 0 if terminal)"""
    T = len(rewards)
    # 1. Compute all n-step returns G_{t:t+n} for all t, n
    G = np.zeros((T, T - 0 + 1))  # G[t, n] = G_{t:t+n}
    for t in range(T):
        for n in range(1, T - t + 1):
            G_tn = sum(gamma ** k * rewards[t + k] for k in range(n))
            if t + n < T:
                G_tn += gamma ** n * values[t + n]
            G[t, n] = G_tn

    # 2. λ-return
    G_lambda = np.zeros(T)
    for t in range(T):
        for n in range(1, T - t):
            G_lambda[t] += (1 - lam) * (lam ** (n - 1)) * G[t, n]
        G_lambda[t] += lam ** (T - t - 1) * G[t, T - t]  # MC tail
    return G_lambda
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| λ | 0.5 - 0.95 |
| **PPO 默认 λ** | **0.95** |
| γ | 0.99 |

#### 跟相关算法对比
| 估计 | 公式 | bias-variance |
|---|---|---|
| TD(0) target | $R + γV(s')$ | 大 bias 小方差 |
| n-step | $G_{t:t+n}$ | 中 |
| **λ-return** | 加权平均所有 n | **可调 (λ 控制)** |
| MC return | $G_t$ | 0 bias 大方差 |

#### 反直觉点
- **λ 不是越大越好**——λ=1 (MC) 方差爆，实践中 0.5-0.95 最优
- **λ 是个"信号纯度 vs 噪声"的旋钮**——λ 大信号多噪声大
- **PPO 调 λ 比调其它超参影响大**——直接影响 advantage 质量

#### 高频面试问法
**Q：λ-return 跟 n-step return 区别？**
**A**：n-step 固定 n（如 n=5）；λ-return 是**所有 n-step 的指数加权平均**（weights $(1-λ)λ^{n-1}$）。**λ 替代固定 n，一个超参覆盖整个 TD-MC 谱**。

---

### 3.2 TD(λ) Forward View ⭐⭐⭐

#### 是什么
用 λ-return 当 target 的 TD update：
$$w_{t+1} = w_t + α [G_t^λ - v̂(S_t, w_t)] ∇v̂(S_t, w_t)$$

#### 为什么需要它
- 直觉上更优雅——一个 update target 平滑混合所有 n-step
- bias-variance 比固定 n 好调

#### 数学
跟 §3.1 λ-return 配合 §3.2 的 TD update 即可：
- target = $G_t^λ$
- update 朝 target 走 α 步

#### 实现痛点
**Forward view 是"非因果的"**——computing $G_t^λ$ 需要 future rewards 和 V(s_{t+1}), V(s_{t+2}), ...

**必须等 episode 结束**才能算所有 $G_t^λ$ → 不能 online。

**解决**：backward view + eligibility trace（§3.3）实现 online 等价。

#### 伪代码（offline forward TD(λ)）
```python
def forward_TD_lambda(env, policy, v_hat, w, alpha=0.01, gamma=0.99, lam=0.95, n_episodes=100):
    for ep in range(n_episodes):
        # 1. Rollout
        states, rewards = [], []
        s = env.reset(); done = False
        while not done:
            a = policy(s)
            s_next, r, done = env.step(a)
            states.append(s); rewards.append(r)
            s = s_next
        # 2. Compute λ-returns (需要 future, 所以 offline)
        values = [v_hat(s, w) for s in states]
        G_lambda = compute_lambda_return(rewards, values, gamma, lam)
        # 3. SGD on (G_λ - v̂)²
        for s_t, G_t in zip(states, G_lambda):
            w += alpha * (G_t - v_hat(s_t, w)) * v_hat.grad(s_t, w)
    return w
```

#### 跟相关算法对比
| | Forward TD(λ) | Backward TD(λ) | TD(0) |
|---|---|---|---|
| Online | ❌ 必须 episode 结束 | ✅ 单 pass | ✅ |
| Update target | $G_t^λ$ | TD error × trace | $R + γV(s')$ |
| Equivalence | offline = forward | offline = forward | n/a |

#### 反直觉点
- **Forward view 是"概念上的"——实现需要 backward**
- **offline forward = backward 严格等价**——但 in-trajectory 微差

#### 高频面试问法
**Q：Forward TD(λ) 跟 Backward TD(λ) 是什么关系？**
**A**：数学等价（offline / episode-end），但**forward 不能 online**（需 future rewards），backward 用 eligibility trace 实现 online 同样的 update。**两者像同一公式的两种实现**。

---

### 3.3 Backward View + Eligibility Trace ⭐⭐⭐⭐

#### 是什么
**单 pass O(1) 实现 TD(λ)** 的关键 trick——维护一个 trace $z_t$ 记录"最近活跃的 feature"，TD error 反向分配 credit。

#### 为什么需要它（动机）
- Forward view 不能 online
- 需要"每步立即更新" 但效果等价 forward
- → eligibility trace

#### 数学

**Eligibility trace 递推**：
$$z_t = γλ z_{t-1} + ∇v̂(S_t, w_t)$$

**TD update**：
$$δ_t = R_{t+1} + γv̂(S_{t+1}, w) - v̂(S_t, w)$$
$$w_{t+1} = w_t + α δ_t z_t$$

**直觉**：
- $z_t$ 记录"哪些 feature 最近被访问 + 加权"——访问越近权重越大
- $δ_t$ 是当前 step 的"惊喜"——朝这方向更新所有 trace 上的 feature

#### Linear FA 简化

$v̂(s, w) = w^T x(s)$，$∇v̂ = x(s)$：

$$z_t = γλ z_{t-1} + x(S_t)$$
$$w_{t+1} = w_t + α δ_t z_t$$

**几何**：trace 是"feature 累加的指数衰减总和"——每步加新 feature，旧 feature 衰减 γλ。

#### Episode 边界
**Episode 结束时 z 必须 reset 为 0**——否则 cross-episode credit 漏。

#### 伪代码（Linear TD(λ) Backward）
```python
def linear_backward_TD_lambda(env, policy, n_features, alpha=0.01, gamma=0.99, lam=0.95, n_episodes=1000):
    w = np.zeros(n_features)
    for ep in range(n_episodes):
        z = np.zeros(n_features)  # trace reset per episode
        s = env.reset()
        x = feature(s)
        done = False
        while not done:
            a = policy(s)
            s_next, r, done = env.step(a)
            x_next = feature(s_next) if not done else np.zeros(n_features)
            # 1. Update trace
            z = gamma * lam * z + x
            # 2. TD error
            delta = r + gamma * w @ x_next - w @ x
            # 3. Update w using trace
            w += alpha * delta * z
            # 4. Move
            x = x_next
    return w
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| λ | 0.5 - 0.9 |
| α | 0.01 - 0.1 |
| γ | 0.99 |

#### 跟相关算法对比
| | TD(0) | n-step TD | Backward TD(λ) | Forward TD(λ) |
|---|---|---|---|---|
| Online | ✅ | 延迟 n | **✅** | ❌ |
| Memory | O(1) | O(n) | O(d) trace | O(T) episode |
| 等价于 | n/a | n/a | offline = forward | (concept) |

#### 反直觉点
- **trace z_t 是向量**（跟 w 同 shape），不是 scalar
- **NN 时 trace 同 shape 跟 NN 参数一样** → 巨大（百万级）→ 实践少用
- **Forward / backward 数学等价（offline）**：episode 末同等更新；但 in-trajectory 有微小 bias

#### 高频面试问法

**Q1：Eligibility trace z_t 是什么？为什么这样递推？**
**A**：z_t = γλ z_{t-1} + ∇v̂(S_t)，记录"最近活跃 feature 的累积衰减权重"。访问越近权重越大（exp 衰减 γλ）。**TD error 反向乘 z_t 一次分发给所有最近 active feature**——单 pass 实现 forward 的等价 update。

**Q2：为什么 z 是 trace 而非 buffer？**
**A**：buffer 存历史 features 需要 O(T) 内存；trace 把 weighted sum 压缩成单个 vector，O(d) 内存。**数学上"曾经活跃的 feature 现在该多大权重"的紧凑表示**。

---

### 3.4 True Online TD(λ)

#### 是什么
**van Seijen 2014** 提出——修正 backward TD(λ) 在 in-trajectory 的 bias，实现严格等价 forward。

#### 为什么需要它（动机）
- 普通 backward TD(λ) 跟 forward 只在 episode-end 严格等价
- In-trajectory 有微小 bias（来自 w 在更新过程中变化）
- True online 修正这个 bias

#### 数学（精简版）

$$\delta_t = R_{t+1} + γv̂(S_{t+1}) - v̂(S_t)$$
$$z_t = γλ z_{t-1} + (1 - α γλ z_{t-1}^T x_t) x_t$$（**dutch trace**，跟普通 trace 不同）
$$w_{t+1} = w_t + α δ_t z_t + α (z_t - α z_t^T x_t · x_t)(w_t^T x_t - w_{t-1}^T x_t)$$

——多了一个修正项让 in-trajectory 严格 = forward view。

#### 跟普通 backward TD(λ) 对比
| | Backward TD(λ) | True Online TD(λ) |
|---|---|---|
| Episode-end 等价 forward | ✅ | ✅ |
| In-trajectory 等价 forward | ≈ | **✅ 严格** |
| 实现复杂度 | 简单 | 中（dutch trace + 修正）|
| 计算量 | O(d) | O(d) |
| 实战提升 | baseline | 微（5-10%） |

#### 反直觉点
- **理论完美 vs 工程复杂**——True online 严格但代码长 2x
- **实战大多用普通 backward**——bias 小到无所谓

#### 高频面试问法
**Q：True online TD(λ) 比 普通 backward 强在哪？**
**A**：严格等价 forward view（不只 episode-end，每步都等价）。代价：dutch trace + 修正项实现复杂。**实战提升微（5-10%），多数实现选普通 backward**。

---

### 3.5 SARSA(λ)

#### 是什么
TD(λ) 的 control 版本——SARSA + eligibility trace。

#### 为什么需要它
- SARSA 信用分配只到 1 步 → 学慢
- 加 trace 让 reward 信号传到更早 step → 学快

#### 数学

$$z_t = γλ z_{t-1} + ∇q̂(S_t, A_t, w)$$
$$δ_t = R_{t+1} + γq̂(S_{t+1}, A_{t+1}, w) - q̂(S_t, A_t, w)$$
$$w_{t+1} = w_t + α δ_t z_t$$

#### 伪代码
```python
def SARSA_lambda(env, n_features, alpha=0.1, gamma=0.99, lam=0.9, eps=0.1, n_episodes=500):
    w = np.zeros(n_features)
    for ep in range(n_episodes):
        z = np.zeros(n_features)
        s = env.reset()
        a = epsilon_greedy(q_hat, s, eps, w)
        x = feature(s, a)
        done = False
        while not done:
            s_next, r, done = env.step(a)
            if not done:
                a_next = epsilon_greedy(q_hat, s_next, eps, w)
                x_next = feature(s_next, a_next)
            else:
                x_next = np.zeros(n_features)
            z = gamma * lam * z + x  # trace
            delta = r + gamma * w @ x_next - w @ x
            w += alpha * delta * z
            x, a = x_next, a_next if not done else None
    return w
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| λ | 0.9（control 通常大些）|
| α | 0.1 |
| ε | 0.1 |

#### 反直觉点
- **SARSA(λ=1) ≈ MC control**
- **Mountain Car 上 SARSA(λ=0.9) + tile coding 是经典 baseline**

---

### 3.6 Watkins's Q(λ)

#### 是什么
Q-learning 的 λ 版本——off-policy + eligibility trace。但**遇到 non-greedy action 必须 reset trace**（截断）。

#### 为什么需要它
- Q-learning 学 Q*，target 用 max_a（off-policy）
- 但 trace 是 on-policy 累加的——一旦 behavior 选了非 greedy action，过去的 trace 就**跟 target policy 不一致**
- → reset trace

#### 数学
$$z_t = \begin{cases} γλ z_{t-1} + ∇q̂(S_t, A_t) & \text{if } A_{t-1} = \arg\max_a q̂(S_{t-1}, a) \\ ∇q̂(S_t, A_t) & \text{otherwise (reset)} \end{cases}$$

#### 反直觉点
- **Trace 频繁 reset → 效率低**——ε-greedy 时大约每 1/ε 步就 reset
- **Tree Backup (Sutton §12.10)** 是替代——无需 IS 无需 reset

#### 跟相关算法对比
| | SARSA(λ) | Watkins Q(λ) | Tree Backup(λ) |
|---|---|---|---|
| On/off-policy | on | off | off |
| Trace reset | 无 | 频繁 | 无 |
| 实战 | 主流 | 罕用 | 罕用 |

#### 高频面试问法
**Q：为什么 Q(λ) 要 reset trace？**
**A**：Q-learning 的 target 是 greedy（max_a），如果 behavior 选了 non-greedy action（探索），那么过去 trace 上的"假设 on-policy"的累加就跟 target 不符——必须 reset 让 trace 重新只覆盖 greedy 之后的 step。

---

### 3.7 GAE (Generalized Advantage Estimation) ⭐⭐⭐⭐

#### 是什么
**Schulman 2015**（arXiv 1506.02438）：把 TD(λ) 思想用到 **advantage estimation**——λ-加权所有 n-step advantage。**PPO 默认 λ=0.95**。

#### 为什么需要它（动机）
- PG 算法（[[ch13-policy-gradient]]）需要 advantage $A_t$ 估计
- TD(0) advantage（n=1）：bias 大方差小
- MC advantage（n=∞）：无 bias 方差大
- → GAE：用 λ 平滑混合所有 n，连续可调 bias-variance

#### 数学推导

**Step 1**：定义 TD error
$$δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)$$

**Step 2**：n-step advantage 可写成 δ 的累加
$$A_t^{(n)} = \sum_{l=0}^{n-1} γ^l δ_{t+l}$$

（用伸缩和：$\sum γ^l δ_{t+l} = \sum γ^l R_{t+l+1} + γ^n V(S_{t+n}) - V(S_t)$）

**Step 3**：GAE = λ-加权 n-step advantages
$$A_t^{GAE(γ, λ)} = (1-λ) \sum_{n=1}^∞ λ^{n-1} A_t^{(n)}$$

**化简**（关键代数）：
$$A_t^{GAE} = \sum_{l=0}^∞ (γλ)^l δ_{t+l}$$

——**简洁形式**：所有 future TD errors 的 γλ-加权累加。

**极限**：
- λ=0 → $A_t^{GAE} = δ_t$（TD(0) advantage）
- λ=1 → $A_t^{GAE} = G_t - V(S_t)$（MC advantage，无 bootstrap）

#### 伪代码（backward sweep，O(T)）
```python
def compute_GAE(rewards, values, gamma=0.99, lam=0.95):
    """逆向计算 GAE，O(T) 单 pass
    rewards[0..T-1]: trajectory rewards
    values[0..T]: V(s_0), V(s_1), ..., V(s_T) (V(s_T) = 0 if terminal)
    """
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae  # recursion
        advantages[t] = gae
    returns = advantages + values[:T]  # for critic training
    return advantages, returns
```

**为什么 backward sweep work**：
$$A_t^{GAE} = δ_t + γλ · A_{t+1}^{GAE}$$

—— 递推关系！从 T-1 倒着算到 0。

#### PPO 中怎么用 GAE

```python
# In PPO training step
def PPO_GAE_workflow(rollout, critic, actor, optimizer):
    # 1. Compute V(s) for all states in rollout
    with torch.no_grad():
        values = critic(rollout.states)  # (T+1,)
    # 2. Compute GAE
    advantages, returns = compute_GAE(rollout.rewards, values, gamma=0.99, lam=0.95)
    # 3. Normalize advantages (CRITICAL for stability)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # 4. PPO loss
    ratio = exp(actor.log_prob(rollout.states, rollout.actions) - rollout.old_log_probs)
    surr1 = ratio * advantages
    surr2 = clip(ratio, 1 - eps, 1 + eps) * advantages
    loss_clip = -min(surr1, surr2).mean()
    # 5. Critic loss
    loss_v = ((critic(rollout.states) - returns) ** 2).mean()
    # 6. Total
    loss = loss_clip + 0.5 * loss_v - 0.01 * entropy
    loss.backward(); optimizer.step()
```

#### 实战参数（PPO 默认）
| 超参 | 推荐 | 直觉 |
|---|---|---|
| **λ** | **0.95**（PPO 默认）| 接近 MC 但仍 bootstrap |
| γ | 0.99 | 长视野 |
| **Normalize advantages** | ✅ 必须 | (减均值除标准差) — 训练稳定 |

#### 跟相关概念对比
| Advantage estimator | 公式 | bias | variance |
|---|---|---|---|
| TD(0) advantage | $δ_t$ | 大 | 小 |
| n-step advantage | $A_t^{(n)}$ | 中 | 中 |
| MC advantage | $G_t - V(s_t)$ | 0 | 大 |
| **GAE λ=0.95** | $\sum (γλ)^l δ_{t+l}$ | 小 | 中（**甜点**）|

#### 反直觉点
- **λ=1 不是 GAE 论文推荐**——尽管 unbiased，方差太大反而劣于 λ=0.95
- **Normalize advantages 极其重要**——不 normalize 训练经常崩
- **TD(λ) 跟 GAE 公式一模一样**——只是 GAE 用 δ 估 advantage 而非学 V
- **GAE 的 backward sweep 是 RL 工程师必会代码**

#### 高频面试问法

**Q1：手写 GAE 公式 + 推导**
**A**：$A_t^{GAE(γ, λ)} = \sum_{l=0}^∞ (γλ)^l δ_{t+l}$。推导：n-step advantage = $\sum γ^l δ$（伸缩和）；GAE = 这些 n-step 的 λ-加权 = $\sum (γλ)^l δ$。

**Q2：GAE 跟 TD(λ) 关系？**
**A**：**数学上完全相同形式** $\sum (γλ)^l δ_{t+l}$。区别：TD(λ) 用 δ 学 V；**GAE 用 δ 估 advantage 给 PG 算 actor 梯度**。**思想同源，应用不同**。

**Q3：PPO λ 一般取 0.95，为什么不 1（无 bias）也不 0（minimum variance）？**
**A**：λ=1 → 退化 MC，方差爆 → PG 训练崩；λ=0 → TD(0)，bias 大 → 学不到长视野信号；**λ=0.95 接近 MC 但仍少量 bootstrap → bias-variance 甜点**。

**Q4：GAE 怎么 O(T) 高效实现？**
**A**：递推 $A_t^{GAE} = δ_t + γλ A_{t+1}^{GAE}$，从 T-1 倒着算到 0，单 pass O(T)。**这是 RL 工程师必会代码**。

**Q5：reward 稀疏（LLM 末尾给 1 次 RM 分）时 GAE 怎么算？**
**A**：rewards 大部分是 0，只末尾非 0。δ 大部分 = γV(s') - V(s)（无 reward 贡献）→ GAE 主要由 V 函数差驱动 → critic 质量决定 advantage 质量。**这是 RLHF critic 难学的根**——也是 GRPO 抛弃 critic 改用 group MC 的动机。

---

## 4. 跨概念对比表

### 4.1 本章算法 family tree
```
λ-return (§3.1)
└── TD(λ) Forward (§3.2)  ← 概念优雅，offline 实现
    └── Backward + Eligibility Trace (§3.3)  ← online 等价
        ├── True Online TD(λ) (§3.4)  ← 严格等价（in-trajectory）
        ├── SARSA(λ) (§3.5)  ← control 版
        ├── Q(λ) (§3.6)  ← off-policy（reset trace）
        └── GAE (§3.7) ⭐⭐⭐⭐  ← advantage 版，PPO 默认
```

### 4.2 λ vs n 选型表

| 场景 | 推荐 |
|---|---|
| Tabular prediction | TD(0) 或 TD(λ=0.5) |
| FA prediction | TD(λ=0.7-0.9) |
| **PPO advantage** | **GAE λ=0.95** ⭐ |
| MC alternative | TD(λ=1.0) |
| Long-CoT RLHF reasoning | GAE λ→1（reward 稀疏不用 bootstrap）|

---

## 5. 习题精选

**Exercise 12.1**：证明 G_t^λ 是 n-step return 的指数加权平均（公式验证）。
**Exercise 12.5（forward / backward 等价证明）**：证明 episode-end 时两者产生相同总 update。
**Exercise 12.10（true online TD(λ)）**：实现 + 对比普通 backward。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter12/random_walk.py`（TD(λ) vs n-step）、`mountain_car.py`（SARSA(λ)）
- **GAE 实战参考**：CleanRL 的 PPO 实现里的 `compute_gae` 函数（10 行）
- **必看**：[GAE 论文 1506.02438](https://arxiv.org/abs/1506.02438) 确认本章公式如何变形为 advantage

## 7. 跟 RLHF 的桥（⭐ 关键章节）

| RLHF 组件 | 来自本章 §3.x |
|---|---|
| **PPO advantage 估计** | **§3.7 GAE** |
| PPO λ=0.95 选择 | §3.7 实战参数 |
| RLHF 长 CoT 稀疏 reward 怎么算 advantage | §3.7 高频问 Q5 |
| GRPO 抛弃 critic | §3.7 反直觉——GAE 失效场景 |
| n-step 思想在 RLHF | §3.1 λ-return 是底层 |

## 8. 苏格拉底 Q&A

详见每个 §3.x。综合题：

- **Q：手写 GAE 计算 backward sweep 代码**
  - 见 §3.7 伪代码
- **Q：从 TD(0) 到 GAE 完整推导链**
  - TD(0) δ_t → n-step adv = Σγ^l δ → λ-加权 GAE = Σ(γλ)^l δ
- **Q：为什么 RLHF critic 难学？跟本章关系？**
  - reward 末尾给 → δ 大部分 = γV(s') - V(s) → 几乎无 reward signal → critic 学不准 → GAE 质量差

## 9. 自测清单（闭卷）

- [ ] **写出 λ-return 公式**（含截断尾项）
- [ ] **写出 TD(λ) backward 算法**（含 z 更新）
- [ ] **解释 forward / backward 等价性**
- [ ] **写出 GAE 公式 + backward sweep 代码** ⭐
- [ ] **解释 GAE 跟 TD(λ) 的关系**
- [ ] **解释 PPO 用 λ=0.95 的工程理由**
- [ ] **解释 LLM RLHF 稀疏 reward 下 GAE 失效原因 → GRPO 动机**

---

## 💡 拓展知识点（书外 trivia）

### A. Eligibility trace 的生物学灵感
"Eligibility" 来自神经生物学——突触 "eligible" for plasticity（可塑性）的概念。Sutton 1980s 借用心理学术语。**RL 跟神经科学的双向影响**。

### B. λ-return 的另一种推导：n-step return 的指数贴现
λ 是"贴现率"——离当下越远的 n-step return 权重越小（exp 衰减）。这跟金融的现值贴现思想相同——**RL 跟金融的隐藏交集**。

### C. True Online TD(λ) 论文的"惊喜"
**van Seijen 2014** *True Online TD(λ)* 论文证明普通 backward TD(λ) **不严格等价 forward**——这之前社区以为是等价的。RL 教材 1998 第一版有错，2018 第二版修正。**经典"教材也会错"的例子**。

### D. GAE 的 λ=0.95 是怎么调出来的
GAE 论文（Schulman 2015）实验扫 λ=0.9, 0.92, 0.95, 0.97, 0.99 在 MuJoCo 上——发现 0.95 最稳。之后 PPO 沿用，成默认。**社区魔术数字**。

### E. PPO 实现里 advantage normalization 的关键性
不 normalize advantage 是 PPO 训不起来的 #1 原因——但论文里**没强调**。CleanRL 的代码里特别注释这点。**RL 工程的"隐藏知识"**。

### F. GRPO 为什么抛弃 GAE/critic
- LLM RLHF reward 极稀疏（末尾给 1 次）
- critic 学不到稳定 V → GAE = Σ(γλ)^l δ 主要由 V 函数差驱动 → 噪声大
- GRPO：用 group 内 sample-based 归一化 advantage（纯 MC 思想）→ 跳过 V learning

**这是 RLHF/RL 圈 2024 重大范式转变**——本章 GAE 不再是唯一选择。

### G. Eligibility trace 在 NN RL 不主流的原因
- trace 跟 NN 参数同 shape（百万级）
- 维护 trace 的内存大
- backward sweep 在 SGD framework 不自然
- 所以现代 deep RL（PPO/SAC）多用 **GAE backward sweep 替代 eligibility trace**

### H. Watkins Q(λ) 被 deep RL 抛弃
原因：trace reset 太频繁（ε-greedy ε=0.1 时 ~1/10 reset rate）→ 效率低。Tree Backup（§12.10）虽更好但实现复杂。**Deep Q-learning 直接走 1-step TD + replay 路线**。

### I. Backward sweep 在工程上的意外重要性
- GAE backward sweep 是 PPO 工程代码里**最容易写错的部分**
- 常见 bug：忘了 episode boundary 重置；values[T] 未处理；advantages 反向 indexing 出错
- CleanRL 的 5 行 backward sweep 是 RL 工程师"必抄"代码

---

## 🔗 更多 wikilinks（深挖）

| 主题 | 跳转 |
|---|---|
| n-step 前置 | [[ch07-n-step-bootstrapping]] |
| GAE 在 PPO 完整应用 | [[ch13-policy-gradient]] §3.6, §3.8 |
| GRPO 抛弃 critic | [[ch13-policy-gradient]] §3.9 |
| RLHF 稀疏 reward 推理 | [[../../../brain/Areas/rl-books/rlhf-lambert/ch07-reasoning-inference-time-scaling]] |
| TD error 在 critic 训练 | [[ch06-temporal-difference-learning]] §3.1 |
| AI-Infra: PPO rollout + GAE 工程 | [[../../../brain/Slipbox/prefill-decode-disaggregation]] |

## Related

- **上一章**：[[ch11-off-policy-methods]]
- **下一章**：[[ch13-policy-gradient]]（GAE 应用）
- **n-step 前置**：[[ch07-n-step-bootstrapping]]
- **RLHF 桥**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning]]
- **进度**：[[_chapter-status]] W6

## Sources

- 原书 Ch 12 pp. 287-321
- **GAE 论文（Schulman 2015）** [arXiv 1506.02438](https://arxiv.org/abs/1506.02438) —— 直接应用
- [Silver UCL Lec 4-5](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- ShangtongZhang `chapter12/`
- True Online TD(λ): van Seijen 2014 *True Online TD(λ)*
- PPO 论文（Schulman 2017）—— 用 GAE
