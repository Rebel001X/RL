---
title: "Sutton & Barto Ch 9. On-policy Prediction with Approximation"
created: 2026-05-24
tags: [area/rl, type/note, status/v3, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 9
difficulty: medium
budget_days: 3
priority_for_rlhf: must
silver_lecture: 6
---

# Ch 9. On-policy Prediction with Approximation ⭐⭐⭐

---

## 📋 本章讲了什么（TL;DR）

1. **从 tabular V(s) 升级到 v̂(s, w) 参数化**——线性 / NN 替代查表，**deep RL 的入口**
2. **Semi-gradient TD** 是个"伪梯度"：用 R + γv̂(S', w) 当 target，但**不对 target 取梯度**——这是 NN-based RL 的本质 trick
3. **on-policy + linear FA + bootstrap 收敛到 TD fixed point**——为什么 Ch 11 deadly triad 突然崩，因为加 off-policy 这一刀让 contraction 消失

**为什么必读**：所有 deep RL（DQN/PPO/SAC）都建立在本章 §9.4 的 projection operator 基础上。**PPO 的 critic V_φ 就是本章 v̂——这是 RLHF 数学链关键一环**。

## 🎯 Top 5 关键洞察

| # | 洞察 | 为什么关键 |
|---|---|---|
| 1 | **Mean Squared Value Error (VE) 用 μ(s) 加权** | 不是 uniform error——访问多的 state 误差影响大 |
| 2 | **Semi-gradient 不对 target 取梯度** | 是 TD 的"伪梯度"——破坏 true SGD 但收敛性证明仍 work |
| 3 | **Linear + on-policy 收敛到 TD fixed point** | TD fp 比 VE 最优差 1/(1-γ) 倍——可控 |
| 4 | **§9.4 Projection operator Π** 是 Ch 11 前置 | 把 BV 投影回 v̂ subspace——核心理论工具 |
| 5 | **Bootstrap 在 FA 下危险** | 误差被 v̂ 放大——为 deadly triad 埋伏笔 |

## 🧭 在 Obsidian graph 中的位置

- **上游**：[[ch06-temporal-difference-learning]]（tabular TD）· [[ch08-planning-and-learning]]（DP backup）
- **横向**：[[ch10-on-policy-control]]（prediction → control 升级）
- **下游**：**[[ch11-off-policy-methods]]**（deadly triad 主战场）· [[ch13-policy-gradient]]（NN policy + NN critic）
- **DQN 联想**：本章是 DQN 的理论根

---

## 0. 阅读元信息

- **难度**：medium（数学概念跃迁——从 tabular 到 FA）
- **预算**：3 天
- **必读理由**：**deep RL 入口**；**§9.4 projection 是 Ch 11 前置**
- **配套**：Silver UCL Lec 6（Value Function Approximation）

## 1. 一句话 + 在 RL 体系中的位置

**把 tabular V(s) 升级为 v̂(s, w) 参数化**——线性 / NN 替代查表；引入 **VE 目标函数** + **semi-gradient** 这两个核心概念。

**在 RL 算法家族中的位置**：
```
RL Value-based
├── Tabular (Ch 4-6)         ← 精确表示，理论好
└── Function Approximation (Ch 9-11) ⭐
    ├── Linear FA (本章)       ← 收敛保证（on-policy）
    ├── NN FA (DQN, Ch 11 trick) ← 实战主流
    └── Off-policy + FA (Ch 11) ← deadly triad 风险区
```

## 2. 概念地图（§3 详解目录）

| # | 概念 | §3.x | 重要度 |
|---|---|---|---|
| 1 | **Mean Squared Value Error (VE)** | §3.1 | ⭐⭐⭐ FA 的目标函数 |
| 2 | **SGD vs Semi-gradient** | §3.2 | ⭐⭐⭐⭐ FA 的本质 trick |
| 3 | **Linear FA** | §3.3 | ⭐⭐⭐ 理论分析的脚手架 |
| 4 | **State Aggregation** | §3.4 | ⭐ 最简 feature |
| 5 | **Tile Coding** | §3.5 | ⭐⭐ NN 之前的 feature 王 |
| 6 | **Projection Operator Π & TD fixed point** | §3.6 | ⭐⭐⭐⭐ Ch 11 前置 |
| 7 | **LSTD (Least-Squares TD)** | §3.7 | ⭐ batch 求解 |
| 8 | **NN FA (preview DQN)** | §3.8 | ⭐⭐⭐ 工业主流 |

---

## 3. 核心概念详解（每个从 0-1 讲清）

### 3.1 Mean Squared Value Error (VE) ⭐⭐⭐

#### 是什么
FA 时代的目标函数：
$$\overline{\text{VE}}(w) \doteq \sum_{s \in S} μ(s) \big[ v_π(s) - v̂(s, w) \big]^2$$

—— μ(s) 是 on-policy state distribution（访问频率）。

#### 为什么需要它（动机）
- Tabular RL 目标是"每个 state 学准 V"
- FA 不可能每个 state 都准（参数有限）→ 必须有"加权 trade-off"
- **VE 用 μ 加权**——访问多的 state 误差大代价更高

#### 数学

**展开**：
$$\overline{\text{VE}}(w) = E_{s \sim μ}\big[(v_π(s) - v̂(s, w))^2\big]$$

**μ(s)** 是 on-policy 分布——agent 在 π 下访问 s 的长期频率。

**为什么不是 uniform error**：
- uniform：罕见 state 跟常 state 同等权重 → 不实用
- μ-weighted：实际遇到的 state 误差权重大 → 实用

#### 伪代码（VE 计算）
```python
def compute_VE(v_pi, v_hat, w, mu):
    """v_pi: dict s -> true V; v_hat: parametric V; mu: state distribution"""
    return sum(mu[s] * (v_pi[s] - v_hat(s, w)) ** 2 for s in v_pi)
```

#### 实战
通常**无法直接算 VE**（v_π 未知）——所以 RL 算 surrogate（如 TD fixed point error），定理保证 VE 也接近最优。

#### 跟相关算法对比
| 目标 | 公式 | 用途 |
|---|---|---|
| **VE** | $\sum μ(v_π - v̂)^2$ | 真实"好坏"，但 v_π 未知 |
| MSBE | $\|BV - V\|^2_μ$ | Bellman residual，理论目标 |
| MSPBE | $\|ΠBV - V\|^2_μ$ | TD 实际优化的（Ch 11） |

#### 反直觉点
- **VE 加权用 μ_π 而非 uniform**——RL 是"on-policy 性能"导向
- **VE 是 ground truth 目标，TD 学的是它的 surrogate**
- **off-policy 时 VE 用 target policy 的 μ_π，但训练只能用 behavior 的分布 → 一切混乱开始（Ch 11）**

#### 高频面试问法
**Q：VE 为什么用 μ(s) 加权而非 uniform？**
**A**：RL 在意 "agent 实际遇到 state 的表现"，不是"任意 state"。μ(s) = on-policy 分布，加权后 = expected 性能。Rare state 误差大但对实际 reward 影响小，weight 小是合理的。

---

### 3.2 SGD vs Semi-gradient（FA 的本质 trick）⭐⭐⭐⭐

#### 是什么
- **True SGD**：$w ← w - α ∇L$，L 是真实 loss
- **Semi-gradient TD**：用 R + γv̂(s', w) 当 target，**不对 target 取梯度**——是"伪梯度"

#### 为什么需要它（动机）
- True SGD 需要 v_π（未知）→ 不可行
- 用 R + γv̂(s') 当 target 是合理估计，但**target 也依赖 w** → 对 w 求导要算 target 那项
- TD trick：**假装 target 是 constant**——只对 v̂(s) 那项求梯度

#### 数学

**True SGD on VE**（如果 v_π 已知）：
$$w_{t+1} = w_t + α [v_π(S_t) - v̂(S_t, w_t)] ∇v̂(S_t, w_t)$$

**Monte Carlo SGD**（用 G_t 替代 v_π）：
$$w_{t+1} = w_t + α [G_t - v̂(S_t, w_t)] ∇v̂(S_t, w_t)$$

——这是**真 SGD**（G_t 不依赖 w），无偏估计。

**Semi-gradient TD(0)**：
$$w_{t+1} = w_t + α [R_{t+1} + γv̂(S_{t+1}, w_t) - v̂(S_t, w_t)] ∇v̂(S_t, w_t)$$

——**注意**：$γv̂(S_{t+1}, w_t)$ **也依赖 w**，但我们假装它是 constant，只对 $-v̂(S_t, w_t)$ 那项算梯度 $-∇v̂(S_t)$。

**"semi"** = 半——只算 v̂(S_t) 的梯度，target 那半假装是常数。

#### 伪代码
```python
def semi_gradient_TD0(env, policy, v_hat, w, alpha=0.1, gamma=0.99, n_episodes=1000):
    """v_hat(s, w): callable returning V estimate;
       v_hat.grad(s, w): callable returning ∇v̂(s, w)"""
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = policy(s)
            s_next, r, done = env.step(a)
            # Semi-gradient: 不对 v_hat(s_next) 求梯度
            target = r + gamma * v_hat(s_next, w)  # treated as constant
            delta = target - v_hat(s, w)
            w += alpha * delta * v_hat.grad(s, w)  # 只算 ∇v̂(s)
            s = s_next
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| α | 0.001 - 0.1（FA 比 tabular 小）|
| n_episodes | 1000+ |
| Loss | MSE on (target - v̂) |

#### 跟相关算法对比
| | True SGD on VE | MC SGD | Semi-gradient TD |
|---|---|---|---|
| Target | $v_π$（未知）| $G_t$（无偏）| $R + γv̂(s')$（有 bias）|
| 是 true gradient？ | ✅ | ✅ | ❌ semi |
| Bias | 0 | 0 | 有（bootstrap）|
| Variance | n/a | 大 | 小 |
| 收敛性（tabular） | n/a | ✅ | ✅ |
| 收敛性（linear on-policy）| n/a | ✅ | ✅（到 TD fp）|
| 收敛性（off-policy + FA）| n/a | ✅ | ❌（deadly triad）|

#### 反直觉点
- **TD 学到的 V 不是 minimize VE**——而是 TD fixed point（差 1/(1-γ) 倍）
- **Semi-gradient 不是真梯度**——破坏 SGD 收敛理论，但实践 work
- **MC SGD 是 true SGD 但方差大**——为什么 RL 主流仍 TD？方差更重要

#### 高频面试问法

**Q1：为什么 "semi-gradient" 叫"半"梯度？**
**A**：target $R + γv̂(s', w)$ 也依赖 w，但**只对 -v̂(s, w) 那项求梯度**，target 那半假装是常数。这是 TD 的本质 trick——破坏 SGD 假设但实践 work。

**Q2：为什么 RL 不用 MC SGD（真 SGD）？**
**A**：理论上更严格（无 bias、是 true SGD），但方差爆炸（用真 G_t 全程随机）+ 必须等 episode 结束。**实践中 TD 的低方差 + online 学习更值**。

---

### 3.3 Linear Function Approximation ⭐⭐⭐

#### 是什么
最简单的 FA：v̂(s, w) = w^T x(s)，其中 x(s) 是 feature 向量。

#### 为什么需要它（动机）
- 线性是 FA 的"理论玩具"——大部分收敛证明只对线性可证
- 即使 NN 时代仍重要——理解 NN FA 困难必先理解 linear FA 性质
- 实战上 tile coding + linear 已能解 Mountain Car

#### 数学

$$v̂(s, w) = w^T x(s) = \sum_i w_i x_i(s)$$

**Semi-gradient TD update** 简化：
$$w_{t+1} = w_t + α [R + γ w_t^T x(s') - w_t^T x(s)] · x(s)$$

——梯度 $∇v̂(s, w) = x(s)$（feature 向量本身）。

#### 收敛性（Tsitsiklis & Van Roy 1997）

**Theorem**：on-policy + linear FA + semi-gradient TD 收敛到 **TD fixed point** $w_{\text{TD}}$，且：
$$\overline{\text{VE}}(w_{\text{TD}}) \leq \frac{1}{1-γ} \min_w \overline{\text{VE}}(w)$$

—— TD fp 跟最优解差 $\frac{1}{1-γ}$ 倍。

**γ=0.99** 时差 100x——所以**γ 越大对 FA 越苛刻**。

#### 伪代码
```python
def linear_semi_gradient_TD(env, policy, n_features, alpha=0.01, gamma=0.99, n_episodes=1000):
    w = np.zeros(n_features)
    for ep in range(n_episodes):
        s = env.reset()
        x = feature(s)
        done = False
        while not done:
            a = policy(s)
            s_next, r, done = env.step(a)
            x_next = feature(s_next) if not done else np.zeros(n_features)
            delta = r + gamma * w @ x_next - w @ x
            w += alpha * delta * x  # ∇v̂ = x
            x = x_next
    return w
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| α | 0.01 - 0.1 |
| Feature dim | 取决于 tile coding 配置（如 1000-10000）|

#### 跟相关算法对比
| Feature | 表达力 | 收敛性 |
|---|---|---|
| Tabular（每 state 一 feature）| 完整 | ✅ |
| State aggregation | 弱 | ✅ |
| **Linear（tile / RBF）**| 中 | **✅ on-policy + bootstrap** |
| NN (DQN) | 强 | ❌ off-policy 可能崩 |

#### 反直觉点
- **Linear FA 比看上去强**——配合 tile coding 能解 Mountain Car
- **NN 收敛性证明大多无效**——理论上没保证，靠 trick

#### 高频面试问法
**Q：为什么 RL 理论分析喜欢 linear FA？**
**A**：(1) ∇v̂ = x 不依赖 w → 数学简单；(2) projection operator 是线性算子 → 不动点存在唯一；(3) on-policy 时 contraction 性质保留 → 收敛可证。**Linear 是 RL 理论分析的脚手架**。

---

### 3.4 State Aggregation

#### 是什么
最简 feature：把状态空间分成 K 个 group，每个 group 用一个共享 V 值。

#### 为什么需要它
- 1000-state random walk 实验经典（Sutton 图 9.1）
- state aggregation = 退化版 tile coding（1 个 tiling 一个粗 grid）

#### 数学
设 g(s) ∈ {0, ..., K-1} 是 state s 的 group id。
$$v̂(s, w) = w_{g(s)}$$

等价于 K 个 binary feature：x(s) = e_{g(s)}（one-hot）。

#### 伪代码
```python
def aggregation_feature(s, n_groups=10, n_states=1000):
    group = s * n_groups // n_states
    feat = np.zeros(n_groups)
    feat[group] = 1.0
    return feat
```

#### 反直觉点
- **State aggregation 是 tile coding 的 K=1 退化**
- **简单但能学到合理 V**（Sutton 1000-state random walk）

---

### 3.5 Tile Coding ⭐⭐

#### 是什么
**CMAC** 思想（Albus 1975 → RL）——多个**重叠的 grid**（tilings），每个 cell 一个 binary feature。

#### 为什么需要它（动机）
- 1990s-2000s deep learning 前 RL 标配 feature
- 比 state aggregation 强：**局部支撑** + **平滑泛化**
- 至今仍是某些任务 baseline（Mountain Car）

#### 数学
- N 个 tiling，每个 grid_size × grid_size 大小
- 每 tile 一个 binary feature → 总 feature 数 = N × grid_size²
- v̂(s, w) = w^T x(s)，x(s) 是 N 个 1 + (其余 0)

**重叠**：tiling 之间有 offset → 平滑 generalization。

#### 伪代码
```python
def tile_coding_feature(s, n_tilings=8, tiles_per_dim=8):
    """Mountain Car: 2D state (pos, vel)"""
    pos, vel = s
    pos_range = (-1.2, 0.5)
    vel_range = (-0.07, 0.07)
    feat = np.zeros(n_tilings * tiles_per_dim ** 2)
    for tile_idx in range(n_tilings):
        offset = tile_idx / n_tilings
        pos_tile = int((pos - pos_range[0] + offset) * tiles_per_dim / (pos_range[1] - pos_range[0])) % tiles_per_dim
        vel_tile = int((vel - vel_range[0] + offset) * tiles_per_dim / (vel_range[1] - vel_range[0])) % tiles_per_dim
        idx = tile_idx * tiles_per_dim ** 2 + pos_tile * tiles_per_dim + vel_tile
        feat[idx] = 1.0
    return feat
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| N tilings | 8 |
| Tiles per dim | 8 - 16 |
| Total features | N × tiles² (~500-2000) |

#### 跟相关算法对比
| Feature | 学习速度 | 泛化 | 现代用 |
|---|---|---|---|
| Tabular | 慢（state 多）| 无 | tiny env |
| State aggregation | 快 | 粗 | 教学 |
| **Tile coding** | 中 | **平滑 + 局部** | 经典 baseline |
| RBF | 慢 | 平滑 | 罕用 |
| NN | 慢但表达力强 | 强 | **主流** |

#### 反直觉点
- **Tile coding 跑 Mountain Car 比 DQN 早 20 年**
- **NN 替代 tile coding 是 deep RL 标志**——2013 DQN 之后

#### 高频面试问法
**Q：Tile coding 跟 NN 在 feature 学习上的本质区别？**
**A**：tile coding 是**手工 feature**（人定 grid），局部支撑 + 重叠 → 平滑泛化；NN 是**学到的 feature**，端到端 + 任意复杂。tile coding 收敛性好但表达力受限；NN 表达力强但收敛性差（deadly triad）。

---

### 3.6 Projection Operator Π & TD Fixed Point ⭐⭐⭐⭐

#### 是什么
**§9.4 的核心数学工具**——把 Bellman backup 结果 BV 投影回 v̂ subspace 的算子 Π。**理解这个才能理解 Ch 11**。

#### 为什么需要它（动机）
- BV（Bellman backup）可能跳出 v̂ 表达空间
- TD 不能精确学到 BV——只能投影后比较
- **TD fixed point** 是 V 使得 ΠBV = V

#### 数学

**Bellman operator B**：
$$BV(s) = \sum_a π(a|s) \sum_{s'} p(s'|s,a) [r + γV(s')]$$

**Projection operator Π**：
对任意 V，定义 ΠV = $\arg\min_{\bar{V} \in F} \|\bar{V} - V\|_μ$，F 是 v̂(w) 可表达空间。

**线性 FA 时**：Π 是个线性矩阵 = $X(X^T D X)^{-1} X^T D$，其中 X 是 feature matrix, D 是 diag(μ)。

**TD fixed point**：V 使得 ΠBV = V

**几何**：
```
[Full V space]
       |
       | Π (project)
       ↓
[v̂ subspace]   ← BV may not be here
       |
       V_TD = ΠBV (fixed point)
```

#### 收敛 vs 不收敛的关键

**on-policy**：ΠB 是 contraction → 不动点存在 + TD 收敛

**off-policy**：ΠB **可能不再是 contraction** → 不动点可能不存在 → TD 发散（deadly triad）

#### 伪代码（理论框架，非算法）
```python
def projection_op(V, features, mu):
    """投影 V 到 v̂ 可表达空间"""
    X = features  # (|S|, n_features)
    D = np.diag(mu)
    # 解 min_w ||Xw - V||_μ
    w = np.linalg.solve(X.T @ D @ X, X.T @ D @ V)
    return X @ w  # = ΠV

def TD_fixed_point(B_op, features, mu, max_iter=1000):
    V = np.zeros(features.shape[0])
    for _ in range(max_iter):
        V = projection_op(B_op(V), features, mu)
    return V  # ΠBV = V
```

#### 跟相关概念对比
| 概念 | 公式 | 含义 |
|---|---|---|
| Bellman fixed point | BV = V | 真 V_π（tabular 时存在）|
| **TD fixed point** | **ΠBV = V** | TD 在 FA 下收敛到的解 |
| VE optimum | $\min_V \|v_π - V\|^2_μ$ | 理论最优 V 估计 |
| **TD vs VE optimum** | 差 $\frac{1}{1-γ}$ 倍 | γ 大时差距大 |

#### 反直觉点
- **TD fixed point 跟"真值"差**——γ=0.99 时差 100x
- **on-policy 时差距可控，off-policy 可能不存在**
- **NN 时投影不是线性算子** → 理论分析更难

#### 高频面试问法
**Q：TD 在 FA 下学到的 V 跟 VE 最优 V 是同一个吗？**
**A**：**不是**。TD 学到 ΠBV = V（TD fixed point），不是 minimize VE。两者差 $\frac{1}{1-γ}$ 倍。**γ 大时差距大**——这就是为什么大 γ 在 FA 下危险。

---

### 3.7 LSTD (Least-Squares TD)

#### 是什么
**Bradtke & Barto 1996**：直接 batch 解 TD fixed point 的 closed-form 解。

#### 为什么需要它
- 在线 TD 收敛慢（需要 many episodes）
- 如果有 batch 数据，能直接解
- LSTD 一步解出 TD fixed point

#### 数学

TD fixed point 满足 $w_{\text{TD}} = A^{-1} b$，其中：
$$A = E[x_t (x_t - γx_{t+1})^T]$$
$$b = E[R_{t+1} x_t]$$

LSTD：用 sample 估计 A, b，解线性方程。

#### 伪代码
```python
def LSTD(samples, n_features, gamma=0.99):
    """samples: list of (s, r, s_next) transitions"""
    A = np.zeros((n_features, n_features))
    b = np.zeros(n_features)
    for s, r, s_next in samples:
        x = feature(s)
        x_next = feature(s_next)
        A += np.outer(x, x - gamma * x_next)
        b += r * x
    w = np.linalg.solve(A, b)
    return w
```

#### 实战参数
通常加 regularization：$A + λI$ 避免 singular。

#### 跟相关算法对比
| | 在线 TD | LSTD | LSTD-Q |
|---|---|---|---|
| Online | ✅ | ❌ batch | ❌ |
| Sample 效率 | 中 | **高** | 高 |
| 计算 | O(d) | O(d³) | O(d³) |
| 适用维度 | 大 | 小（d < 1000）| 小 |

#### 反直觉点
- **LSTD sample 效率高但计算贵**——O(d³) 解线性方程
- **不常用**——deep learning 时代直接 SGD on NN，无 closed-form

#### 高频面试问法
**Q：LSTD vs TD 的核心 trade-off？**
**A**：LSTD batch 求解一步到位（sample 效率高），但 O(d³) 矩阵求逆贵；TD 在线增量（sample 多但 O(d) 每步）。**大模型 (d 大) 必用 TD，小问题 (d < 1000) 用 LSTD 更高效**。

---

### 3.8 NN Function Approximation（DQN 前置）⭐⭐⭐

#### 是什么
用神经网络替代 linear FA：v̂(s, w) = NN(s; w)。

#### 为什么需要它
- Linear FA 表达力受限——Atari pixel 需要 conv net
- NN 能学到 feature → end-to-end
- 但 NN + RL 进入 deadly triad 区域（Ch 11）

#### 数学
跟 §3.2 semi-gradient TD 完全相同，只是 ∇v̂(s, w) 现在是 NN backprop 算的 Jacobian。

$$w ← w + α [r + γ NN(s', w) - NN(s, w)] · ∇_w NN(s, w)$$

#### 伪代码
```python
import torch.nn as nn
class VNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, s):
        return self.net(s).squeeze()

def NN_semi_gradient_TD(env, policy, v_net, optimizer, n_episodes=1000, gamma=0.99):
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = policy(s)
            s_next, r, done = env.step(a)
            with torch.no_grad():
                target = r + gamma * v_net(s_next) * (1 - done)
            v_pred = v_net(s)
            loss = (target - v_pred).pow(2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            s = s_next
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| Hidden | 64 - 256 |
| lr | 1e-4 - 1e-3 |
| Loss | MSE 或 Huber |

#### 跟相关算法对比
| | Linear FA | NN FA |
|---|---|---|
| 收敛性（on-policy）| ✅ | 理论无保证 |
| 表达力 | 弱 | 强 |
| Sample 效率 | 中 | 中 |
| 工业用 | 罕 | **主流** |

#### 反直觉点
- **NN FA 收敛性几乎无理论保证**——靠工程 trick
- **DQN 是 NN FA + Q-learning + 工程 trick**

#### 高频面试问法
**Q：NN FA 比 linear FA 强在哪？弱在哪？**
**A**：强：表达力 + end-to-end feature learning；弱：理论无收敛保证 + 易陷 deadly triad（off-policy 时）。**实战仍主流——表达力胜过理论**。

---

## 4. 跨概念对比表

### 4.1 FA 算法 family tree
```
Function Approximation
├── Linear FA
│   ├── State aggregation (粗)
│   ├── Tile coding (重叠 grid)
│   ├── Coarse coding (RBF)
│   └── 收敛保证（on-policy + bootstrap）
├── Polynomial / Fourier basis
├── NN FA (DQN, PPO critic)
│   └── 收敛性靠 trick (Ch 11)
└── 训练算法
    ├── Semi-gradient TD（伪梯度）
    ├── MC SGD（真 SGD）
    └── LSTD（batch closed-form）
```

### 4.2 FA 收敛性表

| FA + 算法 | 收敛性 |
|---|---|
| Linear + MC | ✅ 真 SGD 严格 |
| Linear + on-policy TD | ✅ 到 TD fixed point |
| Linear + off-policy TD | ❌ 可能 Baird（Ch 11）|
| NN + on-policy TD | 实战 work（PPO critic）|
| NN + off-policy TD | 风险（DQN 靠 trick）|

---

## 5. 习题精选

**Exercise 9.1（state aggregation feature design）**：1000-state random walk 上设计 10-group aggregation。
**Exercise 9.6（gradient MC vs semi-grad TD）**：比较两者学习曲线。
**Exercise 9.7（tile coding 配置）**：扫 # tilings × tile width。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter09/random_walk.py`（1000-state random walk）
- **自己**：linear v̂ 5 行 + semi-grad TD update
- **DQN 论文**：NN 版本完整工程

## 7. 跟 RLHF 的桥

| RLHF 组件 | 来自本章 |
|---|---|
| PPO critic V_φ | §3.8 NN FA + semi-grad TD |
| PPO clip 缓解 off-policy 漂移 | §3.6 projection 失效保护 |
| GAE 的 V(s) 学习 | §3.2 + §3.8 |
| DPO ref model frozen | 类似 target net 防 deadly triad（[[ch11-off-policy-methods]]）|

## 8. 苏格拉底 Q&A

详见每个 §3.x。综合题：

- **Q：从 tabular 到 FA 的本质跃迁是什么？**
  - 答：从"精确存储 V" 到"参数化拟合 V" + 引入 μ-weighted VE + semi-gradient trick + 失去 contraction 保证
- **Q：为什么本章不讨论 Q-learning + FA？**
  - 答：留给 Ch 11——off-policy + FA + bootstrap = deadly triad

## 9. 自测清单（闭卷）

- [ ] **写出 VE 定义**（含 μ 加权）
- [ ] **写出 semi-gradient TD(0) 更新公式**
- [ ] **解释 "semi" 的"半"在哪**
- [ ] **解释 on-policy + linear 收敛但 off-policy + nonlinear 可能崩**
- [ ] **解释 projection operator Π 的几何意义**
- [ ] **写出 TD fixed point 跟 VE 最优的差距界**（$\frac{1}{1-γ}$ 倍）
- [ ] 跑通 ShangtongZhang `chapter09/random_walk.py`
- [ ] **解释 PPO critic 用本章哪个公式**（RLHF 桥）

---

## 💡 拓展知识点（书外 trivia）

### A. Semi-gradient 名字的争议
"Semi-gradient" 是 Sutton 起的名字——业界也叫 "TD-style backprop" / "fixed target backprop"。叫法不统一但思想同。

### B. Tile coding 是 1970s CMAC 的现代名
**Albus 1975** *A New Approach to Manipulator Control: The Cerebellar Model Articulation Controller* 提出 CMAC——把小脑结构数学化。1990s Sutton 改名 "tile coding" 用进 RL。

### C. Tsitsiklis & Van Roy 1997 是 RL 理论里程碑
*An Analysis of Temporal-Difference Learning with Function Approximation* 给出 on-policy linear FA TD 收敛证明——奠定 RL FA 的理论基础。

### D. LSTD 在 2000s 一度流行
LSTD 和 LSPI（Least-Squares Policy Iteration）2003 由 Lagoudakis 推广。一度认为是 RL 的"未来"。但 deep learning 时代 NN + SGD 取代——sample 效率没赢过工程效率。

### E. DQN 论文 2013 NIPS Workshop 被几乎所有人忽视
**Mnih 2013** NIPS Workshop 论文展示 DQN on Atari——会场反响平平。2015 Nature 主刊版才出圈。**典型"早期被低估"的工作**。

### F. Adam 不是 RL 默认 optimizer
Tabular Q-learning 用 fixed α；DQN 多用 RMSProp（Mnih 2015 默认）。Adam 在 RL 上的 bias correction 跟 TD bootstrap 有不良互动——实战经常 Adam 不如 RMSProp 稳。

### G. RL 的"deadly triad"是 Sutton 第二版加的术语
第一版（1998）没正式提"deadly triad"——只散落讨论。2018 第二版 §11.3 才正式命名。**反映出社区对此问题理解的深化**。

### H. 为什么 PPO 用 NN critic 仍 work？
- **On-policy**（虽然 K epoch 有 mild off-policy）
- **GAE 部分用 MC（λ=0.95）**——bootstrap 程度比纯 TD(0) 小
- **不取 max**——绕开 Jensen over-est
- 三者合力让 PPO critic 在 deadly triad 边缘安全

### I. NN FA 在 RL 上为什么收敛性差但实战 work？
- replay buffer / target net / clip 等 trick
- 大模型平滑性强
- on-policy stream 自然近 μ_π
- gradient clip + adaptive lr
**集合作用让 NN 实战稳定，但无统一理论**。

---

## 🔗 更多 wikilinks（深挖）

| 主题 | 跳转 |
|---|---|
| Deadly triad 详解 | [[ch11-off-policy-methods]] §3 |
| DQN deep 版工程 | [[../../../brain/Areas/rl-books/deep-rl-hands-on-lapan-3e/_overview\|Lapan Ch 6]] |
| Distributional V/Q learning | [[../../../brain/Areas/rl-books/distributional-rl-bellemare/_overview]] |
| PPO critic 用本章 | [[ch13-policy-gradient]] §3.4, §3.8 |
| FA 在 control 场景 | [[ch10-on-policy-control]] |
| AI-Infra: critic 训练在 LLM 上的实战 | [[../../../brain/Slipbox/training-metrics-mfu-hfu]] |

## Related

- **上一章**：[[ch08-planning-and-learning]]
- **下一章**：[[ch10-on-policy-control]]
- **延伸**：[[ch11-off-policy-methods]]（off-policy + FA = deadly triad）· [[ch13-policy-gradient]]（NN policy）
- **进度**：[[_chapter-status]] W4

## Sources

- 原书 Ch 9 pp. 197-218
- [Silver UCL Lec 6: Value Function Approximation](https://www.youtube.com/watch?v=UoPei5o4fps)
- ShangtongZhang `chapter09/`
- **核心论文**：
  - Tsitsiklis & Van Roy 1997 *An Analysis of TD Learning with FA*
  - Bradtke & Barto 1996 LSTD
  - Albus 1975 CMAC（tile coding 前身）
  - Mnih 2015 Nature DQN

---

## 💀 Top 3 Gotchas (v4)

### 💀 Gotcha 1：semi-gradient ≠ true gradient

True gradient TD 应该对 target 也求导（target 含 $\hat v(S', w)$ 也依赖 $w$）—— 双 gradient。

Semi-gradient **故意忽略 target 对 $w$ 的依赖**：
$$w \leftarrow w + \alpha \cdot \delta_t \cdot \nabla \hat v(S_t, w)$$

→ **不是真梯度** → 没有保证下降的 loss landscape → **可能发散**（Baird counterexample）。

但 semi-gradient 实际比 true-gradient 更 work——真梯度的 Residual gradient 算法收敛到 minimize Bellman residual 的 fixed point，**不是 V_π**，更差。

### 💀 Gotcha 2：linear FA + TD fixed point ≠ V_π 最优

linear FA + on-policy + Robbins-Monro 步长 → 收敛到 **TD fixed point**——但 **不是** $V_\pi$！

TD fixed point 是 **MSPBE**（Projected Bellman Error）的 minimizer，跟 **VE**（true value error）的 minimizer **不同**。它们的差距取决于 feature 表达能力 → approximation error gap。

实战意思：即使你 feature 设计得能完美表达 $V_\pi$，TD 可能也收敛不到（只到投影后的 fixed point）。

### 💀 Gotcha 3：tile coding 工业上比 RBF / polynomial 强

经典 feature 选择：
- ❌ **Polynomial**: 高阶多项式在 RL state 上**泛化差**（局部支持没保证）
- ⚠️ **RBF**: 平滑但 $\sigma$ 难调，新 domain 上需要重 tune
- ✅ **Tile coding**: 稀疏 + 局部 + 分辨率可控 + 计算便宜——Sutton 推荐 baseline

实战 linear FA 几乎都用 tile coding（Mountain Car、Acrobot 经典基线）；现在 deep RL 一般跳过 linear，直接 NN，但理解 tile coding 让你懂"为什么 NN 比 linear 好（自动学 feature）"。

## 链回

- [[_anki/ch09-fa-cards]] — 35 张卡片版
- [[ch11-off-policy-methods]] — 看 Baird 反例（FA + 真发散的具体例子）
