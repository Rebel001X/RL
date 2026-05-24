---
title: "Sutton & Barto Ch 7. n-step Bootstrapping"
created: 2026-05-24
tags: [area/rl, type/note, status/v3, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 7
difficulty: medium
budget_days: 2
priority_for_rlhf: nice
silver_lecture: 5
---

# Ch 7. n-step Bootstrapping ⭐⭐⭐

---

## 📋 本章讲了什么（TL;DR）

1. **n-step 是 TD(0) 和 MC 的连续谱**：n=1 → TD(0)，n=∞ → MC，中间 n 通常最优
2. **n-step return** $G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})$ 用 n 步实 reward + bootstrap
3. **n-step Off-policy**（Tree Backup / Q(σ)）：解决 trajectory IS 方差爆问题；**Rainbow 用 n-step**

**为什么读**：n-step 是 [[ch12-eligibility-traces|TD(λ)]] 和 [[ch13-policy-gradient|GAE]] 的直接前置。**Rainbow / R2D2 等现代 deep RL 都用 n-step**。

## 🎯 Top 5 关键洞察

| # | 洞察 | 为什么关键 |
|---|---|---|
| 1 | **n 越大 bias 越小、方差越大** | TD ↔ MC 谱的物理 |
| 2 | **最优 n 跟 α 强相关** | 小 α 时大 n 更好 |
| 3 | **n-step SARSA** 信用分配跨 n 步 | 比 TD(0) 学得快 |
| 4 | **n-step off-policy IS 累乘 → 方差爆** | 引出 Tree Backup |
| 5 | **n-step Q(σ) 统一框架**：σ ∈ [0,1] 控 sample vs expectation | 数学最优雅统一 |

## 🧭 在 Obsidian graph 中的位置

- **上游**：[[ch05-monte-carlo-methods]] · [[ch06-temporal-difference-learning]]（n-step 是两者插值）
- **下游**：**[[ch12-eligibility-traces]]**（λ 平滑替代固定 n）· [[ch13-policy-gradient]]（GAE）
- **RLHF 联动**：GAE λ=0.95 是本章的 advantage 版

---

## 0. 阅读元信息

- **难度**：medium（概念是 TD 和 MC 的桥）
- **预算**：2 天 / GAE 和 Rainbow 必懂前置
- **必读理由**：n-step 是 GAE 和 Rainbow 的直接前置

## 1. 一句话 + 在 RL 体系中的位置

**n-step = MC 和 TD(0) 的连续插值**——n=1 是 TD(0)，n=∞ 是 MC，中间任意 n 是折中。

```
信用分配谱
├── MC（n=∞）         ← 看全程，无 bootstrap
├── n-step TD          ← 看 n 步，部分 bootstrap ⭐
├── TD(0)（n=1）       ← 看一步，全 bootstrap
└── TD(λ) / GAE        ← 用 λ 平滑混合所有 n
```

## 2. 概念地图（§3 详解目录）

| # | 概念 | §3.x | 重要度 |
|---|---|---|---|
| 1 | **n-step Return** | §3.1 | ⭐⭐⭐ 基础 |
| 2 | **n-step TD Prediction** | §3.2 | ⭐⭐⭐ V learning |
| 3 | **n-step SARSA** | §3.3 | ⭐⭐⭐ on-policy control |
| 4 | **n-step Expected SARSA** | §3.4 | ⭐⭐ 降方差 |
| 5 | **n-step Off-policy + IS** | §3.5 | ⭐⭐ 噩梦版 |
| 6 | **n-step Tree Backup** | §3.6 | ⭐⭐⭐ 无 IS off-policy |
| 7 | **n-step Q(σ)** | §3.7 | ⭐⭐ 统一框架 |
| 8 | **Rainbow's n-step** | §3.8 | ⭐⭐⭐ deep RL 实战 |

---

## 3. 核心概念详解（每个从 0-1 讲清）

### 3.1 n-step Return ⭐⭐⭐

#### 是什么
$$G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})$$

—— 前 n 步用真 reward，最后用 V(S_{t+n}) bootstrap。

#### 为什么需要它
- TD(0) 只看 1 步——信用分配慢
- MC 看全程——方差大
- n-step 中间——bias-variance 折中

#### 数学
- n=1 → $R_{t+1} + γ V(S_{t+1})$ = TD(0) target
- n=∞ → $G_t$ = MC return
- **U 形 bias-variance**：中间 n 最优

#### 实战参数
| n | 适用 |
|---|---|
| n=1 | TD(0)，sparse compute |
| n=3 | Rainbow 默认 |
| n=5-20 | A2C / A3C 常用 |
| n→∞ | MC |

#### 反直觉点
- **n 不是越大越好**——大 n 方差爆（接近 MC）
- **n=3 是 Rainbow 工程甜点**

---

### 3.2 n-step TD Prediction

#### 是什么
$$V(S_t) \leftarrow V(S_t) + α [G_{t:t+n} - V(S_t)]$$

—— 用 n-step return 当 target。

#### 伪代码
```python
def n_step_TD_prediction(env, policy, n=3, alpha=0.1, gamma=0.99, n_episodes=1000):
    V = defaultdict(float)
    for ep in range(n_episodes):
        states = [env.reset()]; rewards = [None]  # rewards[t] = R_t
        T = float('inf')
        t = 0
        while True:
            if t < T:
                a = policy(states[t])
                s_next, r, done = env.step(a)
                states.append(s_next); rewards.append(r)
                if done: T = t + 1
            tau = t - n + 1
            if tau >= 0:
                G = sum(gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T) + 1))
                if tau + n < T: G += gamma ** n * V[states[tau + n]]
                V[states[tau]] += alpha * (G - V[states[tau]])
            if tau == T - 1: break
            t += 1
    return V
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| n | 3-10 |
| α | 0.1 - 0.5 |

#### 反直觉点
- **n-step 需要 buffer 中间 n 步数据**——比 TD(0) 多内存

---

### 3.3 n-step SARSA ⭐⭐⭐

#### 是什么
SARSA 的 n-step 版——Q learning 用 n-step return。

#### 数学
$$G_{t:t+n} = R_{t+1} + ... + γ^{n-1}R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})$$
$$Q(S_t, A_t) ← Q(S_t, A_t) + α [G_{t:t+n} - Q(S_t, A_t)]$$

#### 伪代码
```python
def n_step_SARSA(env, n=3, alpha=0.1, gamma=0.99, eps=0.1, n_episodes=500):
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    for ep in range(n_episodes):
        states = [env.reset()]
        actions = [epsilon_greedy(Q[states[0]], eps)]
        rewards = [None]
        T = float('inf'); t = 0
        while True:
            if t < T:
                s_next, r, done = env.step(actions[t])
                states.append(s_next); rewards.append(r)
                if done: T = t + 1
                else: actions.append(epsilon_greedy(Q[s_next], eps))
            tau = t - n + 1
            if tau >= 0:
                G = sum(gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T) + 1))
                if tau + n < T: G += gamma ** n * Q[states[tau + n]][actions[tau + n]]
                Q[states[tau]][actions[tau]] += alpha * (G - Q[states[tau]][actions[tau]])
            if tau == T - 1: break
            t += 1
    return Q
```

#### 跟 TD(0) SARSA 对比
| | TD(0) SARSA | n-step SARSA (n=5) |
|---|---|---|
| 信用分配 | 1 步 | 5 步 |
| 学习速度 | 慢 | **快**（信号传更远）|
| 方差 | 小 | 中 |

#### 高频面试问法
**Q：n-step SARSA 比 1-step 强在哪？**
**A**：**信用分配跨 n 步**——奖励能传到 n 步前的 action。1-step SARSA 在 sparse reward 任务上信号传播极慢，n-step 大幅加速。

---

### 3.4 n-step Expected SARSA

#### 是什么
n-step SARSA 的 expectation 版——bootstrap 用 $\sum_a π(a|S_{t+n}) Q(S_{t+n}, a)$。

#### 数学
$$G_{t:t+n}^{Exp} = R_{t+1} + ... + γ^{n-1}R_{t+n} + γ^n \sum_a π(a|S_{t+n}) Q(S_{t+n}, a)$$

#### 反直觉点
- **方差小于 sample SARSA**——同 bias

---

### 3.5 n-step Off-policy + IS 噩梦

#### 是什么
Off-policy n-step：用 behavior b 采，target π 学。需要 IS ratio 累乘。

#### 数学
$$ρ_{t:t+n-1} = \prod_{k=t}^{t+n-1} \frac{π(A_k|S_k)}{b(A_k|S_k)}$$

$$V(S_t) ← V(S_t) + α · ρ_{t:t+n-1} · [G_{t:t+n} - V(S_t)]$$

#### 为什么是"噩梦"
- ρ 是 n 项累乘 → **方差随 n 指数爆**
- n=5 时已经经常不可用
- → 需要 Tree Backup 或 weighted IS

#### 跟 Per-decision IS 对比
| | Trajectory IS | Per-decision IS | Tree Backup |
|---|---|---|---|
| Bias | 0 | 0 | 有 |
| Variance | 巨大 | 中 | 小 |
| 需 IS | ✅ | ✅ | ❌ |

#### 高频面试问法
**Q：n-step off-policy 为什么实战难？**
**A**：IS ratio $\prod_t π/b$ 是 n 项累乘——**方差指数随 n 爆**。n=5 时已经经常不可用，n=10 几乎肯定不工作。**实战要么 n 小（n=2-3）+ weighted IS，要么走 Tree Backup（§3.6）**。

---

### 3.6 n-step Tree Backup ⭐⭐⭐

#### 是什么
**off-policy 不用 IS**——用"期望补"代替"sample 的非 target action"。

#### 为什么需要它
- IS 累乘方差爆
- 观察：non-target action 部分可以用 expectation 代替（不必 sample）
- → Tree Backup

#### 数学

定义：
$$G^{TB}_{t:t+n} = R_{t+1} + γ \sum_{a \neq A_{t+1}} π(a|S_{t+1}) Q(S_{t+1}, a) + γ π(A_{t+1}|S_{t+1}) G^{TB}_{t+1:t+n}$$

**关键**：
- "我没选的 action 部分" 用 expectation $\sum_a π · Q$
- "我选了的 action 部分" 递归到下一步 G^TB

#### 跟 IS 对比
| | n-step IS | Tree Backup |
|---|---|---|
| Need ratio | ✅ | ❌ |
| Variance | 指数爆 | 有限 |
| Bias | 0 | 有 |
| 实战 | 罕用 | 适中 |

#### 高频面试问法
**Q：Tree Backup 怎么避免 IS？**
**A**：non-target action 部分用 expectation $\sum_a π(a|s) Q(s, a)$ 代替 sample——不需要 ratio。"target action 部分"递归。**有 bias 但方差有限，比 IS 实战**。

---

### 3.7 n-step Q(σ) 统一框架

#### 是什么
σ ∈ [0, 1] 控制每步是 sample（σ=1, IS 路）还是 expectation（σ=0, Tree Backup 路）。**整本书最一般的 backup 形式**。

#### 数学（简化）

$$G^{Q(σ)}_{t:t+n} = R_{t+1} + γ[(1-σ_{t+1}) \sum_a π(a|S_{t+1}) Q(S_{t+1}, a) + σ_{t+1} · ρ_{t+1} · G^{Q(σ)}_{t+1:t+n}]$$

- σ_k = 1 → 那步走 IS
- σ_k = 0 → 那步走 Tree Backup expectation
- 任意混合

#### 反直觉点
- **σ=1 退化 SARSA + IS**
- **σ=0 退化 Tree Backup**
- **σ 可以 per-step 不同——最大灵活性**

#### 高频面试问法
**Q：Q(σ) 是统一什么？**
**A**：用 σ 控制每步是 "sample（IS-like）"还是"expectation（Tree Backup-like）"——一个公式覆盖 IS / Tree Backup / SARSA 等所有 off-policy n-step backup。**数学最优雅统一**。

---

### 3.8 Rainbow 的 n-step（实战 deep RL）⭐⭐⭐

#### 是什么
**Hessel 2018 Rainbow** 用 **n=3 n-step Q-learning** 当组件之一——加速 DQN 收敛。

#### 为什么 n=3
- 实证扫 n=1, 2, 3, 5, 10 在 Atari 上
- **n=3 是 bias-variance 工程甜点**
- 太大显存爆（要 buffer n 步 trajectory）
- 太小信用分配慢

#### Rainbow 完整 7 组件
1. Double Q-learning（解 over-est）
2. Dueling architecture
3. Prioritized replay
4. **n-step (n=3)** ⭐
5. Distributional RL (C51)
6. Noisy nets（exploration）
7. DQN base

**每个组件涨 5-10%，加起来涨 ~200%**。

#### 反直觉点
- **n=3 是个魔术数字**——没理论指导，工程实验出来
- **Rainbow 是 RL trick 集大成**——但每个 trick 都源自一篇独立论文

#### 高频面试问法
**Q：Rainbow 为什么用 n=3 n-step？**
**A**：(1) **bias-variance 工程甜点**（实证扫出来）；(2) n>3 → 显存爆（buffer trajectory）；(3) n<3 → 信用分配慢。**n=3 是 deep RL n-step 的经验最佳**。

---

## 4. 跨概念对比表

### 4.1 n-step family tree
```
n-step Return G_{t:t+n}
├── n-step TD Prediction（V）
├── n-step SARSA（on-policy）
│   └── n-step Expected SARSA（降方差）
└── n-step Off-policy
    ├── + IS（trajectory）→ 方差爆
    ├── + Per-decision IS → 方差中
    ├── Tree Backup（无 IS）→ 有 bias
    └── Q(σ) → 统一框架
```

### 4.2 选型表
| 场景 | 推荐 |
|---|---|
| Tabular small MDP | n-step SARSA (n=5) |
| Atari (deep RL) | Rainbow n-step (n=3) |
| Off-policy + 大 n | Tree Backup |
| LLM RLHF | GAE λ=0.95（[[ch12-eligibility-traces]] §3.7）|

---

## 5. 习题精选

**Exercise 7.2**：证明 G_{t:t+n} 期望随 n 变化趋势。
**Exercise 7.3**：在 19-state random walk 上扫 n 和 α，复现 figure 7.2。
**Exercise 7.10**：off-policy n-step with control variate。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter07/random_walk.py` —— 复现 figure 7.2
- **CleanRL Rainbow**：看 n-step Q-learning 工程实现

## 7. 跟 RLHF 的桥

| RLHF 组件 | 来自本章 §3.x |
|---|---|
| **GAE = λ-加权 n-step** | §3.1 + [[ch12-eligibility-traces]] §3.7 |
| 长 CoT RLHF 大 n | §3.1（稀疏 reward 必大 n）|
| **GRPO 用 MC（n=∞）style** | §3.1 极限 |

## 8. 苏格拉底 Q&A

详见每个 §3.x。综合：

- **Q：n-step 在 deep RL 主流，eligibility traces 不主流——为什么？**
  - 答：n-step buffer 简单（存 n 步即可），跟 NN backprop 兼容；eligibility trace 跟 NN 参数同 shape（百万级），维护贵
- **Q：从 n-step 到 TD(λ) 到 GAE 的演进**
  - 答：n-step 固定 n（不灵活）→ TD(λ) 用 λ 平滑混合（优雅）→ GAE 用 λ-加权 TD errors 估 advantage（PG 用）

## 9. 自测清单（闭卷）

- [ ] **写 n-step return 公式**
- [ ] **写 n-step SARSA 算法**
- [ ] **解释 n 在 TD-MC 谱上的位置**
- [ ] **解释 off-policy n-step IS 为什么方差爆**
- [ ] **写 Tree Backup 公式**
- [ ] **解释 GAE 跟 n-step 关系**（RLHF 桥）

---

## 💡 拓展知识点（书外 trivia）

### A. n-step 思想最早是 Watkins 1989 的"truncated returns"
Watkins 1989 PhD（Q-learning 同篇）就讨论了 truncated n-step return。**好想法常被人发明但等正确时机才"被看见"**。

### B. n-step 在 deep RL 复兴
- **Rainbow** 整合 n=3
- **A3C/A2C** 用 n-step bootstrap
- **R2D2 / Ape-X** 分布式 RL 都标配 n-step

### C. 为什么 n=3 是 Rainbow 选的"魔术数字"
实证扫 n=1, 2, 3, 5, 10 在 Atari 上——n=3 是工程甜点（详 §3.8）。

### D. n-step 跟 TD(λ) 哪个赢了
**实战 deep RL 主流是 n-step**——λ 实现需要 eligibility trace（per-feature 累积），跟 NN backprop 不太兼容；n-step 实现简单。**GAE = best of both**（λ + n-step 思想合一）。

### E. n-step off-policy 的"悲剧"
- 纯 IS：方差指数爆
- Per-decision IS：好但仍受限
- Tree Backup：无 IS，但有 bias
- 实战：**PPO 单步 IS + clip 是折中赢家**

### F. n-step 跟 Transformer attention 的远亲
- n-step return 是"看未来 n 步加权"
- Transformer attention 是"看 sequence 内所有 token 加权"
- 都是"用一段历史/未来加权回到当前"
- **不是同回事，但思想模式相似**——RL 跟 sequence modeling 在 architecture 哲学上有交集

### G. GAE 的关键洞察（再强调）
$$A^{GAE}_t = \sum_{l=0}^∞ (γλ)^l δ_{t+l}$$
= 所有 n-step advantage 的 λ-加权平均。**PPO λ=0.95 是工业最优**。

---

## 🔗 更多 wikilinks（深挖）

| 主题 | 跳转 |
|---|---|
| TD(λ) 优雅版 | [[ch12-eligibility-traces]] §3.1-3.3 |
| GAE = n-step 思想用到 advantage | [[ch13-policy-gradient]] §3.6 |
| Off-policy n-step 噩梦 | [[ch11-off-policy-methods]] §3.3 |
| Rainbow 用 n-step | [[../../../brain/Areas/rl-books/deep-rl-hands-on-lapan-3e/_overview\|Lapan Ch 8]] |
| GRPO 用 MC（n=∞）思想 | [[ch13-policy-gradient]] §3.9 |

## Related

- **上一章**：[[ch06-temporal-difference-learning]]
- **下一章**：[[ch08-planning-and-learning]]
- **延伸**：[[ch12-eligibility-traces]]（λ-加权）· [[ch13-policy-gradient]]（GAE）
- **进度**：[[_chapter-status]] W3

## Sources

- 原书 Ch 7 pp. 141-156
- [Silver UCL Lec 5](https://www.youtube.com/watch?v=0g4j2k_Ggc4)
- ShangtongZhang `chapter07/`
- **核心论文**：
  - GAE: [Schulman 2015 arXiv 1506.02438](https://arxiv.org/abs/1506.02438)
  - Rainbow: [Hessel 2018 arXiv 1710.02298](https://arxiv.org/abs/1710.02298)
  - A3C: [Mnih 2016 arXiv 1602.01783](https://arxiv.org/abs/1602.01783)
