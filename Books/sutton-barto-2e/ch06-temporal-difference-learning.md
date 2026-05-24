---
title: "Sutton & Barto Ch 6. Temporal-Difference Learning"
created: 2026-05-24
tags: [area/rl, type/note, status/v3, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 6
difficulty: hard
budget_days: 3-4
priority_for_rlhf: must
silver_lecture: 4
---

# Ch 6. Temporal-Difference Learning ⭐⭐⭐

---

## 📋 本章讲了什么（TL;DR）

1. **TD = MC 的"等结束才更新"改成"下一步就更新"**——结合 DP 的 bootstrap 和 MC 的 model-free，online + 单步增量
2. **SARSA（on-policy）vs Q-learning（off-policy）**两大支系奠定 control 方法分裂；cliff walking 完美对比
3. **TD error δ_t = R + γV(S') - V(S)** 是后续 actor-critic / GAE / PPO advantage 的共同根

**为什么这是 RL 中心章**：Sutton 本人称 TD 为 *"the central and novel idea of reinforcement learning"*。**PPO 的 critic 就是 TD V learner**——本章直接进 RLHF 工业链。

## 🎯 Top 5 关键洞察

| # | 洞察 | 为什么关键 |
|---|---|---|
| 1 | **TD 跟 MC 的 bias-variance 完全相反** | MC 无 bias 高方差；TD 有 bias 低方差——RL 算法核心 trade-off |
| 2 | **Q-learning vs SARSA 算法上只差一处（max vs sample）** | 但导致 on-policy vs off-policy 本质分歧 |
| 3 | **Cliff walking：SARSA 学"安全 path"，Q-learning 学"最优 path"但执行崩** | "学到的 ≠ 执行的"是 RL 工程深坑 |
| 4 | **Q-learning over-estimation bias**（Jensen 不等式）| max E[Q] ≥ E[max Q] → Double DQN |
| 5 | **TD 是 actor-critic 的 critic 部分** | PPO critic 用 TD bootstrap 训 |

## 🧭 在 Obsidian graph 中的位置

- **上游**：[[ch04-dynamic-programming]]（借 DP bootstrap）· [[ch05-monte-carlo-methods]]（借 MC model-free）
- **横向**：[[ch07-n-step-bootstrapping]]（TD/MC 谱中间）· [[ch12-eligibility-traces]]（λ 平滑混合）
- **下游**：**[[ch13-policy-gradient]]**（AC 的 critic 来自本章）· [[ch11-off-policy-methods]]（Q-learning + FA = deadly triad）
- **DQN 关联**：[[../../../brain/Areas/rl-books/deep-rl-hands-on-lapan-3e/_overview|Lapan Ch 6-8]]

---

## 0. 阅读元信息

- **难度**：hard（概念密集 + Q-learning 是 RL 最重要算法）
- **预算**：3-4 天
- **必读理由**：TD 是 RL 中心思想；Q-learning 史上最有影响的 RL 算法；**面试必考**
- **配套**：Silver UCL Lec 4 后半 + Lec 5

## 1. 一句话 + 在 RL 体系中的位置

TD = **MC 的 episode-end update 改成 single-step update**——用 $R + γV(S')$ 当 target（bootstrap），不需等 episode 结束，可在线学；衍生出 SARSA / Q-learning / Expected SARSA / Double Q 等大家族。

**在 RL 算法家族中的位置**：
```
RL Value-based
├── DP (Ch 4)            ← 知 model
├── MC (Ch 5)            ← 无 model, 等 episode 结束
├── TD (Ch 6) ⭐          ← 无 model, 单步增量
│   ├── SARSA            ← on-policy
│   ├── Q-learning       ← off-policy
│   ├── Expected SARSA   ← 降方差
│   └── Double Q         ← 解 over-estimation
└── n-step / TD(λ) (Ch 7, 12)  ← MC ↔ TD 谱
```

## 2. 概念地图（§3 详解目录）

| # | 概念 | §3.x | 重要度 |
|---|---|---|---|
| 1 | **TD(0) Prediction** | §3.1 | ⭐⭐⭐ TD 入门 |
| 2 | **SARSA** | §3.2 | ⭐⭐⭐ on-policy 标杆 |
| 3 | **Q-learning** | §3.3 | ⭐⭐⭐⭐ **史上最有影响** |
| 4 | **Expected SARSA** | §3.4 | ⭐⭐ SARSA 优化版 |
| 5 | **Maximization Bias** | §3.5 | ⭐⭐⭐ Q-learning 的根本病 |
| 6 | **Double Q-learning** | §3.6 | ⭐⭐⭐ 修复 max bias |
| 7 | **TD vs MC 全方位对比** | §3.7 | ⭐⭐ bias-variance 经典 |
| 8 | **DQN（preview Ch 11）** | §3.8 | ⭐⭐⭐ Q-learning deep 版 |

---

## 3. 核心概念详解（每个从 0-1 讲清）

### 3.1 TD(0) Prediction

#### 是什么
最基础的 TD：估计 V_π(s)——给定 policy π，用单步 bootstrap 更新 V 估计。

#### 为什么需要它
- DP 要 model（知 p(s', r | s, a)），现实少
- MC 不要 model，但必须等 episode 结束 + 高方差
- **TD：不要 model + online 增量 + 低方差**——综合两者优点

#### 数学推导

**核心更新公式**：
$$V(S_t) \leftarrow V(S_t) + α \big[\underbrace{R_{t+1} + γ V(S_{t+1})}_{\text{TD target}} - V(S_t)\big]$$

TD error：
$$δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)$$

**推导直觉**：
- 真实 V_π(s) 满足 Bellman: V_π(s) = E[R + γV_π(S') | S=s]
- 用一次 sample (R, S') 估计 RHS → TD target
- 朝 TD target 方向走一小步 α

**为什么 TD 有 bias**：V(S') 是估计值，不准则 target 也不准（bootstrap 误差累积）。
**为什么 TD 方差小于 MC**：MC 用全程随机 G_t；TD 只用一步随机 R + bootstrap V(S')（V 是固定函数 → 方差小）。

#### 伪代码
```python
def TD0_prediction(env, policy, alpha=0.1, gamma=0.99, n_episodes=1000):
    V = defaultdict(float)
    V[TERMINAL] = 0
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = policy(s)
            s_next, r, done = env.step(a)
            V[s] += alpha * (r + gamma * V[s_next] - V[s])
            s = s_next
    return V
```

#### 实战参数
| 超参 | 推荐 | 调参直觉 |
|---|---|---|
| α | 0.01 - 0.1 | 太大不稳，太小慢 |
| γ | 0.9 - 0.99 | episodic 短任务用 0.9 |
| n_episodes | 1000+ | tabular 任务足够 |

#### 跟相关算法对比
| 维度 | MC | TD(0) | DP |
|---|---|---|---|
| 需 model | ❌ | ❌ | ✅ |
| Online | ❌ | ✅ | ✅ |
| Bootstrap | ❌ | ✅ | ✅ |
| Bias | 0 | 有 | 0 |
| Variance | 大 | 小 | 0 |

#### 反直觉点
- **TD 跟 MC 的"剪刀差"**（Sutton 图 6.2）：早期 MC 快，后期 TD 反超
- **TD 在 batch setting 下学到的 V 跟 MC 不同**——MC 学 ML，TD 学 certainty-equivalence model 的 V

#### 高频面试问法
**Q：TD(0) 跟 MC bias-variance 特性？**
**A**：MC 无 bias（用真 G_t）但高方差（全程随机）；TD(0) 有 bias（bootstrap V(s') 不准则 target 不准）但低方差（只一步随机）。

---

### 3.2 SARSA（on-policy TD control）

#### 是什么
TD prediction 升级为 control——学 Q(s, a) 而非 V(s)，用 **on-policy** 方式（A_{t+1} 按当前 ε-greedy policy 采）。

#### 为什么需要它
- 想学最优 policy，不只是 V 估计
- 需要 Q(s, a) 才能决策（argmax_a Q）
- **On-policy**：用当前 policy 采的 (S, A, R, S', A') → 学这个 policy 的 Q

#### 数学
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + α \big[ R_{t+1} + γ Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \big]$$

**名字来源**：SARSA = **S**tate-**A**ction-**R**eward-**S**tate-**A**ction（5 个元素更新一次）。

#### 伪代码
```python
def SARSA(env, alpha=0.1, gamma=0.99, eps=0.1, n_episodes=500):
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    for ep in range(n_episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, eps)
        done = False
        while not done:
            s_next, r, done = env.step(a)
            a_next = epsilon_greedy(Q, s_next, eps)
            Q[s][a] += alpha * (r + gamma * Q[s_next][a_next] - Q[s][a])
            s, a = s_next, a_next
    return Q
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| α | 0.1 - 0.5 |
| γ | 0.99 |
| ε | 0.1（or decay）|
| Q(terminal) | 0 |

#### 跟相关算法对比
| | SARSA | Q-learning | Expected SARSA |
|---|---|---|---|
| Target action | sample $A_{t+1}$ | $\max_a Q$ | $E_a Q$ |
| On/off-policy | on | off | on（也可 off）|
| 方差 | 中 | 高（max 引入）| 低 |
| Cliff walking | 安全 path | 最优但执行崩 | 介于 |

#### 反直觉点
- **SARSA 学"我会 ε-greedy 走"的最优 Q**——不是 Q\*
- **on-policy 是 feature**——能学到"考虑 exploration 的 policy"

#### 高频面试问法
**Q：SARSA 是 on-policy 的根本原因？**
**A**：target 用 sample 出来的 $A_{t+1}$（按当前 policy），不是 $\max_a Q$。所以 update 反映当前 policy 行为 → 学 Q^π 而非 Q^*。

---

### 3.3 Q-learning（off-policy TD control）⭐⭐⭐⭐

#### 是什么
**Watkins 1989** 提出，**RL 史上最有影响的算法**之一。SARSA 的 off-policy 版——target 用 $\max_a Q(S_{t+1}, a)$ 而非 sample。

#### 为什么需要它
- SARSA 学 Q^π，不是 Q\*
- 想直接学 Q\*——用 max_a 当 target 就是 Q\* 的 Bellman optimality
- **off-policy**：execution policy 可以是 ε-greedy（探索）；target policy 是 greedy（学最优）

#### 数学

$$\boxed{Q(S_t, A_t) \leftarrow Q(S_t, A_t) + α \big[ R_{t+1} + γ \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \big]}$$

**关键区别**：
- SARSA: $γ Q(S', A')$（A' 是 sample 的）
- Q-learning: $γ \max_{a'} Q(S', a')$（a' 是 argmax）

**为什么是 off-policy**：target $\max_a Q$ 跟 behavior policy 选的 $A_{t+1}$ 无关。

#### 伪代码
```python
def Q_learning(env, alpha=0.1, gamma=0.99, eps=0.1, n_episodes=500):
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = epsilon_greedy(Q, s, eps)
            s_next, r, done = env.step(a)
            Q[s][a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a])
            s = s_next
    return Q
```

#### 实战参数
跟 SARSA 几乎相同；通常 α 可稍小（max 引入额外 noise）。

#### 跟相关算法对比
| | SARSA | Q-learning |
|---|---|---|
| Target | sample A' | max_a Q |
| On/off-policy | on | **off** |
| 学的 Q | Q^π | **Q\*** |
| Cliff walking | 走远路 | 走悬崖边 |
| Over-estimation | 无 | **有**（Jensen）|
| Deep 版 | 较少 | **DQN/Double DQN/Rainbow** |

#### 反直觉点
- **算法上只差一处（max vs sample）但本质不同**
- **学到 Q\* 不代表执行最优**——还要看 execution policy
- **Cliff walking 故事**（Example 6.6）：Q-learning 学到沿悬崖最短路径，但执行时 ε-greedy 偶尔随机推下悬崖 → 在线 reward 反而比 SARSA 差
- **Q-learning 在 tabular 收敛**（Watkins-Dayan 证明），**但 + FA 可能崩**（[[ch11-off-policy-methods|deadly triad]]）

#### 高频面试问法

**Q1：写 Q-learning 更新公式**
**A**：$Q(s, a) ← Q(s, a) + α [r + γ \max_{a'} Q(s', a') - Q(s, a)]$

**Q2：Q-learning vs SARSA 本质区别？**
**A**：算法上只差 max vs sample。max 让 target 不依赖 behavior policy → off-policy 学 Q\*；sample 让 target 跟随 behavior → on-policy 学 Q^π。

**Q3：Cliff walking 上 Q-learning 在线表现差于 SARSA 矛盾吗？**
**A**：不矛盾——Q-learning 学到"最优 deterministic"沿悬崖；执行时 ε-greedy 偶尔随机掉。SARSA 学到"远离悬崖的 robust 路径"。**教训：学到的 policy ≠ 执行的 policy**。

**Q4：DQN 是 Q-learning 吗？**
**A**：是——DQN = Q-learning + NN + 工程 trick（target net / replay / clip）。详 §3.8。

---

### 3.4 Expected SARSA（降方差）

#### 是什么
SARSA 的优化版——target 用 $E_a[Q(S', a)]$ 替代 sample $Q(S', A')$。

#### 为什么需要它
- SARSA target 用一次 sample → 方差大
- policy π 已知，直接算期望就好
- → Expected SARSA

#### 数学
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + α \big[ R_{t+1} + γ \sum_a π(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \big]$$

#### 伪代码
```python
def expected_sarsa_update(Q, s, a, r, s_next, eps, alpha, gamma):
    # E[Q(s', a)] = ε * mean + (1-ε) * max  (for ε-greedy)
    q_next = (1 - eps) * np.max(Q[s_next]) + eps * np.mean(Q[s_next])
    Q[s][a] += alpha * (r + gamma * q_next - Q[s][a])
```

#### 实战参数
跟 SARSA 同，通常 α 可比 SARSA 大（方差小允许大步长）。

#### 跟相关算法对比
| | SARSA | Expected SARSA | Q-learning |
|---|---|---|---|
| Target | sample Q(s',a') | $E_a Q$ | $\max_a Q$ |
| 方差 | 中 | **小** | 大 |
| Bias | 无 | 无 | over-estimation |
| 通常 | 好 | **更好** | 好（有 over-est）|

#### 反直觉点
- **Expected SARSA 通常严格优于 SARSA**——降方差不引 bias
- **可 on-policy 也可 off-policy**——target 用的 π 可跟 behavior 不同
- **target policy = greedy 时 → 退化为 Q-learning**

#### 高频面试问法
**Q：Expected SARSA 比 SARSA 严格更好吗？例外？**
**A**：一般是——同 bias 但方差小。例外：算 expectation 成本高的大 action space。

---

### 3.5 Maximization Bias（最大化偏差）⭐⭐⭐

#### 是什么
Q-learning 的根本病——**$\max_a E[\hat{Q}(s, a)] \leq E[\max_a \hat{Q}(s, a)]$**（Jensen），导致 Q estimate 系统性偏高。

#### 为什么需要懂
- Q-learning 经常"过度乐观"——Q 长大但实际 policy 不优
- stochastic 环境上尤其严重
- 不懂这个就理解不了 Double Q / Double DQN

#### 数学
**Jensen 不等式**：对凸函数 f，$f(E[X]) \leq E[f(X)]$。max 是凸函数。

设 $\hat{Q}$ 是真 Q 的无偏估计：
$$E\left[\max_a \hat{Q}(s, a)\right] \geq \max_a E[\hat{Q}(s, a)] = \max_a Q(s, a)$$

**直觉**：用同一份 Q **选 action** 和 **估 value**——某 action 估计偏高（noise）就被 max 选中且其偏高值当 target → noise 被系统放大。

#### 经典 Example 6.5
两 state MDP：A → B；B 有 100 个 action，reward N(-0.1, 1)。真实最优 = A 直接 terminate (reward 0)。
但 Q-learning 经常学到"先去 B"——因为 max over 100 noisy estimate 容易找到偶然高的。

#### 数值证明（小实验）
```python
import numpy as np
np.random.seed(42)
true_q = np.zeros(100)
noisy_q_estimates = true_q + np.random.randn(1000, 100) * 1.0
max_of_estimates = noisy_q_estimates.max(axis=1).mean()
print(f"E[max Q-hat] = {max_of_estimates:.3f}")  # ~2.5
print(f"max E[Q] = 0; Over-estimation = {max_of_estimates:.3f}")
```

跑出来 ≈ 2.5——**~2.5 的 systematic over-estimation**（真实值应该是 0）。

#### 跟相关算法对比
| | 同 Q 选+评估 | over-est |
|---|---|---|
| Q-learning | 是 | **严重** |
| Double Q | 两 Q 独立选/评估 | 几乎无 |
| SARSA | 不用 max | 无 |
| Double DQN | online + target 分开 | 显著降 |

#### 反直觉点
- **stochastic 环境严重**——deterministic noise 小不显著
- **不阻碍收敛**——但 Q 持续偏高
- **影响在线 reward**——选 action 时高估某些 → 不一定真好

#### 高频面试问法
**Q：Q-learning over-estimation 来自哪？**
**A**：Jensen 不等式：$E[\max_a \hat{Q}] \geq \max_a E[\hat{Q}]$。用同一份 Q 选 action 和 估 value，noise 被 max 选中放大。

---

### 3.6 Double Q-learning（修复 max bias）⭐⭐⭐

#### 是什么
**van Hasselt 2010** 提出。两个独立 Q 函数 Q1, Q2 交替更新——一个选 action，另一个估 value。

#### 为什么需要它
- Q-learning 的 max 引入 systematic over-est
- 用两个独立估计：Q1 选 a*（认为它好），但 Q2 给的估计值无偏（Q2 没参与 a* 选择）
- 期望意义上无 over-est

#### 数学

每步随机更新 Q1 或 Q2：

**Update Q1**：
$$Q_1(s, a) ← Q_1(s, a) + α [r + γ Q_2(s', \arg\max_{a'} Q_1(s', a')) - Q_1(s, a)]$$

**Update Q2**：类似，Q1, Q2 互换。

**关键**：选 action 用一个 Q（argmax），估 value 用**另一个 Q**（取那个 action 的值）。

#### 伪代码
```python
def double_Q_learning(env, alpha=0.1, gamma=0.99, eps=0.1, n_episodes=500):
    Q1 = defaultdict(lambda: np.zeros(env.n_actions))
    Q2 = defaultdict(lambda: np.zeros(env.n_actions))
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            Q_combined = (Q1[s] + Q2[s]) / 2
            a = epsilon_greedy_on(Q_combined, eps)
            s_next, r, done = env.step(a)
            if np.random.random() < 0.5:
                a_star = np.argmax(Q1[s_next])
                Q1[s][a] += alpha * (r + gamma * Q2[s_next][a_star] - Q1[s][a])
            else:
                a_star = np.argmax(Q2[s_next])
                Q2[s][a] += alpha * (r + gamma * Q1[s_next][a_star] - Q2[s][a])
            s = s_next
    return Q1, Q2
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| α | 跟 Q-learning 同 |
| Coin flip prob | 0.5 |
| Behavior policy | ε-greedy on (Q1+Q2)/2 |

#### 跟相关算法对比
| | Q-learning | Double Q | Double DQN |
|---|---|---|---|
| Q 函数数 | 1 | **2 独立** | 2（online + target 复用）|
| Over-est | 严重 | 几乎无 | 显著降 |
| 内存 | 1x | 2x | ~1x |
| 实现 | 简单 | 中 | 简单 |
| Atari 表现 | baseline | n/a | **+10-50% over DQN** |

#### 反直觉点
- **不要把 Q1, Q2 平均当 Q\***——各自仍有 noise，但 noise 不相关 → max 选择无系统偏差
- **Double DQN 比 Double Q 更"懒"**——直接用 target network 当第二个 Q，几乎免费
- **Atari Double DQN 降 over-est ~5x，得分涨 ~30%**

#### 高频面试问法

**Q1：Double Q vs Double DQN 区别？**
**A**：Double Q 维护两个**独立**训练的 Q 表（tabular）；Double DQN 用现成的 **target network**当第二个 Q（NN）——后者实现简单效果接近。

**Q2：Double Q 为什么不 over-estimate？**
**A**：Q1 选的 action 在 Q2 看来是"任意"的（Q2 没参与 Q1 训练 → 它对该 action 的估计无偏）。所以 $E[Q_2(s', \arg\max Q_1)] = Q^*$。

---

### 3.7 TD vs MC 全方位对比

#### 是什么
不是新算法，而是**理解 TD/MC 设计权衡的总结**——所有后续 RL 方法都在 spectrum 上选位置。

#### 8 维对比

| 维度 | MC | TD(0) | n-step TD |
|---|---|---|---|
| **等 episode 结束** | ✅ 必须 | ❌ | ❌ k 步 |
| **bootstrap** | ❌ | ✅ | ✅ 部分 |
| **Bias** | 0 | 大 | 中 |
| **Variance** | 大 | 小 | 中 |
| **Online** | ❌ | ✅ | 延迟 k 步 |
| **Continuing task** | ❌ | ✅ | ✅ |
| **Tabular 收敛** | ✅ | ✅ | ✅ |
| **FA 收敛** | ✅ | 可能崩（deadly triad）| 同 TD(0) |

#### Random Walk 实测（Sutton 图 6.2）
- α 小：MC 和 TD 都收敛，TD 略快
- α 大：MC 不稳，TD 仍能收敛
- 长期：TD 更精

#### 高频面试问法
**Q：什么场景用 MC vs TD？**
**A**：
- **MC**：(1) episode 短 reward 集中末尾；(2) 不能容忍 bias（OPE 评估）；(3) FA 难收敛时退守
- **TD**：(1) continuing task 必用；(2) 长 episode 想 online 学；(3) 现代 deep RL 默认（PPO critic / DQN）

---

### 3.8 DQN（Deep Q-Network，⭐⭐⭐）

#### 是什么
**Mnih 2013/2015**：Q-learning + CNN + 工程 trick = 第一个 end-to-end pixel → Atari 超人类 RL。

#### 为什么需要它
- Q-learning + NN 直接 = [[ch11-off-policy-methods|deadly triad]] → 可能发散
- DQN 用一组工程 trick "暴力镇压" deadly triad
- 开启 deep RL 时代

#### 数学（跟 Q-learning 同）
$$L(\theta) = E\left[\left(r + γ \max_{a'} Q_{\theta^-}(s', a') - Q_θ(s, a)\right)^2\right]$$

注意 target Q 用 **target network** $\theta^-$（trick 1）。

#### DQN 工程 trick 全家桶

| Trick | 解决什么 | 怎么做 |
|---|---|---|
| **Target network** | 破 deadly triad bootstrap 反馈环 | freeze $θ^-$ N 步 |
| **Experience replay** | 破 sample 相关性 | buffer 存 (s,a,r,s'), 随机采 batch |
| **Reward clipping** | 跨游戏 scale 归一 | clamp r ∈ [-1, 1] |
| **Frame stacking** | 近似 Markov | concat 4 帧当 state |
| **Huber loss** | robust to outlier | 替代 MSE |

#### 伪代码（DQN 主循环）
```python
def DQN(env, q_net, target_net, optimizer, replay, n_steps=int(1e6),
        gamma=0.99, eps_init=1.0, eps_final=0.1, target_update_freq=10000):
    s = env.reset()
    eps = eps_init
    for step in range(n_steps):
        # 1. ε-greedy action
        a = epsilon_greedy(q_net(s), eps)
        s_next, r, done = env.step(a)
        replay.add((s, a, np.clip(r, -1, 1), s_next, done))
        s = s_next if not done else env.reset()
        # 2. Update q_net every step (after warmup)
        if step > 1000:
            batch = replay.sample(32)
            with torch.no_grad():
                target_q = batch.r + gamma * target_net(batch.s_next).max(1)[0] * (1 - batch.done)
            pred_q = q_net(batch.s).gather(1, batch.a.unsqueeze(1)).squeeze()
            loss = F.smooth_l1_loss(pred_q, target_q)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        # 3. Update target net periodically
        if step % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())
        # 4. ε decay
        eps = max(eps_final, eps - (eps_init - eps_final) / 1e6)
```

#### 实战参数（Atari benchmark）
| 超参 | 推荐 |
|---|---|
| Replay buffer | 1M |
| Batch size | 32 |
| Target update freq | 10000 |
| ε schedule | 1.0 → 0.1 over 1M |
| Adam lr | 1e-4 |
| Loss | Huber |
| Frame stack | 4 |
| Gamma | 0.99 |
| Total steps | 50M |

#### 跟相关算法对比
| | Q-learning | DQN | Double DQN | Rainbow |
|---|---|---|---|---|
| FA | None | NN | NN | NN |
| Over-est | 有 | **严重** | 减 | 减 |
| Sample 效率 | n/a | 中 | 中 | **高** |
| Atari 平均 | n/a | baseline | +30% | **+200%** |

**Rainbow** (Hessel 2018) 整合：Double / Dueling / Prioritized / N-step / Distributional (C51) / Noisy nets / DQN base。

#### 反直觉点 / 实战陷阱
- **Target update 太频繁** → bootstrap target 一直动 → divergent
- **Replay buffer 太小** → 不够 break correlation
- **DQN 不能直接用 Adam 默认 lr (1e-3)** → 太大；用 1e-4 - 5e-5
- **Reward clip 在某些游戏伤性能**（高分游戏，clip 把"非常好"压到 1）

#### 高频面试问法

**Q1：DQN 是 Q-learning 吗？区别？**
**A**：DQN = Q-learning + NN + 工程 trick。Q-learning tabular 理论收敛；DQN 用 NN 进入 deadly triad，靠 trick 让它实际稳。

**Q2：DQN 的 target network 解决什么？**
**A**：bootstrap target $r + γ \max_a Q(s', a)$ 依赖 θ → 改 θ 同时改 target → 自反馈循环可能发散。Target network 冻结 θ⁻ N 步 → target 暂时变 supervised → 稳。

**Q3：为什么 DQN 替代 tabular Q-learning？**
**A**：tabular 在 state space 大时不可能（Atari 像素 state ~10^7800）。NN 提供 generalization + end-to-end 学 feature → 真实问题唯一可行的 Q-learning。

---

## 4. 跨概念对比表

### 4.1 本章算法 family tree
```
TD(0) Prediction (V learning)
└── TD Control
    ├── SARSA (on-policy)
    │   └── Expected SARSA (降方差)
    ├── Q-learning (off-policy, 学 Q*)
    │   ├── Double Q-learning (修 max bias)
    │   └── DQN (deep 版)
    │       └── Double DQN / Rainbow (Atari SOTA)
    └── n-step / TD(λ) (Ch 7, 12)
```

### 4.2 选型表
| 场景 | 推荐 |
|---|---|
| Tabular 小问题 | Q-learning / Expected SARSA |
| Cliff-walking 类风险敏感 | SARSA |
| Atari / 大 state space | DQN + Rainbow trick |
| LLM RLHF | **❌ 不用 Q-learning**（详 §7）|
| Continuous control | SAC（[[ch13-policy-gradient]] §3.12）|

---

## 5. 习题精选

**Exercise 6.4（step size α）**：Cliff walk 上比较 α=0.1 / 0.5 / 0.9。
**Exercise 6.6（cliff walking 反思）**：见 §3.3 反直觉。
**Exercise 6.10（windy gridworld + king's moves）**：实现。
**Exercise 6.14（double Q-learning bias）**：见 §3.5 数学。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter06/random_walk.py`、`cliff_walking.py`、`windy_grid_world.py`、`maximization_bias.py`
- **必跑 cliff_walking.py**——看 SARSA vs Q-learning 差异
- **自己实现 Q-learning on FrozenLake-v1**：20 行 numpy

## 7. 跟 RLHF 的桥

| RLHF 组件 | 来自本章 §3.x |
|---|---|
| PPO critic V_φ 训练 | §3.1 TD(0) prediction |
| GAE 的 TD error δ_t | §3.1 |
| PPO 用 GAE 而非 Q | §3.7（TD vs MC 选择）|
| **为什么 RLHF 不用 Q-learning？** | (1) action = vocab (50k)，max 贵；(2) reward 稀疏，bootstrap 几乎全猜；(3) LLM 必须 stochastic → §3.3 优势消失 |
| Target network 思想 → DPO ref model | DPO 的 frozen ref 是 target net 精神延伸 |

## 8. 苏格拉底 Q&A

详见每个 §3.x。综合题：

- **Q：从 TD(0) 到 Q-learning 到 DQN 到 Rainbow，每步解决什么前步痛点？**
  - 答：TD(0)→Q-learning：从 V 到 Q（决策友好）；Q→DQN：tabular → 大 state space；DQN→Rainbow：单 trick → 多 trick 集大成
- **Q：手写 SARSA / Q-learning / Expected SARSA / Double Q 4 公式**
- **Q：cliff walking 故事的工程教训？**
  - 答：学到的 policy ≠ 执行的 policy；选算法考虑 execution noise

## 9. 自测清单（闭卷）

- [ ] **手写 TD(0) / SARSA / Q-learning / Expected SARSA 4 公式**
- [ ] **解释 on-policy vs off-policy 差异**
- [ ] **解释 Q-learning over-est 来源**（Jensen）
- [ ] **写 Double Q-learning 算法**（含 coin flip）
- [ ] **解释 cliff walking 故事**
- [ ] **写 DQN 5 个 trick**
- [ ] **实现 Q-learning on FrozenLake** 跑通
- [ ] **解释为什么 RLHF 不用 Q-learning**

---

## 💡 拓展知识点（书外 trivia）

### A. Q-learning 发明人 Watkins 1989 PhD 论文
**Chris Watkins** Cambridge 1989 PhD 论文《Learning from Delayed Rewards》正式化 Q-learning，证明 tabular 收敛性。论文当时几乎无引用——RL 是 niche。Watkins 之后转去做投资。**"一篇论文改变领域"经典例**。

### B. Q-learning 曾申请专利
1990s Watkins 把 Q-learning 申请过专利（已过期）——RL 圈罕见。之后惯例确立所有 RL 算法开源。

### C. SARSA 名字
SARSA = S-A-R-S-A 5 元素。同期还有 R-learning（Schwartz 1993）等被遗忘。

### D. DQN 当年轰动
**Mnih 2013 NIPS Workshop** 论文展示 DQN 学会玩 Atari Breakout——视频 YouTube 几天百万播放。2015 Nature 主刊版让 RL 出圈。

### E. Rainbow 集成
**Hessel 2018 Rainbow** (AAAI) arXiv 1710.02298 整合 7 改进：Double Q / Dueling / Prioritized replay / Multi-step / Distributional C51 / Noisy nets / DQN base。**每个涨 5-10%，加起来涨 ~200%**。

### F. TD vs MC 的剪刀差（Sutton 图 6.2）
- 早期 MC 学得快
- 中后期 TD 反超
- batch setting 下 TD 学 certainty-equivalence model 的 V（跟 MC 不同）

### G. Cliff walking 早期心理学渊源
Cliff walking 不是 Sutton 发明——Watkins 1989 PhD 就有类似 grid world。**RL 早期实验设计借鉴心理学动物迷宫**。

### H. Q-learning 在围棋上失败
2000s 多次尝试 Q-learning + NN 玩围棋——都失败。原因：(1) action 太大（19×19=361）；(2) reward 极稀疏；(3) deadly triad。**AlphaGo 用 PG + MCTS + supervised 才突破**。

### I. DQN target network 思想被 DPO 借鉴
DQN 的 target net = 冻结的"参考"Q net。**DPO 的 reference model** = 冻结的 SFT model——思想同源（防止 bootstrap 自反馈）。

---

## 🔗 更多 wikilinks（深挖）

| 主题 | 跳转 |
|---|---|
| Q-learning + FA = deadly triad | [[ch11-off-policy-methods]] |
| DQN deep 版工程 | [[../../../brain/Areas/rl-books/deep-rl-hands-on-lapan-3e/_overview\|Lapan Ch 6]] |
| Distributional Q-learning (C51) | [[../../../brain/Areas/rl-books/distributional-rl-bellemare/_overview]] |
| AC 的 critic = TD V learner | [[ch13-policy-gradient]] §3.4 |
| TD error → GAE → PPO | [[ch12-eligibility-traces]] |
| Maximization bias 数学根 | [[ch11-off-policy-methods]] §11.3 |
| 为什么 LLM RLHF 不用 Q-learning | [[ch13-policy-gradient]] §7 |

## Related

- **上一章**：[[ch05-monte-carlo-methods]]（MC 对比）
- **下一章**：[[ch07-n-step-bootstrapping]]（TD/MC 谱中间）
- **延伸**：[[ch11-off-policy-methods]]（Q-learning + FA 的 deadly triad）· [[ch13-policy-gradient]]（AC 的 critic）
- **进度**：[[_chapter-status]] W3 ⭐

## Sources

- 原书 Ch 6 pp. 117-138
- [Silver UCL Lec 4-5](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- ShangtongZhang `chapter06/`
- **核心算法论文**：
  - Q-learning：Watkins 1989 PhD thesis
  - DQN: [Mnih 2013 arXiv 1312.5602](https://arxiv.org/abs/1312.5602)、[Mnih 2015 Nature](https://www.nature.com/articles/nature14236)
  - Double DQN: [van Hasselt 2015 arXiv 1509.06461](https://arxiv.org/abs/1509.06461)
  - Double Q-learning（tabular）: van Hasselt 2010 NeurIPS
  - Rainbow: [Hessel 2018 arXiv 1710.02298](https://arxiv.org/abs/1710.02298)
- Sutton 1988 原始 TD 论文

---

## 💀 Top 3 Gotchas (v4)

### 💀 Gotcha 1：Q-learning 学的是 Q\*，**执行用 ε-greedy**

Q-learning update 用 `max_a' Q(s', a')`（target policy = greedy of Q），但 **rollout / behavior** 用 ε-greedy（不然不 explore）→ **off-policy**。

经典 cliff walking 故事的根：训练时 Q-learning 走悬崖边（学到的最优路），但 ε 探索时频繁摔下去 → episode reward 比 SARSA 低。但**部署**（ε=0）时 Q-learning 才是真最优。

→ **deploy 时还有探索 → 用 SARSA；deploy 完全 greedy → 用 Q-learning**。

### 💀 Gotcha 2：Maximization Bias 来自 Jensen 不等式

$\mathbb{E}[\max_a Q(s,a)] \geq \max_a \mathbb{E}[Q(s,a)]$（Jensen on convex max）

同一个 $Q$ 网络**既选 action 又评估**——选偏高 → 评估也偏高 → 正反馈系统性 over-estimation。

**Double Q-learning 解**：维护 $Q_A, Q_B$，用 $Q_A$ 选 action，**用 $Q_B$ 评估**（或反之）→ decorrelate。DQN 的 Double DQN（Hasselt 2015）同思路：online net 选 action，**target net 评估**。

### 💀 Gotcha 3：TD bootstrap → bias；MC 无 bias 但 variance 大

bias-variance trade-off：
- **TD(0)**: bias 高（bootstrap from estimate）+ variance 低 + **可 continuing**
- **MC**: bias 0 + variance 高（累积所有未来噪声）+ **必须 episodic**
- **n-step / GAE**: 中间，n / λ 是旋钮

实战 deep RL 几乎全 bootstrap（MC 在 long-horizon variance 太炸），但 advantage estimation 用 **GAE λ=0.95** 取中间。

## 链回

- [[_anki/ch06-td-cards]] — 38 张卡片版
- [[../../Code/sutton-barto-2e/_v4_notebooks/ch06-td-vs-mc-random-walk.ipynb]] — TD vs MC 数值 trace
- [[../../Code/sutton-barto-2e/_v4_notebooks/ch06-cliff-walking-sarsa-vs-qlearning.ipynb]] — Cliff walking SARSA vs Q
