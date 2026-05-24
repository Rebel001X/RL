---
title: "Sutton & Barto Ch 4. Dynamic Programming"
created: 2026-05-24
tags: [area/rl, type/note, status/v3, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 4
difficulty: medium
budget_days: 2-3
priority_for_rlhf: nice
silver_lecture: 3
---

# Ch 4. Dynamic Programming ⭐⭐⭐

---

## 📋 本章讲了什么（TL;DR）

1. **DP = "知道完整 MDP 时算最优 policy"的标准答案**——给后续 model-free（MC/TD）当对照基线
2. **两个核心算法**：policy iteration（评估+改进交替）和 value iteration（直接 Bellman optimality 迭代）；两者都收敛到 v\*
3. **GPI（Generalized Policy Iteration）** 是大一统视角——所有 RL 算法都是 GPI 的不同实例化

**为什么读 DP**：你永远用不上它（model + 小 state space 太苛刻），**但所有 model-free 方法都是 DP 的"采样近似"**——不懂 DP 看不懂 MC/TD 在妥协什么。

## 🎯 Top 5 关键洞察

| # | 洞察 | 为什么关键 |
|---|---|---|
| 1 | **Policy improvement theorem** 保证 π'=greedy(v_π) 不劣于 π | RL 收敛性的基石；所有 control 算法依赖这个 |
| 2 | **Value iteration = policy iteration 的 eval 只跑 1 步** | 揭示两者本质同源 |
| 3 | **Bellman operator 是 γ-contraction** | 不动点唯一（Banach）→ DP 必收敛 |
| 4 | **Bootstrap 是 DP 的灵魂** | 也是 [[ch11-off-policy-methods\|deadly triad]] 的祸根 |
| 5 | **Curse of dimensionality 是 DP 的天敌** | O(\|S\|²\|A\|) 在 Backgammon 10²⁰ states 就崩 → 必须 function approximation |

## 🧭 在 Obsidian graph 中的位置

- **上游**：[[ch03-finite-mdps]]（MDP + Bellman = DP 的输入）
- **下游**：[[ch05-monte-carlo-methods]]（去 model）· [[ch06-temporal-difference-learning]]（去 model + 增量）· [[ch08-planning-and-learning]]（Dyna 混真经验）
- **延伸**：[[ch09-on-policy-prediction]]（DP 加函数近似）
- **RLHF 联想**：PPO 的 critic V_φ 学的就是"sample-based policy evaluation"——DP 第一步的工程化

---

## 0. 阅读元信息

- **难度**：medium（数学不难但概念关键）
- **预算**：2-3 天
- **必读理由**：所有 model-free 算法的"标准答案"参照
- **配套**：Silver UCL Lec 3

## 1. 一句话 + 在 RL 体系中的位置

**DP = 已知完整 MDP model 时，通过迭代求解 Bellman 方程**算 v\* 和 π\*；两大算法 policy iteration 和 value iteration 是 RL 控制的"原型"。

**在 RL 算法家族中的位置**：
```
RL Value-based
├── DP (Ch 4) ⭐         ← 知 model，理论标准
├── MC (Ch 5)             ← model-free, episode-end
├── TD (Ch 6)             ← model-free, online
└── DP-style + learned model: Dyna / MuZero / Dreamer
```

## 2. 概念地图（§3 详解目录）

| # | 概念 | §3.x | 重要度 |
|---|---|---|---|
| 1 | **Policy Evaluation** | §3.1 | ⭐⭐⭐ |
| 2 | **Policy Improvement Theorem** | §3.2 | ⭐⭐⭐⭐ 基石定理 |
| 3 | **Policy Iteration** | §3.3 | ⭐⭐⭐ |
| 4 | **Value Iteration** | §3.4 | ⭐⭐⭐ |
| 5 | **Generalized Policy Iteration (GPI)** | §3.5 | ⭐⭐⭐⭐ 大一统视角 |
| 6 | **Asynchronous DP** | §3.6 | ⭐⭐ |
| 7 | **In-place Update** | §3.7 | ⭐⭐ 工程优化 |
| 8 | **Bellman Operator 数学** | §3.8 | ⭐⭐⭐ 收敛证明根 |

---

## 3. 核心概念详解（每个从 0-1 讲清）

### 3.1 Policy Evaluation ⭐⭐⭐

#### 是什么
给定 policy π，**迭代求 v_π**——反复用 Bellman expectation 直到收敛。

#### 为什么需要它
- v_π 是 Bellman 方程的解：$v_π(s) = \sum_a π(a|s) \sum_{s', r} p(s', r | s, a)[r + γ v_π(s')]$
- 线性方程组可直接解，但 |S| 大时贵（O(|S|³)）
- **迭代法 O(|S|² · iterations)，通常 10-100 轮收敛**

#### 数学

**Bellman expectation iteration**：
$$v_{k+1}(s) = \sum_a π(a|s) \sum_{s', r} p(s', r | s, a)[r + γ v_k(s')]$$

**收敛性**（Banach 不动点定理）：
- $B^π$ 是 γ-contraction：$\|B^π v_1 - B^π v_2\|_∞ ≤ γ \|v_1 - v_2\|_∞$
- γ < 1 时存在唯一不动点 v_π
- 任意初始 $v_0$ 必收敛到 v_π

#### 伪代码
```python
def policy_evaluation(env, pi, theta=0.01, gamma=0.99):
    V = np.zeros(env.n_states)
    while True:
        Delta = 0
        for s in range(env.n_states):
            v = V[s]
            V[s] = sum(pi[s][a] * sum(env.p(s_next, r | s, a) * (r + gamma * V[s_next])
                                      for s_next, r in env.transitions(s, a))
                        for a in range(env.n_actions))
            Delta = max(Delta, abs(v - V[s]))
        if Delta < theta:
            break
    return V
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| θ（收敛阈值）| 0.001 - 0.01 |
| γ | 0.9 - 0.99 |

#### 反直觉点
- **Iterative 比直接解线性方程慢但更通用**——支持 update-as-you-go
- **γ 越接近 1 收敛越慢**——effective horizon $1/(1-γ)$ 长

#### 高频面试问法
**Q：Policy evaluation 一定收敛吗？为什么？**
**A**：γ<1 时**必收敛**——Bellman expectation operator 是 γ-contraction，由 Banach 不动点定理，存在唯一不动点 v_π，任意初始 V_0 都收敛到它。

---

### 3.2 Policy Improvement Theorem ⭐⭐⭐⭐

#### 是什么
**RL 收敛性基石定理**：如果对所有 s 有 q_π(s, π'(s)) ≥ v_π(s)，则 v_{π'}(s) ≥ v_π(s)——**贪心改进的 policy 必不劣于原 policy**。

#### 为什么需要它
- 所有 RL control 算法都依赖"贪心改进能提升 policy"这个性质
- 没有这个定理 → policy iteration 不收敛
- **是 SARSA / Q-learning / PG 等所有 control 算法的隐含基石**

#### 数学推导

**Setup**：已知 v_π，定义 π'(s) = $\arg\max_a q_π(s, a)$（greedy w.r.t. v_π）。

**Step 1**（贪心选择）：
$$v_π(s) ≤ \max_a q_π(s, a) = q_π(s, π'(s))$$

**Step 2**（unroll 用 π' 一步切回 π，至少不更差）：
$$q_π(s, π'(s)) = E[R + γ v_π(s')]$$
$$≤ E[R + γ q_π(s', π'(s'))]$$
$$≤ E[R + γ E[R' + γ q_π(s'', π'(s''))]] = ...$$

无限 unroll → $v_π(s) ≤ E_{π'}[R + γR' + γ²R'' + ...] = v_{π'}(s)$ ∎

#### 严格改进（如果某 s 严格不等）
若存在 s 使 q_π(s, π'(s)) > v_π(s)（严格），则 v_{π'} > v_π（严格改进）。

#### 反直觉点
- **看似显然——但完整证明用了"无限 unroll"非平凡**
- **保证 policy iteration 不死循环**：每轮 improvement 严格变好 → 必收敛到 v\*
- **是所有 RL 算法的隐含保证**

#### 高频面试问法
**Q：Policy improvement theorem 是什么？为什么重要？**
**A**：if q_π(s, π'(s)) ≥ v_π(s) ∀s, then v_{π'} ≥ v_π。**意义**：greedy 改进必不劣 → policy iteration 严格 monotonic improvement → 必收敛到 v\*。**是所有 RL control 算法的数学基石**。

---

### 3.3 Policy Iteration ⭐⭐⭐

#### 是什么
**Eval + Improvement 交替**直到 policy 稳定：
1. Evaluation: 求 v_π
2. Improvement: π ← greedy(v_π)
3. 若 π 不变 → 收敛到 π\*

#### 为什么需要它
- 给"算最优 policy"的标准算法
- 收敛极快（finite MDP 上经验 3-5 轮）

#### 伪代码
```python
def policy_iteration(env, gamma=0.99, theta=0.01):
    V = np.zeros(env.n_states)
    pi = np.zeros(env.n_states, dtype=int)
    while True:
        # 1. Policy Evaluation
        while True:
            Delta = 0
            for s in range(env.n_states):
                v = V[s]
                V[s] = sum(env.p(s_next, r | s, pi[s]) * (r + gamma * V[s_next])
                            for s_next, r in env.transitions(s, pi[s]))
                Delta = max(Delta, abs(v - V[s]))
            if Delta < theta: break
        # 2. Policy Improvement
        stable = True
        for s in range(env.n_states):
            old_a = pi[s]
            pi[s] = max(range(env.n_actions),
                        key=lambda a: sum(env.p(s_next, r | s, a) * (r + gamma * V[s_next])
                                          for s_next, r in env.transitions(s, a)))
            if old_a != pi[s]: stable = False
        if stable: return V, pi
```

#### 实战参数
| 超参 | 推荐 |
|---|---|
| θ | 0.001 |
| γ | 0.99 |
| Eval 内层迭代数 | 通常 10-100 收敛 |
| 外层 PI 迭代数 | **经验只需 3-5 轮** |

#### 收敛性
- **finite MDP 上有限步收敛到 π\***
- Hoffman & Karp 1966: 理论上限 |A|^|S|，实际 3-5 轮

#### 反直觉点
- **理论上限 |A|^|S|，实际几乎总 3-5 轮**——RL 早期最 famous"理论实践 gap"
- **Eval 是内层主开销**——所以 VI 把 eval 减到 1 步加速

#### 高频面试问法
**Q：Policy iteration 多少轮收敛？理论 vs 实际？**
**A**：理论 |A|^|S|（指数级），**实际几乎总是 3-5 轮**。Hoffman & Karp 1966 的理论上限至今未被反例打破——RL 早期最 famous 的"理论实践 gap"。

---

### 3.4 Value Iteration ⭐⭐⭐

#### 是什么
**Bellman optimality 直接迭代**：
$$v_{k+1}(s) = \max_a \sum_{s', r} p(s', r | s, a)[r + γ v_k(s')]$$

—— 不显式表示 policy，直接逼近 v\*。

#### 为什么需要它
- Policy iteration 内层 evaluation 贵
- VI 把 eval 退化到 1 步 → 等价"policy iteration 的 eval 只跑 1 步"
- 收敛性同样保证（Bellman optimality operator 也是 contraction）

#### 数学

**Bellman optimality iteration**：
$$v_{k+1}(s) = \max_a \big[ \sum_{s', r} p(s', r | s, a)[r + γ v_k(s')] \big]$$

**收敛**：γ < 1 时 $B^*$ 是 γ-contraction → 必收敛到 v\*。

**Extract policy**（收敛后）：
$$π^*(s) = \arg\max_a \sum_{s', r} p(s', r | s, a)[r + γ v^*(s')]$$

#### 伪代码
```python
def value_iteration(env, gamma=0.99, theta=0.01):
    V = np.zeros(env.n_states)
    while True:
        Delta = 0
        for s in range(env.n_states):
            v = V[s]
            V[s] = max(sum(env.p(s_next, r | s, a) * (r + gamma * V[s_next])
                            for s_next, r in env.transitions(s, a))
                        for a in range(env.n_actions))
            Delta = max(Delta, abs(v - V[s]))
        if Delta < theta: break
    # Extract policy
    pi = np.array([max(range(env.n_actions),
                       key=lambda a: sum(env.p(s_next, r | s, a) * (r + gamma * V[s_next])
                                         for s_next, r in env.transitions(s, a)))
                   for s in range(env.n_states)])
    return V, pi
```

#### 跟 PI 对比
| | Policy Iteration | Value Iteration |
|---|---|---|
| 显式 policy | ✅ | ❌（最后 extract）|
| 内层迭代 | Eval 收敛 | 只跑 1 步 |
| 外层轮数 | 3-5 | 通常 100+ |
| 总时间 | 类似 | 类似（trade-off）|
| 现代 deep RL | 较少 | **DQN 基础**（VI + sample + NN）|

#### 反直觉点
- **VI 总轮数 100+ 但 PI 只 3-5——总时间反而相近**（VI 每轮便宜）
- **DQN 是 VI 的 sample-based + NN 版本**——所以 deep RL 主流 VI 派

#### 高频面试问法
**Q：VI 跟 PI 的关系？为什么 deep RL 用 VI 派系？**
**A**：VI = PI 的 evaluation 只跑 1 步。VI 不需要显式表示 policy → 跟 Q 函数自然耦合（argmax Q 就是 policy）→ DQN / Rainbow / Double DQN 全是 VI 的 sample-based 版本。**PI 派系在 deep RL 罕用**。

---

### 3.5 Generalized Policy Iteration (GPI) ⭐⭐⭐⭐

#### 是什么
**所有 RL 算法的统一视角**——eval 和 improvement 任意交错：
- PI：eval 收敛 + improvement 1 步
- VI：eval 1 步 + improvement 隐含在 max
- Dyna：eval 用真+虚 backup + improvement greedy
- AC：eval = critic 学 V + improvement = actor 学 π
- **PPO**：eval = GAE/critic + improvement = clipped PG

#### 为什么是统一视角
**所有 RL 算法都在做**：
1. 学一个 value（V 或 Q）
2. 基于这个 value 改进 policy
3. 用新 policy 收集数据再学 value
4. 循环

GPI 是这个循环的抽象——不要求 eval 和 improvement 完成度。

#### 数学（GPI 收敛条件，弱）
- Eval 朝 v_π 方向（不必到位）
- Improvement 朝 greedy(v) 方向
- 两个方向都"正确" → 最终 v ← v\*, π ← π\*

**几何直觉**（书 figure 4.7）：
```
    v_π* (target)
   / \
  /   \   ← eval 把 v 拉向 v_π
 v_π
  \   /
   \ /    ← improvement 把 π 拉向 greedy(v)
    π → π*
```

#### 跟具体算法对比
| 算法 | Eval | Improvement |
|---|---|---|
| PI | 收敛 | 1 步 greedy |
| VI | 1 步 | 隐含 max |
| MC Control | episode end + 多 sample | greedy |
| SARSA | TD(0) | ε-greedy |
| Q-learning | TD(0) + max | implicit (max) |
| **Actor-Critic** | critic TD | actor PG |
| **PPO** | GAE + critic | clipped PG with K epochs |

#### 反直觉点
- **GPI 不是算法，是 framework**——所有 RL 算法都是 GPI 的实例
- **eval / improvement 不必各自收敛**——交错前进即可

#### 高频面试问法
**Q：GPI 是什么？为什么重要？**
**A**：Generalized Policy Iteration——eval 和 improvement 任意交错的统一 framework。**所有 RL 算法都是 GPI 的实例**：PI、VI、SARSA、Q-learning、AC、PPO 各自选择 eval/improvement 的 "完成度"。理解 GPI 才能从大局看 RL 算法的内在统一。

---

### 3.6 Asynchronous DP

#### 是什么
不按"所有 state 一遍 sweep"的顺序——任意 state 任意频率更新。

#### 为什么需要它
- Tabular DP 要求 sweep 所有 state——大 state space 不可行
- 异步：focus 重要 state（高 visit / 高 |TD error|）
- **为 model-free + FA 铺路**——Dyna / DQN 都是异步 DP 的 sample 版本

#### 数学
**Sweep DP**：每轮按 fixed order 更新所有 state
**Async DP**：可以同一 state 连续更新多次，其它 state 不更新；**只要每 state 被无限次更新就收敛**

#### 反直觉点
- **任意顺序都收敛**——不需要平等对待 state
- **现代 RL 全是异步 DP**——按 trajectory 出现顺序 update

#### 高频面试问法
**Q：Async DP 跟 Dyna 关系？**
**A**：Dyna 的"random 选 (s, a) planning"就是异步 DP 的随机版本。Prioritized Sweeping 是异步 DP + 优先级。**异步 DP 是 RL 大多数算法的"隐藏前置"**。

---

### 3.7 In-place Update

#### 是什么
DP 实现细节：用最新的 V 值就地更新（而非 two-array）。

#### 数学
- **Two-array DP**：维护 V_old, V_new；用 V_old 算 V_new；最后 swap
- **In-place DP**：单数组 V；update V[s] 时立刻覆盖

#### 收敛性
- 两者都收敛
- **In-place 通常收敛更快**——新 V 立刻参与后续 update（信息传播快）

#### 反直觉点
- **In-place 看似简化但实际加速**——不是"trade-off"，是 strict improvement
- **现代 DP 实现默认 in-place**

#### 高频面试问法
**Q：In-place vs two-array DP 哪个收敛快？**
**A**：**In-place 通常更快**——新 V 立刻参与后续 update，信息传播快。两者都收敛但 in-place 是 strict improvement。**现代实现默认 in-place**。

---

### 3.8 Bellman Operator 数学（收敛证明根）⭐⭐⭐

#### 是什么
DP 收敛性的数学基础——Bellman operator 是 **γ-contraction**，由 Banach 不动点定理收敛。

#### 为什么需要懂
- 没这个证明 → DP "可能不收敛"
- 是后续所有 RL 收敛分析的根
- **DQN deadly triad 失败的根源 = γ-contraction 失效**

#### 数学

**Bellman expectation operator $B^π$**：
$$(B^π v)(s) = \sum_a π(a|s) \sum_{s', r} p(s', r | s, a)[r + γ v(s')]$$

**Contraction property**：
$$\|B^π v_1 - B^π v_2\|_∞ ≤ γ \|v_1 - v_2\|_∞$$

**证明**：
$$|(B^π v_1)(s) - (B^π v_2)(s)| = γ |\sum_a π(a|s) \sum_{s'} p(s'|s,a)(v_1(s') - v_2(s'))|$$
$$≤ γ \max_{s'} |v_1(s') - v_2(s')| = γ \|v_1 - v_2\|_∞$$

所以 $\|B^π v_1 - B^π v_2\|_∞ ≤ γ \|v_1 - v_2\|_∞$。

**Banach 不动点定理**：
- $B^π$ 是 contraction（γ<1）
- → 存在唯一不动点 v\*（即 v_π）
- → 任意 v_0 经 iteration 必收敛到 v\*

**收敛速度**：
$$\|v_k - v_π\|_∞ ≤ γ^k \|v_0 - v_π\|_∞$$

—— 指数衰减，γ 接近 1 时慢。

#### Bellman optimality operator $B^*$
$$(B^* v)(s) = \max_a \sum_{s', r} p(s', r | s, a)[r + γ v(s')]$$

也是 γ-contraction（同证明，max 不破坏 contraction），收敛到 v\*。

#### Deadly triad 时 contraction 消失
**off-policy + FA + bootstrap** 三者同时：
- Effective operator 不再是 contraction
- 不动点可能不存在
- → 可能发散（[[ch11-off-policy-methods|Baird's counterexample]]）

#### 反直觉点
- **γ < 1 是 DP 收敛的"魔法"**——没有它（continuing task γ=1）数学不严格
- **NN 时 Bellman operator 不再 contraction**——这是 deadly triad 的本质

#### 高频面试问法

**Q1：DP 为什么必收敛？**
**A**：Bellman operator $B^π$ 是 γ-contraction（$\|B^π v_1 - B^π v_2\|_∞ ≤ γ \|v_1 - v_2\|_∞$），由 Banach 不动点定理存在唯一不动点 v\*，任意 v_0 经 iteration 必收敛到 v\*，速度 $O(γ^k)$。

**Q2：Deadly triad 的数学根源？**
**A**：**γ-contraction 失效**。off-policy + FA + bootstrap 三者同时时，effective operator 不再是 contraction → 不动点可能不存在 → 不收敛。这是 deadly triad 的数学本质。

---

## 4. 跨概念对比表

### 4.1 DP 算法 family tree
```
DP
├── Policy Iteration (eval 到位 + improve 1 步)
├── Value Iteration (eval 1 步 + improve 隐含 max)
├── Async DP (任意 state 任意频率)
└── GPI (eval + improve 任意交错) ⭐ 统一视角
    └── 所有 RL 算法都是 GPI 的实例
```

### 4.2 DP → model-free 演进

| 维度 | DP | MC | TD | DQN |
|---|---|---|---|---|
| Model | ✅ | ❌ | ❌ | ❌ |
| Bootstrap | ✅ | ❌ | ✅ | ✅ |
| Online | ❌ | ❌ | ✅ | ✅ |
| Tabular | ✅ | ✅ | ✅ | ❌（NN）|
| 收敛保证 | ✅ | ✅ | ✅ | ❌（trick）|

---

## 5. 习题精选

**Exercise 4.4（PI bug）**：policy iteration 有 bug 可能死循环（policy 在两个等价 optimal 间来回切）。修复：检查 q-value 是否真的更大。
**Exercise 4.7（Jack's Car Rental）**：在 Example 4.2 基础上加约束，重写 + 重跑 PI。
**Exercise 4.10**：写 q\* 的 value iteration 公式。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter04/grid_world.py`、`car_rental.py`、`gamblers_problem.py`
- 自己写：5x5 gridworld VI，10 行 numpy
- 跑通 `gamblers_problem.py` 看 figure 4.3 "锯齿状"最优 policy

## 7. 跟 RLHF 的桥

| RLHF 概念 | 来自本章 §3.x |
|---|---|
| **GPI 框架** | §3.5——RLHF 也是 GPI（RM 是 eval，policy 是 improve）|
| **Policy improvement theorem** | §3.2——RLHF 改 policy 不能更差的隐含保证 |
| **Bellman contraction** | §3.8——理解 deadly triad 的数学根 |

## 8. 苏格拉底 Q&A

详见每个 §3.x。综合：

- **Q：DP / MC / TD / PG 都是 GPI 的实例吗？怎么看？**
  - 答：是。每个算法选择不同 (eval, improvement) 完成度。DP eval 严格收敛；MC eval episode-end；TD eval 单步；PG eval = critic + improvement = actor。
- **Q：DP 永远用不上，为什么必读？**
  - 答：所有 model-free RL 是 DP 的"采样近似"——不懂 DP 看不懂 MC/TD 在妥协什么。

## 9. 自测清单（闭卷）

- [ ] **写出 policy iteration 算法**（含 eval + improve 两步）
- [ ] **写出 value iteration 算法**
- [ ] **解释 policy improvement theorem + 证明思路**
- [ ] **证明 Bellman operator 是 γ-contraction**
- [ ] **解释 GPI 为什么是统一视角**
- [ ] **跑通 ShangtongZhang `chapter04/`** 至少一个 demo

---

## 💡 拓展知识点（书外 trivia）

### A. Bellman 起名 "DP" 的政治考量
1950s Richard Bellman 在 RAND 工作，老板讨厌"research"和"mathematics"。他给方法起名 **"Dynamic Programming"**——"dynamic"听着工程化、"programming"在那时指"planning"而非编码——纯粹**词汇伪装**。50 年后 RL 教材还在用这个误导性名字。

### B. AlphaZero 的 MCTS 是 DP 变种
MCTS = "asynchronous prioritized sweeping"（[[ch08-planning-and-learning]]）+ NN value function。本质是 DP 在巨大 state space 上的**树搜索化**。

### C. DP 跟最优控制 / HJB 方程
- **离散 RL DP** ↔ **连续 control HJB equation**（Hamilton-Jacobi-Bellman）
- **LQR**（线性二次调节器）= 连续 DP 的解析解
- **MPC**（model predictive control）= "在线 DP"（每步重解优化）—— 自动驾驶 / 机器人控制大量用

### D. DP 在工业的现代应用
| 领域 | 用法 |
|---|---|
| 金融期权定价 | Bellman 算最优行权策略 |
| 库存管理 | (s, S) policy 是 DP 解 |
| 路径规划 | A* / Dijkstra 是 DP 特例 |
| 基因序列比对 | Smith-Waterman = DP |
| NLP parse tree | CKY = DP |

### E. Prioritized Sweeping 是 DQN PER 的祖先
DQN with **prioritized experience replay**（Schaul 2016, arXiv 1511.05952）用 |TD error| 当优先级——直接借鉴 §3 prioritized sweeping。

### F. Hoffman & Karp 1966 反直觉数学结果
**Hoffman & Karp 1966** 证明：policy iteration 在 finite MDP 上**至多需 |A|^|S| 轮**（理论上限），但**实际几乎总是 3-5 轮**——RL 早期最 famous "理论实践 gap"。**理论上限至今未被反例打破**。

### G. RL Bellman 不动点 vs functional analysis 的 Banach
**Banach 不动点定理**（1922）是泛函分析经典结果。RL 借用——所有收敛性证明的根。**RL 数学根在 1920s 数学**。

### H. DP 跟 reinforcement learning 边界模糊
- "Tabular DP（知 model）"和"Tabular RL（不知 model）"边界在 model 是否给定
- 现代用 **learned model + DP** = MuZero / Dreamer——边界又模糊
- **2026 model-based RL 复兴让 DP 思想重要性回升**

---

## 🔗 更多 wikilinks（深挖）

| 主题 | 跳转 |
|---|---|
| Bellman 方程数学根 | [[ch03-finite-mdps]] §3.7 |
| Async DP → DQN | [[../../../brain/Areas/rl-books/deep-rl-hands-on-lapan-3e/_overview\|Lapan Ch 6]] |
| Prioritized sweeping → PER | [[ch08-planning-and-learning]] §3.4 · [[../../../brain/Areas/rl-books/deep-rl-hands-on-lapan-3e/_overview\|Lapan Ch 8]] |
| MCTS = async DP 变种 | [[ch08-planning-and-learning]] §3.7 · [[ch16-applications-case-studies]] |
| Bellman contraction → deadly triad | [[ch11-off-policy-methods]] §3.1 |
| GPI 在 PG 里 | [[ch13-policy-gradient]] §3.4 actor-critic |

## Related

- **上一章**：[[ch03-finite-mdps]]
- **下一章**：[[ch05-monte-carlo-methods]]
- **延伸**：[[ch11-off-policy-methods]]（deadly triad）· [[ch08-planning-and-learning]]（Dyna）
- **进度**：[[_chapter-status]] W2

## Sources

- 原书 Ch 4 pp. 73-92
- [Silver UCL Lec 3: Planning by DP](https://www.youtube.com/watch?v=Nd1-UUMVfz4)
- ShangtongZhang `chapter04/`
- **经典论文**：
  - Bellman 1957 *Dynamic Programming*
  - Hoffman & Karp 1966（PI 收敛上限）
  - Howard 1960 *Dynamic Programming and Markov Processes*
