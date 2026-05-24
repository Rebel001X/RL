---
title: "Sutton & Barto Ch 8. Planning and Learning with Tabular Methods"
created: 2026-05-24
tags: [area/rl, type/note, status/v3, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 8
difficulty: hard
budget_days: 2
priority_for_rlhf: skip
silver_lecture: 8
---

# Ch 8. Planning and Learning ⭐⭐⭐

---

## 📋 本章讲了什么（TL;DR）

1. **Planning（用 model 做 backup）+ Learning（用真实经验做 backup）统一**——Dyna 在每步交替两者，样本利用率提升数十倍
2. **MCTS（Monte Carlo Tree Search）**= AlphaGo / AlphaZero / MuZero 的核心；decision-time planning 跟 background planning（Dyna）的区别
3. **优先级扫描（prioritized sweeping）**：用 |TD error| 当优先级——**DQN PER 的祖先**

**为什么读**：本章是 [[ch16-applications-case-studies|AlphaGo / AlphaZero / MuZero]] 的理论根。**reasoning model（o1/R1）"花更多 compute 想"的精神跟 MCTS 同源**——LLM agent 的 ToT、RAP 等推理方法的起点。

## 🎯 Top 5 关键洞察

| # | 洞察 | 为什么关键 |
|---|---|---|
| 1 | **Dyna 用 n planning steps 提速数十倍** | 同样真经验，用 model 反复"幻想"backup → 样本效率飞升 |
| 2 | **MCTS 的 PUCT 公式**：$Q + c·P·\sqrt{N}/(1+N_a)$ | AlphaZero 选 action 核心 |
| 3 | **Decision-time vs background planning** | MCTS focus 当前 state；Dyna 处理任意 (s,a) |
| 4 | **Prioritized sweeping** | 不平均更新所有 state，按 \|TD error\| 优先 → DQN PER |
| 5 | **MuZero 把 model 也学出来** | latent dynamics + MCTS——model-based RL 工业巅峰 |

## 🧭 在 Obsidian graph 中的位置

- **上游**：[[ch04-dynamic-programming]]（DP backup = planning 核心）· [[ch06-temporal-difference-learning]]（真经验 backup）
- **下游**：**[[ch16-applications-case-studies]]**（AlphaGo / AlphaZero 案例）
- **现代延伸**：MuZero / Dreamer 系列（world models）
- **RLHF 联想**：o1 / R1 长 CoT 推理 ≈ 浅版 MCTS

---

## 0. 阅读元信息

- **难度**：hard（概念多）
- **预算**：2 天 / RLHF 不需要可 skim
- **必读理由**：MCTS / Dyna 是 AlphaGo / MuZero 的根
- **配套**：Silver UCL Lec 8

## 1. 一句话 + 在 RL 体系中的位置

**Planning + Learning 统一框架**——background planning（Dyna，任意 s, a）和 decision-time planning（MCTS，当前 s）是两大支系。

```
RL Model-based
├── Dyna-Q（background planning）
├── Prioritized Sweeping（高效 backup）
├── MCTS（decision-time planning）⭐
└── 现代延伸
    ├── AlphaGo / AlphaZero（MCTS + NN）
    ├── MuZero（MCTS + learned latent model）
    └── Dreamer（learned model + actor-critic）
```

## 2. 概念地图（§3 详解目录）

| # | 概念 | §3.x | 重要度 |
|---|---|---|---|
| 1 | **Models 与 Planning** | §3.1 | ⭐⭐ 基础 |
| 2 | **Dyna-Q** | §3.2 | ⭐⭐⭐ 经典 |
| 3 | **Dyna-Q+ / 处理 stale model** | §3.3 | ⭐⭐ |
| 4 | **Prioritized Sweeping** | §3.4 | ⭐⭐⭐ PER 祖先 |
| 5 | **Expected vs Sample Updates** | §3.5 | ⭐⭐ trade-off |
| 6 | **Trajectory Sampling / RTDP** | §3.6 | ⭐⭐ |
| 7 | **MCTS** | §3.7 | ⭐⭐⭐⭐ AlphaGo 核心 |
| 8 | **Rollout Algorithms** | §3.8 | ⭐⭐ |

---

## 3. 核心概念详解（每个从 0-1 讲清）

### 3.1 Models 与 Planning

#### 是什么
- **Model**：能预测 (s, a) → (s', r) 的函数——可以是学到的（learned）或给定的（known）
- **Planning**：用 model 做 backup 改进 V / Q

#### 数学
- **Distribution model**：给出 $p(s', r | s, a)$ 完整分布 → 期望 backup
- **Sample model**：给出 $(s', r) \sim p$ 一个 sample → sample backup

#### 跟 Learning 对比
| | Model | Real experience |
|---|---|---|
| 数据来源 | model（fake）| 环境（real）|
| 用途 | planning backup | learning backup |
| Sample 成本 | 几乎免费 | 贵（real env interaction）|

#### 高频面试问法
**Q：Planning 和 Learning 是同一件事吗？**
**A**：**算法上等价**——都是 Bellman backup。区别只是 **数据来源**：planning 用 model 生成 fake transition，learning 用真实 env 的 transition。Dyna 把两者统一。

---

### 3.2 Dyna-Q ⭐⭐⭐

#### 是什么
**Sutton 1990** 提出的经典架构——每步交替：(1) 用真实经验 update Q（learning）；(2) 学 model；(3) 用 model 生成 n 个 fake transition 再 update Q（planning）。

#### 为什么需要它（动机）
- 真实 env interaction 贵（机器人、Atari 跑得慢）
- 学一个 model 后能"无限模拟" → 样本效率飞升
- 简单加 n planning steps 就能跑得快几十倍

#### 数学

**Real experience update**（Q-learning 风格）：
$$Q(S, A) ← Q(S, A) + α[R + γ \max_{a'} Q(S', a') - Q(S, A)]$$

**Model learning**（tabular case）：
$$\text{Model}(S, A) ← (R, S')$$

**Planning step**（n 次）：
```
随机选已访问过的 (S_p, A_p)
(R_p, S'_p) = Model(S_p, A_p)
Q(S_p, A_p) += α[R_p + γ max_a' Q(S'_p, a') - Q(S_p, A_p)]
```

#### 伪代码（完整 Dyna-Q）
```python
def Dyna_Q(env, n_planning=50, alpha=0.1, gamma=0.99, eps=0.1, n_episodes=500):
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    Model = {}  # (s, a) -> (r, s_next)
    visited = []  # all (s, a) ever experienced
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            # 1. Real experience
            a = epsilon_greedy(Q[s], eps)
            s_next, r, done = env.step(a)
            Q[s][a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a])
            # 2. Model learning
            Model[(s, a)] = (r, s_next)
            visited.append((s, a))
            # 3. Planning (n fake updates)
            for _ in range(n_planning):
                s_p, a_p = visited[np.random.randint(len(visited))]
                r_p, s_np = Model[(s_p, a_p)]
                Q[s_p][a_p] += alpha * (r_p + gamma * np.max(Q[s_np]) - Q[s_p][a_p])
            s = s_next
    return Q
```

#### 实战参数
| 超参 | 推荐 | 调参直觉 |
|---|---|---|
| **n_planning** | **5 - 50** | 越大学得快但 model 错时崩 |
| α | 0.1 | TD step |
| ε | 0.1 | exploration |

#### 跟相关算法对比
| | Q-learning | Dyna-Q (n=50) |
|---|---|---|
| Real experience 用次数 | 1 次 backup | 1 + 50 fake backup |
| Sample 效率 | baseline | **几十倍** |
| Memory | O(\|S\|·\|A\|) Q | + O(\|S\|·\|A\|) Model |
| Model 错怎么办 | n/a | planning 学到错的 Q |

#### 反直觉点
- **Dyna 在 maze 上 n=50 比 n=0 快几十倍**（Sutton 图 8.2 经典）
- **Tabular Dyna 简单但威力巨大**
- **Model 错时 planning 反而坏 Q**——Dyna-Q+ 解决（§3.3）

#### 高频面试问法
**Q：Dyna-Q 怎么 work？n=0 vs n=50 差别？**
**A**：每步真经验更新 Q + 学 model + 用 model 生成 n 个 fake transition 再 update Q。**n=50 在 maze 任务上比 n=0 快几十倍**——同样真经验，model 反复"幻想"backup 让 Q 加速收敛。

---

### 3.3 Dyna-Q+（处理 stale model）

#### 是什么
Dyna-Q 升级版——给"很久没访问的 (s, a)"加 **exploration bonus**，处理 env 变化（如 maze 墙改变）。

#### 为什么需要它
- Dyna-Q 的 model 不会自动 detect env 变化
- 如果 env 变了（如 maze 通路改变），旧 model 误导 planning
- → 加 bonus 鼓励"重访旧地方"

#### 数学
**Bonus reward**：
$$R^+_p = R_p + κ \sqrt{τ(s, a)}$$

其中 $τ(s, a)$ 是 (s, a) 上次访问后的时间步数；κ 是 bonus 系数（如 0.001）。

#### 实战参数
| 超参 | 推荐 |
|---|---|
| κ | 0.0001 - 0.01 |

#### 高频面试问法
**Q：Dyna-Q+ 解决什么 Dyna-Q 不能解决的问题？**
**A**：env 变化（如 maze 墙挪了）—— Dyna-Q 的 model 不更新会一直用旧 model planning，学到过时 Q。Dyna-Q+ 给"久未访问的 (s, a)" exploration bonus，鼓励重访检测变化。

---

### 3.4 Prioritized Sweeping ⭐⭐⭐

#### 是什么
不平均更新所有 (s, a)——按 **|TD error|** 优先排序，先更新"误差大"的。

#### 为什么需要它（动机）
- Dyna 随机选 (s, a) planning → 浪费在已稳定的
- 观察：TD error 大的更新最有价值
- → priority queue 按 |TD error| 排

#### 数学
**Priority**：
$$P(s, a) = |r + γ \max_{a'} Q(s', a') - Q(s, a)|$$

#### 伪代码
```python
def prioritized_sweeping(env, n_planning=10, theta=0.01, alpha=0.1, gamma=0.99, eps=0.1):
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    Model = {}
    PQueue = []  # priority queue of (priority, (s, a))
    predecessors = defaultdict(set)  # s' -> set of (s, a) that lead to s'
    s = env.reset()
    for step in range(int(1e5)):
        a = epsilon_greedy(Q[s], eps)
        s_next, r, done = env.step(a)
        Model[(s, a)] = (r, s_next)
        predecessors[s_next].add((s, a))
        # Compute TD error
        td_error = abs(r + gamma * np.max(Q[s_next]) - Q[s][a])
        if td_error > theta:
            heapq.heappush(PQueue, (-td_error, (s, a)))
        # n planning steps from priority queue
        for _ in range(n_planning):
            if not PQueue: break
            _, (s_p, a_p) = heapq.heappop(PQueue)
            r_p, s_np = Model[(s_p, a_p)]
            Q[s_p][a_p] += alpha * (r_p + gamma * np.max(Q[s_np]) - Q[s_p][a_p])
            # Propagate to predecessors
            for (s_pred, a_pred) in predecessors[s_p]:
                r_pred, _ = Model[(s_pred, a_pred)]
                td_pred = abs(r_pred + gamma * np.max(Q[s_p]) - Q[s_pred][a_pred])
                if td_pred > theta:
                    heapq.heappush(PQueue, (-td_pred, (s_pred, a_pred)))
        s = s_next if not done else env.reset()
    return Q
```

#### 跟相关算法对比
| | Dyna-Q | Prioritized Sweeping | PER (DQN) |
|---|---|---|---|
| 选 (s,a) 顺序 | 随机 | 按 \|TD error\| | 按 \|TD error\| |
| 数据结构 | list | priority queue | sumtree |
| 加速 | n × | **n × 但更优** | similar |

#### 反直觉点
- **比 Dyna-Q 快 5-10x**（在 maze 任务）
- **DQN PER（Schaul 2016）是 prioritized sweeping 的 deep 版**——直接借鉴

#### 高频面试问法
**Q：Prioritized Sweeping 跟 PER 关系？**
**A**：PER（Prioritized Experience Replay, Schaul 2016 arXiv 1511.05952）= 把 Dyna 章节的 prioritized sweeping 思想用到 DQN replay buffer。从历史 transition 中按 |TD error| 加权 sample → 优先学误差大的 → 加速 DQN 收敛。**默认在所有现代 DQN 变种里**。

---

### 3.5 Expected vs Sample Updates

#### 是什么
- **Expected update**：用 $p(s', r | s, a)$ 完整分布算期望——精确但贵
- **Sample update**：从 model 采 1 个 $(s', r)$——近似但便宜

#### 数学

**Expected**:
$$Q(s, a) ← \sum_{s', r} p(s', r | s, a)[r + γ V(s')]$$

**Sample**:
$$Q(s, a) ← Q(s, a) + α[r + γ V(s') - Q(s, a)]$$（$(s', r) \sim p$）

#### Trade-off

| | Expected | Sample |
|---|---|---|
| 计算成本 | O(\|S\|) per update | O(1) |
| 准确性 | 精确 | 噪声 |
| 适用 | 小 branching factor | 大 branching factor |
| 现代 RL | 罕（model 大）| **主流**（DQN/AlphaGo） |

#### 高频面试问法
**Q：Sample update 比 Expected 弱在准确性，为什么仍主流？**
**A**：现代 RL state/action space 巨大（Atari 像素），expected update 不可行（要 enumerate 所有 s'）。Sample update O(1) per step，靠 SGD-like 平均最终收敛。**计算效率胜准确性**。

---

### 3.6 Trajectory Sampling / Real-time DP

#### 是什么
- **Trajectory sampling**：沿真 trajectory 做 backup（不随机选 state）
- **RTDP (Real-time DP)**：on-policy trajectory + DP backup

#### 为什么需要它
- Dyna 随机选 state planning → 浪费在 unreachable state
- 沿 on-policy trajectory backup → focus 重要 state

#### 跟 Dyna 对比
| | Dyna | Trajectory Sampling |
|---|---|---|
| 选 state | 任意已访问 | on-policy trajectory |
| 关注 | 全部 state space | reachable state |
| 收敛 | 全局 | 局部（reachable）|

#### 反直觉点
- **Trajectory sampling 比 Dyna 更聚焦**——但只学 reachable state，可能错过 better state（exploration trade-off）

---

### 3.7 MCTS（Monte Carlo Tree Search）⭐⭐⭐⭐

#### 是什么
**Coulom 2006 / Kocsis & Szepesvári 2006** 提出（UCT）——decision-time planning 的主流：从当前 state 滚 simulation 树，构建 partial search tree 选最佳 action。**AlphaGo / AlphaZero / MuZero 核心**。

#### 为什么需要它（动机）
- 围棋 state 太大（10²³⁰），tabular Q-learning 不可能
- 但**从当前 state 出发**做局部 search 可行
- → MCTS：incremental 构建 search tree

#### 数学（4 步骤）

**Step 1: Selection**（从 root 下行到 leaf）：
$$a^* = \arg\max_a \left[ Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}} \right]$$

—— UCB1（Upper Confidence Bound）公式。c 是 exploration 系数。

**Step 2: Expansion**：到 leaf 后，添加一个新 child node

**Step 3: Simulation (Rollout)**：从 leaf 用 default policy（如 random）滚到终态，得 reward

**Step 4: Backpropagation**：reward 沿路径回传更新 Q 和 N
```
for node in path:
    N(node) += 1
    Q(node) += (reward - Q(node)) / N(node)  # incremental average
```

#### AlphaZero 的 PUCT 公式
AlphaGo / AlphaZero 用 **PUCT**（Polynomial UCB Tree）替代 UCB1：
$$a^* = \arg\max_a \left[ Q(s, a) + c · P(s, a) · \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]$$

其中 $P(s, a)$ 是 policy net 的先验（替代 UCB1 的均匀先验）→ exploration 被引导。

#### 伪代码（标准 MCTS）
```python
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> child node
        self.Q = 0
        self.N = 0

def mcts(root_state, n_simulations=1000, c=1.4):
    root = MCTSNode(root_state)
    for _ in range(n_simulations):
        node = root
        # 1. Selection
        while node.children and not is_terminal(node.state):
            a, node = select_ucb(node, c)
        # 2. Expansion
        if not is_terminal(node.state):
            for a in legal_actions(node.state):
                s_next = transition(node.state, a)
                node.children[a] = MCTSNode(s_next, parent=node)
            a, node = random_choice(list(node.children.items()))
        # 3. Simulation
        reward = rollout(node.state)
        # 4. Backpropagation
        while node:
            node.N += 1
            node.Q += (reward - node.Q) / node.N
            node = node.parent
    # Return action with max visit count (more robust than max Q)
    return max(root.children, key=lambda a: root.children[a].N)

def select_ucb(node, c):
    """UCB1: Q + c·sqrt(ln N / N_a)"""
    best_score = -np.inf; best_a = None; best_child = None
    for a, child in node.children.items():
        if child.N == 0:
            return a, child  # 优先未访问
        ucb = child.Q + c * np.sqrt(np.log(node.N) / child.N)
        if ucb > best_score:
            best_score = ucb; best_a = a; best_child = child
    return best_a, best_child
```

#### AlphaZero MCTS（与上面区别）
```python
def alphazero_mcts(root_state, network, n_simulations=800):
    """network outputs (P, V) for each state"""
    root = MCTSNode(root_state)
    P_root, V_root = network(root_state)
    root.priors = P_root
    for _ in range(n_simulations):
        node = root
        path = [node]
        # 1. Selection with PUCT
        while node.children and not is_terminal(node.state):
            a, node = select_puct(node, c=1.0)
            path.append(node)
        # 2. Expansion + Evaluation（NN 一次性给 P, V）
        if not is_terminal(node.state):
            P, V = network(node.state)
            for a in legal_actions(node.state):
                s_next = transition(node.state, a)
                node.children[a] = MCTSNode(s_next, parent=node)
            node.priors = P
        else:
            V = terminal_value(node.state)
        # 3. NO simulation step!（用 V net 替代 random rollout）
        # 4. Backpropagation
        for n in reversed(path):
            n.N += 1
            n.Q += (V - n.Q) / n.N
            V = -V  # 双方对弈，alternating sign
    return max(root.children, key=lambda a: root.children[a].N)
```

**AlphaZero 关键 vs 标准 MCTS**：
- 用 NN 输出的 P 当 prior（PUCT）
- 用 NN 输出的 V 替代 random rollout（rollout 慢且不准）
- **search tree 由 NN 引导**——质量飞跃

#### 实战参数
| 超参 | 推荐 |
|---|---|
| c (UCB)| 1.4 |
| c (PUCT) | 1.0 - 2.5 |
| n_simulations | 800 - 8000 |
| Rollout policy | uniform / NN |

#### 跟相关算法对比
| | Dyna-Q | MCTS | AlphaZero MCTS |
|---|---|---|---|
| Planning timing | background | decision-time | decision-time |
| Focus | 全 state | 当前 state | 当前 state |
| Tree explicit | ❌ | ✅ | ✅ |
| NN 引导 | ❌ | ❌ | ✅ |

#### 反直觉点
- **MCTS 不需要 V 函数也能 work**（random rollout）
- **但 NN-guided MCTS 是 AlphaGo 突破**——rollout 质量决定 search 效果
- **MCTS 返回 action 用 visit count 而非 Q**——visit count 更鲁棒

#### 高频面试问法

**Q1：MCTS 4 步骤？**
**A**：(1) Selection: UCB 从 root 下行到 leaf；(2) Expansion: 添加 child；(3) Simulation: rollout 到终态得 reward；(4) Backpropagation: reward 沿路径回传更新 Q 和 N。

**Q2：UCB1 vs PUCT 区别？**
**A**：UCB1 用均匀 prior $\sqrt{\ln N / N_a}$；PUCT 用 policy net 的 prior $P(s,a) · \sqrt{N}/(1+N_a)$——**NN 引导 exploration**，AlphaZero 用这个。

**Q3：AlphaZero MCTS 跟标准 MCTS 区别？**
**A**：(1) PUCT 替代 UCB1（NN prior）；(2) **NN 的 V 替代 random rollout**（rollout 慢且不准 → V net 精确且快）；(3) **没有 simulation 步**——backprop 直接用 V 估的 value。

**Q4：reasoning model（o1/R1）的"思考"是 MCTS 吗？**
**A**：精神类似但不是显式 MCTS。o1/R1 训练时用 RL（GRPO + RLVR）学到"长 CoT" → 推理时模型自己"想"（生成长思考过程）——**不是 tree search，是 sequential 思考**。但"花更多 inference compute"的思想跟 MCTS 同源。

---

### 3.8 Rollout Algorithms

#### 是什么
**Tesauro & Galperin 1996**——给定 base policy π_0，对当前 state 跑 N 个 rollout 选最佳 action：
$$a^* = \arg\max_a \frac{1}{N} \sum_{i=1}^N G_i(a)$$

其中 $G_i(a)$ 是第 i 次 rollout（首 action=a，后续用 π_0）得到的 return。

#### 为什么需要它
- 当前 policy 不一定最优——但可以"测试"每个 action 看 N 个 rollout 结果
- 简单 + 易实现

#### 跟 MCTS 对比
| | Rollout | MCTS |
|---|---|---|
| 用 search tree | ❌ | ✅ |
| 复用 simulation 信息 | ❌ | ✅（tree node Q）|
| 实现 | 简单 | 中 |
| 性能 | 中 | 强 |

#### Best-of-N (BoN) 在 LLM 是什么？
**LLM 的 BoN ≈ rollout algorithm 的最简版**：
- 同 prompt 跑 N 个 response（rollout）
- 用 RM 打分选最高
- 等价于 "rollout + RM as reward"

详 [[../../../brain/Areas/rl-books/rlhf-lambert/ch09-rejection-sampling]]。

#### 高频面试问法
**Q：BoN 跟本章 Rollout Algorithm 关系？**
**A**：BoN = LLM 版的 rollout algorithm。同 prompt 跑 N 个 response（rollout），用 RM 评分选最佳——精神跟 Tesauro 1996 一致。**RL 的老思想在 LLM 时代复活**。

---

## 4. 跨概念对比表

### 4.1 Planning 算法 family tree
```
Planning (用 model backup)
├── Background planning（任意 s,a）
│   ├── Dyna-Q（random）
│   ├── Dyna-Q+（处理 stale）
│   └── Prioritized Sweeping（|TD error| 优先）→ DQN PER
├── Decision-time planning（当前 s）
│   ├── Rollout Algorithms（简单 → LLM 的 BoN）
│   ├── MCTS（UCB 引导）
│   │   └── AlphaZero MCTS（PUCT + NN）
│   └── Trajectory Sampling / RTDP
└── Model-based + Learning
    ├── Dyna-style（learned model）
    ├── AlphaGo / AlphaZero（known game model）
    └── MuZero（learned latent model）
```

### 4.2 选型表

| 场景 | 推荐 |
|---|---|
| 真 env 贵（机器人）| Dyna |
| Tabular small MDP | Prioritized Sweeping |
| 棋类游戏 | MCTS / AlphaZero |
| LLM agent reasoning | rollout / BoN / ToT |
| Continuous control | Dreamer / world model |

---

## 5. 习题精选

**Exercise 8.1（Dyna-Q+）**：理解 κ bonus 怎么鼓励"老 (s,a)" 复访。
**Exercise 8.4（prioritized sweeping 优先级）**：实现优先队列把 |TD error| 大的 (s, a) 优先 backup。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter08/maze_dyna_q.py`、`maze_prioritized_sweeping.py`
- 跑 `maze_dyna_q.py` 看 n=0/5/50 学习曲线差异
- **AlphaZero 实现**：参考 LeelaChessZero 或 alpha-zero-general 开源库

## 7. 跟 RLHF 的桥

| RLHF 组件 | 来自本章 §3.x |
|---|---|
| **BoN (Best-of-N)** | §3.8 Rollout algorithms |
| **Rejection Sampling FT** | §3.8 + offline RL |
| **Reasoning model "思考"** | §3.7 MCTS 精神（非显式 tree） |
| **Tree of Thoughts (ToT)** | §3.7 浅版 MCTS for LLM |
| **MuZero-style learned model** | §3.7 + learned dynamics |

## 8. 苏格拉底 Q&A

详见每个 §3.x。综合：

- **Q：Dyna vs MCTS 区别？**
  - 答：Dyna = background planning（任意 s,a，random）；MCTS = decision-time planning（当前 s，UCB 引导 tree search）。
- **Q：AlphaGo 用了本章哪些 §3.x？**
  - 答：§3.7 MCTS + policy/value net 引导（PUCT）。

## 9. 自测清单（闭卷）

- [ ] **写 Dyna-Q 算法**（含 n planning steps）
- [ ] **解释 model-based vs model-free trade-off**
- [ ] **画 MCTS 4 步骤**
- [ ] **写 UCB1 / PUCT 公式**
- [ ] **解释 AlphaZero MCTS 跟标准 MCTS 区别**
- [ ] **解释 Prioritized Sweeping → PER 的传承**
- [ ] **解释 BoN 跟 Rollout Algorithm 关系**

---

## 💡 拓展知识点（书外 trivia）

### A. Dyna 名字的由来
Sutton 1990 起名 "Dyna"——动力学（dynamics）+ 学习（learning）的合体词。当时 RL 圈认为 "model-free" 是优势，Dyna 提出 model + learning 协同是离经叛道。**今天 Dyna 思想（learned world model）是 model-based RL 主流**。

### B. UCT (UCB applied to Trees) 是 MCTS 的关键
**Kocsis & Szepesvári 2006** *Bandit Based Monte-Carlo Planning* —— 把 UCB1 用到 tree search → UCT 算法。这是 MCTS 走向主流的关键节点。

### C. AlphaGo Zero 比 AlphaGo 强多少？
**100-0**——完全无人类数据 + 自学几小时就超 AlphaGo。揭示人类棋谱在某些任务上反而是桎梏。

### D. MuZero 是 model-based RL 的工业巅峰
**Schrittwieser 2020 Nature** *Mastering Atari, Go, chess and shogi by planning with a learned model*：
- 不学 ground-truth dynamics
- 学 **latent dynamics**：在 abstract space 里预测 reward + value + policy 就够
- **关键**：model 只需让 planning 出对的 action，不必模拟世界

### E. Dreamer 系列（DeepMind world models）
**DreamerV1/V2/V3** (Hafner 2019/2020/2023)：
- 在 latent space 做 RL
- 跨 Minecraft / Atari / DM Control 都 SOTA
- V3 是第一个不调参就能跨多任务 SOTA 的 model-based RL

### F. AlphaTensor / AlphaFold 也用 MCTS
- **AlphaTensor (2022)**：RL + MCTS 发现新矩阵乘法算法（4×4 从 49 步降到 47 步）
- AlphaFold 2 用 attention 不用 MCTS；AlphaFold 3 据传引入 search
- **MCTS 不只玩游戏——是发现新算法的工具**

### G. Reasoning model 的 inference-time compute = decision-time planning
- **o1 / R1** 推理时生成长 CoT = "花更多 compute 想清楚"
- 不是显式 MCTS（无 tree），但精神类似 decision-time planning
- 训练时用 RL（GRPO + RLVR）让模型"学会 search"
- **RL × LLM agent 最有意思的交集**

### H. ToT (Tree of Thoughts) 是 LLM 版浅 MCTS
**Yao 2023** *Tree of Thoughts*：让 LLM 在 reasoning 时显式做 tree search（分支思考）。是 MCTS 在 LLM 上的"快糙好"实现——但 inference 慢，未成主流。

### I. RAP (Reasoning via Planning) 等后续工作
2023-2024 多个工作把 MCTS 思想用于 LLM reasoning：RAP, LATS, ToT 等。**RL + LLM 交叉前沿**。

---

## 🔗 更多 wikilinks（深挖）

| 主题 | 跳转 |
|---|---|
| AlphaGo / AlphaZero / MuZero 案例详解 | [[ch16-applications-case-studies]] |
| Q-learning + PER | [[../../../brain/Areas/rl-books/deep-rl-hands-on-lapan-3e/_overview\|Lapan Ch 8]] |
| Reasoning model RL | [[../../../brain/Areas/rl-books/rlhf-lambert/ch07-reasoning-inference-time-scaling]] |
| BoN / Rejection Sampling | [[../../../brain/Areas/rl-books/rlhf-lambert/ch09-rejection-sampling]] |
| Bellman backup（planning 基础）| [[ch04-dynamic-programming]] |

## Related

- **上一章**：[[ch07-n-step-bootstrapping]]
- **下一章**：[[ch09-on-policy-prediction]]
- **AlphaGo 案例**：[[ch16-applications-case-studies]]
- **进度**：[[_chapter-status]] W5

## Sources

- 原书 Ch 8 pp. 159-186
- [Silver UCL Lec 8: Integrating Learning and Planning](https://www.youtube.com/watch?v=ItMutbeOHtc)
- ShangtongZhang `chapter08/`
- **核心论文**：
  - AlphaGo: Silver 2016 Nature
  - AlphaZero: Silver 2018 Science
  - MuZero: Schrittwieser 2020 Nature
  - UCT: Kocsis & Szepesvári 2006
  - PER: [Schaul 2016 arXiv 1511.05952](https://arxiv.org/abs/1511.05952)
  - DreamerV3: [Hafner 2023 arXiv 2301.04104](https://arxiv.org/abs/2301.04104)
