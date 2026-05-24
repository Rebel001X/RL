---
title: "Sutton & Barto Ch 16. Applications and Case Studies"
created: 2026-05-24
tags: [area/rl, type/note, status/v3, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 16
difficulty: easy
budget_days: 2
priority_for_rlhf: nice
silver_lecture: 10
---

# Ch 16. Applications and Case Studies ⭐⭐⭐

---

## 📋 本章讲了什么（TL;DR）

1. **RL 真实落地案例集锦**——从 1990s TD-Gammon 到 2016 AlphaGo / 2017 AlphaZero / 2020 MuZero / 2022 ChatGPT
2. **每个 case 展示 RL 框架的某个独特强项**：self-play / function approximation / model-based / multi-agent
3. **AlphaGo / AlphaZero / MuZero 详解** = 面试 must-know；DQN 在 Atari 是 deep RL 起点

**为什么必读**：理论懂了不等于会用——本章看 RL 在真实问题上的"工程战术"。**AlphaGo 段是 RL 面试必问**。

## 🎯 Top 5 关键洞察

| # | 洞察 | 为什么关键 |
|---|---|---|
| 1 | **TD-Gammon 1992 用 NN + TD(λ) + self-play 超人类** | RL 第一次"超人类"成就，证明 deep RL 雏形 |
| 2 | **DQN 用 end-to-end pixel → action** | 第一个 general RL agent，开启 deep RL 时代 |
| 3 | **AlphaGo = MCTS + Policy/Value Net + Self-Play** | 4 个 RL 关键技术合一 |
| 4 | **AlphaZero 比 AlphaGo 100-0**（无人类数据 + 单 NN） | 揭示人类棋谱反成桎梏 |
| 5 | **MuZero 把 model 也学出来** | model-based RL 工业巅峰；reasoning model 灵感 |

## 🧭 在 Obsidian graph 中的位置

- **上游**：[[ch04-dynamic-programming]] · [[ch06-temporal-difference-learning]] · [[ch08-planning-and-learning]] · [[ch12-eligibility-traces]] · [[ch13-policy-gradient]]
- **延伸**：[[ch17-frontiers]]（未来方向）
- **现代关联**：[[../../../brain/Areas/rl-books/deep-rl-hands-on-lapan-3e/_overview|Lapan]]（DQN/PPO 实战）
- **LLM 联想**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch07-reasoning-inference-time-scaling]]（推理时 search 跟 MCTS 思想）

---

## 0. 阅读元信息

- **难度**：easy（叙述性）
- **预算**：2 天 / 必读 AlphaGo 段
- **必读理由**：**AlphaGo 详解是 RL 面试必问**
- **配套**：Silver UCL Lec 10

## 1. 一句话 + 在 RL 体系中的位置

**RL 应用集锦** + **里程碑案例详解**——把前 15 章理论落到真实成就上。

```
RL 应用历史时间线
├── 1959 Samuel's Checkers（早期 RL 雏形）
├── 1992 TD-Gammon（第一个超人类 RL）
├── 2011 Watson on Jeopardy（RL 用于 wagering）
├── 2013/2015 DQN on Atari（end-to-end pixel→action）
├── 2016 AlphaGo Lee Sedol（围棋世界冠军）
├── 2017 AlphaGo Zero（无人类数据）
├── 2017 AlphaZero（通用框架：围棋+象棋+将棋）
├── 2020 MuZero（learned model + planning）
└── 2022 ChatGPT（RLHF 第一个出圈应用）
```

## 2. 概念地图（§3 详解目录）

| # | 案例 | §3.x | 重要度 |
|---|---|---|---|
| 1 | **TD-Gammon**（Tesauro 1992）| §3.1 | ⭐⭐ 历史里程碑 |
| 2 | **Samuel's Checkers**（1959）| §3.2 | ⭐ 历史 |
| 3 | **Watson on Jeopardy**（2011）| §3.3 | ⭐ |
| 4 | **DQN on Atari**（2013/2015）| §3.4 | ⭐⭐⭐⭐ deep RL 起点 |
| 5 | **AlphaGo**（2016）| §3.5 | ⭐⭐⭐⭐ 完整算法 |
| 6 | **AlphaGo Zero**（2017）| §3.6 | ⭐⭐⭐ |
| 7 | **AlphaZero**（2017）| §3.7 | ⭐⭐⭐⭐ 通用框架 |
| 8 | **MuZero**（2020）| §3.8 | ⭐⭐⭐⭐ learned model + MCTS |

---

## 3. 核心概念详解（每个从 0-1 讲清）

### 3.1 TD-Gammon（Tesauro 1992）⭐⭐

#### 是什么
**Gerald Tesauro** 在 IBM 用 **TD(λ) + NN + self-play** 训练 backgammon 程序——超过当时世界冠军水平。RL 第一个"超人类"成就。

#### 为什么重要
- 1992 年——deep learning 还没有，NN 是小型 MLP
- 证明 RL + NN + self-play 能解决复杂问题
- 比 DQN 早 20 年
- 影响 AlphaGo 设计哲学

#### 算法

**Setup**：
- State：backgammon 棋盘状态（28 维 hand-crafted feature）
- Action：合法移动
- Reward：游戏结束 +1/-1
- Network：小型 MLP（40-80 hidden units）输出 V(s)

**Training**：
- **Self-play**：模型跟自己对弈数百万局
- **TD(λ)**：用 λ-return 当 target 更新 NN
- λ ≈ 0.7（经验值）

**Decision-time**：
- 一步前瞻：枚举所有合法 move，选 V(next_state) 最高的

#### 关键技术贡献
- **首次系统化 NN + RL**——影响所有后续 deep RL
- **Self-play 训练范式**——AlphaGo 沿用
- **TD(λ) 在大状态空间** work——证明 FA 可行

#### 反直觉点
- **小 MLP（40 units）就能超人类**——backgammon 因含 dice noise 适合 RL
- **deterministic 游戏（chess）当时 RL 不 work**——要等 AlphaGo

#### 高频面试问法
**Q：TD-Gammon 为什么 backgammon 上超人类但跳棋不行？**
**A**：backgammon 自带 stochasticity（骰子），exploration 天然 + 反映期望 value；deterministic 游戏（如跳棋）需要更复杂的 search + value function。TD-Gammon 的 NN 不够强支撑这个。**AlphaGo 用 MCTS + 更深 NN 才突破 deterministic 棋类**。

---

### 3.2 Samuel's Checkers（1959）

#### 是什么
**Arthur Samuel** IBM 工程师写的跳棋程序——**RL 的史前祖先**。

#### 创新
- "Learning by playing itself"（self-play 早期雏形）
- 用 linear function 估 board value
- minimax search + learned eval function

#### 历史意义
- 第一个 ML 程序（早 ML 几十年）
- RL 的"前 RL"——没有正式 framework 但思想都在

---

### 3.3 Watson on Jeopardy（2011）

#### 是什么
IBM Watson 2011 在 Jeopardy! 击败人类冠军。**RL 用在 wagering（下注）决策**——不是答题本身。

#### RL 应用
- 答题策略（buzz in or not, 题目难度选择）
- Daily Double bet amount 决策
- 用 reinforcement learning 学最优 wagering policy

#### 反直觉点
- **Watson 主要是 IR + ML（不是 RL）**——但 wagering 模块用 RL
- **RL 的"小角色"也能造就出圈成就**

---

### 3.4 DQN on Atari（2013/2015）⭐⭐⭐⭐

#### 是什么
**Mnih et al. 2013 NeurIPS Workshop / 2015 Nature**——DeepMind 的 DQN：第一次 end-to-end RL（raw pixel → action）跨多个 Atari games 达到人类水平。

#### 为什么重要
- **打破 RL "需要人工 feature" 的局限**——CNN 学到 feature
- **single architecture 跨 49 games**——通用性
- **开启 deep RL 时代**

#### 算法（详 [[ch06-temporal-difference-learning]] §3.8 + [[ch11-off-policy-methods]] §3.7）

**核心**：Q-learning + CNN + 4 个工程 trick：
1. **Experience replay**：buffer 1M transitions, batch sample
2. **Target network**：冻结 $θ^-$ 每 10000 步同步
3. **Reward clipping**：r ∈ [-1, 1]
4. **Frame stacking**：concat 4 帧近似 Markov

**Network**：3 个 conv layer + 2 个 FC → Q 值输出（每 action 一个）

#### 实战参数（Atari）
| 超参 | 值 |
|---|---|
| Replay buffer | 1M |
| Batch size | 32 |
| Target update freq | 10000 steps |
| Adam lr | 1e-4（不能用默认 1e-3）|
| ε schedule | 1.0 → 0.1 over 1M steps |
| γ | 0.99 |
| Total steps | 50M (200M on hard games) |

#### 实证表现（Atari benchmark）
- 49 games 中 **29 个达到人类水平**
- Breakout / Pong / Space Invaders 超人类
- Montezuma's Revenge 等 sparse-reward 游戏失败

#### 跟相关算法对比
| | DQN | Double DQN | Rainbow |
|---|---|---|---|
| Over-est | 严重 | 减 | 减 |
| 平均得分 | baseline | +30% | **+200%** |
| 训练 trick 数 | 4 | 5 | 11 |

#### 反直觉点
- **2013 NeurIPS Workshop 版被几乎所有人忽视**——2015 Nature 主刊版才出圈
- **DQN 单 architecture 跨 49 games 是 deep learning + RL 的 generalization 力量**
- **Montezuma's Revenge 等 sparse reward 游戏 DQN 仍学不会**——deep RL 仍有大量未解问题

#### 高频面试问法

**Q1：DQN 论文为什么是 RL 里程碑？**
**A**：(1) **end-to-end pixel → action**——破"人工 feature"局限；(2) **single architecture 跨 49 games**——通用 agent；(3) 开启 deep RL 时代——AlphaGo / PPO / SAC 都是后继。

**Q2：DQN 跟 tabular Q-learning 关键差异？**
**A**：DQN = Q-learning + NN + 4 工程 trick（replay / target net / clip / frame stack）。tabular Q-learning 在 state space 大时不可能（Atari 像素 state ~10^7800）；DQN 用 NN 提供 generalization 让 Q-learning 在真实问题 work。

---

### 3.5 AlphaGo（2016）⭐⭐⭐⭐

#### 是什么
**Silver et al. 2016 Nature** *Mastering the game of Go with deep neural networks and tree search*——DeepMind 的围棋程序，2016 年 3 月 4-1 击败李世石。

#### 为什么重要
- 围棋之前被认为"10 年内 AI 不可能"——10^230 state space
- AlphaGo 用 RL + MCTS + NN 突破——震撼世界
- 把 AI 推向公众视野（比 ChatGPT 早 6 年）

#### 完整算法（4 部分）

**1. SL Policy Net $P_σ$**：
- 从 16M 人类业余高手棋谱训
- 输入 board state（19×19 + features），输出每个位置的下子概率
- 准确率 57%（预测人类下一手）

**2. RL Policy Net $P_ρ$**：
- 初始化 = $P_σ$
- 用 self-play + REINFORCE 微调
- Reward = +1 赢 / -1 输（终局）

**3. Value Net $V_θ$**：
- 从 $P_ρ$ self-play 数据训
- 输入 board state，输出 V(s) ∈ [-1, 1]（预测胜负）

**4. MCTS at decision time**：
- 用 **PUCT** 公式：$a^* = \arg\max[Q + c·P·\sqrt{N}/(1+N_a)]$
- Selection: PUCT 下行
- Expansion: 添加 child node
- Evaluation: $V_θ$(leaf) + fast rollout（mixed）
- Backprop: 更新 Q, N

#### 完整流程图
```
[棋局 input]
    ↓
[CNN feature extractor]
    ↓
    ├── Policy Net P (action prior)
    ├── Value Net V (state value)
    └── Fast Rollout Policy (quick simulation)
            ↓
        [MCTS with PUCT]
            ↓
        [Action（visit count argmax）]
```

#### 实战参数（AlphaGo Lee Sedol 版）
| 项 | 值 |
|---|---|
| Hardware | 1920 CPU + 280 GPU |
| MCTS simulations | 几千次 per move |
| 思考时间 | ~30s per move |

#### 反直觉点
- **AlphaGo 不"理解"围棋——它学到了模式**
- **MCTS 是关键——没有 MCTS 单 NN 弱很多**
- **Value net 替代 random rollout 是质变**——之前 MCTS 用 random rollout 评估 leaf，慢且不准；NN V net 又快又准

#### 高频面试问法（AlphaGo 面试 must-know）

**Q1：AlphaGo 4 部分是什么？**
**A**：(1) SL Policy Net $P_σ$（人类棋谱 SL，57% 预测准确）；(2) RL Policy Net $P_ρ$（self-play REINFORCE 微调）；(3) Value Net $V_θ$（self-play 数据预测胜负）；(4) MCTS（PUCT 引导 search，用 P 当 prior，V 当 leaf 评估）。

**Q2：MCTS 的 PUCT 公式？**
**A**：$a^* = \arg\max_a [Q(s, a) + c · P(s, a) · \frac{\sqrt{N(s)}}{1 + N(s, a)}]$。Q 是平均 value，c·P·√N/(1+N_a) 是 exploration bonus——P 是 NN policy prior，N(s) 是父节点访问数，N(s,a) 是该 action 访问数。

**Q3：AlphaGo 用 RL 还是 SL？**
**A**：**两者都用**：SL 初始化 policy（从人类棋谱）→ RL self-play 微调（REINFORCE）→ MCTS 决策。SL 是冷启动加速，RL 是真正提升超越人类。

**Q4：MCTS 为什么不直接用 random rollout 而要 NN V net？**
**A**：random rollout 慢（要走到终局）+ 不准（random 走棋远离 optimal）；NN V net 一次 forward 给出精确 value 评估。**NN-guided MCTS 是 AlphaGo 突破的关键技术贡献**。

---

### 3.6 AlphaGo Zero（2017）⭐⭐⭐

#### 是什么
**Silver et al. 2017 Nature** *Mastering the game of Go without human knowledge*——AlphaGo 的"无人类数据版"：完全从随机 self-play 起，3 天就超 AlphaGo Lee 版。

#### 为什么重要
- **完全无人类数据**——证明人类棋谱反而是桎梏
- **100-0 击败 AlphaGo Lee**
- 简化 architecture（单 NN 同时输出 P 和 V）

#### 跟 AlphaGo 区别

| | AlphaGo (2016) | AlphaGo Zero (2017) |
|---|---|---|
| 人类数据 | ✅（16M 棋谱）| ❌ 完全无 |
| Policy Net | 单独 P_σ + P_ρ | 单一 NN（P + V 共享 backbone）|
| Value Net | 单独 V_θ | 同 NN |
| Rollout | mixed (fast policy + V) | **纯 V**（无 rollout）|
| MCTS | UCB 变种 | PUCT |
| 训练时间 | 数月 | **3 天**（超过 AlphaGo Lee）|

#### 关键创新

**Single network**：
$$(P(s), V(s)) = f_θ(s)$$

—— 一个 NN 同时输出 policy prior + value——共享 representation 学得更好。

**Pure self-play**：
- 不用任何人类棋谱
- 第一局完全随机
- 几小时后超随机；3 天超 AlphaGo Lee；40 天 SOTA

**Training loop**：
1. 用当前 NN + MCTS 玩 self-play games
2. 收集 (state, MCTS visit count distribution, game outcome)
3. 训 NN：P 拟合 MCTS visit count（更好的 policy），V 拟合 game outcome

#### 反直觉点
- **无人类数据反而强**——AlphaGo Zero 不学到人类棋手的偏见 / 错误模式
- **单 NN 比双 NN 强**——shared representation 是关键
- **3 天超数月**——证明 self-play scaling 极强

#### 高频面试问法

**Q1：AlphaGo Zero 比 AlphaGo 强多少？为什么？**
**A**：**100-0**（无人类数据，3 天训出）。原因：(1) **不被人类棋谱误导**——人类有偏见和盲点；(2) **single network 共享 representation**——P 和 V 互相帮助学；(3) **pure self-play scaling**——只要 compute 够，performance 持续提升。

**Q2：AlphaGo Zero 的 MCTS 训练有什么独特？**
**A**：用 **MCTS visit count** 当 policy improvement 信号——NN 输出的 P 跟 MCTS 强化后的 visit count distribution 拟合 → NN 越来越像 "弱 MCTS"。这是 self-distillation 思想：用 MCTS 强化弱 policy 后回训 policy。

---

### 3.7 AlphaZero（2017）⭐⭐⭐⭐

#### 是什么
**Silver et al. 2018 Science**——AlphaGo Zero 的**通用框架**：用同一套算法 + 同 hyper-parameter 训围棋（Go）、象棋（Chess）、将棋（Shogi），全部超过当时 SOTA。

#### 为什么重要
- **通用性**——一套算法解决多个棋类，不需要 game-specific tuning
- 击败 Stockfish（顶级象棋引擎，hand-engineered eval function）
- 证明 "**learned eval + MCTS** > **手工 eval + alpha-beta search**"

#### 跟 AlphaGo Zero 区别
| | AlphaGo Zero | AlphaZero |
|---|---|---|
| 目标 game | 围棋 | 围棋 + 象棋 + 将棋 |
| 算法 | 围棋专用细节 | 通用框架 |
| 超参 | 围棋调过 | 跨 game 同 |
| Tournament 设置 | self-play | self-play |

#### 跟传统棋类引擎对比

| | Stockfish (传统 chess engine) | AlphaZero |
|---|---|---|
| Search | alpha-beta + handcrafted eval | MCTS + NN eval |
| Eval function | 手工（几千行 C++）| NN 学到 |
| Nodes/sec | 数千万 | ~80K |
| 强度（Elo）| ~3400 | ~3500 |
| 可推广性 | 仅 chess | **多 game 通用** |

**关键观察**：AlphaZero 用 **1/1000 的 search nodes** 但更强——因为每个 node 评估精度高（NN > 手工 eval）。

#### 反直觉点
- **NN eval + MCTS 击败 hand-crafted eval + alpha-beta**——deep learning 对传统 AI 的胜利
- **AlphaZero 在 chess 上偏好 "long-term planning"**——下出人类大师都不下的"奇怪但深远"棋
- **影响传统棋类引擎**：今天 Stockfish 也加 NN eval（NNUE）了

#### 高频面试问法

**Q1：AlphaZero 跟 AlphaGo Zero 区别？**
**A**：AlphaZero = AlphaGo Zero 的**通用版**——同一套算法 + 同 hyperparameters 训围棋 / 象棋 / 将棋，跨 game 通用。AlphaGo Zero 是围棋专用。

**Q2：AlphaZero 击败 Stockfish 但 search nodes 少 1000 倍，为什么仍赢？**
**A**：**精度胜数量**。Stockfish 每秒搜几千万 node 但每个 node 用手工 eval（粗糙）；AlphaZero 每秒搜 80K node 但每个 node 用 NN eval（精确）。**总评估质量 = 节点数 × 节点精度**，AlphaZero 后者远超前者。

---

### 3.8 MuZero（2020）⭐⭐⭐⭐

#### 是什么
**Schrittwieser et al. 2020 Nature** *Mastering Atari, Go, chess and shogi by planning with a learned model*——MuZero 是 AlphaZero 的进化：**model 也学出来**（latent dynamics），不需要 know game rules。

#### 为什么重要
- 之前 AlphaZero 需要 game rules（知道下一步 state 怎么转移）
- MuZero **学一个 abstract dynamics model**——在 latent space 里 plan
- 在 Atari + 棋类 + Pacman 都 SOTA——证明通用性

#### 关键创新

**Three Functions**（一起学）：
- **Representation function $h$**：obs → latent state $s_0$
- **Dynamics function $g$**：(s_t, a_t) → (r_{t+1}, s_{t+1})  — 在 **latent space**
- **Prediction function $f$**：s → (policy P, value V)

**MCTS in latent space**：
```python
def MuZero_MCTS(obs, h, g, f, n_simulations=50):
    root_state = h(obs)  # latent state
    root = MCTSNode(root_state)
    P_root, V_root = f(root_state)
    root.priors = P_root
    for _ in range(n_simulations):
        node = root; path = [node]
        # 1. Selection (PUCT)
        while node.children:
            a, node = select_puct(node, c=1.0)
            path.append(node)
        # 2. Expansion in latent space
        if not is_terminal(node.state):
            r_pred, s_next = g(node.parent.state, node.action)  # learned dynamics
            P, V = f(s_next)
            node.state = s_next  # latent, not observable
            node.priors = P
        # 3. Backpropagation
        for n in reversed(path):
            n.N += 1; n.Q += (V - n.Q) / n.N
            V = r_pred + gamma * V  # accumulate latent reward
    return max(root.children, key=lambda a: root.children[a].N)
```

#### 关键观察
- **Model 不是"模拟世界"——只需让 planning 出对的 action**
- 在 latent space 学 dynamics 比 raw obs space 容易（语义压缩）
- 训练 loss = policy loss (MCTS visit count fit) + value loss + **reward prediction loss**

#### 跟 AlphaZero / Dreamer 对比

| | AlphaZero | MuZero | Dreamer |
|---|---|---|---|
| 需要 game rules | ✅ | ❌（学 latent dynamics）| ❌ |
| Planning | MCTS in true state | MCTS in latent | 想象 rollout |
| 适用 | 棋类 | 棋类 + Atari | continuous control |

#### 反直觉点
- **Learned model 没有"真实"含义**——latent state $s_t$ 不对应 obs，但保证 (r, V, P) 预测正确
- **比 AlphaZero 通用**——不需要 game-specific rules → 真正"通用 RL agent"
- **影响 LLM 时代**：world models 概念延续到 Dreamer / Genie / Sora-like 视频生成

#### 高频面试问法

**Q1：MuZero 跟 AlphaZero 关键差异？**
**A**：AlphaZero 需要 **knowable game model**（围棋 / 象棋的规则可以 perfect simulate）；MuZero **学一个 latent dynamics**，不需要知道规则——所以能跨 Atari（没有 "rule" 给 search）+ 棋类。

**Q2：MuZero 的 latent state 是什么？**
**A**：**抽象表示**——不直接对应 obs。MuZero 学到的 dynamics function $g(s, a) → (r, s')$ 在 latent space，只要保证 (reward, policy, value) 预测正确就行。**model 不模拟世界，只引导 planning**。

**Q3：MuZero 影响 LLM 时代什么？**
**A**：world model 思想——**Dreamer V3** 在 latent space 做 RL；**Genie** 学世界 model；**Sora / Veo** 视频生成本质是 world model。MuZero 是这条线的"工业起点"。

---

## 4. 跨概念对比表

### 4.1 RL 里程碑时间线
```
1959 Samuel's Checkers ── 早期 RL 雏形
1992 TD-Gammon       ── 第一个超人类 RL
2011 Watson           ── RL 工业出圈（部分）
2013/15 DQN          ── deep RL 起点
2016 AlphaGo Lee     ── 围棋世界冠军（SL + RL + MCTS）
2017 AlphaGo Zero    ── 无人类数据（self-play only）
2017 AlphaZero       ── 通用框架（围棋 + 象棋 + 将棋）
2020 MuZero          ── learned model + MCTS
2022 ChatGPT         ── RLHF 出圈，RL 进入 LLM 时代
2024-26 DeepSeek R1, o1 ── RLVR 复兴
```

### 4.2 各案例关键技术对比

| 案例 | 算法 | 关键创新 | 现代延续 |
|---|---|---|---|
| TD-Gammon | TD(λ) + NN + self-play | 第一次 RL+NN | AlphaGo |
| DQN | Q-learning + CNN + trick | end-to-end pixel | Rainbow, Apex |
| AlphaGo | MCTS + NN（P+V）+ self-play | NN-guided MCTS | AlphaZero |
| AlphaGo Zero | 同 + no human data | self-play scaling | MuZero |
| AlphaZero | 同 + 通用 framework | 跨 game | Stockfish NNUE |
| MuZero | 同 + learned latent model | model-based RL | Dreamer, Genie |

---

## 5. 习题精选

无算法习题。**思考题**：

- **Q：为什么 backgammon 比 chess 早被 RL 解决？**
- **Q：AlphaGo Zero 比 AlphaGo 强 100-0，揭示什么哲学问题？**

## 6. 代码伴侣

- ShangtongZhang 无对应 case study 代码（叙述章）
- **必读论文**：
  - Tesauro 1995 *TD-Gammon*
  - Mnih 2013/2015 DQN
  - Silver 2016 AlphaGo Nature
  - Silver 2017 AlphaGo Zero Nature
  - Silver 2018 AlphaZero Science
  - Schrittwieser 2020 MuZero Nature
  - Hafner 2023 DreamerV3
- **开源实现**：
  - alpha-zero-general（PyTorch AlphaZero）
  - LeelaChessZero（生产级 AlphaZero）
  - muzero-general（学术 MuZero 实现）

## 7. 跟 RLHF 的桥

| RLHF 概念 | 来自本章 §3.x |
|---|---|
| Self-play training | §3.5 AlphaGo / §3.6 Zero |
| Iterative RLHF | §3.6 self-play loop 思想 |
| Best-of-N + RM | §3.4 DQN replay / §3.5 MCTS rollout 简化 |
| Reasoning model "思考" | §3.5-3.8 MCTS 精神 |
| World models for agents | §3.8 MuZero learned dynamics |

## 8. 苏格拉底 Q&A

详见每个 §3.x。综合题：

- **Q：RL 里程碑 5 个**
  - 答：TD-Gammon (1992) / DQN (2013) / AlphaGo (2016) / AlphaZero (2017) / MuZero (2020)
- **Q：每个里程碑的关键技术贡献**
  - 答：NN+RL+selfplay / end-to-end pixel / NN-guided MCTS / 通用框架 / learned latent model
- **Q：从 AlphaGo 到 MuZero 的演进逻辑**
  - 答：AlphaGo（SL + RL + MCTS）→ Zero（去人类数据）→ AlphaZero（跨 game）→ MuZero（去 game rules，学 model）

## 9. 自测清单（闭卷）

- [ ] **画 AlphaGo 4 部分流程图**
- [ ] **写出 PUCT 公式**
- [ ] **解释 MCTS 4 步骤**
- [ ] **解释 AlphaGo Zero 比 AlphaGo 强的 3 个原因**
- [ ] **解释 MuZero 跟 AlphaZero 关键差异**
- [ ] **列 RL 5 个里程碑及关键技术贡献**

---

## 💡 拓展知识点（书外 trivia）

### A. AlphaGo Lee Sedol 2016 "第 37 手"
2016 年 3 月 AlphaGo vs Lee Sedol Game 2 第 37 手——AlphaGo 下出人类 2500 年围棋史从未有的"奇怪一手"。事后被职业棋手分析认为是"divine move"。**这是 AI 第一次给人类带来"创造性"启发**。

### B. AlphaGo Lee 之后人类围棋水平显著提升
2016-2020 年人类职业棋手开始用 AlphaGo / Leela 学习——发现很多"长期被认为不好"的下法其实优——人类围棋整体水平提升一个台阶。**AI 反过来教人类**。

### C. AlphaZero 的"chess 风格"震惊业界
AlphaZero 下 chess 偏好"long-term sacrifices"（弃子换长期优势）——人类大师都不下这种风格。被认为揭示了 chess 此前未被人类完全探索的领域。

### D. DQN 论文当年的轰动 vs 沉寂
**Mnih 2013 NIPS Workshop** 展示 DQN 玩 Breakout 视频——**会场反响平平**（其它人忙关注其它 paper）。2015 Nature 主刊版才出圈。**典型"早期被低估"的工作**。

### E. AlphaTensor / AlphaFold 也用 MCTS 思想
- **AlphaTensor (2022)**：RL + MCTS 发现新矩阵乘法算法（4×4 从 49 步降到 47 步，超过 Strassen 1969 经典）
- **AlphaFold 3 (2024)**：据传引入 search
- **AlphaProof (2024)**：IMO 数学奥赛获奖
- **MCTS 不只玩游戏——是发现新算法 / 解数学问题的工具**

### F. Waymo 等自动驾驶用 RL（部分）
- 大多数模块是 supervised learning + IL（imitation learning）
- RL 主要用于 **edge case** decision making（如左转抢道）
- 不是端到端 RL（safety 风险大）
- **现实工业 RL 多是"模块化"应用，不是"端到端"**

### G. ChatGPT 是 RL 影响最大应用
ChatGPT（2022.11）→ 让所有人意识到 RL 在 LLM 后训练的关键性。**InstructGPT 论文**（Ouyang 2022）是 RL 应用史影响最深远的一篇。

### H. AlphaProof 2024 IMO 数学奥赛
DeepMind 2024 用 RL + 形式化数学（Lean）训出 AlphaProof，在 IMO 拿银牌。**RL + formal reasoning 是 2026 新前沿**。

### I. MuZero 的"latent model 不需有意义"是个深刻洞察
之前 model-based RL 都假设 model 应"准确模拟世界"。MuZero 证明 **model 只需让 planning 输出对的 action 就够**——不需要 obs 重构、不需要 dynamics 物理准确。**这是 deep learning 哲学的胜利**。

---

## 🔗 更多 wikilinks（深挖）

| 主题 | 跳转 |
|---|---|
| MCTS 详解 | [[ch08-planning-and-learning]] §3.7 |
| DQN 工程详解 | [[ch06-temporal-difference-learning]] §3.8 · [[ch11-off-policy-methods]] §3.7 |
| Reasoning model 跟 MCTS | [[../../../brain/Areas/rl-books/rlhf-lambert/ch07-reasoning-inference-time-scaling]] |
| BoN（rollout 简化）| [[../../../brain/Areas/rl-books/rlhf-lambert/ch09-rejection-sampling]] |
| Self-play 思想 → iterative RLHF | [[../../../brain/Areas/rl-books/rlhf-lambert/ch03-training-overview]] |
| ChatGPT / RLHF 里程碑 | [[../../../brain/Areas/rl-books/rlhf-lambert/ch02-history]] |

## Related

- **上一章**：[[ch15-neuroscience]]
- **下一章**：[[ch17-frontiers]]
- **理论根**：[[ch08-planning-and-learning]]（MCTS）· [[ch13-policy-gradient]]（PG）
- **进度**：[[_chapter-status]] W8

## Sources

- 原书 Ch 16 pp. 421-448
- [Silver UCL Lec 10](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- **核心论文**：
  - Tesauro 1995 *TD-Gammon*
  - Mnih 2013 [arXiv 1312.5602](https://arxiv.org/abs/1312.5602) · Mnih 2015 [Nature DQN](https://www.nature.com/articles/nature14236)
  - Silver 2016 Nature AlphaGo
  - Silver 2017 Nature AlphaGo Zero
  - Silver 2018 Science AlphaZero
  - Schrittwieser 2020 Nature MuZero
  - Hafner 2023 [DreamerV3 arXiv 2301.04104](https://arxiv.org/abs/2301.04104)
- **开源实现**：
  - LeelaChessZero（生产级 AlphaZero）
  - alpha-zero-general（学术 AlphaZero）
  - muzero-general
  - CleanRL（DQN / PPO）
