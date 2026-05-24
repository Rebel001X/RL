---
title: "Sutton & Barto Ch 14. Psychology"
created: 2026-05-24
tags: [area/rl, type/note, status/v3, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 14
difficulty: easy
budget_days: 1
priority_for_rlhf: skip
silver_lecture: 9
---

# Ch 14. Psychology ⭐⭐

---

## 📋 本章讲了什么（TL;DR）

1. **RL 数学框架跟动物行为学习高度对应**——classical conditioning（巴甫洛夫）≈ value learning，instrumental conditioning（操作条件反射）≈ policy/control
2. **Rescorla-Wagner 模型（1972 心理学经典）= TD(0) 退化版**——1980s Sutton 发现并打通 RL ↔ 心理学
3. **Habits vs Goal-directed = Model-free vs Model-based**——心理学早 RL 几十年就有这区分

**为什么读**：理解 RL 的"前世"——动物学习心理学；**alignment 工程的 reward shaping 直觉很多来自本章**。

## 🎯 Top 5 关键洞察

| # | 洞察 | 为什么关键 |
|---|---|---|
| 1 | **Rescorla-Wagner = TD(0) when γ=0** | 心理学 1972 提出，比 RL 早 16 年 |
| 2 | **Classical conditioning ↔ TD prediction** | 学 cue→outcome 的关联 |
| 3 | **Instrumental conditioning ↔ TD control** | 学 action→reward 的关联 |
| 4 | **Habits vs Goal-directed = Model-free vs Model-based** | 大脑里两套 RL system 并存 |
| 5 | **Delay discounting in humans is hyperbolic, not exponential** | RL γ 模型不匹配人类 |

## 🧭 在 Obsidian graph 中的位置

- **上游**：[[ch04-dynamic-programming]] · [[ch06-temporal-difference-learning]]（数学跟心理学打通的地方）
- **下游**：[[ch15-neuroscience]]（神经基础）
- **RLHF 联想**：reward shaping / constitutional AI 启发来自行为学习

---

## 0. 阅读元信息

- **难度**：easy（叙述性）
- **预算**：1 天 / skim
- **必读理由**：理解 RL 的"前身"——动物学习心理学
- **配套**：Silver UCL Lec 9

## 1. 一句话 + 在 RL 体系中的位置

**RL 数学框架 = 动物学习心理学的形式化**——心理学早 RL 几十年提出"prediction error"等核心概念，Sutton 1988 用数学打通两者。

```
学习理论
├── 心理学（行为学派 + 认知派）
│   ├── Classical conditioning (Pavlov 1904)
│   ├── Instrumental conditioning (Thorndike 1898 / Skinner)
│   └── Rescorla-Wagner model (1972)
├── 数学化（Sutton 1988+）
│   ├── TD prediction
│   ├── TD control
│   └── Model-based RL
└── 神经科学（Ch 15）
```

## 2. 概念地图（§3 详解目录）

| # | 概念 | §3.x | 重要度 |
|---|---|---|---|
| 1 | **Classical Conditioning（巴甫洛夫）** | §3.1 | ⭐⭐⭐ |
| 2 | **Rescorla-Wagner 模型** | §3.2 | ⭐⭐⭐⭐ = TD(0) |
| 3 | **TD Model of Classical Conditioning** | §3.3 | ⭐⭐⭐ |
| 4 | **Instrumental Conditioning（操作条件反射）** | §3.4 | ⭐⭐⭐ |
| 5 | **Delayed Reinforcement + Eligibility traces** | §3.5 | ⭐⭐ |
| 6 | **Habits vs Goal-directed** | §3.6 | ⭐⭐⭐ |
| 7 | **Latent Learning（Tolman）** | §3.7 | ⭐⭐ Model-based 灵感 |

---

## 3. 核心概念详解（每个从 0-1 讲清）

### 3.1 Classical Conditioning（巴甫洛夫）⭐⭐⭐

#### 是什么
**Pavlov 1904 诺贝尔奖经典**：动物学到"中性刺激（铃声）→ 有意义结果（食物）"的关联。

#### 实验
- **Phase 1**：铃响 → 给食物（动物流口水）；反复多次
- **Phase 2**：单独响铃 → 动物流口水（**learned association**）

#### 跟 RL 对应
| 心理学 | RL |
|---|---|
| Stimulus (US, e.g. food) | reward signal |
| Conditioned Stimulus (CS, e.g. bell) | state feature / cue |
| Conditioned Response (drooling) | predicted value |
| **Association** | V(s) |

#### 反直觉点
- **动物学到的是"prediction"——不是"反应"**：流口水是对"将来会有食物"的预测
- **跟 RL 的 prediction problem 完全对应**——学 V_π

#### 高频面试问法
**Q：Pavlov 的狗在做什么 RL 任务？**
**A**：**TD prediction**——学 "stimulus（铃声）→ outcome（食物）" 的关联，本质是学 V(s)。动物的"流口水"是对"将来 reward" 的预测——跟 RL 的 V_π(s) 完全对应。

---

### 3.2 Rescorla-Wagner Model（1972）⭐⭐⭐⭐

#### 是什么
**心理学经典**：Rescorla & Wagner 1972 用数学描述 classical conditioning——**实际上是 TD(0) 的退化版**。

#### 数学

**Rescorla-Wagner 公式**（per trial）：
$$\Delta V_X = α(λ - \sum_X V)$$

其中：
- $V_X$ = cue X 跟 outcome 的关联强度
- $λ$ = 实际 outcome（onset = 1 / absence = 0）
- $\sum_X V$ = 所有 active cue 的 V 总和（aggregate prediction）
- α = learning rate

#### 跟 TD(0) 等价

**TD(0)**（γ=0 退化）：
$$V(S_t) ← V(S_t) + α(R_{t+1} - V(S_t))$$

**Rescorla-Wagner**:
$$V_X ← V_X + α(λ - \sum V)$$

**等价点**：
- $λ$ ↔ $R_{t+1}$（actual outcome）
- $\sum V$ ↔ $V(S_t)$（predicted outcome）
- "prediction error" $λ - \sum V$ ↔ TD error $R - V$
- 都是"朝预测误差方向走 α 步"

**Sutton 1988** 论文 *Learning to Predict by the Methods of Temporal Differences* 首次发现这个等价——把心理学跟 RL 打通。

#### 反直觉点
- **心理学 1972 比 RL 1988 早 16 年**——但完整 TD（含 γ）是 Sutton 贡献
- **Rescorla-Wagner 是"单步"，TD 是"多步"**：γ>0 的 TD 能处理 delay
- **本是"分析动物行为"的工具——居然成了 AI 的核心数学**

#### 高频面试问法
**Q：Rescorla-Wagner 模型跟 TD(0) 是什么关系？**
**A**：**RW 模型 = TD(0) when γ=0**——心理学 1972 提出，比 RL 1988 早 16 年。两者本质都是"朝 prediction error 方向走 α 步"。**Sutton 1988 论文首次发现这个等价**，奠定 RL ↔ 心理学双向 influence。

---

### 3.3 TD Model of Classical Conditioning ⭐⭐⭐

#### 是什么
**Sutton & Barto 1981/1990** 用完整 TD（含 γ）建模 classical conditioning——比 Rescorla-Wagner 多个"时间因素"，能处理 inter-stimulus interval。

#### 为什么需要它
- Rescorla-Wagner 单步，不区分"cue 早多少时间出现"
- 实际动物对 cue 出现时间敏感（早 0.5s vs 5s 效果不同）
- TD 模型用 γ 自然处理 temporal credit

#### 数学
$$V(S_t) ← V(S_t) + α[R_{t+1} + γ V(S_{t+1}) - V(S_t)]$$

#### TD 模型 vs Rescorla-Wagner 实验区分
- 实验：cue A 提前 0.5s，cue B 提前 5s——动物对 A 反应更强
- RW 模型：无法区分（无时间维度）
- **TD 模型**：自然预测"近 cue 反应更强"——跟实验匹配

#### 反直觉点
- **TD 模型在心理学比 Rescorla-Wagner 更准**——是 RL 数学反哺心理学
- **多巴胺 = TD error**假说（Ch 15）直接来自这个 TD 模型

#### 高频面试问法
**Q：TD model 比 Rescorla-Wagner 在心理学上强在哪？**
**A**：TD 含 γ，能处理 inter-stimulus interval（cue 跟 outcome 的时间间隔）。实验上动物对"近 cue"反应强、"远 cue"反应弱——RW 不能解释，TD 自然预测。**RL 数学反哺心理学的经典案例**。

---

### 3.4 Instrumental Conditioning（操作条件反射）⭐⭐⭐

#### 是什么
**Thorndike 1898 / Skinner**：动物学到"动作 → reward"的关联——猫从 puzzle box 逃出来、鸽子按 lever 得食。

#### 跟 RL 对应
| 心理学 | RL |
|---|---|
| Operant response | action |
| Reinforcer (food/shock) | reward |
| Schedule of reinforcement | reward function |
| **Law of Effect (Thorndike)** | policy improvement |

**Thorndike's Law of Effect**：reward 跟动作绑定 → 加强该动作。**= policy improvement 的心理学版**。

#### Skinner Box 实验
- 鸽子按 lever：give food → 按 lever 频率增加
- 按 lever：give shock → 按 lever 频率降低
- 这是 **Q-learning 的动物版**——学 Q(状态, 按 lever)

#### 跟 classical conditioning 对比
| | Classical (Pavlov) | Instrumental (Thorndike/Skinner) |
|---|---|---|
| 学什么 | stimulus → outcome | action → outcome |
| 动物角色 | 被动 | 主动 |
| RL 对应 | TD prediction | TD control |

#### 反直觉点
- **两种 conditioning 同时存在于动物 + 人**：classical 学预测，instrumental 学控制
- **AI 的 RL 跟心理学 instrumental 是同一回事**

#### 高频面试问法
**Q：Classical vs Instrumental conditioning 在 RL 有什么对应？**
**A**：**Classical → TD prediction**（学 cue → outcome 关联，对应 V learning）；**Instrumental → TD control**（学 action → reward 关联，对应 Q learning 或 policy gradient）。**RL 不仅是计算 framework，也是动物学习理论的形式化**。

---

### 3.5 Delayed Reinforcement + Eligibility Traces

#### 是什么
**心理学发现**：reward 延迟 N 秒 → 动物仍能学（但 N 大时弱）——大脑用"trace"记忆最近的 action。

#### 跟 Eligibility Trace 关联
- 心理学的"trace decay"概念——动物大脑有"短期记忆痕迹"
- RL 的 eligibility trace（[[ch12-eligibility-traces]] §3.3）= 这个思想的数学化
- **eligibility 这个词就来自心理学**

#### 反直觉点
- **RL 的 eligibility trace 不是凭空发明**——直接借用心理学的"突触可塑性"概念
- **Hebbian learning** 跟 eligibility trace 思想相通

#### 高频面试问法
**Q：Eligibility trace 的"eligibility"是什么意思？**
**A**：心理学/神经生物学术语——突触"eligible"（合格）for plasticity（可塑性）。动物大脑里"最近活跃的突触"被标记为可被未来 reward 调整。**Sutton 借用这个术语 = RL 的 trace z_t**，记录"最近 active feature"的累积权重。

---

### 3.6 Habits vs Goal-directed ⭐⭐⭐

#### 是什么
**心理学早 RL 几十年就有的区分**：
- **Habit**（习惯）：刺激→反应直接映射，不考虑未来后果
- **Goal-directed**：基于对结果的预期来决策

**Dickinson 1985** 经典实验：
- 训鼠按 lever → 食物
- 让食物变难吃（reward devaluation）
- 短训鼠仍按 lever（habit）；长训鼠不按了（goal-directed）

#### 跟 RL 对应
| 心理学 | RL |
|---|---|
| Habit | **Model-free RL**（cached Q value）|
| Goal-directed | **Model-based RL**（plan with model）|

#### 大脑里两套系统并存
- **Striatum**（纹状体）：model-free，habit 控制
- **Prefrontal Cortex**（前额叶）：model-based，goal-directed
- 两套竞争 + 协同——**这跟 Dyna（Ch 8）思想完全一致**

#### 反直觉点
- **心理学早 RL 几十年就区分两种 system**——RL 借鉴
- **现代 RL 的 Dyna / MuZero / Dreamer 都在做"habit + goal-directed"协同**
- **人类 dual-process theory** (Kahneman System 1/2) 跟这个对应

#### 高频面试问法
**Q：Habits vs goal-directed 在 RL 怎么对应？**
**A**：**Habit = Model-free**（cached Q，快但不灵活）；**Goal-directed = Model-based**（用 model plan，慢但适应变化）。**大脑里两套 RL system 并存**（striatum 管 habit，PFC 管 goal-directed）——RL 的 Dyna / MuZero 是这两套协同的工程化。

---

### 3.7 Latent Learning（Tolman）⭐⭐

#### 是什么
**Tolman 1948** 经典实验：老鼠在迷宫"逛"几天（无 reward），第 N 天加 reward → 它**立刻**走最短路径——**说明它在"逛"时已学到了 model**。

#### 为什么是 model-based RL 灵感
- 行为派认为"无 reward 不学"——但 Tolman 证明动物能 learn without reinforcement
- 它学的是 **cognitive map**（环境结构）= **model**
- **Model-based RL 的心理学根**

#### 跟 RL 对应
| 心理学 | RL |
|---|---|
| Cognitive map（Tolman 1948）| **Model** in MDP |
| Latent learning (no reward) | **Model learning** in Dyna |
| Sudden goal-directed behavior | Model-based planning |

#### 反直觉点
- **1948 提出"动物学到 cognitive map"——当时行为学派认为是异端**
- **Model-based RL 是 70 年前心理学思想的工程化**

#### 高频面试问法
**Q：Tolman 1948 Latent Learning 跟 Model-based RL 关系？**
**A**：Tolman 证明动物能在"无 reward 时"学到环境结构（**cognitive map**）—— 加 reward 后立刻 goal-directed 行动。**这是 Model-based RL 的心理学根**——动物先学 model 再 plan。Dyna / MuZero 是这个思想的现代实现。

---

## 4. 跨概念对比表

### 4.1 心理学 ↔ RL 对照大全
| 心理学概念 | RL 概念 | 关键人物/年份 |
|---|---|---|
| Classical conditioning | TD prediction | Pavlov 1904 |
| Instrumental conditioning | TD control | Thorndike 1898 |
| Law of Effect | Policy improvement | Thorndike 1898 |
| Rescorla-Wagner | TD(0), γ=0 | 1972 |
| Cognitive map | Model | Tolman 1948 |
| Latent learning | Model-based RL | Tolman 1930s |
| Habit | Model-free policy | Dickinson 1985 |
| Goal-directed | Model-based decision | Dickinson 1985 |
| Eligibility trace | TD(λ) z_t | 心理学早期 |
| Delay discounting (hyperbolic) | γ-discount (exponential) | — |
| Schedule of reinforcement | Reward design | Skinner 1957 |

### 4.2 心理学贡献 vs RL 借鉴

| 心理学贡献到 RL | RL 反哺心理学 |
|---|---|
| Rescorla-Wagner → TD(0) | TD 含 γ → 解释 inter-stimulus interval |
| Eligibility trace 术语 | TD(λ) 形式化 |
| Habits vs Goal-directed | Dyna / MuZero 工程化 |
| Cognitive map | Model-based RL |
| **Sutton 1988** | 双向打通 |

---

## 5. 习题精选

**Exercise 14.1**：手算证明 Rescorla-Wagner = TD(0) when γ=0。
**Exercise 14.2**：用 TD model 解释 inter-stimulus interval 实验。

## 6. 代码伴侣

- ShangtongZhang 无对应代码（叙述章）
- **必读论文**：
  - Sutton 1988 *Learning to Predict by the Methods of Temporal Differences*
  - Sutton & Barto 1990 *Time-Derivative Models of Pavlovian Reinforcement*
  - Schultz/Dayan/Montague 1997 Science（多巴胺 = TD error）

## 7. 跟 RLHF 的桥（间接）

| RLHF 概念 | 心理学根（本章）|
|---|---|
| Reward shaping | Skinner 的 schedule of reinforcement |
| Constitutional AI principles | superego（道德/规则内化） |
| Sycophancy 谄媚 | social reinforcement（讨好反馈强化） |
| Reward hacking | 操作条件反射的 dark pattern |
| Iterative RLHF | shaping reward 渐进塑造 |
| Long-CoT reasoning | goal-directed planning（System 2 思维）|

## 8. 苏格拉底 Q&A

详见每个 §3.x。综合：

- **Q：心理学跟 RL 谁先谁后？**
  - 答：心理学早 RL 几十年——Rescorla-Wagner 1972 vs TD 1988（16 年）；Thorndike 1898 vs Q-learning 1989（91 年）。**RL 是心理学的数学化**。
- **Q：Sutton 1988 论文为什么重要？**
  - 答：首次发现 Rescorla-Wagner = TD(0) → 打通 RL ↔ 心理学双向 influence。**奠定 RL 不只是 AI 算法，也是动物学习的数学模型**。

## 9. 自测清单（闭卷）

- [ ] **写出 Rescorla-Wagner 公式 + 跟 TD(0) 等价证明**
- [ ] **列 6+ 个心理学 ↔ RL 对照**
- [ ] **解释 Tolman 1948 跟 Model-based RL 关系**
- [ ] **解释 Habits vs Goal-directed 跟 Model-free/based 对应**
- [ ] **解释为什么 RLHF 的 reward shaping 跟心理学有关**

---

## 💡 拓展知识点（书外 trivia）

### A. Pavlov 1904 诺贝尔奖
Pavlov 因消化生理学（不是 conditioning）拿诺贝尔奖。**Conditioning 是他后来研究的副产品**——成为 RL 思想的鼻祖。

### B. Sutton 跟心理学的关系
Sutton 本科在斯坦福读心理学——奠定他对 RL ↔ 心理学双向理解的基础。**RL 教父深受心理学影响**。

### C. RL 跟"Bitter Lesson"的反讽
Sutton 2019 *Bitter Lesson*：scale > clever algorithm。但他本人的 TD 算法**借鉴了 100 年的心理学**——一个**反 Bitter Lesson 的案例**。

### D. Constitutional AI 的心理学血缘
Anthropic 的 Constitutional AI 让模型"内化 principles" → 跟弗洛伊德 superego 概念惊人相似。**心理学→ AI alignment 直接桥梁**。

### E. 人类 delay discounting 不是 exponential
- RL 用 $γ^t$ 指数折扣
- 人类用 hyperbolic discount $\frac{1}{1+kt}$
- 这就是为什么人类有"现在的诱惑"问题（procrastination）
- **RL γ 模型对人类不严格匹配**

### F. Schedule of Reinforcement 影响 RLHF
Skinner 的 schedule of reinforcement（间歇 reward 比连续 reward 更稳）→ **影响 RLHF 的 reward design**（reward 频率 / sparsity 选择）。

### G. 当代心理学跟 RL 的反向影响
- Behavioral economics 借用 RL 概念解释人类决策
- Reinforcement learning theory of depression（多巴胺 RPE 失调）
- **RL 反过来形塑心理学**——双向输出

### H. AlphaGo 的"创造力"挑战心理学
AlphaGo 下出第 37 手——人类无法解释的"创造"。**心理学被迫思考"什么是 creativity"**——是否所有创造性都能被 RL + search 解释？开放问题。

---

## 🔗 更多 wikilinks（深挖）

| 主题 | 跳转 |
|---|---|
| 神经科学版（dopamine = TD error）| [[ch15-neuroscience]] |
| TD(0) 数学 | [[ch06-temporal-difference-learning]] §3.1 |
| Eligibility trace 数学 | [[ch12-eligibility-traces]] §3.3 |
| Model-based RL（Tolman 工程化）| [[ch08-planning-and-learning]] §3.2 |
| RLHF reward shaping | [[../../../brain/Areas/rl-books/rlhf-lambert/ch14-style-and-information]] |

## Related

- **上一章**：[[ch13-policy-gradient]]
- **下一章**：[[ch15-neuroscience]]
- **进度**：[[_chapter-status]] W8

## Sources

- 原书 Ch 14 pp. 357-378
- **经典论文**：
  - Rescorla & Wagner 1972
  - Sutton 1988 *Learning to Predict by TD*
  - Tolman 1948 *Cognitive maps in rats and men*
  - Dickinson 1985 *Actions and habits*
  - Sutton & Barto 1990 *Time-Derivative Models of Pavlovian Reinforcement*
