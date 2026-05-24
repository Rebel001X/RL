---
title: "Sutton & Barto Ch 14. Psychology"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 14
difficulty: easy
budget_days: 1
priority_for_rlhf: skip
silver_lecture: 9
---

# Ch 14. Psychology

## 0. 阅读元信息

- **难度**：easy（叙述性，无数学难关）
- **预算**：1 天 / skim
- **必读理由**：理解 RL 的"前身"——动物学习心理学；多巴胺 = TD error 假说在 Ch 15
- **配套**：Silver UCL Lec 9（高级主题）
- **跟 RLHF 的桥**：弱直接——但 reward shaping、social learning 等概念对 alignment 有启发

## 1. 一句话

RL 的数学框架跟动物行为学习高度对应——classical conditioning（巴甫洛夫）≈ value learning，instrumental conditioning（操作条件反射）≈ policy/control；本章讲两个领域的术语对照与历史交融。

## 2. 关键概念地图

| 心理学概念 | RL 对应 |
|---|---|
| **Classical conditioning** | TD prediction（学 cue → outcome）|
| **Pavlov's dog** | Rescorla-Wagner model = TD(0) special case |
| **Instrumental conditioning** | TD control（学 action → reward）|
| **Thorndike's law of effect** | "reward → 加强行为" = policy improvement |
| **Latent learning** | model-based RL（先学环境再用） |
| **Habitual vs goal-directed** | model-free vs model-based |
| **Secondary reinforcement** | 学到的 value 当 reward |
| **Generalization** | function approximation 的心理学版 |
| **Delay discounting** | 折扣因子 γ |
| **Cognitive map** | Tolman 1948——内化的 model |

## 3. 必背公式与推导

### Rescorla-Wagner Model（1972，心理学经典）

$$ΔV_X = α(λ - \sum_X V)$$

其中 V_X 是 cue X 与 outcome 的关联强度，λ 是实际 outcome——**等价于 RL 的 TD(0) error 更新**。

### 跟 TD(0) 的等价

$$V(S_t) ← V(S_t) + α [R_{t+1} + γV(S_{t+1}) - V(S_t)]$$

—— 1981 Sutton 发现 Rescorla-Wagner 是 TD(0) 的 γ=0 退化版（无 bootstrap），从此 RL ↔ 心理学打通。

## 4. 关键算法（伪代码）

无算法（叙述章）。但理解 Rescorla-Wagner 与 TD 等价后，可写 9 行 Python 复现 Pavlov experiment：

```python
V_bell = 0; α = 0.1
for trial in range(100):
    cue = bell rings
    R = food
    V_bell += α (R - V_bell)
# V_bell 收敛到 R——dog 听见 bell 期待 food
```

## 5. 习题精选

**Exercise 14.1（Rescorla-Wagner → TD 推导）**：手算证明 Rescorla-Wagner = TD(0) when γ=0。

## 6. 代码伴侣

- **ShangtongZhang**: 本章无对应代码（叙述章）
- 推荐：Schultz/Dayan/Montague 1997 Science 论文（多巴胺神经元 = TD error 假说，跨到 Ch 15）

## 7. 反直觉点 / 易错点

- **反直觉**：心理学早 RL 几十年提出"prediction error"——RL 是数学化版本
- **反直觉**：habits（model-free）vs goal-directed（model-based）在心理学早有划分，2020s 才系统化进 RL
- **反直觉**：delay discounting 的"hyperbolic" 形式跟 RL 的"exponential γ"不一致——人类不严格 exponential discount

## 8. 跟 RLHF 的桥

- **弱直接**——但 alignment 哲学受心理学影响：
  - "shaping rewards" 来自 Skinner box 实验
  - constitutional AI 类比"教 dog 守规矩"
  - sycophancy（谄媚）跟"behavioral reinforcement"的 dark pattern 类似
- **不必背本章**为 RLHF 服务，但读完会让 reward design 直觉更好

## 9. 苏格拉底 Q&A

- **Q：Rescorla-Wagner 1972 跟 TD(0) 哪个先？**
  - 提示：心理学先，TD(0) 是数学化（Sutton 1981/1988）
- **Q：habits vs goal-directed 在 LLM agent 里有对应吗？**
  - 提示：cached responses（habits）vs reasoning model（goal-directed）；o1/R1 是后者
- **Q：delay discounting 不是 exponential，RL 用 exponential γ 为什么没出问题？**
  - 提示：工程上 hyperbolic 可近似为 piecewise exponential；且 RL 不是为模拟人类

## 10. 自测清单（闭卷）

- [ ] 解释 Rescorla-Wagner 跟 TD(0) 等价关系
- [ ] 列出 4 个心理学 ↔ RL 对应
- [ ] 答 Q1-Q3 至少 1 个

## Related

- **上一章**：[[ch13-policy-gradient]]
- **下一章**：[[ch15-neuroscience]]
- **进度**：[[_chapter-status]] W8 (skim)

## Sources

- 原书 Ch 14 pp. 357-378
- Rescorla & Wagner 1972 原始论文
- Sutton 1988 "Learning to Predict by the Methods of Temporal Differences"

## 附：本章 RL ↔ 心理学全对照表

| 心理学术语 | RL 术语 | 关键来源 |
|---|---|---|
| Classical conditioning | TD prediction | Pavlov 1927 |
| Instrumental conditioning | TD control | Thorndike 1898 |
| Rescorla-Wagner model | TD(0), γ=0 | Rescorla & Wagner 1972 |
| Cognitive map | Model | Tolman 1948 |
| Latent learning | Model-based RL（无 reward 也学）| Tolman 1930s |
| Secondary reinforcer | 学到的 V (s) 当 reward 替代 | — |
| Goal-directed action | Model-based decision | — |
| Habit | Model-free policy | Dickinson 1985 |
| Delay discounting | γ < 1 折扣 | — |
| Generalization | Function approximation | — |
| Habituation / Sensitization | Pre-training 的副作用 | — |
| Negative reinforcement | 学避险 action | Mowrer 1960 |
| Schedule of reinforcement | Reward design | Skinner 1957 |

## 附：从 RL 看"动物为什么学得快"

- 人/动物 RL 比 RL agent **样本效率高几千倍**——靠 prior + 强先验 + episodic memory
- 模仿学习（imitation）+ social learning 让"探索"几乎免费——LLM SFT 类似
- "Curiosity" 是内禀 reward——RL 的 intrinsic motivation 流派直接借

## 附：从心理学反推 RLHF 的设计选择

读完本章后看 RLHF 工程，会发现很多设计是"心理学常识"的形式化：

| RLHF 工程 | 心理学/行为学根 |
|---|---|
| KL penalty（贴 ref model）| habit dominance——人不轻易偏离已有 pattern |
| Reward hacking | 操作条件反射的 dark pattern（如赌博成瘾）|
| Sycophancy 谄媚 | social reinforcement——讨好反馈强化 |
| Constitutional AI principles | superego（道德 / 规则内化）|
| Iterative RLHF | shaping reward 渐进塑造 |
| Long-CoT reasoning | goal-directed planning（系统 2 思维）|

**启示**：alignment 不只是技术问题，也是行为塑造的工程。心理学 100 年的经验值得 RLHF 工程借鉴。

## 附：本章是否值得深读？

- **目标 = RL 工程师**：skim 即可，本章不影响算法实现
- **目标 = alignment researcher**：值得深读——理解 reward 设计背后的人类行为模型
- **目标 = 认知科学跨界**：必读——这是 RL 跟 cognitive science 的桥
