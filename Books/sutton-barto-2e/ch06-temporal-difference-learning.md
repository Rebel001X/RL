---
title: "Sutton & Barto Ch 6. Temporal-Difference Learning"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 6
difficulty: hard
budget_days: 3-4
priority_for_rlhf: must
silver_lecture: 4
---

# Ch 6. Temporal-Difference Learning ⭐⭐⭐

## 0. 阅读元信息

- **难度**：hard（概念密集 + Q-learning 是 RL 最重要算法）
- **预算**：3-4 天 / ~7 小时
- **必读理由**：**TD 是 RL 的中心思想**——结合 DP（bootstrap）和 MC（model-free）的优点；Q-learning 是史上最有影响的 RL 算法；面试必考
- **配套**：Silver UCL Lec 4 后半 + Lec 5
- **跟 RLHF 的桥**：PPO 的 GAE / critic 用 TD error 估 advantage；理解 TD 才能理解 critic 训练

## 1. 一句话

TD = MC 的"等结束才更新"改成"下一步就更新"——用 $R_{t+1} + γV(S_{t+1})$ 当 target（bootstrap），不需要 episode 结束，可在线学习；衍生出 SARSA（on-policy）和 Q-learning（off-policy）两大支系。

## 2. 关键概念地图

| 概念 | 一句话 |
|---|---|
| **TD(0) update** | V(S_t) += α [R_{t+1} + γV(S_{t+1}) - V(S_t)] |
| **TD error** | δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t) —— bootstrap target 与当前估计的差 |
| **TD target** | R_{t+1} + γV(S_{t+1}) —— 用"下一步 reward + 下一步 V"代替真 G_t |
| **SARSA** | on-policy TD control：Q(S,A) += α [R + γQ(S',A') - Q(S,A)]，A' 按当前 π 采 |
| **Q-learning** | off-policy TD control：Q(S,A) += α [R + γ max_a' Q(S',a') - Q(S,A)]，独立于 behavior |
| **Expected SARSA** | Q(S,A) += α [R + γ Σ_a' π(a'\|S') Q(S',a') - Q(S,A)]，方差更小 |
| **Double Q-learning** | 两个 Q 表交替更新，消除 max 引入的 over-estimation bias |
| **Cliff walking** | Example 6.6，演示 SARSA vs Q-learning 的"安全 vs 最优"差异 |

## 3. 必背公式与推导

### TD(0) Prediction

$$V(S_t) \leftarrow V(S_t) + \alpha \big[\underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD target}} - V(S_t)\big]$$

**为什么 TD < MC 方差**：MC 用真 G_t（depends on whole future randomness），TD 只用 R_{t+1} + V(S_{t+1})（一步随机 + bootstrap）。

**为什么 TD 有 bias**：V(S_{t+1}) 是估计值，不准则 target 不准 → 偏；MC 无 bias。

### Q-learning

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\big]$$

**off-policy**：target 用 max_a'，与 behavior policy 选的 A' 无关——直接学最优 Q\*。

### SARSA

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\big]$$

**on-policy**：A_{t+1} 按当前 π（含 ε-greedy 的随机性）—— 学的是 π 的 Q，不是 Q\*。

### TD error 的几何

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

这是 V 的"预测误差"——所有现代 actor-critic（含 PPO）的 advantage 估计基于 δ。

## 4. 关键算法（伪代码）

**Q-learning**：
```
init Q(s, a) arbitrarily; Q(terminal) = 0
loop for each episode:
  S = init state
  loop for each step:
    A = ε-greedy(Q, S)
    take A, observe R, S'
    Q(S, A) += α [R + γ max_a' Q(S', a') - Q(S, A)]
    S = S'
  until S terminal
```

**SARSA**：
```
init Q
loop for each episode:
  S = init state
  A = ε-greedy(Q, S)
  loop for each step:
    take A, observe R, S'
    A' = ε-greedy(Q, S')
    Q(S, A) += α [R + γ Q(S', A') - Q(S, A)]
    S = S'; A = A'
  until S terminal
```

**Double Q-learning**：维护 Q1, Q2，每步 50% 更新一个用另一个估 target。

## 5. 习题精选

**Exercise 6.4（cliff walking step size）**：在 Cliff walk 上比较 α=0.1 vs 0.5 vs 0.9——大 α 学得快但不稳。

**Exercise 6.6（cliff walking 反思）**：为什么 Q-learning 学到最优 path 但在线 reward 比 SARSA 差？**答案**：Q-learning 学的是 deterministic optimal（沿悬崖走），但 ε-greedy 执行时偶尔随机 → 掉悬崖；SARSA 学到"考虑 ε 的安全 path"（走远点）。

**Exercise 6.10（windy gridworld with king's moves）**：扩展 wind grid 加 8 方向 + stay action。代码题。

**Exercise 6.14（double Q-learning bias）**：证明 max E[Q] ≥ E[max Q]（Jensen）—— 这是 Q-learning over-estimation 的根。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter06/random_walk.py`（TD vs MC）、`chapter06/cliff_walking.py`（SARSA vs Q-learning）、`chapter06/windy_grid_world.py`、`chapter06/maximization_bias.py`（Double Q）
- **必跑**：cliff_walking.py 跑完看 figure 6.4——SARSA 走"安全路径"，Q-learning 走"悬崖边"
- **自己写**：Q-learning on FrozenLake-v1（gym），20 行 numpy

## 7. 反直觉点 / 易错点

- **反直觉**：TD 比 MC 方差小但**有 bias**——bias-variance 经典权衡
- **反直觉**：Q-learning 用 max → **systematic over-estimation bias**——所有 Q 都被高估，特别 stochastic env 严重
- **反直觉**：SARSA 看似比 Q-learning 弱（只学到次优），但**实际在线表现更好**（在悬崖类风险环境）
- **反直觉**：Expected SARSA 方差比 SARSA 小 + bias 跟 SARSA 同——通常更优
- **易错**：忘记 terminal state 的 V = 0
- **易错**：on-policy vs off-policy 不是"算法属性"——是"target policy 跟 behavior 是否同一个"

## 8. 跟 RLHF 的桥（⭐ 必懂）

- **PPO 的 critic V_φ(s)** 用 TD-style 训：$V_φ(s) \leftarrow V_φ(s) + \alpha [R + γV_φ(s') - V_φ(s)]$ —— 跟本章 TD(0) prediction 完全相同
- **TD error δ_t** 是 GAE 的基础：GAE(λ) = $\sum_{l} (γλ)^l δ_{t+l}$ —— λ-weighted TD errors
- **GRPO 不用 critic**：直接用 group 内 reward 归一化估 advantage——绕过 TD learning
- **Q-learning 思想在 RLHF**：DPO 也间接学了 Q（隐式 reward = β log[π/π_ref]）

## 9. 苏格拉底 Q&A

- **Q：TD 比 MC 强在哪？**
  - 提示：(1) 不等 episode 结束（continuing 任务也行）；(2) 方差小；(3) 在线（每步学）
- **Q：TD 比 MC 弱在哪？**
  - 提示：bias（bootstrap 误差传播）；理论收敛性要求更严
- **Q：Q-learning 和 SARSA 算法上只差一处（max vs sample），为什么本质不同？**
  - 提示：max → 学最优 policy 的 Q\*（off-policy）；sample → 学当前 policy 的 Q_π（on-policy）
- **Q：Cliff walking 里 SARSA"更好"是道德故事还是数学故事？**
  - 提示：是**学到的 policy 与执行 policy 一致性**——SARSA 学到"我会 ε-greedy 走"的最优，Q-learning 学到"deterministic 最优"但执行时是 ε-greedy
- **Q：Q-learning over-estimation 怎么解决？**
  - 提示：Double Q-learning（two Q tables）/ Double DQN（target net + online net 解耦）
- **Q：为什么 RLHF 不直接用 Q-learning？**
  - 提示：action space 是 vocab（~50k），max_a 太贵；reward 稀疏 → bootstrap 没用；policy gradient 更适合

## 10. 自测清单（闭卷）

- [ ] 写出 TD(0), SARSA, Q-learning, Expected SARSA 四个更新公式
- [ ] 解释 on-policy vs off-policy 差异
- [ ] 解释 Q-learning over-estimation bias 来源（Jensen）
- [ ] 解释 cliff walking 里 SARSA vs Q-learning 差异的原因
- [ ] 不看 Zhang 自己实现 Q-learning on FrozenLake
- [ ] **解释 PPO critic 跟本章 TD prediction 的关系**
- [ ] 答 Q1-Q6 至少 4 个

## Related

- **上一章**：[[ch05-monte-carlo-methods]]
- **下一章**：[[ch07-n-step-bootstrapping]]
- **RLHF 桥**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning]]
- **进度**：[[_chapter-status]] W3 ⭐

## Sources

- 原书 Ch 6 pp. 117-138
- [Silver UCL Lec 4 后半 + Lec 5](https://www.youtube.com/watch?v=0g4j2k_Ggc4)
- ShangtongZhang `chapter06/`
- DQN 论文（Mnih 2015）—— Q-learning 的深度版
- Double DQN 论文（van Hasselt 2015）
