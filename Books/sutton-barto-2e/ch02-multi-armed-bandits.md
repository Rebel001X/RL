---
title: "Sutton & Barto Ch 2. Multi-armed Bandits"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 2
difficulty: easy
budget_days: 2
priority_for_rlhf: nice
silver_lecture: 2
---

# Ch 2. Multi-armed Bandits

## 0. 阅读元信息

- **难度**：easy（数学最少的一章，但概念基础）
- **预算**：2 天 / ~3 小时
- **必读理由**：RL 的"退化版"——只有 1 个 state，方便聚焦"exploration vs exploitation"这一根本张力；为后续 full RL 打概念基础
- **配套**：Silver UCL Lec 2（前半部分讲 bandits），CS234 Lec 1-2
- **跟 RLHF 的桥**：best-of-N（BoN）/ rejection sampling 本质都是 bandit；RM 评分选最优 response 是 contextual bandit

## 1. 一句话

Bandit 是只有一个 state 的退化 RL——agent 反复在 k 个 action 中选，每个 action 有未知的 reward 分布，目标是最大化长期累积 reward；核心张力是**探索新 action 收集信息 vs 利用已知最好 action 立即收益**。

## 2. 关键概念地图

| 概念 | 一句话 |
|---|---|
| **k-armed bandit** | k 个 action，每个 action 的 reward 服从某固定分布 q*(a) |
| **Action value Q_t(a)** | 在第 t 步前对 action a 的真实 value q*(a) 的估计 |
| **Greedy action** | argmax_a Q_t(a) |
| **ε-greedy** | 以概率 ε 随机选 action（探索），其余 greedy（利用） |
| **Sample-average method** | Q_t(a) = 已 sampled 的 reward 平均 |
| **Incremental update** | Q_{n+1} = Q_n + (1/n)(R_n - Q_n) — 不存历史只存 Q |
| **Stationary vs non-stationary** | reward 分布固定 vs 随时间变化（用 fixed step-size α 替代 1/n） |
| **Optimistic initial values** | 初始 Q 设很大，逼模型早期探索 |
| **UCB (Upper Confidence Bound)** | A_t = argmax [Q_t(a) + c √(ln t / N_t(a))] —— 不确定性越大越想试 |
| **Gradient bandit** | 学 policy 偏好 H_t(a)，softmax 选 action |

## 3. 必背公式与推导

### 增量平均（incremental average）

$$Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)$$

**推导**：
$$Q_{n+1} = \frac{1}{n} \sum_{i=1}^n R_i = \frac{1}{n}(R_n + (n-1) Q_n) = Q_n + \frac{1}{n}(R_n - Q_n)$$

**通用形式**：`new_estimate = old + step × (target - old)`——这是后续 TD、Q-learning 等所有学习规则的母版。

### Non-stationary：fixed step-size α

$$Q_{n+1} = Q_n + \alpha (R_n - Q_n)$$

展开后：$Q_n = (1-\alpha)^n Q_0 + \sum \alpha (1-\alpha)^{n-i} R_i$ —— **指数加权平均**，旧 reward 衰减。

### UCB

$$A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$

**直觉**：第二项是"不确定度"——访问少的 action（N_t(a) 小）+ 时间越长（ln t 大），更值得试。c 控制探索强度。

### Gradient bandit（softmax policy）

$$\pi_t(a) = \frac{e^{H_t(a)}}{\sum_b e^{H_t(b)}}$$

更新：
$$H_{t+1}(a) = H_t(a) + \alpha (R_t - \bar{R}_t)(1_{a = A_t} - \pi_t(a))$$

**这是 policy gradient 的最简退化版**（一个 state 的 PG）——预演 Ch 13。

## 4. 关键算法（伪代码）

**ε-greedy（增量版）**：

```python
for each step t:
    A = argmax(Q) if rand() > ε else random_action()
    R = bandit(A)
    N[A] += 1
    Q[A] += (1/N[A]) * (R - Q[A])
```

**UCB**：

```python
for each step t:
    if min(N) == 0: A = first unvisited action
    else: A = argmax(Q + c * sqrt(ln(t) / N))
    R = bandit(A)
    N[A] += 1; Q[A] += (1/N[A]) * (R - Q[A])
```

## 5. 习题精选（cherry-picked）

**Exercise 2.5（nonstationary 测试）**：用 10-armed testbed 但 q*(a) 每步加随机游走。比较 sample-average 和 α=0.1 fixed step。**关键答案**：α-fixed 显著更好，因为 sample-average 给所有过去 reward 等权重，跟不上变化。

**Exercise 2.9（gradient bandit 推导）**：证明 H 更新是 expected reward 的 stochastic gradient ascent。**关键步骤**：
$$\frac{\partial E[R_t]}{\partial H_t(a)} = E\left[ (R_t - b)(1_{a=A_t} - \pi_t(a)) \right]$$
对任意 baseline b 都成立（baseline 不改梯度无偏性，只降方差）。这是 Ch 13 baseline trick 的预演。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter02/ten_armed_testbed.py` — 复现书中 figure 2.1, 2.2, 2.3, 2.5, 2.6
- 自己 reimplement：写 5 行 ε-greedy + 5 行 UCB，对比 1000 步累积 reward
- 关键 diff：Zhang 用 numpy 向量化，自己写时先用 Python list 理解，再 numpy 加速

## 7. 反直觉点 / 易错点

- **反直觉**：ε=0（纯 greedy）几乎总是输给 ε=0.01——一点点探索就足够，但不能没有
- **反直觉**：optimistic initial values 在 stationary 任务上能"自动"探索（高估必然下降，逼访问），但在 non-stationary 上无效
- **反直觉**：UCB 的 ln t 增长很慢，所以 c 通常取 2 这种"小数"才有效
- **易错**：gradient bandit 的 baseline 不是"必须"才能学，它**只是降方差**——任何 baseline 都不破坏无偏性
- **易错**：fixed α 不收敛到 true Q*（永远在抖动），但对 non-stationary 是 feature 不是 bug

## 8. 跟 RLHF 的桥

- **Best-of-N (BoN)** ≈ k-armed bandit：从 N 个候选 response 中用 RM 选最高分——等价于一次性 bandit，没有 exploration 也没有时间维度
- **Rejection sampling fine-tuning (RFT)** = bandit + 把好 sample 喂回 SFT
- **Reward model 训练** 用 Bradley-Terry pairwise loss——本质是 dueling bandit（成对比较）的 RL 推广
- **PPO ratio** 涉及 importance sampling，这里 gradient bandit 章节的 baseline trick 是其精神前身

## 9. 苏格拉底 Q&A

- **Q：为什么 RL 教材开篇要讲一个"非 RL"的 bandit？**
  - 提示：剥离 state 维度，纯粹研究 exploration vs exploitation；后续 RL 算法的核心思想（增量更新、ε-greedy）都从这里来
- **Q：ε-greedy 跟 UCB 在什么场景下差别最大？**
  - 提示：reward 方差大时 UCB 更优——它能识别"不是真的不好，只是 sample 少"；ε-greedy 只是机械随机
- **Q：UCB 公式里的 ln t 改成 t 会怎样？**
  - 提示：探索项随时间快速增长，让"老 action 也被反复试"——理论上 regret bound 变差
- **Q：gradient bandit 不用 Q，那它"学"到了什么？**
  - 提示：直接学 policy 参数 H——这是 policy-based 思想的最早出现，跳过了 value estimation
- **Q：bandit 跟 contextual bandit 区别？**
  - 提示：contextual bandit 多了个 context（state-like）但仍是单步——是 bandit 到 full RL 的桥

## 10. 自测清单（闭卷）

- [ ] 不看书 1 分钟解释 exploration vs exploitation 给非技术朋友
- [ ] 闭眼写出增量平均公式 + 推导
- [ ] 闭眼写出 UCB 公式 + 解释每一项
- [ ] 实现 ε-greedy 和 UCB on 10-armed testbed，画累积 reward 曲线
- [ ] 答 Q1-Q5 中至少 3 个
- [ ] 解释 BoN 跟 bandit 关系（RLHF 桥）

## Related

- **first-pass**：（无——Ch 1 跳到 Ch 2 直接深读）
- **上一章**：[[ch01-introduction]]
- **下一章**：[[ch03-finite-mdps]]
- **RLHF 关联**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch09-rejection-sampling]]
- **进度**：[[_chapter-status]] W1

## Sources

- 原书 Ch 2 pp. 25-44
- [Silver UCL Lec 2 (Markov Decision Processes, 含 bandits 复习)](https://www.youtube.com/watch?v=lfHX2hHRMVQ)
- ShangtongZhang `chapter02/ten_armed_testbed.py`
- [planetbanatt § Ch 2](https://planetbanatt.net/articles/sutton.html)
- Lattimore & Szepesvári《Bandit Algorithms》(进阶参考)
