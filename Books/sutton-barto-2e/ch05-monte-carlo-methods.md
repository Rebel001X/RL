---
title: "Sutton & Barto Ch 5. Monte Carlo Methods"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 5
difficulty: medium
budget_days: 3
priority_for_rlhf: must
silver_lecture: 4
---

# Ch 5. Monte Carlo Methods

## 0. 阅读元信息

- **难度**：medium（§5.5 importance sampling 是难点，且**RLHF 必懂**）
- **预算**：3 天 / ~6 小时
- **必读理由**：MC 是第一个 model-free 算法——用 sample average 替代期望；**§5.5 importance sampling 是 PPO ratio 的根**
- **配套**：Silver UCL Lec 4（前半 Model-free Prediction）
- **跟 RLHF 的桥**：PPO 的 importance sampling ratio 直接来自 §5.5；off-policy 评估的核心思想

## 1. 一句话

Monte Carlo (MC) = 跑完整 episode → 算 G_t → 取平均估 V/Q；不需要 model（无 bootstrap），但**只能用于 episodic 任务**，且需 episode 完成才能更新。

## 2. 关键概念地图

| 概念 | 一句话 |
|---|---|
| **First-visit MC** | 一个 episode 内同一 state 第一次出现时记 G_t；多 episode 取平均估 V(s) |
| **Every-visit MC** | 每次访问都记——理论略不同但实际差不多 |
| **Exploring Starts** | 强制每个 (s, a) 都有 nonzero 起始概率——保证 q 全覆盖 |
| **On-policy** | 用同一个 policy 采数据 + 优化 |
| **Off-policy** | 用 behavior policy b 采数据，优化 target policy π |
| **Importance Sampling (IS)** | 用 ρ = π(a\|s)/b(a\|s) 加权修正分布差异 |
| **Ordinary IS** | 直接 ρG 平均——无偏但方差大 |
| **Weighted IS** | (Σ ρG) / (Σ ρ)——有偏但方差小，实践首选 |
| **Incremental MC** | 增量 V(s) += α (G_t - V(s)) |

## 3. 必背公式与推导

### First-visit MC

$$V(s) \leftarrow \text{average of all } G_t \text{ following first visits to } s$$

### Off-policy Importance Sampling

$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

$$V_\pi(s) = E_b[\rho_{t:T-1} G_t | S_t = s]$$

**关键证明**：
$$E_b[\rho G] = \sum_\tau b(\tau) \cdot \frac{\pi(\tau)}{b(\tau)} G(\tau) = \sum_\tau \pi(\tau) G(\tau) = E_\pi[G]$$

—— **任何分布的期望都可通过另一分布的加权样本估计**。这是 PPO/GRPO 的根。

### Weighted IS

$$V(s) = \frac{\sum_t \rho_t G_t}{\sum_t \rho_t}$$

有偏但渐进无偏 + 方差远小于 ordinary IS。**实践中默认用 weighted**。

## 4. 关键算法（伪代码）

**MC Control with Exploring Starts**：
```
init Q(s, a), π(s) arbitrarily; Returns(s, a) = []
loop forever:
  generate episode using π with exploring start (S_0, A_0 random)
  G = 0
  for t = T-1 down to 0:
    G = γG + R_{t+1}
    if (S_t, A_t) not in earlier steps:
      Returns(S_t, A_t).append(G)
      Q(S_t, A_t) = average(Returns(S_t, A_t))
      π(S_t) = argmax_a Q(S_t, a)
```

**Off-policy MC with Weighted IS**：
```
init Q(s, a), C(s, a) = 0; π = greedy(Q); b = ε-soft
loop:
  generate episode from b: S_0, A_0, R_1, ..., S_{T-1}, A_{T-1}, R_T
  G = 0; W = 1
  for t = T-1 down to 0:
    G = γG + R_{t+1}
    C(S_t, A_t) += W
    Q(S_t, A_t) += (W/C(S_t, A_t)) (G - Q(S_t, A_t))
    π(S_t) = argmax_a Q(S_t, a)
    if A_t ≠ π(S_t): break    # importance ratio 0
    W *= 1/b(A_t|S_t)
```

## 5. 习题精选

**Exercise 5.5（first vs every visit）**：在一个 10 步 episode，比较两种 estimator 对 V 的估计。Every-visit 偏差但方差更小。

**Exercise 5.10（weighted IS 增量更新）**：推导 weighted IS 的增量公式（如上算法中）。**关键**：分子分母同时累加，比例自动正确。

**Exercise 5.12（Racetrack）**：实现 race track 控制——经典 off-policy MC 应用。代码题。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter05/blackjack.py`（Example 5.1, 5.3）、`chapter05/infinite_variance.py`（IS 方差爆炸演示）、`chapter05/racetrack.py`
- **必看**：`infinite_variance.py` 演示 ordinary IS 方差可能无限大——理解为什么实践用 weighted

## 7. 反直觉点 / 易错点

- **反直觉**：MC 不需要 model，但需要"episode 终止"——continuing 任务直接不能用 MC
- **反直觉**：every-visit 略偏（同一 episode 内 G 相关），但实践跟 first-visit 表现相近
- **反直觉**：ordinary IS 方差可能**无限大**（即使期望有限）——见 figure 5.4
- **易错**：importance ratio 是 trajectory 累乘——长 trajectory 数值不稳（exp 或 underflow）；这是 PPO clip 的动机之一
- **易错**：off-policy MC 必须 b 在 π 支持上 nonzero（"absolute continuity"），否则 ratio 无意义

## 8. 跟 RLHF 的桥（⭐ 必懂）

**PPO ratio = single-step importance ratio**：
$$r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$$

—— 这就是 MC 章节的 importance ratio，只是从 trajectory 退化到单步 + clip。

**RLHF 的 off-policy 视角**：
- "behavior policy" = 收集 rollout 时用的旧 policy（每轮 update 前的）
- "target policy" = 想优化的新 policy
- 不重新 rollout 用旧数据多 epoch update —— 就是 off-policy + IS
- **clip 是为了控制 ratio 太大 → IS 方差爆炸的 RL 版"weighted IS"**

读完本章你能"打通" PPO clip 的数学根。

## 9. 苏格拉底 Q&A

- **Q：MC 比 DP 强在哪？**
  - 提示：不需要 model；可用真实 sample（Backgammon 没法 enumerate state）
- **Q：MC 比 TD 弱在哪？**
  - 提示：必须等 episode 结束；方差更大（用真实 G，不 bootstrap）
- **Q：ordinary IS 为什么方差可能无限大？**
  - 提示：ratio 的乘积可以任意大（数学上无界）；某些 trajectory 极少出现但有巨大 weight
- **Q：weighted IS 偏在哪？**
  - 提示：sample 数有限时分母 Σρ 不等于 N，比例略偏；N→∞ 时偏消失
- **Q：RLHF 不能用 MC 吗？为什么主流用 PPO（TD-style）？**
  - 提示：理论可以（一个 trajectory = 一个 episode），但 trajectory 短、reward 稀疏（只末尾给）—— MC 方差爆炸；PPO 的 GAE 用 bootstrap 折中

## 10. 自测清单（闭卷）

- [ ] 写出 first-visit MC 算法
- [ ] 推导 importance sampling 等式（任意分布的期望可换分布估）
- [ ] 解释 ordinary vs weighted IS 的偏差/方差权衡
- [ ] **解释 PPO ratio 来自本章哪个公式**（RLHF 桥必答）
- [ ] 跑通 ShangtongZhang `chapter05/infinite_variance.py`，体会方差爆炸
- [ ] 答 Q1-Q5 至少 3 个

## Related

- **上一章**：[[ch04-dynamic-programming]]
- **下一章**：[[ch06-temporal-difference-learning]]
- **RLHF 桥**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning]]
- **进度**：[[_chapter-status]] W2

## Sources

- 原书 Ch 5 pp. 91-115
- [Silver UCL Lec 4: Model-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA)
- ShangtongZhang `chapter05/`
- PPO 论文（Schulman 2017）—— 引用本章 IS 思想
