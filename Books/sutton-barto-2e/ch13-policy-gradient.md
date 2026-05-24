---
title: "Sutton & Barto Ch 13. Policy Gradient Methods"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 13
difficulty: hard
budget_days: 5-7
priority_for_rlhf: must
silver_lecture: 7
---

# Ch 13. Policy Gradient Methods ⭐⭐⭐

## 0. 阅读元信息

- **难度**：hard（PG 定理推导是 RL 经典数学难关）
- **预算**：5-7 天 / **手推 PG 定理 ≥ 2 次**
- **必读理由**：**RLHF 的全部数学基础**——PPO/GRPO/DPO 都从 PG 定理派生
- **配套**：Silver UCL Lec 7、lcalem Ch 13 神文
- **跟 RLHF 的桥**：本章 = RLHF 的"理论根"

## 1. 一句话

Policy Gradient = 直接对 π_θ(a|s) 参数化 + 沿 J(θ) = E[G] 的梯度上升优化——**policy gradient theorem** 给出 ∇J 不依赖 state distribution 的梯度对策略参数的导数的优雅形式，让 sample-based 估计可行。

## 2. 关键概念地图

| 概念 | 一句话 |
|---|---|
| **Parametrized policy** | π_θ(a\|s) 由参数 θ 控制（softmax / Gaussian / NN） |
| **Performance objective** | J(θ) = v_{π_θ}(s_0) 或 average reward |
| **Policy gradient theorem** | $∇J(θ) = E_π[\sum_t ∇log π_θ(a_t\|s_t) Q^{π_θ}(s_t, a_t)]$ |
| **REINFORCE** | 用 G_t 替代 Q^π 的 sample-based PG |
| **Baseline b(s)** | $∇J = E[∇log π · (Q - b)]$，不改无偏性，降方差 |
| **Actor-Critic** | Actor = π_θ，Critic = V_φ（学 baseline）|
| **Advantage A(s,a)** | A = Q(s,a) - V(s) —— 自然 baseline |
| **TRPO** | trust region 约束的 PG |
| **PPO** | TRPO 简化版，clip ratio 实现 trust region |

## 3. 必背公式与推导

### Policy Gradient Theorem

$$∇J(θ) = E_{s \sim μ_π, a \sim π_θ}\left[ Q^{π_θ}(s, a) ∇_θ \log π_θ(a|s) \right]$$

### 推导路径（白板必会）

设 J(θ) = v_π(s_0)，从 Bellman 出发：
$$v_π(s) = \sum_a π(a|s) q_π(s, a)$$

对 θ 求导（乘积法则）：
$$∇v_π(s) = \sum_a [∇π(a|s) q_π(s,a) + π(a|s) ∇q_π(s,a)]$$

q_π 展开 = r + γ Σ p · v_π(s')，递归：
$$∇q_π(s, a) = γ \sum_{s'} p(s'|s,a) ∇v_π(s')$$

把递归展开到无穷（注意每步乘 π · p · γ），合并 state-visit distribution μ_π：
$$∇J = \sum_s μ_π(s) \sum_a ∇π(a|s) q_π(s, a)$$

用 score function trick `∇π = π · ∇log π`：
$$∇J = E_{s \sim μ_π, a \sim π}[Q · ∇log π]$$

**关键 trick**：μ_π 不显式出现 → sample-based 估计可行（rollout 样本天然按 μ_π 分布）。

### REINFORCE

$$θ ← θ + α G_t ∇log π(A_t|S_t, θ)$$

### REINFORCE with Baseline

$$θ ← θ + α (G_t - b(S_t)) ∇log π(A_t|S_t, θ)$$

—— b 可以是任意 state-dep function，最优 b ≈ V(s)。

### Actor-Critic（One-step TD）

$$δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)$$
$$θ ← θ + α δ_t ∇log π(A_t|S_t, θ)$$
$$w ← w + β δ_t ∇V(S_t, w)$$

—— A = δ (TD error 当 advantage)，critic 学 V。

## 4. 关键算法（伪代码）

**REINFORCE**：
```
init θ
loop for each episode:
  generate trajectory τ following π_θ
  for t = 0, ..., T-1:
    G_t = Σ_{k=t}^{T-1} γ^{k-t} R_{k+1}
    θ += α γ^t G_t ∇log π_θ(A_t|S_t)
```

**One-step Actor-Critic**：
```
init θ, w
loop for each episode:
  init S; I = 1
  loop:
    A = π_θ(S)
    take A, observe R, S'
    δ = R + γ V_w(S') - V_w(S)   (V_w(terminal) = 0)
    w += β δ ∇V_w(S)
    θ += α I δ ∇log π_θ(A|S)
    I *= γ
    S = S'
  until terminal
```

**PPO（书外补充，但本章基础）**：
```
for iter:
  collect rollout under π_old
  for K epochs:
    compute advantage A_t (GAE)
    ratio_t = π_θ(A_t|S_t) / π_old(A_t|S_t)
    L = min(ratio_t A_t, clip(ratio_t, 1-ε, 1+ε) A_t)
    θ += α ∇L
  π_old = π_θ
```

## 5. 习题精选

**Exercise 13.1（PG 定理推导第一步）**：写出 v_π(s) 关于 θ 的展开。

**Exercise 13.2（baseline 不变 unbiasedness）**：证明任意 b(s) 不破坏 PG 无偏性。**关键步骤**：$E_a[b(s) ∇log π(a|s)] = b(s) \sum_a ∇π(a|s) = b(s) ∇\sum_a π(a|s) = b(s) ∇1 = 0$。

**Exercise 13.3（continuous action）**：把 PG 推广到 Gaussian policy。

**Exercise 13.5（actor-critic on cliff walking）**：实现 one-step actor-critic on cliff，比较 SARSA / Q-learning / AC。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter13/short_corridor.py`（Example 13.1）、 `chapter13/mountain_car.py`、`chapter13/continuous_mountain_car.py`
- **自己实现 REINFORCE on CartPole**：50 行 PyTorch，必练
- 现代实现：CleanRL 仓库（CleanRL/cleanrl, 单文件 PPO/A2C/SAC）
- lcalem Ch 13 神文：[lcalem.github.io/blog/2019/03/21/sutton-chap13](https://lcalem.github.io/blog/2019/03/21/sutton-chap13)

## 7. 反直觉点 / 易错点

- **反直觉**：PG 直接学 policy 不学 V/Q——绕过 deadly triad
- **反直觉**：baseline 不改梯度无偏性——任意 b(s) 都行
- **反直觉**：γ^t 在 episodic 公式里出现——长时间不能直接用 G_t，要 discount
- **反直觉**：optimal baseline 不是 V(s) 严格，是更复杂的 weighted V
- **易错**：忘了 score function trick ∇π = π ∇log π
- **易错**：REINFORCE 高方差——MC return 加噪声，必须 baseline 才实用
- **易错**：on-policy 严格——每次 update 都要新 sample；off-policy PG 需 IS

## 8. 跟 RLHF 的桥（⭐⭐⭐ 直接根）

**RLHF 的 PG**：
- State s = (prompt + 已生成 tokens)
- Action a = next token
- π_θ(a|s) = LLM next-token distribution
- Reward R = 末尾给一次 RM 分（中间 R=0）
- Advantage A_t = GAE（用 critic 学 V）或 group normalize（GRPO）

**PPO 公式跟本章关系**：
- ratio = π_θ/π_old 来自 importance sampling（Ch 5）
- advantage A_t = TD error 累积（Ch 6 + 12）
- clip 是 trust region 的工程化（TRPO 简化）
- KL penalty 是 reference-policy 约束（防止 deadly triad）

**DPO 关系**：
- 把 PPO + KL constraint 的最优 policy 求 closed-form
- 反代回去得到不需 RM 的 loss
- 数学上等价于隐式 PG

**手撕 PG 定理是 AI-Infra/算法岗 RLHF 题第一关**。

## 9. 苏格拉底 Q&A

- **Q：PG 比 value-based 强在哪？**
  - 提示：(1) 直接优化目标；(2) 处理 continuous action；(3) 学 stochastic policy；(4) 避免 max → 绕开 deadly triad
- **Q：PG 比 value-based 弱在哪？**
  - 提示：高方差、sample 效率低、易陷局部最优
- **Q：score function trick 为什么关键？**
  - 提示：让 ∇π 变成 π · ∇log π → expectation 形式 → sample-based 估计
- **Q：baseline 怎么选？**
  - 提示：通常 V(s)——降方差且自然；optimal baseline 更复杂但收益小
- **Q：actor-critic 跟 PG + baseline 的区别？**
  - 提示：AC 用 critic 学到 V 作 baseline（且用 TD bootstrap）；纯 PG + baseline 用 fixed 或 MC V
- **Q：PPO 的 clip 等价 trust region 吗？**
  - 提示：approximate——hard constraint 改 soft；clip ε 控制等价 trust radius
- **Q：GRPO 比 PPO 简化在哪？**
  - 提示：去 critic——用 group reward 内部归一化算 advantage，省一半网络
- **Q：DPO 没 PG 没 V，为什么仍能 alignment？**
  - 提示：PPO+KL 最优解 closed-form → 反向工程把 RM 折叠进 policy loss
- **Q：RLVR 跟传统 PG 区别？**
  - 提示：reward 不靠学到的 RM，靠规则验证（math correct / code pass）；otherwise 同 PG
- **Q：long-CoT reasoning RL 训练特殊性？**
  - 提示：trajectory 长（千 token）+ 稀疏 reward → MC-style advantage 优于 TD；GRPO group baseline 更稳

## 10. 自测清单（闭卷）

- [ ] **手推 PG 定理**（从 v_π 展开到 score function trick）—— 必做
- [ ] 写出 REINFORCE、REINFORCE + baseline、one-step AC 三个算法
- [ ] 证明 baseline 不改 unbiasedness（习题 13.2）
- [ ] 实现 REINFORCE on CartPole 跑通
- [ ] **写出 PPO loss 公式并指出每个组件来自 S&B 哪一章**
- [ ] 答 Q1-Q10 至少 6 个

## Related

- **上一章**：[[ch12-eligibility-traces]]（GAE 前置）
- **下一章**：[[ch14-psychology]]
- **IS 前置**：[[ch05-monte-carlo-methods]] §5.5
- **RLHF 全链**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning]] [[ch08-direct-alignment-algorithms]]
- **进度**：[[_chapter-status]] W7 ⭐⭐⭐

## Sources

- 原书 Ch 13 pp. 321-343
- [Silver UCL Lec 7: Policy Gradient](https://www.youtube.com/watch?v=KHZVXao4qXs)
- [lcalem Ch 13 神文](https://lcalem.github.io/blog/2019/03/21/sutton-chap13)
- PG Theorem 原始论文（Sutton et al. NeurIPS 1999）
- TRPO 论文（Schulman 2015）
- PPO 论文（Schulman 2017）
- GAE 论文（Schulman 2015）
- ShangtongZhang `chapter13/`
- CleanRL 单文件 PPO 实现
