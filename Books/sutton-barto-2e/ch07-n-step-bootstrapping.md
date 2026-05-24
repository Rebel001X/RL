---
title: "Sutton & Barto Ch 7. n-step Bootstrapping"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 7
difficulty: medium
budget_days: 2
priority_for_rlhf: nice
silver_lecture: 5
---

# Ch 7. n-step Bootstrapping

## 0. 阅读元信息

- **难度**：medium（概念是 TD 和 MC 的桥）
- **预算**：2 天 / ~4 小时
- **必读理由**：n-step 是 TD 和 MC 的统一视角，**为 Ch 12 eligibility traces / Ch 13 GAE 铺路**
- **配套**：Silver UCL Lec 5
- **跟 RLHF 的桥**：GAE = λ-加权的 n-step returns，本章是 GAE 直接前置

## 1. 一句话

n-step bootstrapping = MC 和 TD 的连续谱：n=1 是 TD(0)，n=∞ 是 MC，中间任意 n 是折中——通常某个中间 n 在 bias-variance 上最优。

## 2. 关键概念地图

| 概念 | 一句话 |
|---|---|
| **n-step return** | $G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})$ |
| **n-step TD** | V(S_t) += α [G_{t:t+n} - V(S_t)] |
| **n-step SARSA** | Q(S_t, A_t) += α [G_{t:t+n}^Q - Q(S_t, A_t)] |
| **n-step Expected SARSA** | Σ_a π(a\|S_{t+n}) Q(S_{t+n}, a) 当尾估计 |
| **n-step off-policy** | 用 importance sampling 修正 |
| **n-step Tree Backup** | off-policy 无 IS 的精巧 backup |
| **n-step Q(σ)** | 统一框架，σ ∈ [0,1] 控制 sample（IS）vs expectation（Tree Backup） |

## 3. 必背公式与推导

### n-step Return

$$G_{t:t+n} = R_{t+1} + γR_{t+2} + \cdots + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})$$

- n=1 → R_{t+1} + γV(S_{t+1}) = TD(0) target
- n=∞ → MC return G_t
- 中间 n → 折中

### n-step TD Update

$$V(S_t) \leftarrow V(S_t) + α [G_{t:t+n} - V(S_t)]$$

**延迟**：要等到 t+n 步才能更新 V(S_t)。

### n-step off-policy IS

$$ρ_{t:t+n-1} = \prod_{k=t}^{t+n-1} \frac{π(A_k|S_k)}{b(A_k|S_k)}$$

$$V(S_t) \leftarrow V(S_t) + α ρ_{t:t+n-1} [G_{t:t+n} - V(S_t)]$$

### n-step Tree Backup（off-policy 不用 IS）

不用 importance sampling、但仍能 off-policy 学。核心思想：把没采到的 action 用其期望补上：

$$G^{TB}_{t:t+n} = R_{t+1} + γ \sum_{a \neq A_{t+1}} π(a|S_{t+1}) Q(S_{t+1}, a) + γ π(A_{t+1}|S_{t+1}) G^{TB}_{t+1:t+n}$$

—— 比纯 IS 方差小，但有 bias；实践上 n-step Q(σ) 给出统一权衡。

### n-step Q(σ) 统一框架

σ ∈ [0, 1] 控制每一步是 sample（σ=1, 走 IS 路）还是 expectation（σ=0, 走 tree backup 路）。整本书最一般的 backup 形式。

### bias-variance 在 n 维度的 U 形

定性：
- 小 n（接近 1）：bias 大（bootstrap V 不准），但 variance 小（少步随机）
- 大 n（接近 ∞ = MC）：bias 0，但 variance 大（全程随机累积）
- 中间某 n\* 最优 —— ShangtongZhang figure 7.2 在 random walk 上可见明显 U 形

## 4. 关键算法（伪代码）

**n-step SARSA**：
```
init Q; π = ε-greedy(Q)
for each episode:
  S_0 = init; A_0 = π(S_0)
  T = ∞
  for t = 0, 1, 2, ...:
    if t < T:
      take A_t, observe R_{t+1}, S_{t+1}
      if S_{t+1} terminal: T = t+1
      else: A_{t+1} = π(S_{t+1})
    τ = t - n + 1     # 当前要更新的时间
    if τ ≥ 0:
      G = Σ_{i=τ+1}^{min(τ+n, T)} γ^{i-τ-1} R_i
      if τ + n < T: G += γ^n Q(S_{τ+n}, A_{τ+n})
      Q(S_τ, A_τ) += α (G - Q(S_τ, A_τ))
    if τ == T-1: break
```

## 5. 习题精选

**Exercise 7.2（n-step return 性质）**：证明 G_{t:t+n} 的期望随 n 变化的趋势。

**Exercise 7.3（random walk n-step 比较）**：在 19-state random walk 上扫 n 和 α，复现 figure 7.2 看哪个 n 最优。

**Exercise 7.10（off-policy n-step with control variate）**：用 control variate 降方差版本。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter07/random_walk.py` —— 复现 figure 7.2 的 n vs α 网格
- 体会"**中间 n 最优**"——bias-variance 在 n 维度上的 U 形曲线

## 7. 反直觉点 / 易错点

- **反直觉**：n 不是越大越好——大 n 方差爆（接近 MC），小 n bias 大（TD(0)）
- **反直觉**：最优 n 跟 step size α 强相关——小 α 时大 n 更好（更稳）
- **易错**：n-step 算法要"延迟"更新——要 buffer 中间 n 步数据
- **易错**：episode 结束时 t+n 可能超 T，公式要 truncate

## 8. 跟 RLHF 的桥

- **GAE (Generalized Advantage Estimation)** = λ-加权的 n-step advantage：$A^{GAE}_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}$
- 当 λ=0 → n=1 TD(0) advantage；当 λ=1 → MC advantage（无 bootstrap）
- **PPO 用 GAE λ=0.95** —— 在 bias-variance 上的实用甜区
- 本章是 GAE 的概念前置，必读
- **RLHF 场景特殊性**：reward 极稀疏（只末尾给一次 RM 分），中间 R=0；这种情况下大 n（接近 MC）更合理——因为 bootstrap 的 V(S_{t+n}) 也只是把 reward 往前传，没有额外信号；这就是为什么 GRPO 直接走 MC-style advantage 也能 work
- **trajectory-level reward 设计**：现代 RLHF（特别 reasoning）有 process reward（中间步打分）—— 让 TD/n-step 重新有意义

## 9. 苏格拉底 Q&A

- **Q：n=1 跟 n=∞ 各有什么 trade-off？**
  - 提示：n=1 = TD(0) bias 大方差小；n=∞ = MC 无 bias 方差大；中间最优
- **Q：n-step 比 TD(0) 强在哪？**
  - 提示：信用分配更长——奖励能传到 n 步前的 action
- **Q：n-step 比 eligibility traces 弱在哪？**
  - 提示：固定 n 不灵活；eligibility traces 用 λ 自适应混合所有 n
- **Q：n-step off-policy 用 IS 累乘——方差会怎样？**
  - 提示：随 n 爆炸——所以实践 n 小（或用 weighted IS / clip）

## 10. 自测清单（闭卷）

- [ ] 写出 n-step return 公式
- [ ] 解释 n 在 TD-MC 谱上的位置
- [ ] 解释 GAE 跟 n-step 的关系（RLHF 桥）
- [ ] 跑通 ShangtongZhang `chapter07/random_walk.py`
- [ ] 答 Q1-Q4 至少 2 个

## Related

- **上一章**：[[ch06-temporal-difference-learning]]
- **下一章**：[[ch08-planning-and-learning]]
- **延伸**：[[ch12-eligibility-traces]]（λ-加权）
- **RLHF 桥**：GAE 论文（Schulman 2015）
- **进度**：[[_chapter-status]] W3

## Sources

- 原书 Ch 7 pp. 141-156
- [Silver UCL Lec 5](https://www.youtube.com/watch?v=0g4j2k_Ggc4)
- ShangtongZhang `chapter07/`
- GAE 论文（Schulman 2015）—— 应用 n-step 思想
