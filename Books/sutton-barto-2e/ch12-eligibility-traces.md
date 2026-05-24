---
title: "Sutton & Barto Ch 12. Eligibility Traces"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 12
difficulty: medium
budget_days: 3
priority_for_rlhf: must
silver_lecture: 4-5
---

# Ch 12. Eligibility Traces

## 0. 阅读元信息

- **难度**：medium（概念优雅但数学密集）
- **预算**：3 天 / §12.1-12.3 必读，余可 skim
- **必读理由**：**λ-return 是 GAE 的直接前置**——理解 TD(λ) 才理解 PPO 用的 GAE
- **配套**：Silver UCL Lec 4-5
- **跟 RLHF 的桥**：GAE = λ-加权 TD errors = 本章 TD(λ) 的 advantage 版

## 1. 一句话

Eligibility traces = 把"过去 N 步"的 credit assignment 用 λ ∈ [0,1] 平滑混合——TD(0) (λ=0) 到 MC (λ=1) 的连续谱；同时给出**单 pass 高效算法**（不需 n-step buffer）。

## 2. 关键概念地图

| 概念 | 一句话 |
|---|---|
| **λ-return** | $G_t^λ = (1-λ) \sum_{n=1}^∞ λ^{n-1} G_{t:t+n}$ —— n-step return 的指数加权平均 |
| **TD(λ)** | 用 λ-return 当 target |
| **Forward view** | 看未来 n 步，用 λ-return 计算 update |
| **Backward view** | 维护 eligibility trace z_t，当前 δ 反向分配信用 |
| **Eligibility trace z_t** | z_t = γλ z_{t-1} + ∇v̂(S_t, w) —— "最近活跃的 feature 多少 credit" |
| **TD(0)** | λ=0 → z_t = ∇v̂(S_t)（无 trace） |
| **TD(1)** | λ=1 → 全 trace，等价 MC |
| **True online TD(λ)** | 解决 backward 的 bias 问题，等价 forward |
| **SARSA(λ)** | TD(λ) 的 Q-version |
| **Watkins's Q(λ)** | off-policy Q 的 λ 版本，遇 non-greedy action 截断 trace |

## 3. 必背公式与推导

### λ-return

$$G_t^λ = (1-λ) \sum_{n=1}^{T-t-1} λ^{n-1} G_{t:t+n} + λ^{T-t-1} G_t$$

- λ=0 → 全 weight 在 n=1（TD(0)）
- λ=1 → 全 weight 在 G_t（MC）
- 中间 λ → 平滑混合

### TD(λ) Backward View

$$z_t = γλ z_{t-1} + ∇v̂(S_t, w)$$
$$w_{t+1} = w_t + α δ_t z_t$$

其中 $δ_t = R_{t+1} + γv̂(S_{t+1}) - v̂(S_t)$。

**直觉**：trace z_t 记录"哪些 feature 最近活跃"，TD error δ_t 反向分配给这些 feature。

### GAE 公式（Schulman 2015，本章直接应用）

$$A_t^{GAE(γ, λ)} = \sum_{l=0}^∞ (γλ)^l δ_{t+l}$$

—— 完全是 TD(λ) backward view 的 advantage 形式。**PPO 默认 λ=0.95**。

## 4. 关键算法（伪代码）

**Linear TD(λ) Backward**：
```
init w = 0
loop for each episode:
  init S; z = 0
  loop:
    A = π(S)
    take A, observe R, S'
    z = γλ z + x(S)
    δ = R + γ w^T x(S') - w^T x(S)
    w += α δ z
    S = S'
  until S terminal
```

**SARSA(λ)**：类似但 z 累积 ∇q̂(S, A, w)，δ 用 Q 版本。

## 5. 习题精选

**Exercise 12.1（λ-return 性质）**：证明 G_t^λ 是 n-step return 的加权平均。

**Exercise 12.5（TD(λ) → forward 等价）**：证明 forward 和 backward TD(λ) 在 episode 结束后产生相同总 update（offline 等价）。

**Exercise 12.10（true online TD(λ)）**：实现 true online 版本，比较跟普通 TD(λ) 在 random walk 上学习曲线。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter12/random_walk.py`（TD(λ) vs n-step），`chapter12/mountain_car.py`（SARSA(λ)）
- 自己实现：linear TD(λ) 5 行更新逻辑
- **看 GAE 论文**（Schulman 2015）确认本章公式如何变形为 advantage

## 7. 反直觉点 / 易错点

- **反直觉**：backward view 用 trace 实现 forward view 的等价 update——优雅
- **反直觉**：true online TD(λ) 解决了 backward 的"in-trajectory bias"——细节但重要
- **反直觉**：λ 的最优值通常 0.5-0.95——边界 0 和 1 都次优
- **易错**：trace z_t 是 vector（跟 w 同 shape），不是 scalar
- **易错**：episode 切换时 z 要 reset 为 0

## 8. 跟 RLHF 的桥（⭐ 直接相关）

**GAE 是本章的"advantage 版"**：
$$A_t^{GAE(γ, λ)} = \sum_{l=0}^∞ (γλ)^l δ_{t+l}$$

跟 TD(λ) forward 完全同形——只是把 V 换成 advantage A，δ 仍是 TD error。

**PPO 用 GAE 是因为**：
- λ=0 → 高 bias 低方差（纯 TD）
- λ=1 → 低 bias 高方差（纯 MC）
- λ=0.95 → 工业甜区

**实现细节**：PPO 算 GAE 是 backward sweep（从 trajectory 末尾倒推），等价 backward view。**会写 PPO 必须懂本章**。

## 9. 苏格拉底 Q&A

- **Q：TD(λ) 比 n-step 优在哪？**
  - 提示：λ 平滑混合所有 n，避免选单 n 的限制；backward view 单 pass O(1) update
- **Q：forward vs backward view 是数学等价还是 trick 等价？**
  - 提示：episode-end 等价；in-trajectory 有微小差（true online 修正）
- **Q：λ=0 和 λ=1 各等价什么？**
  - 提示：λ=0 → TD(0)，λ=1 → MC（offline 等价）
- **Q：GAE 跟 TD(λ) 公式完全一样吗？**
  - 提示：结构同，只是把 V 换 A——GAE 是把 TD(λ) 思想用到 advantage estimation 上
- **Q：PPO 为什么默认 λ=0.95 而不 0.5？**
  - 提示：高 λ → 接近 MC → 用 trajectory 真实信号；适合 reward 稀疏长 trajectory 场景
- **Q：reasoning model 长 CoT，λ 应该多大？**
  - 提示：经验上接近 1（甚至 1.0）—— reward 末尾给一次，bootstrap 没意义

## 10. 自测清单（闭卷）

- [ ] 写出 λ-return 公式
- [ ] 写出 TD(λ) backward view 算法（含 z 更新）
- [ ] 解释 forward vs backward view 等价性
- [ ] **写出 GAE 公式并指出跟 TD(λ) 的关系**（RLHF 桥必答）
- [ ] 解释 PPO 用 λ=0.95 的工程理由
- [ ] 答 Q1-Q6 至少 4 个

## Related

- **上一章**：[[ch11-off-policy-methods]]
- **下一章**：[[ch13-policy-gradient]]
- **n-step 前置**：[[ch07-n-step-bootstrapping]]
- **RLHF 桥**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning]]
- **进度**：[[_chapter-status]] W6

## Sources

- 原书 Ch 12 pp. 287-321
- GAE 论文（Schulman 2015, arXiv 1506.02438）—— 本章直接应用
- [Silver UCL Lec 4-5](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- ShangtongZhang `chapter12/`

## 附：True Online TD(λ) 详解

普通 backward TD(λ) 跟 forward TD(λ) 在 episode-end 之外有微小偏差（in-trajectory bias）。**True online TD(λ)** 用更复杂的 trace update 完美等价 forward view：

$$w_{t+1} = w_t + α δ_t z_t + α (z_t - α(z_t^T x_t) x_t)(w_t^T x_t - w_{t-1}^T x_t)$$

—— 多了一个修正项。实践上轻量级实现选普通 TD(λ) 就够（误差小）。Sutton 在第 12 章 §12.5 给出完整证明 + 算法。
