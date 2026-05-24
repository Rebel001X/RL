---
title: "Sutton & Barto Ch 4. Dynamic Programming"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 4
difficulty: medium
budget_days: 2-3
priority_for_rlhf: nice
silver_lecture: 3
---

# Ch 4. Dynamic Programming

## 0. 阅读元信息

- **难度**：medium（数学不难但概念关键）
- **预算**：2-3 天 / ~5 小时
- **必读理由**：DP 给的是"知 model 时怎么算最优 policy"的标准答案，后续 model-free（MC/TD）都是它的近似；理解 DP 才能理解 model-free 在妥协什么
- **配套**：Silver UCL Lec 3（Planning by DP）
- **跟 RLHF 的桥**：DP 是"完美 model"假设——RLHF 没有 model，但 RM 扮演类似"价值评估"角色

## 1. 一句话

Dynamic Programming = 已知完整 MDP model 时，**通过迭代求解 Bellman 方程**计算 v\* 和 π\*；核心两个算法是 policy iteration（policy eval + policy improvement 交替）和 value iteration（直接 Bellman optimality 迭代）。

## 2. 关键概念地图

| 概念 | 一句话 |
|---|---|
| **Policy evaluation** | 给定 π，迭代求 v_π —— 反复用 Bellman expectation 直到收敛 |
| **Policy improvement** | 给定 v_π，构造 π' = greedy(v_π) —— **policy improvement theorem** 保证 π' ≥ π |
| **Policy iteration** | 交替 eval + improvement，必收敛到最优 |
| **Value iteration** | 把 Bellman optimality 当迭代规则直接跑 —— 跳过显式 policy 表示 |
| **Generalized Policy Iteration (GPI)** | eval 和 improvement 任意交错（不必各自收敛）—— RL 大家族的统一视角 |
| **In-place update** | 用最新 V 值就地更新（比 two-array 收敛更快） |
| **Asynchronous DP** | 不同 state 不同 frequency 更新——为 model-free 铺路 |
| **Bootstrap** | 用 V(s') 估 V(s)——DP 的核心，**也是 deadly triad 的根** |

## 3. 必背公式与推导

### Policy Evaluation 迭代

$$v_{k+1}(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]$$

**收敛性**：γ < 1 时 Bellman expectation operator 是 γ-contraction，必收敛到唯一不动点 v_π（Banach 不动点定理）。

### Policy Improvement Theorem

如果对所有 s 有 q_π(s, π'(s)) ≥ v_π(s)，则 v_{π'}(s) ≥ v_π(s)。

**推导直觉**：用 π' 一步后切回 π，至少不更差 → 一直用 π' 也至少不更差（unrolling）。

### Value Iteration

$$v_{k+1}(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]$$

**关键**：把 Bellman optimality 当 iteration rule。等价于 "policy iteration 的 evaluation 只跑 1 步就 improvement"。

## 4. 关键算法（伪代码）

**Policy Iteration**：
```
init V(s) = 0, π(s) = arbitrary
loop:
  # Policy Evaluation
  repeat:
    Δ = 0
    for each s:
      v = V(s)
      V(s) = Σ_{s', r} p(s', r | s, π(s)) [r + γ V(s')]
      Δ = max(Δ, |v - V(s)|)
  until Δ < θ

  # Policy Improvement
  stable = True
  for each s:
    old = π(s)
    π(s) = argmax_a Σ_{s', r} p(s', r | s, a) [r + γ V(s')]
    if old ≠ π(s): stable = False
  if stable: return V, π
```

**Value Iteration**：
```
init V(s) = 0
repeat:
  Δ = 0
  for each s:
    v = V(s)
    V(s) = max_a Σ_{s', r} p(s', r | s, a) [r + γ V(s')]
    Δ = max(Δ, |v - V(s)|)
until Δ < θ

# Extract policy
for each s:
  π(s) = argmax_a Σ_{s', r} p(s', r | s, a) [r + γ V(s')]
```

## 5. 习题精选

**Exercise 4.4**：policy iteration 有 bug 可能死循环（policy 在两个等价 optimal 间来回切）。修复：检查是否 q-value 真的更大，而不仅是 action 改变。

**Exercise 4.7（Jack's Car Rental）**：在 Example 4.2 基础上加额外规则（每天 2 辆车免费转移），重写 transition + reward 重跑 policy iteration。考验你**把现实约束建模成 MDP 的能力**。

**Exercise 4.10**：写出 q_*(s, a) 的 value iteration 更新公式。**答案**：
$$q_{k+1}(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} q_k(s', a')]$$

## 6. 代码伴侣

- **ShangtongZhang**: `chapter04/grid_world.py`（Example 4.1 复现）、`chapter04/car_rental.py`（Example 4.2）、`chapter04/gamblers_problem.py`（Example 4.3）
- 自己写：5x5 gridworld 的 value iteration，10 行 numpy
- 跑通 `gamblers_problem.py` 后体会 figure 4.3 那个"锯齿状"最优 policy——反直觉

## 7. 反直觉点 / 易错点

- **反直觉**：policy iteration 通常只需 ~3-5 轮就收敛，policy eval 内层迭代才是主开销
- **反直觉**：value iteration 把 eval 退化到 1 步，居然也收敛——因为每步都隐式做了 improvement
- **反直觉**：DP 对 state space 是 O(|S|²|A|) 复杂度——"curse of dimensionality"使其难以 scale 到大问题（Backgammon 10²⁰ states 就崩）
- **易错**：in-place vs two-array 更新在收敛速度上有微妙差异（in-place 更快）
- **易错**：asynchronous DP 不保证按"state 顺序"更新——任意顺序只要每个 state 被无限次更新就收敛

## 8. 跟 RLHF 的桥

- **RLHF 没有 model**（不知道 LLM token → next token 真实分布的所有概率）→ 必须 model-free
- 但 RM 的"评估"过程类似 policy evaluation——固定 policy 估计其价值
- **PPO 的 critic** = 学的 V 函数，本质是 policy iteration 的 evaluation step（用 bootstrap 估 V_π）
- **GRPO 跳过 critic**——直接用 group 内 reward 归一化估 advantage，绕开了 DP-style value learning

## 9. 苏格拉底 Q&A

- **Q：DP 既然能算最优，为什么实际很少用？**
  - 提示：需要完整 model（转移概率全已知）+ state space 小（O(|S|²|A|) tabular）。真实问题这两个都不满足
- **Q：policy iteration vs value iteration 哪个快？**
  - 提示：policy iteration 总轮数少但每轮贵（内层 eval），value iteration 总轮数多但每轮便宜——总时间相近，task-dependent
- **Q：bootstrap 是 DP 的"灵魂"——为什么这是个问题？**
  - 提示：用 V(s') 估 V(s) 时 V(s') 本身可能不准——错误传播；deadly triad 之一
- **Q：GPI 把 eval 和 improvement"任意交错"，为什么仍收敛？**
  - 提示：只要 eval 和 improvement 都在"对的方向"上推动，最终趋于 v\*——不需要严格 eval 完才 improve

## 10. 自测清单（闭卷）

- [ ] 不看书讲清 policy iteration 两步是什么
- [ ] 写出 value iteration 公式
- [ ] 解释 policy improvement theorem 为什么 π' 不会更差
- [ ] 跑通 ShangtongZhang `chapter04/car_rental.py`
- [ ] 答 Q1-Q4 至少 3 个

## Related

- **上一章**：[[ch03-finite-mdps]]
- **下一章**：[[ch05-monte-carlo-methods]]
- **进度**：[[_chapter-status]] W2

## Sources

- 原书 Ch 4 pp. 73-92
- [Silver UCL Lec 3: Planning by DP](https://www.youtube.com/watch?v=Nd1-UUMVfz4)
- ShangtongZhang `chapter04/`
