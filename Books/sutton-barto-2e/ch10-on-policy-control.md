---
title: "Sutton & Barto Ch 10. On-policy Control with Approximation"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 10
difficulty: medium
budget_days: 2
priority_for_rlhf: nice
silver_lecture: 6
---

# Ch 10. On-policy Control with Approximation

## 0. 阅读元信息

- **难度**：medium（Ch 9 的 control 版扩展）
- **预算**：2 天 / ~4 小时
- **必读理由**：semi-gradient SARSA 是 NN 时代之前最实用的 control 算法；continuing task 的 average reward 形式化
- **配套**：Silver UCL Lec 6
- **跟 RLHF 的桥**：on-policy 思想跟 PPO 一致；但 PPO 是 policy-based，本章仍是 value-based

## 1. 一句话

把 Ch 9 的 prediction（v̂(s, w)）升级为 control（q̂(s, a, w)）——semi-gradient SARSA + ε-greedy = NN 之前最常用的 RL 控制框架；同时引入 **average reward formulation** 处理无终态 continuing task。

## 2. 关键概念地图

| 概念 | 一句话 |
|---|---|
| **Semi-gradient SARSA** | q̂(S, A, w) 替代 tabular Q，update 用 R + γq̂(S', A', w) target |
| **Semi-gradient Expected SARSA** | target 用 Σ_a π(a\|S') q̂(S', a, w) |
| **n-step semi-gradient SARSA** | 推广到 n-step return |
| **Average reward** | continuing task 中 r̄(π) = lim E[R]，不用折扣 |
| **Differential return** | G_t = R_{t+1} - r̄ + R_{t+2} - r̄ + ... |
| **Differential value** | $v_π(s) = E_π[Σ (R_{t+k+1} - r̄)]$ —— 相对平均的剩余 reward |
| **Discounted setting limitation** | §10.4 强调 discounted 在 function approximation + continuing 下"几乎无意义" |

## 3. 必背公式与推导

### Semi-gradient SARSA

$$w_{t+1} = w_t + α [R_{t+1} + γ q̂(S_{t+1}, A_{t+1}, w_t) - q̂(S_t, A_t, w_t)] ∇q̂(S_t, A_t, w_t)$$

### Average reward MDP

$$r̄(π) = \lim_{h→∞} \frac{1}{h} \sum_{t=1}^h E[R_t | A_{0:t-1} \sim π]$$

### Differential semi-gradient SARSA

$$δ_t = R_{t+1} - \bar{R}_t + q̂(S_{t+1}, A_{t+1}, w) - q̂(S_t, A_t, w)$$
$$w_{t+1} = w_t + α δ_t ∇q̂(S_t, A_t, w_t)$$
$$\bar{R}_{t+1} = \bar{R}_t + β δ_t$$

—— 用 differential value 替代 discounted value，**无 γ 出现**。

## 4. 关键算法（伪代码）

**Semi-gradient SARSA (episodic)**：
```
init w arbitrary
loop for each episode:
  S = init; A = ε-greedy(q̂(S, ·, w))
  loop:
    take A, observe R, S'
    if S' terminal:
      w += α [R - q̂(S, A, w)] ∇q̂(S, A, w)
      break
    A' = ε-greedy(q̂(S', ·, w))
    w += α [R + γ q̂(S', A', w) - q̂(S, A, w)] ∇q̂(S, A, w)
    S, A = S', A'
```

## 5. 习题精选

**Exercise 10.1（Mountain Car）**：用 tile coding + semi-gradient SARSA 解 Mountain Car；扫 α 找最优。

**Exercise 10.7（differential SARSA）**：实现 average reward 版本在 access-control queuing task 上。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter10/mountain_car.py`（Example 10.1）、`chapter10/access_control.py`（Example 10.2，average reward）
- Mountain Car 是经典 RL 试金石——能跑通 SARSA + tile coding 算 RL 入门毕业
- 现代版：用 NN 替代 tile coding 跑同任务

## 7. 反直觉点 / 易错点

- **反直觉**：Mountain Car 用 SARSA + tile coding 跑通比 DQN 早 20 年——简单方法在简单任务上够用
- **反直觉**：discounted 在 continuing + function approximation 下"几乎无意义"——§10.4 论证
- **反直觉**：average reward 不存在 fixed γ，但仍能学
- **易错**：忘记 q̂(terminal, ·, w) = 0
- **易错**：differential SARSA 的 r̄ 也是学的——多一个 step size β

## 8. 跟 RLHF 的桥

- 弱直接关联——RLHF 用 policy gradient 不用 SARSA
- **概念上**：on-policy + function approximation = PPO 的"族"——PPO clip 是这个族里防止"step too far"的 trick
- **average reward** 视角对 LLM agent 有启发：长 trajectory chat 没有"终态"，differential value 概念可借鉴
- **continuing chat 场景的 reward 累计**：multi-turn agent 长跑 trajectory 没自然终点，类比 continuing MDP；average reward 提供数学框架（虽然 RLHF 工程实践仍切 episode）
- **Differential value 在 RLHF 评测中的影子**：reward 的"平均水平"作为 baseline——跟 PPO/GRPO 用的 advantage normalization（减均值除标准差）思想相通

## 9.5 跟"现代 RL 实践"的 gap

本章是 NN 之前的"经典"on-policy control，跟现代 deep RL 的 gap：
- **现代**：DQN + replay buffer 是 off-policy + value-based 的工业标准（违反本章的 on-policy 前提）
- **现代**：PPO / SAC 是 policy-based + actor-critic（绕过纯 value-based SARSA）
- **现代**：tile coding 几乎全被 NN feature 替代
- **本章价值**：理解概念框架，不是直接落地——Mountain Car SARSA + tile coding 已是博物馆级

## 9. 苏格拉底 Q&A

- **Q：semi-gradient SARSA vs Q-learning + FA 哪个更稳？**
  - 提示：SARSA on-policy 更稳；Q-learning + FA 可能进入 deadly triad（Ch 11）
- **Q：为什么 §10.4 反对 discounted continuing 任务？**
  - 提示：γ < 1 时 average reward 是 ε-依赖的常数；γ 选择跟 task scale 错配
- **Q：Mountain Car 跑 SARSA 时 γ 应该多大？**
  - 提示：~0.99（任务长，需远视）；γ=0.9 收敛慢

## 10. 自测清单（闭卷）

- [ ] 写出 semi-gradient SARSA 更新公式
- [ ] 写出 differential semi-gradient SARSA 公式（含 r̄ 更新）
- [ ] 解释 average reward 跟 discounted 的根本差异
- [ ] 跑通 Mountain Car SARSA
- [ ] 解释为什么本章方法"博物馆化"了——现代 deep RL 用什么替代了
- [ ] 答 Q1-Q3 至少 2 个

## 跑通指标

完成 Mountain Car 实战的"过关标准"：
- 在 200-step 限制下能 reach goal（reward > -200）
- 学习曲线在 ~500 episodes 内收敛
- 看 tile coding feature 可视化（书 figure 10.1）

## Related

- **上一章**：[[ch09-on-policy-prediction]]
- **下一章**：[[ch11-off-policy-methods]]
- **进度**：[[_chapter-status]] W4

## Sources

- 原书 Ch 10 pp. 243-262
- [Silver UCL Lec 6: Value Function Approximation](https://www.youtube.com/watch?v=UoPei5o4fps)
- ShangtongZhang `chapter10/`
