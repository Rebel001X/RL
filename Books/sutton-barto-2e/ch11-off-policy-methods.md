---
title: "Sutton & Barto Ch 11. Off-policy Methods with Approximation"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 11
difficulty: hardest
budget_days: 4-6
priority_for_rlhf: nice
silver_lecture: 6
---

# Ch 11. Off-policy Methods with Approximation ⚠️

## 0. 阅读元信息

- **难度**：**hardest**（全书最难——deadly triad 是 RL 著名理论难点）
- **预算**：4-6 天 / ~10-15 小时；§11.3 重读 3 遍
- **必读理由**：理解为什么 RL 训练这么难稳定；DQN 的所有 trick（target net / replay buffer / Double DQN）都是绕过 deadly triad
- **配套**：Silver UCL Lec 6 后半 + planetbanatt § 11 段落
- **跟 RLHF 的桥**：弱直接，但 RLHF 工程的"为什么这么多 trick"答案在本章

## 1. 一句话

off-policy + function approximation + bootstrap 三者同时出现 = **deadly triad**——理论上可能发散，没有通用解决方案；本章解释为什么 + 给出 gradient TD 类方法尝试解决（但实践上 DQN 的工程 trick 更主流）。

## 2. 关键概念地图

| 概念 | 一句话 |
|---|---|
| **Deadly Triad** | off-policy + function approximation + bootstrap → 可能发散 |
| **Baird's counterexample** | §11.2 经典反例——TD weights 真的发散到无穷 |
| **Projection operator Π** | 把 Bellman backup 结果投影回 v̂ 可表达空间 |
| **Bellman residual** | 不投影，直接最小化 (BV - V)² |
| **Mean Square Bellman Error (BE)** | $\overline{\text{BE}}(w) = \|BV - V\|_μ^2$ |
| **Mean Square Projected Bellman Error (PBE)** | $\overline{\text{PBE}}(w) = \|ΠBV - V\|_μ^2$ |
| **Gradient TD (GTD2, TDC)** | 真 SGD on PBE——保证收敛但慢 |
| **Emphatic-TD** | 用 followon trace 修正分布——理论收敛 |

## 3. 必背公式与推导

### Deadly Triad 三个要素

1. **Off-policy**：训练分布 ≠ target policy 分布
2. **Function approximation**：v̂ 不是 tabular（不能精确表达 v_π）
3. **Bootstrap**：用 v̂(s', w) 估 target（vs MC 用真 G）

**为什么三者同时致命**：每个单独都 ok，三个组合时 update 不再是 contraction → 可能发散。

### Baird's counterexample 直觉

构造 7-state MDP + linear v̂，off-policy semi-gradient TD 的 w 指数发散。**关键**：on-policy 时收敛，仅改 behavior 分布就崩。

### MSPBE 最小化（GTD2）

$$\overline{\text{PBE}}(w) = (BV - V)^T D M (BV - V)$$

其中 M = (Φ^T D Φ)^{-1}（投影到 feature 空间）。**真 SGD on PBE 保证收敛**到 TD fixed point。

### Emphatic TD

引入 emphatic weight：$F_t = γ ρ_{t-1} F_{t-1} + i(S_t)$，用 $F_t · δ_t$ 当 update 强度——修正 off-policy 分布偏差。

## 4. 关键算法（伪代码）

**Off-policy semi-gradient TD（**会发散**，仅作对照）**：
```
w += α ρ_t [R + γ w^T x(S') - w^T x(S)] x(S)
```

**Emphatic TD(0)**：
```
init w, F = 0
loop:
  F = γ ρ_{prev} F + i(S)
  δ = R + γ w^T x(S') - w^T x(S)
  w += α F δ x(S)
  ρ_{prev} = π(A|S) / b(A|S)
```

**DQN 的工程 trick（实际生产用）**：
- Experience replay buffer（打破 sample 相关性）
- Target network（freeze target Q 网络 N 步）
- Reward clipping
- Double DQN（解 over-estimation）

—— 都是绕过 deadly triad 的"工程妥协"。

## 5. 习题精选

**Exercise 11.3（Baird 复现）**：手算 Baird counterexample 第一步 update，验证 w 真的偏离最优。

**Exercise 11.4（Bellman error 与 VE 关系）**：证明 BE ≥ VE，等号何时成立。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter11/baird_counterexample.py`（看 w 发散）、`chapter11/random_walk.py`（GTD vs TD 对比）
- **必跑** baird_counterexample.py——亲眼看 w 飞起来

## 7. 反直觉点 / 易错点

- **反直觉**：deadly triad 不是"小概率事件"——大量实际 RL 系统都隐式碰到，工程 trick 都是 workaround
- **反直觉**：DQN 能 work 不是因为解决了 deadly triad，而是 **trick 让它"足够稳"**——理论仍无保证
- **反直觉**：Bellman residual 看似自然（最小化 BV - V），但实际**不好**——梯度需要 double sampling
- **易错**：MSPBE vs MSBE 是两个不同目标，最优点不同
- **易错**：Emphatic TD 理论收敛但实践少用——慢且复杂

## 8. 跟 RLHF 的桥

- 弱直接，但**揭示 RLHF 工程为什么这么多 trick**：
  - **PPO clip** ≈ 防止 step too far → 类似"软性 contraction"
  - **KL penalty** ≈ 约束在 ref model 附近 → 避免发散
  - **target net** 思想在 DQN（DPO 的 ref model 类似）
  - **experience replay** 思想在 offline RLHF
- DPO 没有 bootstrap（直接对偏好 likelihood 优化）→ **绕开 deadly triad**

## 9. 苏格拉底 Q&A

- **Q：为什么 on-policy 不发散？**
  - 提示：on-policy 时 update 分布跟 TD operator 的"自然分布"一致 → 仍是 contraction
- **Q：Bootstrap 的"问题"具体是什么？**
  - 提示：target 依赖当前 w → update 改变 target → 自我反馈循环可能正反馈
- **Q：DQN 的 target net 解决了什么？**
  - 提示：把 target Q 冻结 N 步 → target 不动 → bootstrap "暂时变" supervised → 稳
- **Q：MC 为什么没 deadly triad？**
  - 提示：MC 不 bootstrap（用真 G），triad 的第三条不满足 → 安全
- **Q：DPO 是怎么"绕开"deadly triad 的？**
  - 提示：不学 V/Q（无 bootstrap），不 off-policy（数据是 preference 对，不是 trajectory）
- **Q：实践中如何监控 deadly triad 发生？**
  - 提示：监控 Q value 是否爆炸增长；梯度 norm；replay buffer 的 TD error 分布

## 10. 自测清单（闭卷）

- [ ] 解释 deadly triad 三要素
- [ ] 跑通 ShangtongZhang `chapter11/baird_counterexample.py`，看 w 发散
- [ ] 解释 MSPBE 跟 VE 的区别
- [ ] 列出 DQN 的 3 个工程 trick 及其各解决什么
- [ ] **解释 PPO clip 和 KL penalty 是 deadly triad 的"工程缓解"**（RLHF 桥）
- [ ] 答 Q1-Q6 至少 4 个

## Related

- **上一章**：[[ch10-on-policy-control]]
- **下一章**：[[ch12-eligibility-traces]]
- **DQN 关联**：[[../../../brain/Areas/rl-books/deep-rl-hands-on-lapan-3e/ch06-dqn]]
- **进度**：[[_chapter-status]] W5 ⚠️

## Sources

- 原书 Ch 11 pp. 257-283
- planetbanatt § Ch 11 段落
- ShangtongZhang `chapter11/`
- DQN 论文（Mnih 2015）—— 工程 workaround
- Double DQN 论文（van Hasselt 2015）
- "Deep RL that Matters"（Henderson 2018）—— deadly triad 工程综述
