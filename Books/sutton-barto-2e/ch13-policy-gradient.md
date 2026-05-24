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

> 🚀 **高阶深读版 jupyter notebook** ：[[../../Code/sutton-barto-2e/ch13-policy-gradient/ch13-policy-gradient.ipynb|ch13.ipynb]]
>
> 含完整 PG 定理推导 + sympy 符号验证 + REINFORCE PyTorch 实现 + 数值 baseline 方差对比 + 9 篇核心论文链接 + TRPO/PPO/GRPO/DPO 拓展推导链。
> 本 md 是闭卷自测用速查卡。

---

## 📋 本章讲了什么（TL;DR）

**3 句话总览**：

1. **从 value-based 转向 policy-based**：不再学 Q(s,a) 再 derive policy，而是直接参数化 π_θ(a|s) 并沿 J(θ) 梯度上升——这是 RL 的另一支主流路线
2. **Policy Gradient Theorem**（§13.2）是本章核心数学结果：$∇J = E_π[∇\log π · Q^π]$——**state distribution μ 不出现在梯度里**，让 sample-based 估计成为可能
3. **REINFORCE → Actor-Critic → TRPO → PPO → GRPO/DPO**：所有现代 LLM 后训练算法都从本章公式派生

**为什么这章是 RL 圣经第一章**（个人观点）：S&B 17 章里只有 Ch 3（MDP）和 Ch 13（PG）是真正"绕不开"的——前者是语言，后者是 LLM 时代的 RL 入口。

---

## 🎯 Top 5 关键洞察

> 闭卷应该能复述这 5 点 + 各自的"为什么"。

| # | 洞察 | 为什么关键 |
|---|---|---|
| 1 | **Score function trick**：$∇π = π · ∇\log π$ | 把"对 π 求导"变成"对 log π 求导 + 当 expectation"——sample-based PG 的根技 |
| 2 | **State distribution μ_π 不显式出现** | Rollout 样本天然按 μ 分布，所以 sample 平均就是 ∇J 的无偏估计——这是 Sutton 1999 的革命 |
| 3 | **Baseline 不变 unbiasedness** | 任意 b(s) 都能减——因为 $E_a[∇\log π · b(s)] = b(s) \nabla\sum_a π = b·∇1 = 0$。**降方差不要钱** |
| 4 | **PG 绕开 max → 绕开 deadly triad** | value-based + off-policy + FA = 可能发散（[[ch11-off-policy-methods]]）；PG 不取 max，对 NN 更友好 |
| 5 | **PPO clip 是 trust region 的工程化** | TRPO 的 KL hard constraint → PPO 的 ratio clip 软约束——损失理论严格性换实现简洁，**这是 RLHF 工业标准** |

---

## 🧭 在 Obsidian graph 中的位置

本章是 RL vault 的**第一中心节点**（Ch 3 是第二）。打开 graph view 应该看到 Ch 13 连出大量边：

**上游**（前置）：
- [[ch03-finite-mdps]]（MDP 形式化）
- [[ch05-monte-carlo-methods]]（importance sampling，PPO ratio 的根）
- [[ch09-on-policy-prediction]]（function approximation，PG 的 NN 化）
- [[ch12-eligibility-traces]]（GAE 直接前置）

**横向**（同层并列）：
- [[ch11-off-policy-methods]]（PG 是绕开 deadly triad 的另一支）

**下游**（应用）：
- [[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning|RLHF Book Ch 6]]（PPO for LLM）
- [[../../../brain/Areas/rl-books/rlhf-lambert/ch08-direct-alignment-algorithms|RLHF Book Ch 8 DPO]]
- [[../../../brain/Slipbox/inference-metrics-ttft-tpot]]（RLHF 训练 rollout 用 inference infra）

**brain vault first-pass**：[[../../../brain/Areas/rl-books/sutton-barto-2e/ch01-introduction|brain Ch 1（first-pass）]]（Ch 13 first-pass 待写）

---

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

---

## 💡 拓展知识点（书外 trivia）

> 这一段是"读完本章后，吹牛的本钱"——书没写但 RLHF 圈子都知道的。

### A. PG 是怎么从冷门变成 LLM 时代主角的

- **1992 Williams REINFORCE 论文**当年只有几十引用——neural net 都不流行
- **1999 Sutton PG Theorem** 也是不温不火——deep learning 还没起
- **2013 Mnih DQN** 让 value-based 出圈，但很快**2015 Schulman 系列 (TRPO + GAE)** 让 PG 反超
- **2017 PPO** 论文+OpenAI Gym benchmark → 工业标准
- **2022 InstructGPT/ChatGPT** → PG 成了 LLM 后训练唯一标准（直到 DPO 出现）
- **2024 GRPO + DeepSeek R1** → PG 复兴在 reasoning model

**洞察**：技术不是"对就胜出"，是要等到合适的应用场景。PG 等了 30 年。

### A.1 Sutton 1999 论文的现场反响

据 RLDM 老人回忆——那篇 NeurIPS 论文当年现场反应平平，没人意识到二十年后它会驱动 OpenAI 估值千亿。Sutton 本人在 2017 年 reddit AMA 里说："I didn't realize it would become this important."

### B. Score Function Trick 的其它名字

同一个东西在不同领域不同名字：
- **统计**：Score function（Fisher information 的根）
- **OR / Simulation**：Likelihood ratio estimator（Glynn 1990）
- **RL**：REINFORCE estimator
- **VAE**：reparameterization 之外的 SF estimator（用于 discrete latent）
- **NAS**：search policy 用 RL 时也是这套

**结论**：score function trick 是统计学里的"勾股定理"——多次被独立发现。

### C. PG 在 RL 之外的应用

| 领域 | 用法 |
|---|---|
| **GAN training** | discriminator → reward，generator → policy，但实际用 BP 而非 PG |
| **NAS（神经网络结构搜索）** | controller 是 policy，validation acc 是 reward — Zoph 2016 |
| **超参调优** | autoML 用 RL 选超参 — Google Vizier |
| **数据增广** | AutoAugment 用 PG 学最优增广 policy |
| **分子设计** | RL for drug discovery — 化学结构是 action |
| **网络协议** | TCP congestion control 用 RL 学 — Pantheon |
| **LLM tool use** | tool calling policy 是 PG 训出来的 — Toolformer |

### D. 一个被低估的细节：γ^t in REINFORCE

S&B 算法里 REINFORCE 的更新是 `θ += α γ^t G_t ∇log π`——**很多实现忘了 γ^t**。理论上必须（episodic discounted），实践上经常省略且没大影响。

**RLHF 里通常 γ=1**，所以这个 trick 不显著。但是对短-episode 任务（CartPole γ=0.99）忽略 γ^t 会让早 steps 权重过大。**面试 trap**。

### E. Vanishing variance 现象

PPO 训稳后，advantage 接近 0，gradient 信号变弱——容易过早收敛到次优。**解决**：reward shaping / entropy bonus / 周期性 reset critic。这是 PPO 实际部署的隐藏痛点。

### F. 为什么 LLM RLHF 不用 SAC / TD3 这些"现代"算法

- **SAC** 设计给 continuous action（机器人控制）；LLM 是 discrete token，离散版 SAC 不主流
- **TD3** 强调 deterministic policy + twin critic；LLM 必须 stochastic（多样性 + exploration）
- **PPO** 在 discrete + 大 action space + 稀疏 reward 场景仍最稳——所以是 RLHF 默认

### G. RLVR 的"反 RLHF"哲学

传统 RLHF：RM 是核心，但容易 reward hacking。
**RLVR**（DeepSeek R1 路线）：reward 来自**可验证规则**（数学对错 / 代码 unit test），跳过 RM。
**新趋势**：reasoning model 用 RLVR + GRPO 训长 CoT，让模型"自己想自己验证"。

这是 2024-2026 RL 最有意思的方向——**reward source 革命**。

### H. PG 跟 backprop 是什么关系？

- **backprop** = 算梯度的算法（chain rule）
- **PG** = 用 backprop 来估计 ∇J 的"trick"（特别 score function trick）

混淆点：很多人以为 "PPO = 用 PG 训 LLM"。**更准确**：PPO 用 backprop 算 ∇log π，然后乘以 advantage 当 reward signal——backprop 是工具，PG 是公式。

---

## 🔗 更多 wikilinks（深挖）

| 主题 | 跳转 |
|---|---|
| RLHF 训练 pipeline 全图 | [[../../../brain/Areas/rl-books/rlhf-lambert/ch03-training-overview]] |
| Reward Modeling 数学 | [[../../../brain/Areas/rl-books/rlhf-lambert/ch05-reward-modeling]] |
| 推理时的 reasoning model（reasoning RL）| [[../../../brain/Areas/rl-books/rlhf-lambert/ch07-reasoning-inference-time-scaling]] |
| Direct Alignment (DPO/IPO/KTO)| [[../../../brain/Areas/rl-books/rlhf-lambert/ch08-direct-alignment-algorithms]] |
| Over-optimization / reward hacking | [[../../../brain/Areas/rl-books/rlhf-lambert/ch16-over-optimization]] |
| RLHF 评测 | [[../../../brain/Areas/rl-books/rlhf-lambert/ch17-evaluation]] |
| AI-Infra: PPO rollout 与推理引擎 | [[../../../brain/Slipbox/inference-metrics-ttft-tpot]] [[../../../brain/Slipbox/prefill-decode-disaggregation]] |
| AI-Infra: FP8 训练（RLHF 训练也用）| [[../../../brain/Slipbox/fp8-training-pipeline]] |
| 面试系统设计：PPO rollout 服务 | [[../../../brain/Areas/ai-infra/_system-design]] |
| DeepSeek V3 技术报告（GRPO 来源）| [[../../../brain/Papers/arxiv-2412.19437]] |

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
