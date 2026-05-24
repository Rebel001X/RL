---
chapter: 13
deck: Sutton-Barto::Ch13-PG
cards: 42
format: cloze
created: 2026-05-24
---

# Ch 13 — Policy Gradient Cards (42)

> ⭐ 全 deck 最 interview-load-bearing 章——PPO / GRPO / DPO 大量手撕公式题。

## §A. Policy Gradient 基础

%%Card 1%%
PG 目标 J(θ) = E_{π_θ}[{{c1::G_0}}] 或 V_{π_θ}(s_0)——希望 maximize expected return。

%%Card 2%%
**Policy Gradient Theorem**: ∇J(θ) = E_{s∼d_π, a∼π}[∇log π(a|s) · {{c1::Q_π(s, a)}}]——这是所有 PG 算法的根。

%%Card 3%%
PG theorem 的关键是 ∇log π 出现——叫 {{c1::log-derivative trick}} 或 score function。

%%Card 4%%
d_π(s) 是 on-policy state distribution——这就是 PG **天然 on-policy** 的根（用 b≠π 采的 sample 估计 d_π 是 {{c1::off-policy}}问题）。

%%Card 5%%
PG 推导关键：∇π = π · {{c1::∇ log π}}（恒等式）——log 转换把 expectation 内的 ∇ 移到 log π。

## §B. REINFORCE

%%Card 6%%
**REINFORCE** update: θ ← θ + α · {{c1::G_t}} · ∇log π(A_t|S_t)——用整 episode return 作 PG 估计。

%%Card 7%%
REINFORCE 是 **MC PG**——无 bias 但方差**爆炸**（G_t 累积所有未来噪声）。

%%Card 8%%
REINFORCE 收敛保证：满足 Robbins-Monro 条件 → 收敛到 {{c1::local optimum of J(θ)}}（不一定 global）。

%%Card 9%%
REINFORCE 必须等 episode 终止才能 update——所以**不适合 continuing task** 跟 {{c1::长 horizon}}。

## §C. Baseline + Actor-Critic

%%Card 10%%
**Baseline subtraction**: ∇J = E[∇log π · (Q - {{c1::b(s)}})]——减 b(s) 不引 bias（只要 b 不依赖 a），但**降方差**。

%%Card 11%%
Baseline 不引 bias 的证明关键：E_a[∇log π(a|s) · b(s)] = b(s) · ∇_θ Σ_a π(a|s) = b(s) · ∇_θ {{c1::1}} = 0。

%%Card 12%%
最常用 baseline：{{c1::V_π(s)}}——这就是 actor-critic 的来源。Q - V = advantage。

%%Card 13%%
**Actor-critic** = PG (actor) + critic 估 V——actor: θ ← θ + α · ∇log π · {{c1::Â}}（Â = advantage）。

%%Card 14%%
critic 用 TD(0) update: w ← w + β · δ · ∇v̂(S, w)——actor-critic 是 PG + {{c1::TD}}（不是 MC）。

## §D. A2C / A3C / GAE

%%Card 15%%
**A2C** = synchronous Advantage Actor-Critic；用 GAE 或 n-step advantage——多个 env 并行采，{{c1::同步 update}}。

%%Card 16%%
**A3C** = asynchronous A2C（多 worker 各自 update shared params）——OpenAI 后来证明 {{c1::同步}}的 A2C 反而效果一样好（且更易复现）。

%%Card 17%%
A2C / PPO / TRPO 都用 **GAE** 算 advantage——Â^GAE_t = Σ (γλ)^l · {{c1::δ_{t+l}^V}}。

## §E. TRPO

%%Card 18%%
**TRPO**（Schulman 2015）思想：每步 update **限制** new π 跟 old π 的 KL 距离——硬约束 {{c1::trust region}}。

%%Card 19%%
TRPO 优化形式: max_θ Ê[π_new/π_old · Â] s.t. {{c1::E[KL(π_old || π_new)]}} ≤ δ——含约束的优化。

%%Card 20%%
TRPO 求解：natural gradient + line search——计算贵（要算 Fisher 矩阵或近似）→ {{c1::PPO}} 把它简化了。

## §F. PPO ⭐⭐⭐

%%Card 21%%
**PPO clip loss** L^CLIP(θ) = Ê[min(r(θ) · Â, clip(r(θ), 1-ε, {{c1::1+ε}}) · Â)]，其中 r(θ) = π_new/π_old。

%%Card 22%%
PPO clip 中的 ratio r(θ) = {{c1::π_θ(a|s) / π_θ_old(a|s)}}——重要性采样比。

%%Card 23%%
PPO clip 把 r 限制在 [1-ε, 1+ε]——隐式 {{c1::trust region}}，避开 TRPO 的 Fisher 矩阵。

%%Card 24%%
PPO ε 推荐 {{c1::0.2}}（原论文）；ε 越小越保守、收敛慢但稳；ε 越大越激进、易崩。

%%Card 25%%
PPO 每 batch update 推荐做 **{{c1::4-10}} 个 epoch**（multiple gradient steps）——能比 TRPO 高效复用样本。

%%Card 26%%
PPO 实战还加：(1) value loss (2) {{c1::entropy bonus}}（鼓励探索）(3) gradient clipping (4) Adam optimizer。

%%Card 27%%
PPO loss 完整形式: L = L^CLIP - c_1 · L^VF + c_2 · {{c1::S[π]}}（S 是 entropy）。

%%Card 28%%
PPO 在 RLHF 中是 **工业标准**——OpenAI ChatGPT / Anthropic Claude / 大部分国内 LLM 都用 PPO 做 RLHF。

## §G. GRPO ⭐⭐⭐

%%Card 29%%
**GRPO**（Group Relative Policy Optimization，DeepSeek 2024）核心：**去掉 critic**，用 group 内 relative reward 算 advantage。

%%Card 30%%
GRPO advantage: Â_i = (r_i - {{c1::mean(r_group)}}) / std(r_group)——同 prompt 多 sample 组内归一化。

%%Card 31%%
GRPO 优势 vs PPO：(1) 不需要 critic（省一半参数）(2) reward {{c1::标准化}}使 update 更稳 (3) group 内 baseline 自适应不同难度。

%%Card 32%%
GRPO 实战在 DeepSeek-R1 训 reasoning 上效果显著——成为 {{c1::RLVR}}（Reward from Verifiable Rewards）流派代表算法之一。

## §H. DPO ⭐⭐⭐

%%Card 33%%
**DPO**（Direct Preference Optimization，Rafailov 2023）核心：**完全绕开 RL**，直接 supervised loss 从 pairwise preference 学。

%%Card 34%%
DPO loss: L_DPO = -log σ(β · log [π_θ(y_w|x) / π_ref(y_w|x)] - β · log {{c1::[π_θ(y_l|x) / π_ref(y_l|x)]}})——y_w 是 winner，y_l 是 loser。

%%Card 35%%
DPO 推导关键: 从 PPO+KL 最优解的 closed-form 反推 → π* ∝ π_ref · exp(r/β) → 用 Bradley-Terry preference 把 r 消去——{{c1::得到只关于 π 的 loss}}。

%%Card 36%%
DPO 优势 vs PPO：(1) no reward model (2) no RL 训练 (3) {{c1::稳定 + 简单}} (4) 单步 supervised。

%%Card 37%%
DPO 劣势：on-policy 信号弱（只用静态 preference 数据）→ 不能在线 explore；可能 {{c1::reward hacking}} preference 数据 distribution。

## §I. DDPG / TD3 / SAC

%%Card 38%%
**DDPG** / **TD3** 是 deterministic policy gradient 在 continuous control 上的实现——actor 输出 a 而非 π(·|s)，{{c1::off-policy}} + replay。

%%Card 39%%
**SAC** 加 max entropy 项 J = E[r + α · {{c1::H(π(·|s))}}]——鼓励探索，practice 中超稳。

## §J. 面试 gold

%%Card 40%%
**Q：手写 PPO clip loss + 解释 clip？**
A：L = Ê[min(r·Â, clip(r, 1-ε, 1+ε)·Â)]，其中 r=π/π_old。clip 阻止 π 在 advantage 大时偏离过远；min 让 clip {{c1::只在 advantage 跟 r 同号且 r 偏离时}}生效（pessimistic bound）。

%%Card 41%%
**Q：PPO vs DPO 怎么选？**
A：PPO 强但训练栈复杂（reward model + critic + RL loop）；DPO 简单稳定但 {{c1::只能从静态 preference 学}}（无在线探索）。LLM RLHF 业内目前 PPO/GRPO 跟 DPO/SimPO/KTO 并存。

%%Card 42%%
**Q：GRPO 为什么去掉 critic 还能 work？**
A：用 group 内 relative reward 作 advantage——同一 prompt 多 sample 共享 baseline (mean of group)。本质是 {{c1::Monte Carlo baseline}}的特殊形式，无需 V critic。

## 链回

- [[ch13-policy-gradient]]
- [[ch12-gae-cards]] — GAE 在 §3.6
- [[_index]]
