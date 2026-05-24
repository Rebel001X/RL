---
chapter: 5
deck: Sutton-Barto::Ch05-MC
cards: 35
format: cloze
created: 2026-05-24
---

# Ch 5 — Monte Carlo Cards (35)

## §A. MC 基础

%%Card 1%%
MC 估值是把多个 episode 的 return G_t 取 {{c1::样本平均}}——不需要 environment model。

%%Card 2%%
**first-visit MC**：episode 里每个 state 只用 {{c1::首次访问}}后的 return。

%%Card 3%%
**every-visit MC**：episode 里每个 state 的 {{c1::每次访问}}都贡献一次 return 样本。

%%Card 4%%
first-visit MC 估计是 {{c1::无偏}}的；every-visit MC 在小样本下有 bias 但渐近一致。

%%Card 5%%
MC 估计 V 的方差为 {{c1::O(1/N)}}（N 是样本数）——standard error 衰减 1/√N。

%%Card 6%%
MC 必须 episode 终止才能算 G_t——所以**不能用于 {{c1::continuing}}** task。

%%Card 7%%
incremental MC update：V(S_t) ← V(S_t) + α · [{{c1::G_t - V(S_t)}}]——括号里是 MC error。

## §B. MC for Q + Exploration

%%Card 8%%
MC for V 需要 model 才能做 control；MC for Q **不需要 model**——这就是 control 用 {{c1::Q 不用 V}} 的原因。

%%Card 9%%
**Exploring Starts (ES)** 假设：每个 (s, a) 对都有 {{c1::非零概率}}作为 episode 起点——实际中往往不现实。

%%Card 10%%
**ε-greedy** 策略：1-ε 概率贪心 + ε 概率随机；保证 π(a|s) ≥ {{c1::ε/|A|}}。

%%Card 11%%
ε-soft policy：所有 π(a|s) > 0 的策略族——ε-greedy 是其中一种特殊形式。

%%Card 12%%
GLIE 条件（保证收敛到最优）：所有 (s,a) 访问无限次 **且** ε 最终 {{c1::趋于 0}}。

## §C. Importance Sampling 基础

%%Card 13%%
**off-policy** 学习：用 behavior policy b 采样，估计 target policy {{c1::π}} 的值。

%%Card 14%%
覆盖性 (coverage) 假设：π(a|s) > 0 → {{c1::b(a|s) > 0}}（b 必须包含 π 的支持）。

%%Card 15%%
single-step IS ratio：ρ_t = {{c1::π(A_t|S_t) / b(A_t|S_t)}}——一步的概率比。

%%Card 16%%
trajectory IS ratio：ρ_{t:T-1} = Π_{k=t}^{T-1} π(A_k|S_k) / {{c1::b(A_k|S_k)}}——整段路径的乘积比。

%%Card 17%%
注意：环境 dynamics p(s'|s,a) 在 ratio 里 {{c1::消掉}}！——所以 IS 不需要 model。

## §D. Ordinary vs Weighted IS

%%Card 18%%
**Ordinary IS** 估计：V(s) = (1/|T(s)|) Σ ρ_{t:T-1} · {{c1::G_t}}——简单平均。

%%Card 19%%
**Weighted IS** 估计：V(s) = Σ ρ · G_t / {{c1::Σ ρ}}——加权平均（分母是 ρ 之和）。

%%Card 20%%
Ordinary IS 是 {{c1::无偏}}但方差**可能无限大**（极端 ratio 撕方差）。

%%Card 21%%
Weighted IS 是 {{c1::有偏}}（小样本下 biased to behavior）但方差有界——一般 practice 用 weighted。

%%Card 22%%
Weighted IS 在 N→∞ 时收敛到 {{c1::true V_π}}——渐近无偏。

%%Card 23%%
**Per-decision IS** 思想：reward 只取决于到该时刻的 {{c1::前面}}动作——所以只要乘到该 reward 时刻的 ratio。

## §E. 算法 / 收敛

%%Card 24%%
Off-policy MC control 用 weighted IS 更新 Q：Q(S_t, A_t) ← Q(S_t, A_t) + (W/C) · [G_t - {{c1::Q(S_t, A_t)}}]，其中 C 是累积权重。

%%Card 25%%
behavior policy 要 ε-soft + cover target；常见做法是 b = {{c1::ε-greedy of Q}}，π = greedy of Q。

%%Card 26%%
MC + ε-soft + GLIE → 收敛到 {{c1::π*}}（ε 必须最终 → 0）。

## §F. 反直觉 / 陷阱

%%Card 27%%
MC vs DP：DP 用 expectation（model）；MC 用 {{c1::sample}}（model-free）——这是 RL 跟 planning 的分水岭。

%%Card 28%%
MC error = G_t - V(S_t)；TD error = R_{t+1} + γV(S_{t+1}) - V(S_t)——MC 不 {{c1::bootstrap}}。

%%Card 29%%
MC 估计**间独立**（不同 state 用不同 return）——所以 backup tree {{c1::很深}}。

%%Card 30%%
长 trajectory 上 trajectory IS ratio 是 |T| 个数的乘积——极易撕方差，所以 long-horizon off-policy MC {{c1::几乎不能用}}。

%%Card 31%%
Behavior b 跟 π 差越大 → IS variance 越 {{c1::大}}。差太大基本无法学（advisor exploration → 学不到 target）。

## §G. 面试 gold

%%Card 32%%
**Q：为什么 MC 不能做 continuing task？**
A：必须等 episode 终止才能算 {{c1::G_t}}；continuing 没终止。

%%Card 33%%
**Q：weighted IS 跟 ordinary IS 怎么选？**
A：weighted 在 practice 永远占优——variance 有界，渐近无偏。Ordinary 只有 {{c1::理论}}研究 unbiased 才用。

%%Card 34%%
**Q：MC 跟 TD 的 trade-off？**
A：MC 无 bias + 高 variance + episode 终止；TD 有 {{c1::bootstrap bias}} + 低 variance + 可在线。

%%Card 35%%
**Q：Importance Sampling 为什么 RLHF 几乎不直接用？**
A：long-horizon ratio 撕方差；RLHF 用 {{c1::PPO clip}} 替代 (ratio clip 在 [1-ε, 1+ε])，本质是 IS 的稳健化变体。

## 链回

- [[ch05-monte-carlo-methods]]
- [[_index]]
