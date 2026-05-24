---
chapter: 12
deck: Sutton-Barto::Ch12-ET-GAE
cards: 35
format: cloze
created: 2026-05-24
---

# Ch 12 — Eligibility Traces + GAE Cards (35)

## §A. λ-return + TD(λ) Forward View

%%Card 1%%
n-step return G^{(n)}_t = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + {{c1::γ^n V(S_{t+n})}}

%%Card 2%%
**λ-return** G^λ_t = (1 - λ) · Σ_{n=1}^∞ λ^{n-1} · {{c1::G^{(n)}_t}}——加权平均所有 n-step return，权重几何衰减。

%%Card 3%%
λ-return 中 λ=0 → G^λ = G^{(1)} = {{c1::TD(0)}}；λ=1 → G^λ = G^∞ = MC return。

%%Card 4%%
λ-return 的 (1-λ) 系数保证 {{c1::几何级数收敛}}到 1（权重归一化）。

%%Card 5%%
TD(λ) **forward view** update: V(S_t) ← V(S_t) + α · [{{c1::G^λ_t - V(S_t)}}]——理论形式，但需要 episode 终止才能算。

## §B. TD(λ) Backward View（eligibility trace）

%%Card 6%%
**eligibility trace** e_t(s) 是每个 state 的"近期是否被访问"的指数衰减计数。

%%Card 7%%
**Accumulating** trace update: e_t(s) = γλe_{t-1}(s) + {{c1::1}}_{S_t = s}——访问就 +1。

%%Card 8%%
**Replacing** trace update: e_t(s) = γλe_{t-1}(s) if S_t ≠ s, else {{c1::1}}——访问就 reset to 1，不累加。

%%Card 9%%
TD(λ) backward view update: V(s) ← V(s) + α · δ_t · {{c1::e_t(s)}}——单 step δ 传播给所有 active state。

%%Card 10%%
TD(λ) backward view 跟 forward view 在 offline（episode 终止后 update）下 {{c1::数学等价}}（Sutton 12.2 定理）。

%%Card 11%%
backward view 优势：{{c1::online}}（不需要等 episode 终止）+ 计算高效（一次扫所有 active state）。

## §C. True Online TD(λ)

%%Card 12%%
经典 TD(λ) backward view 在 **online** 下跟 forward view **不严格等价**（有 offline-online 差）——叫 {{c1::forward-backward 差距}}。

%%Card 13%%
**True Online TD(λ)** (van Seijen 2014) 修正了这个 gap：online 下也跟 forward view {{c1::严格等价}}。

%%Card 14%%
True Online TD(λ) 用 **dutch trace**（不是 accumulating 或 replacing）：trace 更新涉及 last update 的偏移项。

## §D. SARSA(λ) / Q(λ)

%%Card 15%%
**SARSA(λ)** 是 SARSA + ET：每步算 δ + 把 δ 沿 trace 传到所有近期 (s, a)——比 SARSA 收敛 {{c1::更快}}。

%%Card 16%%
SARSA(λ) trace update: e_t(s, a) = γλe_{t-1}(s, a) + 1（访问 (s,a) 时）——是 (s, a) 对的 trace，不止 s。

%%Card 17%%
**Q(λ)** 有两种：Watkins's Q(λ)（trace 在 non-greedy action 时 {{c1::reset to 0}}）vs Peng's Q(λ)（不 reset，理论不严但实用）。

%%Card 18%%
Q(λ) 跟 SARSA(λ) 的区别就是 SARSA vs Q-learning 的区别——target 是 Q(S', A') 还是 max_a Q(S', a)，但 trace 处理上 {{c1::Q-learning 要 reset}}。

## §E. GAE（最重要！）

%%Card 19%%
**GAE**（Generalized Advantage Estimation，Schulman 2015）定义: Â^GAE(γ, λ)_t = Σ_{l=0}^∞ {{c1::(γλ)^l · δ_{t+l}^V}}

%%Card 20%%
GAE 公式中 δ_{t+l}^V = R_{t+l+1} + γV(S_{t+l+1}) - {{c1::V(S_{t+l})}}——TD residual。

%%Card 21%%
GAE λ=0 → Â = δ_t = TD(0) advantage（{{c1::低 variance, 高 bias}}）。

%%Card 22%%
GAE λ=1 → Â = G_t - V(S_t) = MC advantage（{{c1::高 variance, 低 bias}}）——λ 是 bias-variance 旋钮。

%%Card 23%%
GAE 实战 λ 推荐 **0.95**——经验最优 sweet spot for PPO / TRPO。

%%Card 24%%
GAE 推导核心：用 (γλ) 加权所有 n-step advantage——n-step adv 跟 n-step return - V 等价，但 GAE 用 δ 累积 {{c1::更高效}}。

%%Card 25%%
GAE 计算上是 reverse-time 累积：Â_t = δ_t + γλ · {{c1::Â_{t+1}}}——从 episode 末尾倒回算，O(T)。

## §F. 反直觉 + 易错

%%Card 26%%
λ 的两种语义都对应同一个 λ 值——**TD(λ) 的 λ** 控制 eligibility trace 衰减；**GAE 的 λ** 控制 advantage estimator——本质一样。

%%Card 27%%
ET 是 **mechanistic** view（每 step 更新所有近期 state）；λ-return 是 **theoretical** view（forward 加权）——两个视角，{{c1::同一个事}}。

%%Card 28%%
ET 在 **linear FA** 下也有清晰理论；在 **NN FA** 下理论不严但实际 work（PPO + GAE 就是）。

%%Card 29%%
NN 实战中很少用经典 TD(λ) 算 V——但 GAE 是**算 advantage**的标配（PPO / TRPO / A2C），与 V critic 配合。

%%Card 30%%
GAE 跟 N-step advantage 在 truncated 情况下等价——λ = 1 退化 {{c1::MC adv}}，λ = 0 退化 1-step adv。

## §G. 面试 gold

%%Card 31%%
**Q：手写 GAE 公式？**
A：Â^GAE_t = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}^V，其中 δ_t^V = {{c1::R_{t+1} + γV(S_{t+1}) - V(S_t)}}。或递归形式 Â_t = δ_t + γλ · Â_{t+1}。

%%Card 32%%
**Q：GAE λ 怎么调？**
A：λ → 1 高方差低偏差（更 MC-like）；λ → 0 低方差高偏差（更 TD-like）；实战 PPO 用 {{c1::λ=0.95}}。γ 跟 λ 独立但 advantage 累积中 γλ 是 effective decay。

%%Card 33%%
**Q：为什么 GAE 比单纯 MC adv 或 TD adv 好？**
A：在 bias-variance 上 **任意可调**——λ 当旋钮，实战 0.95 sweet spot；MC adv 高 var，TD adv 高 bias，{{c1::GAE 中间最优}}。

%%Card 34%%
**Q：eligibility trace 跟 GAE 关系？**
A：eligibility trace 是 TD(λ) 的 {{c1::mechanism}} 实现，GAE 把同一思想用到 advantage 上——本质是同一个 (γλ)^l 加权框架。

%%Card 35%%
**Q：GAE 必须 V critic 吗？**
A：必须——GAE 中的 δ_t = R + γV(S') - V(S) 含 V——所以 PPO/TRPO 都要 critic（actor-critic 架构）。MC return + baseline 是其极端 {{c1::λ=1}} 情况。

## 链回

- [[ch12-eligibility-traces]]
- [[ch13-policy-gradient]] — GAE 用在 PG 上
- [[_index]]
