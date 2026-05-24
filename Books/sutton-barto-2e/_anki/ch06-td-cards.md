---
chapter: 6
deck: Sutton-Barto::Ch06-TD
cards: 38
format: cloze
created: 2026-05-24
---

# Ch 6 — Temporal Difference Cards (38)

## §A. TD(0) 基础

%%Card 1%%
TD(0) update: V(S_t) ← V(S_t) + α · [{{c1::R_{t+1} + γV(S_{t+1})}} - V(S_t)]

%%Card 2%%
TD **target** = {{c1::R_{t+1} + γV(S_{t+1})}}；TD **error** δ_t = target - V(S_t)。

%%Card 3%%
TD 跟 MC 的关键差异：TD 用 V(S_{t+1}) 估计 future，叫 {{c1::bootstrap}}；MC 直接拿真实 G_t。

%%Card 4%%
TD bias-variance 跟 MC 相反：TD bias {{c1::高}}（bootstrap from estimate）+ variance {{c1::低}}（只一步 reward）。

%%Card 5%%
TD 可以**在线** + **continuing task** 用；MC 必须 {{c1::episode 终止}}。

%%Card 6%%
TD(0) 在 tabular + α 满足 Robbins-Monro 条件下 collected 收敛到 {{c1::V_π}}。

## §B. SARSA

%%Card 7%%
SARSA 是 **on-policy** TD control；名字来自 update 用到 (S, A, R, {{c1::S', A'}})。

%%Card 8%%
SARSA update: Q(S, A) ← Q(S, A) + α · [R + γ · {{c1::Q(S', A')}} - Q(S, A)]，其中 A' 由当前策略选。

%%Card 9%%
SARSA 学的是 {{c1::当前策略}}的 Q（包括探索成本）——所以 cliff walking 它会"绕远路"。

%%Card 10%%
Expected SARSA: target = R + γ · Σ_a {{c1::π(a|S') · Q(S', a)}}——用期望代替 sample，方差比 SARSA 低。

%%Card 11%%
Expected SARSA 当 π = greedy 时退化为 {{c1::Q-learning}}——这就是 Q-learning 是 Expected SARSA 特例的来源。

## §C. Q-learning

%%Card 12%%
Q-learning update: Q(S, A) ← Q(S, A) + α · [R + γ · {{c1::max_{a'} Q(S', a')}} - Q(S, A)]

%%Card 13%%
Q-learning 是 **off-policy**：target 用 {{c1::max}}（greedy of Q），behavior 用 ε-greedy。

%%Card 14%%
Q-learning 学的是 {{c1::Q*}}（最优 Q），无视当前策略——所以 cliff walking 它走"悬崖边"（最优但 ε 探索时会摔）。

%%Card 15%%
Q-learning 收敛保证：满足 Robbins-Monro + 所有 (s,a) {{c1::访问无限次}} → Q → Q*（即使 behavior 是任意 ε-greedy）。

%%Card 16%%
Q-learning vs SARSA 收敛速度：SARSA 通常更快（on-policy），但 Q-learning 学到的 policy 渐近 {{c1::更优}}。

## §D. Maximization Bias + Double Q

%%Card 17%%
**Maximization bias**：E[max_a Q(s,a)] ≥ {{c1::max_a E[Q(s,a)]}}（Jensen 不等式）——所以 Q-learning over-estimate true value。

%%Card 18%%
Maximization bias 的根：同一个 Q 网络**既选 action 又评估 action**——选偏高 → 评估也偏高，{{c1::正反馈}}。

%%Card 19%%
**Double Q-learning** 解法：维护两个 Q 网络 Q_A, Q_B；用 {{c1::Q_A 选 action，Q_B 评估}}（或反之，随机切换）。

%%Card 20%%
Double Q-learning update: Q_A(S, A) ← Q_A + α · [R + γ · Q_B(S', {{c1::argmax_a Q_A(S', a)}}) - Q_A(S, A)]——选用 A，评估用 B。

%%Card 21%%
Double DQN（Hasselt 2015）应用同思路：用 online net 选 action，{{c1::target net}} 评估——降低 over-estimation。

%%Card 22%%
Double Q 在 stochastic env 下显著改善——deterministic env 下 max bias 不严重，{{c1::收益小}}。

## §E. 函数近似预告（DQN）

%%Card 23%%
DQN 三大稳定 trick：experience replay / {{c1::target network}} / reward clipping。

%%Card 24%%
**Target network** 思想：Q_target 每 N 步同步一次，避免"追自己尾巴"——降低 {{c1::moving target}} 问题。

%%Card 25%%
Experience replay 解决两个问题：(1) **打破相关性** (2) {{c1::样本复用}}——提升样本效率。

%%Card 26%%
DQN 在 Atari 上首次证明 Q-learning + neural network + 这些 trick 可以 {{c1::稳定}}训练——破除了 deadly triad 的诅咒。

## §F. n-step TD 衔接

%%Card 27%%
n-step TD target = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + {{c1::γ^n V(S_{t+n})}}

%%Card 28%%
n=1 退化 {{c1::TD(0)}}；n→∞ 退化 MC——n 是 bias-variance 的连续旋钮。

%%Card 29%%
n-step 比 TD(0) 通常**更快收敛**——但要等 n 步才能更新，{{c1::增加延迟}}。

## §G. 面试 gold

%%Card 30%%
**Q：SARSA vs Q-learning 谁更好？**
A：取决于**部署时 ε**——如果 deploy 时还有探索 → SARSA（学到带 cost 的）；如果 deploy 时 greedy → {{c1::Q-learning}}（学到最优 Q*）。

%%Card 31%%
**Q：手写 Q-learning update？**
A：Q(S, A) ← Q(S, A) + α · [R + γ · max_{a'} Q(S', a') - Q(S, A)]——记住 target 是 {{c1::max_a'}} 不是 Q(S', A')。

%%Card 32%%
**Q：手写 SARSA update？**
A：Q(S, A) ← Q(S, A) + α · [R + γ · Q(S', A') - Q(S, A)]——其中 A' 由 {{c1::当前策略}}选（on-policy）。

%%Card 33%%
**Q：为什么 Q-learning 是 off-policy？**
A：update 用 max_{a'} Q(S', a')——target 是 {{c1::greedy（target）policy}}，behavior 可以是 ε-greedy（不同 policy）。

%%Card 34%%
**Q：Cliff walking 中 SARSA 跟 Q-learning 学到的策略差异？**
A：SARSA 走 {{c1::远离悬崖}}的安全路（因为考虑 ε 探索时会摔下去）；Q-learning 走悬崖边（最优但 ε 时会摔，sum reward 低）。

%%Card 35%%
**Q：DQN 为什么需要 target network？**
A：用 online Q 同时算 target 和 prediction → target {{c1::非稳定}}（追自己尾巴），训练发散。Target network 定期同步，提供稳定 target。

%%Card 36%%
**Q：Double Q-learning 解的是什么问题？**
A：Q-learning over-estimation（max bias）——同 Q 选 action 和评估时 {{c1::Jensen 不等式}}导致系统性高估。

%%Card 37%%
**Q：Expected SARSA 相比 SARSA 优势？**
A：用 expected over actions 代替 sample → 方差更{{c1::低}}，同样 on-policy 但学习更稳定（特别 stochastic env）。

%%Card 38%%
**Q：TD 跟 MC 的本质差异？**
A：TD **bootstrap**（用 V(S') 估计），MC 用真实 {{c1::G_t}}。TD bias 高 / variance 低 / continuing 可用；MC 无 bias / variance 高 / 需 episode 终止。

## 链回

- [[ch06-temporal-difference-learning]]
- [[_index]]
