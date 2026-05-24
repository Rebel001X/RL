---
chapter: 11
deck: Sutton-Barto::Ch11-DeadlyTriad
cards: 35
format: cloze
created: 2026-05-24
---

# Ch 11 — Deadly Triad Cards (35)

## §A. 致命三元组

%%Card 1%%
**Deadly triad** 三条腿：function approximation / bootstrapping / {{c1::off-policy}}——三者**同时**出现才可能发散。

%%Card 2%%
缺任意一条腿都 safe：tabular（无 FA）/ MC（无 bootstrap）/ on-policy（无 off-policy）——这就是为什么 PPO / A2C {{c1::稳定}}（on-policy）。

%%Card 3%%
DQN 同时具备 FA（NN）+ bootstrap（Q-learning target）+ off-policy（replay buffer）——三腿俱全，必须靠 {{c1::trick}} 才稳。

%%Card 4%%
**bootstrap** 的定义：用 estimate 而非真实 sample 作 target——TD(0)、Q-learning 都是 bootstrap；MC {{c1::不是}}。

%%Card 5%%
**off-policy** = behavior policy b ≠ target policy π；on-policy 即 {{c1::b = π}}。

## §B. Baird's Counterexample

%%Card 6%%
Baird 反例是 **7-state MDP + linear FA + off-policy semi-gradient TD**——w 会指数发散到 {{c1::±∞}}。

%%Card 7%%
Baird 7 个 state，特征 8 维，关键是 state 7 的 feature 跟其他 state {{c1::共享参数}}（非独立）。

%%Card 8%%
Baird 中 behavior 是 uniform random（6/7 概率去 state 7）；target 是总去 {{c1::state 7}}——典型 off-policy。

%%Card 9%%
Baird 反例的核心教训：linear FA + bootstrap + off-policy → 可能 {{c1::无限发散}}（不止不收敛，是越跑越炸）。

%%Card 10%%
即使 V_π 完全可表示（在 representable 集合内），Baird 还是发散——说明问题在 {{c1::优化路径}}不是 representation。

## §C. MSBE vs MSPBE

%%Card 11%%
**MSBE**（Mean Squared Bellman Error）= ||T_π v̂ - {{c1::v̂}}||²_μ——直接最小化 Bellman 算子残差。

%%Card 12%%
**MSPBE**（Mean Squared Projected Bellman Error）= ||Π(T_π v̂) - v̂||²_μ——把 T_π v̂ 投影回 representable space 后比较。

%%Card 13%%
TD fixed point 等于 {{c1::MSPBE}} 的最小化器（不是 MSBE）——这是 Sutton 9.4 关键定理。

%%Card 14%%
MSBE 不**可学** (not learnable)：只用 trajectory 样本无法估 MSBE（需要两条独立 trajectory 从同一 state，但 sample 中 {{c1::只有一条}}）。

%%Card 15%%
MSPBE 是 {{c1::可学}}的——这就是为什么 TD 算法实际能 work。

## §D. GTD / GTD2 / TDC

%%Card 16%%
**GTD** 系列（Sutton 2009）目标：**真梯度** of MSPBE——保证 off-policy 下 linear FA + bootstrap {{c1::收敛}}。

%%Card 17%%
GTD2 比 GTD 更新更平滑（用两个 step size α, β）——通常 β << α，叫 {{c1::two-timescale}}。

%%Card 18%%
**TDC** (TD with gradient Correction) update: w ← w + α(δx - γx' · (x^T u))；u 是辅助参数——比 GTD2 更 {{c1::样本高效}}。

%%Card 19%%
GTD/GTD2/TDC 都保证 off-policy linear FA + bootstrap {{c1::收敛}}——理论漂亮但实际 deep RL 圈用得少。

%%Card 20%%
Practice 中 deep RL 偏向用 {{c1::trick}}（target net + replay）而非 GTD——trick 简单且 NN 上 work，GTD 是 linear 故事。

## §E. Emphatic-TD

%%Card 21%%
**Emphatic-TD** (Sutton 2016)：用 follow trace F_t 强调"重要"的 (s,a) update——保证 off-policy {{c1::收敛}}。

%%Card 22%%
Emphatic-TD 关键 trick：给每个 state 算一个 emphasis weight m_t，update 用 m_t · δ_t · {{c1::x(S_t)}}。

%%Card 23%%
Emphatic-TD 比 GTD 更 {{c1::简洁}}（不需要辅助参数 u）——但 m_t 估计本身可能高方差。

## §F. DQN / PPO 的解法

%%Card 24%%
DQN 解 deadly triad 的 trick：(1) experience replay (2) {{c1::target network}}——但仍是 semi-gradient，不是真梯度。

%%Card 25%%
**Target network** 思想：用一份延迟同步的 w_target 算 target——降低 {{c1::moving target}} 问题，破坏 bootstrap 的"自指"环。

%%Card 26%%
DQN 没解决 deadly triad 的**根本数学问题**——只是用工程 trick {{c1::降低}}发散概率。

%%Card 27%%
**PPO 绕开**：on-policy（无 off-policy 腿）+ clip 阻止 ratio 偏离 1 → 隐式约束 {{c1::policy 变化}}——避开 deadly triad。

%%Card 28%%
**DPO 绕开**：完全没 RL bootstrap——直接 supervised loss → 三腿一条不沾，{{c1::稳定}}。

## §G. 反直觉 / 面试 gold

%%Card 29%%
**Q：DQN 算 off-policy 吗？**
A：算——replay buffer 里的样本是 {{c1::旧 policy}}采的，target 用 max（greedy of current Q）。三腿俱全是 DQN 的本质。

%%Card 30%%
**Q：为什么 PPO 这么稳？**
A：on-policy（无 off-policy 腿）+ clip ratio + early stop on KL → 即使 NN FA + bootstrap 也 {{c1::稳定}}。

%%Card 31%%
**Q：为什么 SAC 这么稳？尽管它是 off-policy？**
A：用 {{c1::target network + soft Q}}（log-sum-exp 隐式平滑）+ replay——本质上还是 trick 路线，不是 true gradient。

%%Card 32%%
**Q：linear FA 在 on-policy 下一定收敛吗？**
A：满足 (1) on-policy (2) feature 线性独立 (3) Robbins-Monro step size——可证收敛到 {{c1::TD fixed point}}（不是 V_π，是 MSPBE min）。

%%Card 33%%
**Q：MSBE 跟 MSPBE 怎么选？**
A：理论上 MSBE 更 desire（直接 minimize Bellman error），但**不可学**——practice 中只能学 {{c1::MSPBE}}。

%%Card 34%%
**Q：deadly triad 跟 RLHF 的关系？**
A：RLHF 里 PPO 是 on-policy（avoid off-policy 腿），所以**没踩 deadly triad**——这是 PPO 在 LLM RLHF 占主流的根因之一。GRPO 也 on-policy。{{c1::DPO}} 干脆 no-RL，更彻底。

%%Card 35%%
**Q：Baird's counterexample 学了能怎么用到面试？**
A：是 RL 理论"为什么三腿俱全要小心"的 canonical example——能讲清 Baird 就证明懂 deadly triad 的 {{c1::真实数学根源}}，比死记 trick 强百倍。

## 链回

- [[ch11-off-policy-methods]]
- [[_index]]
