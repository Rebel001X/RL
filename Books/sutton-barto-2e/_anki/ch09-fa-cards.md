---
chapter: 9
deck: Sutton-Barto::Ch09-FA
cards: 35
format: cloze
created: 2026-05-24
---

# Ch 9 — Function Approximation Cards (35)

## §A. 为什么需要 FA

%%Card 1%%
Tabular RL 在 |S| 大时**两个不可行**：(1) 内存爆 (2) 每个 state 要 {{c1::访问无限次}}才能收敛。

%%Card 2%%
FA 把 V(s) 用参数化函数 v̂(s, w) 替代，w 维度 << |S|——目标是 {{c1::generalize}}（一个 update 影响多个 state）。

%%Card 3%%
FA 的代价是 **可能不收敛**——经典 tabular 收敛保证全部失效（这是 [[deadly triad]] 的核心命题）。

## §B. VE 目标函数

%%Card 4%%
**Value Error** VE(w) = Σ_s μ(s) · [{{c1::V_π(s) - v̂(s, w)}}]² ——state distribution μ 加权的均方误差。

%%Card 5%%
on-policy 时 μ 是 {{c1::stationary distribution of π}}——访问越多的 state 加权越高。

%%Card 6%%
VE 是 mean **squared** value error；其平方根 √VE 是 RMS VE，更易解读（单位跟 V 同）。

## §C. 梯度 vs Semi-gradient

%%Card 7%%
**true gradient** TD update 应该是 -∇_w (target - v̂)²/2——但 target 含 v̂(S', w) **也依赖 w**，导致 {{c1::双梯度}}问题。

%%Card 8%%
**semi-gradient** TD：故意**忽略 target 对 w 的依赖**，只取 prediction 部分梯度——w ← w + α · δ · {{c1::∇_w v̂(S, w)}}。

%%Card 9%%
Semi-gradient TD(0) update: w ← w + α · [R + γ · v̂(S', w) - v̂(S, w)] · {{c1::∇_w v̂(S, w)}}

%%Card 10%%
Semi-gradient 不是 true gradient → **不一定**对应一个稳定的 loss landscape → 这就是为什么会 {{c1::发散}}。

%%Card 11%%
Linear FA + semi-gradient TD 在 **on-policy** 下收敛到 TD fixed point——但 off-policy 可能 {{c1::发散}}（Baird 反例）。

## §D. Linear FA

%%Card 12%%
Linear FA: v̂(s, w) = w^T · {{c1::x(s)}}——特征向量 x(s) ∈ R^d 跟 w 线性组合。

%%Card 13%%
Linear FA 梯度：∇_w v̂(s, w) = {{c1::x(s)}}——计算几乎免费。

%%Card 14%%
Linear semi-gradient TD(0) update: w ← w + α · δ · x(S_t)——简洁形式 = update step + {{c1::特征}}。

%%Card 15%%
Linear FA 的 on-policy fixed point 不是 VE 的最小化器；而是 {{c1::MSPBE}}（Projected Bellman Error）的最小化器。

## §E. 经典特征

%%Card 16%%
**Tile coding**：用多个 offset 的网格覆盖 state space；每个 tile 是 binary feature——好处是 {{c1::稀疏 + 局部 + 可控分辨率}}。

%%Card 17%%
**Coarse coding**：每个 feature 覆盖 state space 的一块（不必网格）——多 feature 重叠区域 → 分辨率高。

%%Card 18%%
**RBF**：feature x_i(s) = exp(-||s - c_i||² / 2σ²)——连续 + 平滑，但计算贵。

%%Card 19%%
Polynomial features 一般 **不推荐**——high order polynomial 在 RL state 上 {{c1::泛化差}}（局部支持没保证）。

## §F. LSTD + 高级线性

%%Card 20%%
**LSTD(0)** 是 closed-form 线性 TD：w = A^{-1} · b，其中 A = E[x(S)(x(S) - γx(S'))^T]——一次性解，{{c1::样本效率高}}。

%%Card 21%%
LSTD 缺点：每步 O(d²) update + O(d³) solve——d 大时 {{c1::太贵}}。

%%Card 22%%
LSTD vs semi-gradient TD 样本效率：LSTD 收敛**快**但每步贵；semi-gradient TD {{c1::慢}}但每步便宜。

## §G. NN FA

%%Card 23%%
NN FA：v̂(s, w) 用 deep net；用 SGD on 梯度估计 update——但 deadly triad 风险更大，需要 {{c1::稳定 trick}}（target net + replay）。

%%Card 24%%
NN FA 没有理论收敛保证——但 DQN/PPO/SAC 在 practice 中 work，主要靠 {{c1::算法工程}}（trick）+ 大批量数据。

%%Card 25%%
overparameterized NN 表现得**像 linear**（NTK 理论）——这是 deep RL 没完全炸的部分理论解释。

## §H. 反直觉 + 易错

%%Card 26%%
FA 不保证收敛——但 **on-policy linear semi-gradient TD** 在 ergodic 假设下收敛到 {{c1::TD fixed point}}。

%%Card 27%%
TD fixed point ≠ VE 最优——差距叫 {{c1::approximation error gap}}（depends on feature 表达能力）。

%%Card 28%%
即使 V_π 落在 representable 集合内，TD fixed point 也**不一定**是 V_π——只是 {{c1::projection onto representable space along Π}}。

%%Card 29%%
linear FA + on-policy → 收敛；linear FA + off-policy → **可能发散**（[[Baird's counterexample]]）。

%%Card 30%%
NN FA + on-policy → practice 中能 work（A2C/PPO）；NN FA + off-policy（DQN）→ 需要 {{c1::target network + replay}}。

## §I. 面试 gold

%%Card 31%%
**Q：semi-gradient 和 full gradient 区别？**
A：full gradient 把 target 对 w 的依赖也算进去（双梯度）；semi-gradient 故意忽略 target 对 w 的依赖——不是 true gradient，所以可能 {{c1::发散}}，但实际 work 得更好（Sutton 9.4 详解）。

%%Card 32%%
**Q：什么时候 linear FA + TD(0) 收敛？**
A：(1) on-policy (2) feature 满足某线性独立条件 (3) step size 满足 Robbins-Monro——这三条**都满足**才收敛到 {{c1::TD fixed point}}。

%%Card 33%%
**Q：为什么 tile coding 比 RBF 更工业化？**
A：(1) 稀疏 → 计算便宜 (2) 分辨率明确可调 (3) 不需要选 σ——RBF 那些 hyperparam 在新 domain 上不易调。

%%Card 34%%
**Q：用 NN FA 做 Q-learning 容易遇到什么问题？**
A：(1) target 漂移（DQN 用 {{c1::target network}} 解）(2) 样本相关（用 replay）(3) over-estimation（用 Double DQN）(4) 训练不稳定（用 Huber loss / reward clipping）。

%%Card 35%%
**Q：LSTD 跟 SGD-TD 怎么选？**
A：state 小 / feature 维度 d 不大（< 几千）+ 想要快收敛 → LSTD；state 大 / feature 高维 + 要 online → {{c1::SGD-TD}}。

## 链回

- [[ch09-on-policy-prediction]]
- [[_index]]
