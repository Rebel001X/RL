---
chapter: 3
deck: Sutton-Barto::Ch03-MDP
cards: 35
format: cloze
created: 2026-05-24
---

# Ch 3 — MDP Cards (35)

> 兼容 Obsidian Spaced Repetition 插件；每行一卡，cloze 遮单 term。

## §A. MDP 定义 & 基础符号

%%Card 1%%
MDP 五元组 (S, A, P, R, γ)，其中 P(s'|s,a) 是 {{c1::转移概率}}，R(s,a,s') 是即时奖励。
<!--SR:!2026-05-25,1,250-->

%%Card 2%%
Markov 性质：未来只依赖当前 state，即 P(S_{t+1}|S_t, A_t, S_{t-1}, ..., S_0) = P(S_{t+1}|{{c1::S_t, A_t}})。
<!--SR:!2026-05-25,1,250-->

%%Card 3%%
γ ∈ [0, 1] 是 {{c1::折扣因子}}——越小越短视，γ=0 是 myopic，γ=1 是 undiscounted（episodic 才合法）。

%%Card 4%%
return G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ_{k=0}^∞ γ^k · {{c1::R_{t+k+1}}}

%%Card 5%%
recursive return: G_t = R_{t+1} + γ · {{c1::G_{t+1}}}

%%Card 6%%
γ < 1 保证 G_t {{c1::有界收敛}}（即使 reward 上界存在）——这就是为什么 continuing task 一般用 γ < 1。

## §B. Value Functions

%%Card 7%%
状态价值 V_π(s) = E_π[{{c1::G_t}} | S_t = s]——从 s 开始，遵从 π，未来 return 的期望。

%%Card 8%%
动作价值 Q_π(s, a) = E_π[G_t | S_t = s, {{c1::A_t = a}}]——比 V 多固定了首动作。

%%Card 9%%
V 和 Q 的关系：V_π(s) = Σ_a π(a|s) · {{c1::Q_π(s, a)}}

%%Card 10%%
Q 和 V 的关系：Q_π(s, a) = Σ_{s', r} p(s', r|s, a) · [r + γ · {{c1::V_π(s')}}]

%%Card 11%%
V* 和 Q* 关系：V*(s) = {{c1::max_a}} Q*(s, a)（最优策略一定贪心地选 Q*）

## §C. Bellman Equations

%%Card 12%%
Bellman expectation (V)：V_π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) · [r + γ · {{c1::V_π(s')}}]

%%Card 13%%
Bellman expectation (Q)：Q_π(s,a) = Σ_{s',r} p(s',r|s,a) · [r + γ · Σ_{a'} {{c1::π(a'|s') · Q_π(s', a')}}]

%%Card 14%%
Bellman **optimality** (V)：V*(s) = max_a Σ_{s',r} p(s',r|s,a) · [r + γ · {{c1::V*(s')}}]

%%Card 15%%
Bellman optimality (Q)：Q*(s,a) = Σ_{s',r} p(s',r|s,a) · [r + γ · {{c1::max_{a'} Q*(s', a')}}]

%%Card 16%%
Bellman 算子在 sup-norm 下是 {{c1::γ-contraction}}——这就是 value iteration 收敛性的根。

## §D. 策略 & 最优性

%%Card 17%%
deterministic policy 写作 a = π(s)；stochastic policy 写作 π({{c1::a|s}})。

%%Card 18%%
π 至少跟 π' 一样好 (π ≥ π') 当且仅当对所有 s 有 V_π(s) {{c1::≥}} V_{π'}(s)。

%%Card 19%%
最优策略 π* 至少存在 {{c1::一个 deterministic 的}}（可能多个最优 policy，但一定有 deterministic 的之一）。

%%Card 20%%
π*(s) = {{c1::argmax_a}} Q*(s, a)——给定 Q*，最优策略就是贪心。

%%Card 21%%
所有最优策略共享同一个 V*——因为最优定义就是 max_π V_π(s) = {{c1::V*(s)}}。

## §E. Episodic vs Continuing

%%Card 22%%
episodic task 终止于 {{c1::terminal state}}；γ 可以 = 1（return 自然有限）。

%%Card 23%%
continuing task 没有终态；必须 γ < 1（不然 G_t {{c1::发散}}）。

%%Card 24%%
unified notation：吸收态自循环 reward = 0，等效于把 episodic 当 {{c1::continuing}} 处理。

## §F. 易错点

%%Card 25%%
G_t 的下标 t+1 是关键：奖励是 *离开* 状态后得到的，所以 R_{t+1} = r(S_t, A_t, {{c1::S_{t+1}}})——别写错下标。

%%Card 26%%
Bellman optimality 是 **非线性**（含 max）——所以不能像 Bellman expectation 那样直接解线性系统；要 iterate。

%%Card 27%%
Bellman expectation 给定 π，是 |S| 个未知数的 {{c1::线性方程组}}——理论上可以直接 solve（O(|S|^3)）。

%%Card 28%%
V 是 |S| 维向量；Q 是 |S|×|A| 矩阵——Q 大但 control 更直接（不需要 model）。

%%Card 29%%
有 model（知道 p）→ 可以用 {{c1::V}} 做 planning；没 model → 必须用 Q（不然没法把 V 转回 action）。

## §G. 面试 gold cards

%%Card 30%%
**Q：γ 接近 1 vs 接近 0 的 trade-off？**
A：γ→1 看长远 + variance {{c1::高}}（return 累积更多噪声）+ 收敛慢；γ→0 myopic + variance 低 + 收敛快。

%%Card 31%%
**Q：为什么 Bellman optimality 不能直接 solve？**
A：含 {{c1::max 算子}}，是非线性的；只能 iterate（value iteration 或 policy iteration）。

%%Card 32%%
**Q：MDP 假设 vs POMDP？**
A：MDP 假设 state 是 full observable；POMDP 中 agent 只看到 {{c1::observation}}，state 需要 belief state 维护。

%%Card 33%%
**Q：reward shaping 是什么 + 有什么风险？**
A：加 potential-based reward 加速收敛；风险是 {{c1::改变最优策略}}（必须用 F(s, s') = γΦ(s') - Φ(s) 这种 potential form 才不改最优）。

%%Card 34%%
**Q：partial vs full episode？**
A：MC 必须等 episode 终止；TD 可以 {{c1::bootstrap}} 即时更新——这就是 TD 比 MC 适合 continuing task 的核心原因。

%%Card 35%%
**Q：state 设计有什么关键？**
A：必须满足 Markov 性质——若 state 不 Markov（如 Atari 单帧），就用 {{c1::frame stack 或 RNN/Transformer}} 把历史压进 state 里。

## 链回

- [[ch03-finite-mdps]] — 章节正文
- [[_index]] — Anki deck 总览
