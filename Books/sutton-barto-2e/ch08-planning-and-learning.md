---
title: "Sutton & Barto Ch 8. Planning and Learning with Tabular Methods"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 8
difficulty: hard
budget_days: 2
priority_for_rlhf: skip
silver_lecture: 8
---

# Ch 8. Planning and Learning

## 0. 阅读元信息

- **难度**：hard（概念多但 RLHF 不直接用）
- **预算**：2 天 / skim 1 天 + 选读 1 天
- **必读理由**：model-based RL 范式入门；理解 Dyna / MCTS / 优先级扫描；AlphaGo 用了 MCTS
- **配套**：Silver UCL Lec 8（Integrating Learning and Planning）
- **跟 RLHF 的桥**：弱——RLHF 是 model-free；但 reasoning model 的 tree search / o1 是 MCTS 复兴

## 1. 一句话

把 **planning**（用 model 做 backup）和 **learning**（用真实经验做 backup）统一进同一个框架——Dyna 在每步交替"真实经验更新 Q + 用学到的 model 模拟 backup"，让样本利用率提升数十倍。

## 2. 关键概念地图

| 概念 | 一句话 |
|---|---|
| **Model** | 学一个 p̂(s', r \| s, a)，能"幻想"未来 |
| **Planning** | 用 model 做 backup（不需要新真实经验） |
| **Learning** | 用真实经验做 backup |
| **Dyna-Q** | TD + model + planning 三合一 |
| **Dyna-Q+** | 加 exploration bonus 处理 model 过时 |
| **Prioritized sweeping** | 优先更新 TD error 大的 state-action |
| **Expected vs Sample updates** | 用全分布 vs 单 sample 做 backup |
| **Trajectory sampling** | 沿真 trajectory 做 backup 而非随机 |
| **Real-time DP (RTDP)** | 沿 on-policy trajectory 做 DP backup |
| **MCTS (Monte Carlo Tree Search)** | 从当前 state 滚 simulation 树 |
| **Rollout algorithms** | 用 policy 做 N 次 rollout 选 action |

## 3. 必背公式与推导

### Dyna-Q backup

实经验 backup（同 Q-learning）：
$$Q(S, A) \leftarrow Q(S, A) + α [R + γ \max_{a'} Q(S', a') - Q(S, A)]$$

planning step（用 model 模拟）：
```
随机选已见过的 (S, A)
(S', R) = Model(S, A)
Q(S, A) += α [R + γ max_a' Q(S', a') - Q(S, A)]
```

### MCTS 四步骤

1. **Selection**：从根用 UCB 下行直到 leaf
2. **Expansion**：leaf 拓展一个新 child node
3. **Simulation**：从 child node 用 default policy rollout 到终态
4. **Backpropagation**：rollout reward 沿路径回传更新 Q 和 N

UCB for trees：$\text{argmax}_a \left[ Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}} \right]$

## 4. 关键算法（伪代码）

**Dyna-Q**：
```
init Q, Model
loop:
  S = current state
  A = ε-greedy(Q, S)
  take A, observe R, S'
  Q(S, A) += α [R + γ max_a' Q(S', a') - Q(S, A)]
  Model(S, A) = (R, S')
  # Planning: n times
  for _ in range(n):
    S_p, A_p = random from previously experienced
    R_p, S'_p = Model(S_p, A_p)
    Q(S_p, A_p) += α [R_p + γ max_a' Q(S'_p, a') - Q(S_p, A_p)]
  S = S'
```

**MCTS（伪代码骨架）**：
```
def mcts(root, iterations):
    for _ in range(iterations):
        node = select(root)         # UCB 下行
        child = expand(node)
        reward = simulate(child)    # rollout
        backprop(child, reward)
    return argmax_a N(root, a)      # 选访问最多的 action
```

## 5. 习题精选

**Exercise 8.1（Dyna-Q+ 模型新鲜度）**：理解 Dyna-Q+ 的 κ bonus 怎么鼓励"老（s,a）"被复访。

**Exercise 8.4（prioritized sweeping 优先级）**：实现优先队列把 |TD error| 大的 (s, a) 优先 backup。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter08/maze_dyna_q.py`（Dyna-Q 在 maze）、`chapter08/maze_prioritized_sweeping.py`
- 跑 `maze_dyna_q.py` 看 n=0/5/50 时的学习曲线差异——n=50 收敛快几十倍
- AlphaGo 论文（DeepMind）—— MCTS + value/policy net 的工业级应用
- MuZero 论文——把 model 也学出来（latent dynamics model）+ MCTS

## 7. 反直觉点 / 易错点

- **反直觉**：planning step n 不是越大越好——学的 model 不准时反而坏 Q
- **反直觉**：Dyna 在 maze 上 n=50 比 n=0 快**几十倍**——同样样本利用率天差地别
- **反直觉**：MCTS 的"selection"用 UCB 而非 greedy——保证 exploration
- **易错**：model 学错时，planning 把错误放大——需要监控 prediction error
- **易错**：MCTS 的 simulation 用 random / heuristic / learned policy 各有 trade-off

## 8. 跟 RLHF 的桥

- **直接用**：弱——RLHF 是 model-free（不学 LLM 的环境转移模型）
- **间接关联**：
  - **OpenAI o1 / DeepSeek R1** 推理 = 隐式 search——decode 时多步推理像 MCTS
  - **Best-of-N + RM = rollout algorithm**（Ch 8.10）的简化版
  - **MCTS for LLM**（如 ToT、RAP）研究方向——本章是其理论根
- **跳过决策**：纯 RLHF 路径可跳本章；做 reasoning model 研究可读

## 9. 苏格拉底 Q&A

- **Q：为什么 model-based 比 model-free 数据效率高？**
  - 提示：学到 model 后可"无限模拟"；真实经验只用一次但模拟可反复 backup
- **Q：MCTS 跟 Dyna 区别？**
  - 提示：Dyna 是 background planning（任意 (s,a)）；MCTS 是 decision-time planning（focus 当前 state 的 action）
- **Q：AlphaGo 为什么用 MCTS 而非 Q-learning？**
  - 提示：围棋 state space 太大 tabular 不可能；MCTS 用当前 state 局部 search，配 value/policy net 泛化
- **Q：reasoning model（o1）的"思考"是 MCTS 吗？**
  - 提示：训练时可能用 search 类数据 + RL；推理时是顺序 CoT 不是显式 tree——但精神是类似的"花时间想"

## 10. 自测清单（闭卷）

- [ ] 写出 Dyna-Q 算法（带 n planning steps）
- [ ] 解释 model-based vs model-free trade-off
- [ ] 画 MCTS 四步骤
- [ ] 解释 prioritized sweeping 提速原理
- [ ] 答 Q1-Q4 至少 2 个

## Related

- **上一章**：[[ch07-n-step-bootstrapping]]
- **下一章**：[[ch09-on-policy-prediction]]
- **AlphaGo 案例**：[[ch16-applications-case-studies]]
- **进度**：[[_chapter-status]] W5

## Sources

- 原书 Ch 8 pp. 159-186
- [Silver UCL Lec 8: Integrating Learning and Planning](https://www.youtube.com/watch?v=ItMutbeOHtc)
- ShangtongZhang `chapter08/`
- AlphaGo 论文（Silver 2016 Nature）
- AlphaZero 论文（Silver 2018 Science）
