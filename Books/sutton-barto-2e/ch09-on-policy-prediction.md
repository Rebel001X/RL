---
title: "Sutton & Barto Ch 9. On-policy Prediction with Approximation"
created: 2026-05-24
tags: [area/rl, type/note, status/v1, book/sutton-barto-2e]
book: sutton-barto-2e
chapter: 9
difficulty: medium
budget_days: 3
priority_for_rlhf: must
silver_lecture: 6
---

# Ch 9. On-policy Prediction with Approximation

## 0. 阅读元信息

- **难度**：medium（数学概念跃迁——从 tabular 到函数近似）
- **预算**：3 天 / ~6 小时
- **必读理由**：**没函数近似就没现代 RL**——所有 NN-based RL（DQN/PPO/SAC）都基于本章；**§9.4 projection operator 是 Ch 11 deadly triad 的前置**
- **配套**：Silver UCL Lec 6（Value Function Approximation）
- **跟 RLHF 的桥**：PPO 的 critic V_φ 就是用 NN 做函数近似——本章是其理论根

## 1. 一句话

把 V(s) 从查表换成 v̂(s, w)——参数化函数（线性 / NN）替代 tabular V；引入两件关键事：(1) 必须最小化某种**预测误差目标函数**（Mean Squared Value Error）；(2) bootstrap + 函数近似 ≠ 梯度下降——所以叫 "semi-gradient"。

## 2. 关键概念地图

| 概念 | 一句话 |
|---|---|
| **Function approximation** | v̂(s, w) ≈ v_π(s)，w 是参数（向量） |
| **Mean Squared Value Error (VE)** | $\overline{\text{VE}}(w) = \sum_s μ(s) [v_π(s) - v̂(s, w)]^2$，μ 是 on-policy 分布 |
| **Stochastic Gradient Descent** | w += -½α ∇VE = α [v_π(S_t) - v̂(S_t, w)] ∇v̂(S_t, w) |
| **Semi-gradient TD(0)** | 用 R + γv̂(S', w) 当 target，但**不对 target 取梯度**（伪梯度） |
| **Linear approximation** | v̂(s, w) = w^T x(s)，x 是 feature vector |
| **Tile coding** | 经典 feature construction：重叠 grid 的 binary feature |
| **Coarse coding / RBF** | 平滑 feature 替代 |
| **ANN (neural net)** | 现代主流——NN 当 v̂(s, w) |
| **Projection operator Π** | 把 Bellman 算子的结果投影到 v̂ 可表达空间 —— §9.4 重点 |
| **TD fixed point** | semi-gradient TD 不收敛到 VE 最优，而是 TD 不动点（差距由 §11.3 阐述） |

## 3. 必背公式与推导

### Mean Squared Value Error

$$\overline{\text{VE}}(w) = \sum_s μ(s) [v_π(s) - v̂(s, w)]^2$$

μ(s) = on-policy state distribution——访问越多权重越大。

### True SGD（如果知道 v_π）

$$w_{t+1} = w_t + α [v_π(S_t) - v̂(S_t, w_t)] ∇v̂(S_t, w_t)$$

### Monte Carlo SGD（用 G_t 替 v_π）

$$w_{t+1} = w_t + α [G_t - v̂(S_t, w_t)] ∇v̂(S_t, w_t)$$

**无偏**——SGD 收敛性保证（适当 α）。

### Semi-gradient TD(0)

$$w_{t+1} = w_t + α [R_{t+1} + γ v̂(S_{t+1}, w_t) - v̂(S_t, w_t)] ∇v̂(S_t, w_t)$$

**"semi"** 因为 target $R + γv̂(S', w)$ 也依赖 w，但**不对它求梯度**（pretend it's a constant）—— 这才是 TD 的精髓，但破坏了真 SGD。

### 线性近似 + semi-gradient TD 收敛性

linear v̂(s, w) = w^T x(s)，on-policy semi-gradient TD 收敛到 **TD fixed point**：
$$\overline{\text{VE}}(w_{\text{TD}}) \leq \frac{1}{1-γ} \min_w \overline{\text{VE}}(w)$$

—— 跟最优解差一个 γ-依赖的因子；on-policy + linear 仍能收敛是核心保证。

## 4. 关键算法（伪代码）

**Semi-gradient TD(0) for v̂**：
```
init w arbitrary; π = given
loop for each episode:
  S = init
  loop for each step:
    A = π(S)
    take A, observe R, S'
    w += α [R + γ v̂(S', w) - v̂(S, w)] ∇v̂(S, w)
    S = S'
  until S terminal
```

**Linear semi-gradient TD(0)**：
```
v̂(S, w) = w^T x(S)
∇v̂ = x(S)
w += α [R + γ w^T x(S') - w^T x(S)] x(S)
```

## 5. 习题精选

**Exercise 9.1（feature design）**：在 random walk on 1000 states 上设计 state aggregation feature（10 group）。

**Exercise 9.6（gradient MC vs semi-gradient TD）**：比较在 random walk 上两者的学习曲线和最终 VE。

**Exercise 9.7（tile coding 配置）**：扫 # tilings × tile width 对 VE 的影响。

## 6. 代码伴侣

- **ShangtongZhang**: `chapter09/random_walk.py`（1000-state random walk，state aggregation + tile coding）
- 自己写：1 行 linear v̂(s, w) = w^T x(s)，5 行 semi-gradient TD update
- DQN 论文（Mnih 2015）是 NN 版本的本章 + Q-learning + experience replay

## 7. 反直觉点 / 易错点

- **反直觉**：semi-gradient TD **不**最小化 VE——而是 TD fixed point；两者有 gap
- **反直觉**：on-policy + linear 必收敛；off-policy + nonlinear 可能发散（Ch 11 deadly triad）
- **反直觉**：μ(s) 加权很关键——VE 是"按访问频率加权"的误差，rare state 误差大也无所谓
- **易错**：忘记 v̂(terminal) = 0
- **易错**：semi-gradient 把 target 当常数算梯度——不是 true gradient

## 8. 跟 RLHF 的桥

- **PPO critic V_φ(s)** = NN 函数近似 v̂(s, φ)
- **训练 target** = R + γV_φ(s') 或 GAE → 跟 semi-gradient TD 同形式
- **GRPO 没 critic**——回到 MC-style estimation，回避了 §9 的所有复杂性
- **关键认识**：RLHF 用 NN 必然进入 nonlinear function approximation 区域——理论上无收敛保证，工程上靠 PPO clip / KL penalty 维持稳定

## 9. 苏格拉底 Q&A

- **Q：为什么不直接 SGD 在 v_π 上？**
  - 提示：v_π 未知——这是 RL 的核心难题；只能用 sample（MC）或 bootstrap（TD）替代
- **Q：semi-gradient 为什么不是 true gradient？**
  - 提示：target 也依赖 w，但忽略这个依赖求梯度——这是"伪梯度"
- **Q：on-policy 收敛 vs off-policy 发散——根源是什么？**
  - 提示：on-policy 分布 μ 跟 update 一致；off-policy μ ≠ behavior，权重不匹配 → 不动点可能不存在
- **Q：tile coding 跟 NN 的区别？**
  - 提示：tile = linear feature + 局部支撑；NN = 学到的非线性 feature。tile 收敛性好但 expressivity 受限
- **Q：μ(s) 是什么？为什么加权？**
  - 提示：on-policy state distribution——agent 访问多的 state 误差影响大；理论保证依赖这个加权

## 10. 自测清单（闭卷）

- [ ] 写出 VE 定义和 semi-gradient TD(0) 更新
- [ ] 解释 "semi-gradient" 的"半"在哪
- [ ] 解释 on-policy + linear 收敛但 off-policy + nonlinear 可能发散
- [ ] 跑通 ShangtongZhang `chapter09/random_walk.py`
- [ ] **解释 PPO critic 跟本章关系**（RLHF 桥）
- [ ] 答 Q1-Q5 至少 3 个

## Related

- **上一章**：[[ch08-planning-and-learning]]
- **下一章**：[[ch10-on-policy-control]]
- **延伸**：[[ch11-off-policy-methods]]（deadly triad，看本章 §9.4 后再看）
- **RLHF 桥**：[[../../../brain/Areas/rl-books/rlhf-lambert/ch06-reinforcement-learning]]
- **进度**：[[_chapter-status]] W4

## Sources

- 原书 Ch 9 pp. 197-218
- [Silver UCL Lec 6: Value Function Approximation](https://www.youtube.com/watch?v=UoPei5o4fps)
- ShangtongZhang `chapter09/`
- DQN 论文（Mnih 2015）—— 工业级 NN 函数近似
