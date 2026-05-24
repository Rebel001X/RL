# v4 Numerical Trace Notebooks

5 个完整 trace 的 notebook，每个亲眼看一个核心算法在数值上怎么演化。

## 列表

| 文件 | 章节 | 核心 | cells |
|---|---|---|---|
| `ch06-td-vs-mc-random-walk.ipynb` | Ch 6 | 5-state random walk：TD(0) vs MC RMSE 收敛对比 | 11 |
| `ch06-cliff-walking-sarsa-vs-qlearning.ipynb` | Ch 6 | Cliff Walking：SARSA 绕远路 vs Q-learning 走悬崖 | 11 |
| `ch11-baird-counterexample.ipynb` | Ch 11 | Baird 反例：亲眼看 \|\|w\|\| 指数发散 | 9 |
| `ch12-gae-step-by-step.ipynb` | Ch 12 | GAE 手算 λ ∈ {0, 0.5, 0.95, 1} 数值 trace + 方差验证 | 11 |
| `ch13-reinforce-vs-baseline-cartpole.ipynb` | Ch 13 | REINFORCE ± baseline 在 50 seed 上方差降低 | 9 |

## 跑

```bash
cd C:/Users/jianm/Desktop/RL/Code/sutton-barto-2e
pip install -r requirements.txt  # numpy + matplotlib (+ torch for ch13 if extending to NN)
jupyter notebook _v4_notebooks/
```

## 设计原则（mtrazzi 风格简化版）

1. **复现书中关键图**——RL 笔记圈最高 ROI 学法
2. **打印每步 trace**——抽象公式 vs 看到 V[s]=0.5→0.6→... 是完全不同的理解层次
3. **interview 启示**——每个 notebook 末尾收一段面试 gold
4. **链回正章**——`[[ch06-temporal-difference-learning]] §3.x` 形式

## 不做什么

- 不复现书中所有图（mtrazzi 花 400h 太重）
- 不写完整 PyTorch / gym 依赖（保持 numpy + matplotlib 简洁）
- 不重复 ch13-policy-gradient/ 里已有的 PG 公式推导 notebook

## 链回

- [[../README]] — Code 目录总览
- [[../../Books/sutton-barto-2e/_anki/_index]] — Anki 卡片 deck
