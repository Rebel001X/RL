# Sutton & Barto 2e — 章节代码与 Jupyter notebooks

每章对应一个子目录，含：
- `chXX-*.ipynb` — 深读 jupyter notebook（math + code + plots + citations）
- `*.py` — 算法独立实现
- `figures/` — 生成的图

## 环境

```bash
pip install -r requirements.txt
jupyter notebook
```

## 结构

```
Code/sutton-barto-2e/
├── README.md
├── requirements.txt
├── _common/                          ← 共用工具
│   └── plotting.py                    ← matplotlib helpers
├── ch13-policy-gradient/
│   ├── ch13-policy-gradient.ipynb     ← 主笔记本（math + code + plots）
│   ├── reinforce_cartpole.py          ← 独立 REINFORCE
│   └── pg_theorem_verification.py     ← 数值验证 PG 定理
└── （其它章按需添加）
```

## 跟 `Books/sutton-barto-2e/chXX-*.md` 的关系

- **md 文件**：快速参考、闭卷自测
- **ipynb 文件**：深度推导、可运行代码、可视化（本目录）

两者**互链**：md 顶部链 ipynb，ipynb 第一格 markdown 链回 md。
