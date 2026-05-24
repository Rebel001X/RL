# RL vault — Conventions

This is an Obsidian vault + git repo for my Reinforcement Learning learning workspace. When invoked here, follow these rules.

## Purpose

- **Code experiments**: DQN/PPO/GRPO impl, paper reproductions → `Code/`
- **Per-book chapter notes** (RL-focused, going deeper than brain vault): `Books/<book-slug>/`
- **Atomic RL concepts** (algorithm cards, Bellman variants): `Slipbox/`
- **RL papers** with my notes: `Papers/<arxiv-id>/`
- **Daily learning log**: `Journal/YYYY/MM-DD.md`

## Relation to brain vault

`Desktop/brain/` is the main knowledge vault. **This RL vault complements, not replaces it**:

- Brain vault `Areas/rl-books/` = my **first-pass** chapter notes (concise, 1-page, no code)
- This RL vault `Books/` = **deeper second-pass** notes when I want to add code / exercises / experiments per chapter
- Brain vault `Slipbox/` = general AI-Infra Slipbox (KV cache, FA, etc.)
- This RL vault `Slipbox/` = RL-specific atomic concepts (Bellman variants, advantage estimators, etc.)

If a topic only needs reading-level notes → brain. If it needs runnable code → here.

## Organization

- `Books/<book-slug>/chXX-title.md` — deeper chapter notes
- `Books/<book-slug>/chXX/` — sub-directory if a chapter generates code/data
- `Code/<topic>/` — algorithm impl (e.g. `Code/dqn-cartpole/`, `Code/grpo-math/`)
- `Slipbox/<concept>.md` — atomic notes
- `Papers/<arxiv-id>/` — paper-style notes
- `Journal/YYYY/MM-DD.md` — daily log
- `_templates/` — note templates
- `_attachments/` — images, gifs, training curves

## Naming

- kebab-case for everything except book PDFs (already named by Z-Library)
- Frontmatter required: `title`, `created`, `tags`
- Tags hierarchical: `area/rl`, `type/code|note|paper|journal`, `status/v1|v2|stub`, `book/<slug>`

## Don't touch

- `*.pdf` / `*.epub` (教科书原件，gitignored)
- `.obsidian/` (Obsidian config, partially gitignored)
- `.git/`
- `desktop.ini`

## Workflow

1. Reading new chapter → first-pass note in **brain vault**
2. If chapter has code/experiments → second-pass note here + Code/
3. Daily: `Journal/YYYY/MM-DD.md` 记进度 + 卡点
4. 周末：commit + push

## Git

This is a public repo: `github.com/Rebel001X/RL`. Be mindful what gets committed.
- PDFs / EPUBs: excluded (copyrighted)
- Model weights: excluded (.pth, .pt, .ckpt, .safetensors)
- Wandb logs: excluded
- `.obsidian/workspace.json` etc personal state: excluded (see .gitignore)
- Notes / code / configs: tracked
