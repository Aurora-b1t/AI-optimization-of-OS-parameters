# 基于常见机器学习算法实现系统参数的AI调优
全国大学生计算机系统能力大赛OS功能挑战赛道proj288

本项目以“虚拟页式缓存技术中的缓存淘汰”为场景，提供:
- 一个无 Gym 依赖的向量化页面缓存环境 `VectorPageCacheEnv`
- 一个用于该环境的 DQN 实现
- 若干传统缓存淘汰基线（FIFO/LRU/LFU）

通过 `run.py` 可训练 DQN 并观察命中率；通过 `run2.py` 可离线对比传统策略的效果。

## 目录结构

- `page_cache_vector_env.py`：向量化页面缓存环境（RL 环境）
- `theDQN.py`：DQN 网络、经验回放与训练逻辑
- `run.py`：DQN 训练与评估脚本
- `run2.py`：FIFO/LRU/LFU 基线评测

## 环境与依赖

- Python 3.11
- 依赖包:
  - numpy
  - torch
- 可选: GPU（自动选择 `cuda`，否则回退 `cpu`）