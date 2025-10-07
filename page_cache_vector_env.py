# -*- coding: utf-8 -*-
"""
一个按“向量状态/向量动作”定义的页面缓存 RL 环境（无 Gym 依赖）

状态 S: 长度 N=x+y 的 float32 向量
  - s[i] = 1.0  -> 上一步刚好用到了第 i 页（与是否在内存无关）
  - s[i] = 0.5  -> 第 i 页当前在内存中（但不是上一步用到）
  - s[i] = 0.0  -> 第 i 页不在内存中

动作 A: 长度 N 的实数向量
  - 选 j = argmin{ a[k] | k 在内存 } 作为淘汰候选
  - 选 m = argmax{ a[k] | k 不在内存 或 k == j } 作为装入候选
  - 若 j == m: 不执行任何置换
  - 否则: 将 j 移出内存，将 m 移入内存
  - 如此可保持内存容量恒定为 capacity

转移:
  - 执行动作决定下一步开始时的内存内容
  - 生成下一次请求页 req
  - 奖励: 命中则 reward_hit（默认 1.0），否则 reward_miss（默认 0.0）
  - s' 依据“更新后的内存 + 刚请求的页=1.0”构造

注意:
  - tie-breaking（值相等时）统一采用“索引最小优先”，保证确定性
  - 初始时内存填满 capacity 个页，并确保初始 last 请求页在内存（便于早期学习）

用法:
  env = VectorPageCacheEnv(cfg); s = env.reset()
  for ...:
      a = policy(s)  # 形状 (N,)
      s', r, done, info = env.step(a)
      buffer.add(s, a, r, s')
      s = s'
"""

from dataclasses import dataclass
from typing import Optional, Set, Dict, Tuple, List
import numpy as np
import random


# -------- 请求序列生成器（带重复/局部性，便于可控实验） --------
@dataclass
class GeneratorConfig:
    num_pages: int
    p_repeat: float = 0.5
    p_local: float = 0.3
    local_window: int = 3
    seed: Optional[int] = None


class PageRequestGenerator:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.last: Optional[int] = None

    def reset(self, init_page: Optional[int] = None) -> int:
        if init_page is None:
            self.last = self.rng.randrange(self.cfg.num_pages)
        else:
            self.last = init_page % self.cfg.num_pages
        return self.last

    def next(self) -> int:
        assert self.last is not None, "Call reset() before next()."
        n = self.cfg.num_pages
        r = self.rng.random()

        if r < self.cfg.p_repeat:
            page = self.last
        else:
            r2 = self.rng.random()
            if r2 < self.cfg.p_local:
                lo = max(0, self.last - self.cfg.local_window)
                hi = min(n - 1, self.last + self.cfg.local_window)
                page = self.rng.randrange(lo, hi + 1)
            else:
                page = self.rng.randrange(n)

        self.last = page
        return page


# -------- 环境定义 --------
@dataclass
class EnvConfig:
    num_pages: int                 # N = x + y
    capacity: int                  # 内存容量 x（要求 0 < x < N）
    horizon: int = 200             # 每个 episode 的步数
    reward_hit: float = 1.0        # 命中奖励
    reward_miss: float = 0.0       # 未命中奖励（或惩罚，按需可设为负）
    seed: Optional[int] = None
    generator: Optional[GeneratorConfig] = None


class VectorPageCacheEnv:
    def __init__(self, cfg: EnvConfig):
        assert 0 < cfg.capacity < cfg.num_pages, "capacity 必须在 1..num_pages-1"
        self.cfg = cfg
        self.N = cfg.num_pages
        self.C = cfg.capacity
        self.horizon = cfg.horizon

        gen_cfg = cfg.generator or GeneratorConfig(
            num_pages=cfg.num_pages,
            seed=(None if cfg.seed is None else cfg.seed + 1001)
        )
        self.gen = PageRequestGenerator(gen_cfg)

        self.py_rng = random.Random(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed)

        # 状态
        self.t: int = 0
        self.memory: Set[int] = set()     # 当前在内存的页集合
        self.last_req: Optional[int] = None  # 上一步请求的页
        self._obs_vec: Optional[np.ndarray] = None  # 缓存上一观测

        # 统计
        self.hit_count: int = 0
        self.miss_count: int = 0

    # ---------- 工具 ----------
    def _build_obs(self, last_req: Optional[int]) -> np.ndarray:
        """
        构造长度 N 的状态向量:
          - 先对内存页置 0.5
          - 再将 last_req 位置覆盖为 1.0（若存在）
        """
        s = np.zeros(self.N, dtype=np.float32)
        if self.memory:
            mem_idx = np.fromiter(self.memory, dtype=np.int32)
            s[mem_idx] = 0.5
        if last_req is not None:
            s[last_req] = 1.0
        return s

    @staticmethod
    def _argmin_over_candidates(values: np.ndarray, candidates: List[int]) -> int:
        """
        在 candidates 中，按 values 取最小值；若并列，返回索引最小的候选。
        """
        if not candidates:
            raise ValueError("argmin candidates is empty")
        # 先找最小的 value
        vals = values[candidates]
        min_val = vals.min()
        # 把并列的候选过滤出来，再取最小索引
        ties = [idx for idx in candidates if values[idx] == min_val]
        return min(ties)

    @staticmethod
    def _argmax_over_candidates(values: np.ndarray, candidates: List[int]) -> int:
        """
        在 candidates 中，按 values 取最大值；若并列，返回索引最小的候选。
        """
        if not candidates:
            raise ValueError("argmax candidates is empty")
        vals = values[candidates]
        max_val = vals.max()
        ties = [idx for idx in candidates if values[idx] == max_val]
        return min(ties)

    # ---------- 外部接口 ----------
    def reset(self) -> np.ndarray:
        self.t = 0
        self.memory.clear()
        self.hit_count = 0
        self.miss_count = 0

        # 初始化 last 请求页，并填充内存，确保 last 在内存中
        init_last = self.gen.reset(init_page=self.py_rng.randrange(self.N))
        self.last_req = init_last

        # 随机填满容量，确保包含 init_last
        all_pages = list(range(self.N))
        all_pages.remove(init_last)
        chosen = self.py_rng.sample(all_pages, k=self.C - 1)
        self.memory = set(chosen + [init_last])

        self._obs_vec = self._build_obs(self.last_req)
        return self._obs_vec.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        action: shape (N,) 的实数向量（list/np.ndarray 均可）
        返回: (s', r, done, info)
        """
        assert self._obs_vec is not None, "Call reset() before step()."

        a = np.asarray(action, dtype=np.float32)
        if a.shape != (self.N,):
            raise ValueError(f"action 形状应为 ({self.N},)，但收到 {a.shape}")

        # 1) 根据动作决定置换 j/m
        in_mem = sorted(self.memory)  # 排序后在并列时索引最小优先
        not_mem = [i for i in range(self.N) if i not in self.memory]

        j = self._argmin_over_candidates(a, in_mem)  # 仅在内存集合上取 argmin
        cand_m = set(not_mem)
        cand_m.add(j)  # 按规则：j 被视为“可装入”的候选
        m = self._argmax_over_candidates(a, sorted(list(cand_m)))  # 仅在 cand_m 上取 argmax

        if j != m:
            # 执行置换，保持容量恒定
            self.memory.remove(j)
            self.memory.add(m)

        # 2) 生成下一请求并计奖惩
        requested = self.gen.next()
        hit = requested in self.memory
        reward = self.cfg.reward_hit if hit else self.cfg.reward_miss
        if hit:
            self.hit_count += 1
        else:
            self.miss_count += 1

        # 3) 更新时间与观察
        self.t += 1
        self.last_req = requested
        next_obs = self._build_obs(self.last_req)
        self._obs_vec = next_obs  # 保存以便下步使用

        done = (self.t >= self.horizon)
        info = {
            "t": self.t,
            "requested": requested,
            "hit": hit,
            "evict_j": j,
            "load_m": m,
            "memory": sorted(self.memory),
            "hits": self.hit_count,
            "misses": self.miss_count,
        }
        return next_obs.copy(), float(reward), done, info

    # 便捷方法
    @property
    def num_pages(self) -> int:
        return self.N

    @property
    def capacity(self) -> int:
        return self.C


# --------- 演示用简单策略和运行函数 ---------
class RandomVectorPolicy:
    """
    简单演示策略：对每步产生一个均匀[0,1)的动作向量。
    """
    def __init__(self, N: int, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.N = N

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.random(self.N, dtype=np.float32)


def run_episode(env: VectorPageCacheEnv, policy, render: bool = False) -> Dict:
    s = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        a = policy.act(s)
        s, r, done, info = env.step(a)
        total_reward += r
        steps += 1
        if render:
            print(f"t={info['t']:3d} req={info['requested']:3d} "
                  f"hit={int(info['hit'])} evict={info['evict_j']:3d} load={info['load_m']:3d} "
                  f"reward={r:+.2f}")
        if done:
            break

    return {
        "steps": steps,
        "total_reward": total_reward,
        "avg_reward": total_reward / max(steps, 1),
        "hit_rate": info["hits"] / steps if steps > 0 else 0.0,
        "hits": info["hits"],
        "misses": info["misses"],
    }


if __name__ == "__main__":
    # 一个简单配置：N=20，容量=5，episode 长度=500
    cfg = EnvConfig(
        num_pages=20,
        capacity=5,
        horizon=500,
        reward_hit=1.0,
        reward_miss=0.0,
        seed=42,
        generator=GeneratorConfig(
            num_pages=20,
            p_repeat=0.55,
            p_local=0.35,
            local_window=2,
            seed=123
        )
    )
    env = VectorPageCacheEnv(cfg)
    policy = RandomVectorPolicy(env.num_pages, seed=0)

    print("=== Run with RandomVectorPolicy ===")
    res = run_episode(env, policy, render=False)
    print(res)