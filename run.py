import time
from typing import List, Set
import numpy as np
import torch

from theDQN import DQNAgent, DQNConfig, Transition
from page_cache_vector_env import VectorPageCacheEnv, EnvConfig, GeneratorConfig


def mem_from_state_first_step(s: np.ndarray) -> Set[int]:
    return set(np.where(s >= 0.5)[0].tolist())


def not_mem_from_memory_set(N: int, memory_set: Set[int]) -> List[int]:
    return [i for i in range(N) if i not in memory_set]


def build_next_not_mem_mask(N: int, memory_list_sorted: List[int]) -> np.ndarray:
    mask = np.ones(N, dtype=np.float32)
    for idx in memory_list_sorted:
        mask[idx] = 0.0
    return mask


def main():
    N = 20
    capacity = 5
    cfg_env = EnvConfig(
        num_pages=N,
        capacity=capacity,
        horizon=20000,
        reward_hit=1.0,
        reward_miss=0.0,
        seed=42,
        generator=GeneratorConfig(
            num_pages=N,
            p_repeat=0.7,
            p_local=0.2,
            local_window=3,
            seed=123
        )
    )
    env = VectorPageCacheEnv(cfg_env)

    cfg_dqn = DQNConfig(
        input_dim=N,
        hidden_dims=[128, 128],
        lr=0.001,
        gamma=0.99,
        batch_size=128,
        target_update_interval=1000,
        replay_capacity=200_000,
        min_replay_size=2000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=100_000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        grad_clip_norm=5.0,
    )
    agent = DQNAgent(cfg_dqn)

    episodes = 15
    max_steps_per_episode = cfg_env.horizon
    print_interval = 1

    global_step = 0
    t0 = time.time()

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        ep_return = 0.0
        loss_sum = 0.0
        loss_count = 0

        # memory set for t=0
        memory_set = mem_from_state_first_step(s)

        step_in_ep = 0
        info_last = None

        while not done and step_in_ep < max_steps_per_episode:
            # Determine not-in-memory indices from memory_set
            not_mem_idx = not_mem_from_memory_set(env.num_pages, memory_set)
            # Select action
            a_idx, a_vec, _ = agent.select_action(s, not_mem_idx)

            # Step the env
            s_next, r, done, info = env.step(a_vec)

            # Prepare replay fields
            next_not_mem_mask = build_next_not_mem_mask(env.num_pages, info["memory"])
            tr = Transition(
                s=s.astype(np.float32),
                a_idx=int(a_idx),
                a_vec=a_vec.astype(np.float32),
                r=float(r),
                s_next=s_next.astype(np.float32),
                done=bool(done),
                next_not_mem_mask=next_not_mem_mask.astype(np.float32),
            )
            agent.replay.push(tr)

            # Optimize
            loss = agent.optimize()
            if loss is not None:
                loss_sum += loss
                loss_count += 1

            # Book-keeping
            ep_return += r
            s = s_next
            info_last = info
            memory_set = set(info["memory"])  # exact memory from env
            step_in_ep += 1
            global_step += 1

        hit_rate = (info_last["hits"] / step_in_ep) if info_last and step_in_ep > 0 else 0.0

        if ep % print_interval == 0 or ep == 1:
            avg_loss = (loss_sum / max(1, loss_count))
            elapsed = time.time() - t0
            print(f"[Ep {ep:4d}] return={ep_return:.2f} steps={step_in_ep:4d} "
                  f"hit_rate={hit_rate:.3f} eps={agent.eps:.3f} avg_loss={avg_loss:.5f} "
                  f"time={elapsed:.1f}s")

    eval_episodes = 5
    print("\n=== Evaluation (greedy) ===")
    saved_eps = agent.eps
    agent.eps = 0.0
    for ep in range(1, eval_episodes + 1):
        s = env.reset()
        done = False
        ep_return = 0.0
        memory_set = mem_from_state_first_step(s)
        steps = 0
        info_last = None
        while not done:
            not_mem_idx = not_mem_from_memory_set(env.num_pages, memory_set)
            a_idx, a_vec, _ = agent.select_action(s, not_mem_idx)
            s, r, done, info = env.step(a_vec)
            memory_set = set(info["memory"])
            ep_return += r
            steps += 1
            info_last = info
        hit_rate = (info_last["hits"] / steps) if info_last else 0.0
        print(f"[Eval Ep {ep}] return={ep_return:.2f} steps={steps} hit_rate={hit_rate:.3f}")
    agent.eps = saved_eps


if __name__ == "__main__":
    main()