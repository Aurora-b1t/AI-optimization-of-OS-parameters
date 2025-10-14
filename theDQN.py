from dataclasses import dataclass
from typing import List, Tuple, Optional, Deque
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    s: np.ndarray                 # (N,), float32
    a_idx: int                    # chosen page index m
    a_vec: np.ndarray             # (N,), float32 (the action vector sent to env)
    r: float
    s_next: np.ndarray            # (N,), float32
    done: bool
    next_not_mem_mask: np.ndarray # (N,), float32; 1.0 for not-in-memory at s', else 0.0


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, tr: Transition):
        self.buffer.append(tr)

    def sample(self, batch_size: int) -> List[Transition]:
        idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idxs]


@dataclass
class DQNConfig:
    input_dim: int
    hidden_dims: List[int]
    lr: float = 0.001
    gamma: float = 0.99
    batch_size: int = 64
    target_update_interval: int = 1000  # steps
    replay_capacity: int = 100_000
    min_replay_size: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip_norm: Optional[float] = 5.0


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.policy_net = DQNNet(cfg.input_dim, cfg.hidden_dims, cfg.input_dim).to(cfg.device)
        self.target_net = DQNNet(cfg.input_dim, cfg.hidden_dims, cfg.input_dim).to(cfg.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_capacity)
        self.step_count: int = 0

        self.eps = cfg.eps_start

    def update_epsilon(self):
        # Linear decay
        self.step_count += 1
        if self.cfg.eps_decay_steps > 0:
            frac = min(1.0, self.step_count / self.cfg.eps_decay_steps)
            self.eps = self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)
        else:
            self.eps = self.cfg.eps_end

    def select_action(self, state_vec: np.ndarray, not_mem_indices: List[int]) -> Tuple[int, np.ndarray, np.ndarray]:
        self.update_epsilon()

        s_t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.cfg.device)  # [1, N]
        with torch.no_grad():
            q = self.policy_net(s_t).squeeze(0)  # [N]
        q_np = q.detach().cpu().numpy().astype(np.float32)

        if len(not_mem_indices) == 0:
            m_idx = int(np.argmax(q_np))
        else:
            if np.random.rand() < self.eps:
                m_idx = int(np.random.choice(not_mem_indices))
            else:
                mask_vals = q_np[not_mem_indices]
                m_idx = int(not_mem_indices[int(np.argmax(mask_vals))])

        action_vec = q_np.copy()
        action_vec[m_idx] = float(np.max(q_np) + 1.0)

        return m_idx, action_vec.astype(np.float32), q_np

    def optimize(self):
        if len(self.replay) < self.cfg.min_replay_size:
            return None

        batch = self.replay.sample(self.cfg.batch_size)
        device = self.cfg.device

        states = torch.from_numpy(np.stack([t.s for t in batch], axis=0)).float().to(device)            # [B, N]
        actions = torch.tensor([t.a_idx for t in batch], dtype=torch.long, device=device).unsqueeze(1) # [B, 1]
        rewards = torch.tensor([t.r for t in batch], dtype=torch.float32, device=device).unsqueeze(1)  # [B, 1]
        next_states = torch.from_numpy(np.stack([t.s_next for t in batch], axis=0)).float().to(device) # [B, N]
        dones = torch.tensor([float(t.done) for t in batch], dtype=torch.float32, device=device).unsqueeze(1)  # [B,1]
        next_not_mem_masks = torch.from_numpy(
            np.stack([t.next_not_mem_mask for t in batch], axis=0)
        ).float().to(device)

        q_values = self.policy_net(states)                                # [B, N]
        q_sa = q_values.gather(1, actions)                                 # [B, 1]

        with torch.no_grad():
            q_next_target = self.target_net(next_states)  # [B, N]
            # mask invalid actions by adding a large negative number
            masked_next = q_next_target + (1.0 - next_not_mem_masks) * (-1e9)
            max_next, _ = masked_next.max(dim=1, keepdim=True)  # [B,1]
            target = rewards + (1.0 - dones) * self.cfg.gamma * max_next  # [B,1]

        loss = nn.functional.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        # Periodic hard update
        if self.step_count % self.cfg.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())