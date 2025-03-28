import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import settings as st
import random

# GPU设置
if torch.cuda.is_available():
    device = "cuda"
    print(torch.cuda.get_device_name(0))
else:
    device = "cpu"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(st.N_state, 20)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(20, 20)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(20, 20)
        self.fc3.weight.data.normal_(0, 0.1)
        self.action_head = nn.Linear(20, st.N_action)
        self.action_head.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一层 ReLU 激活
        x = F.relu(self.fc2(x))  # 第二层 ReLU 激活
        x = F.relu(self.fc3(x))  # 第三层 ReLU 激活
        action = self.action_head(x)  # 输出动作空间
        return action


class DQN(object):
    def __init__(self):
        # 创建评估网络和目标网络
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step_counter = 0  # 学习步数记录
        self.memory_counter = 0  # 记忆量计数
        self.memory = []  # 使用 list 存储经验
        self.optimzer = torch.optim.Adam(self.eval_net.parameters(), lr=st.LR)
        self.loss_func = nn.MSELoss().to(device)  # 使用均方损失函数
        self.e = 0.05  # epsilon的初始值
        self.device = device

    def choose_action(self, state):
        """选择动作：epsilon-greedy策略"""
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)  # 将状态转换为 tensor 并添加批次维度

        if self.e < st.EPSILON_MAX:
            self.e = 0.995 * self.e + 0.005  # epsilon 随时间逐渐减小
            print(f"epsilon: {self.e}")

        if np.random.uniform() < self.e and self.memory_counter > 100:
            action_value = self.eval_net(state)  # 使用评估网络预测当前状态的 Q 值
            action = torch.argmax(action_value, dim=1).item()  # 获取最大 Q 值对应的动作
        else:
            action = np.random.randint(0, st.N_action)  # 随机选择动作

        return action

    def store_transition(self, state, action, reward, next_state, slot):
        """存储经验"""
        transition = np.hstack((state, action, reward, next_state))
        self.memory.append(torch.tensor(transition, device=device))  # 将过渡数据存储到列表中
        self.memory_counter += 1

    def sample_batch(self):
        """随机采样一个批次的经验"""
        batch = random.sample(self.memory, st.BATCH_SIZE)  # 从记忆池中随机采样
        batch = torch.stack(batch)
        b_s = batch[:, :st.N_state].float().to(device)  # 当前状态
        b_a = batch[:, st.N_state:st.N_state + 1].long().to(device)  # 当前动作
        b_r = batch[:, st.N_state + 1:st.N_state + 2].float().to(device)  # 当前奖励
        b_s_ = batch[:, -st.N_state:].float().to(device)  # 下一个状态
        return b_s, b_a, b_r, b_s_

    def learn(self):
        """训练DQN网络"""
        # 随机采样一个批次的记忆
        b_s, b_a, b_r, b_s_ = self.sample_batch()

        # 计算 q_eval（评估网络的输出）
        q_eval = self.eval_net(b_s).gather(1, b_a)

        # 计算 q_target（目标网络的输出）
        q_next = self.target_net(b_s_)
        q_target = b_r + st.GAMMA * q_next.max(1)[0].view(-1, 1)

        # 计算损失（均方误差损失）
        loss = self.loss_func(q_eval, q_target)
        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()

        return loss.item()  # 返回损失值，用于后续的绘图
