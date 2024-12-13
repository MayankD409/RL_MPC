import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super(TD3Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = torch.tanh(self.fc3(x))
        return a

class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super(TD3Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q = nn.Linear(hidden,1)

    def forward(self, s, a):
        x = torch.cat([s,a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

class TD3Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, buffer_size=100000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size

        self.actor = TD3Actor(state_dim, action_dim)
        self.actor_target = TD3Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = TD3Critic(state_dim, action_dim)
        self.critic2 = TD3Critic(state_dim, action_dim)
        self.critic1_target = TD3Critic(state_dim, action_dim)
        self.critic2_target = TD3Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)

        self.replay_buffer = deque(maxlen=buffer_size)
        self.total_it = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training SAC Agent on {self.device}")
        self.to(self.device)

    def to(self, device):
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic1.to(device)
        self.critic2.to(device)
        self.critic1_target.to(device)
        self.critic2_target.to(device)

    def get_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        a = self.actor(state_t)
        # Scale action from [-1,1] to [0.1,5.0]
        def scale(x): return 0.1 + (x+1)*(5.0-0.1)/2
        W_s = scale(a[0,0].item())
        W_e = scale(a[0,1].item())
        W_c = scale(a[0,2].item())
        return W_s, W_e, W_c

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)).to(self.device),
                torch.FloatTensor(np.array(actions)).to(self.device),
                torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device),
                torch.FloatTensor(np.array(next_states)).to(self.device),
                torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device))

    def update_networks(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.total_it += 1
        states, actions, rewards, next_states, dones = self.sample_batch()

        with torch.no_grad():
            noise = (torch.randn_like(actions)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_states)
            next_actions = (next_actions + noise).clamp(-1,1)

            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + (1 - dones)*self.gamma*q_next

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # Policy update
        if self.total_it % self.policy_freq == 0:
            actor_actions = self.actor(states)
            actor_loss = -self.critic1(states, actor_actions).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Soft update targets
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def update_networks_end_episode(self):
        # TD3 updates continuously, no special end-of-episode update needed
        pass

    def save_model(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict()
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
