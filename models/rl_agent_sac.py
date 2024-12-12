import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.97, tau=0.005, alpha=0.2, buffer_size=100000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        self.actor = MLP(state_dim, action_dim*2) # mean and log_std
        self.critic1 = MLP(state_dim+action_dim, 1)
        self.critic2 = MLP(state_dim+action_dim, 1)
        self.critic1_target = MLP(state_dim+action_dim, 1)
        self.critic2_target = MLP(state_dim+action_dim, 1)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)

        self.replay_buffer = deque(maxlen=buffer_size)

        # copy targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training SAC Agent on {self.device}")
        self.to(self.device)

    def to(self, device):
        self.actor.to(device)
        self.critic1.to(device)
        self.critic2.to(device)
        self.critic1_target.to(device)
        self.critic2_target.to(device)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean_logstd = self.actor(state)
            mean, log_std = mean_logstd[:,:self.action_dim], mean_logstd[:,self.action_dim:]
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            normal = torch.randn_like(mean)
            action = mean + std*normal
            action = torch.tanh(action) # action in [-1,1]
            # Scale action to positive weights range if needed
            # For now, let's map [-1,1] to [0.1, 5.0]
            W_s, W_e, W_c = action[0]
            W_s = 0.1 + (W_s.item()+1)* (5.0-0.1)/2
            W_e = 0.1 + (W_e.item()+1)* (5.0-0.1)/2
            W_c = 0.1 + (W_c.item()+1)* (5.0-0.1)/2
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
        states, actions, rewards, next_states, dones = self.sample_batch()

        with torch.no_grad():
            # next action
            mean_logstd = self.actor(next_states)
            mean, log_std = mean_logstd[:,:self.action_dim], mean_logstd[:,self.action_dim:]
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            normal = torch.randn_like(mean)
            next_action = torch.tanh(mean + std*normal)
            # next q
            next_q1 = self.critic1_target(torch.cat([next_states, next_action],1))
            next_q2 = self.critic2_target(torch.cat([next_states, next_action],1))
            next_q = torch.min(next_q1, next_q2) - self.alpha * (-(0.5* ((normal**2).sum(dim=1))) )
            target_q = rewards + (1 - dones)*self.gamma * next_q

        # Critic update
        current_q1 = self.critic1(torch.cat([states, actions],1))
        current_q2 = self.critic2(torch.cat([states, actions],1))
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        critic_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.step()

        # Actor update
        mean_logstd = self.actor(states)
        mean, log_std = mean_logstd[:,:self.action_dim], mean_logstd[:,self.action_dim:]
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        normal = torch.randn_like(mean)
        action = torch.tanh(mean + std*normal)
        q1 = self.critic1(torch.cat([states, action],1))
        q2 = self.critic2(torch.cat([states, action],1))
        q = torch.min(q1,q2)
        actor_loss = (self.alpha * (0.5*(normal**2).sum(dim=1)) - q).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # soft update targets
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def save_model(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
