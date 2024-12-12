import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std.clamp(-20, 2)
        return mean, log_std

    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.randn_like(mean)
        action = torch.tanh(mean + std*normal)
        # Scale action appropriately if needed
        W_s, W_e, W_c = action[0]
        # Map [-1,1] to [0.1,5.0] as before
        def scale(x): return 0.1 + (x+1)*(5.0-0.1)/2
        return scale(W_s.item()), scale(W_e.item()), scale(W_c.item())

    def log_prob(self, state, action_raw):
        # action_raw expected in [-1,1]
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = (action_raw - mean)/std
        log_prob = -0.5 * normal.pow(2) - log_std - np.log(np.sqrt(2*np.pi))
        return log_prob.sum(dim=-1)

class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.v(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2, update_iters=80, target_kl=0.01, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.update_iters = update_iters
        self.target_kl = target_kl
        self.batch_size = batch_size

        self.actor = PPOActor(state_dim, action_dim)
        self.critic = PPOCritic(state_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.done_flags = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def get_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.actor.forward(state_t)
        std = torch.exp(log_std)
        normal = torch.randn_like(mean)
        action_raw = torch.tanh(mean + std*normal)
        # Scale action back
        def scale(x): return 0.1 + (x+1)*(5.0-0.1)/2
        W_s, W_e, W_c = scale(action_raw[0,0].item()), scale(action_raw[0,1].item()), scale(action_raw[0,2].item())

        # Store log_prob
        # To get log_prob:
        action_flat = action_raw
        log_prob = -0.5 * ((action_flat - mean)/std).pow(2) - log_std - np.log(np.sqrt(2*np.pi))
        log_prob = log_prob.sum(dim=-1)

        value = self.critic(state_t).squeeze(0)
        self.last_value = value.item()

        # Store transition temporarily, append after step execution in run_episode
        self.current_state = state
        self.current_action_raw = action_raw[0].detach().cpu().numpy()
        self.current_log_prob = log_prob.detach().cpu().numpy().item()
        self.current_value = value.detach().cpu().item()

        return W_s, W_e, W_c

    def store_transition(self, state, action, reward, next_state, done):
        # action here is scaled weights, we need to store raw action or re-compute log prob if needed.
        # We already have current_action_raw from get_action step.
        self.states.append(self.current_state)
        self.actions.append(self.current_action_raw)
        self.rewards.append(reward)
        self.values.append(self.current_value)
        self.log_probs.append(self.current_log_prob)
        self.done_flags.append(done)

    def finish_path(self, last_val=0):
        # Compute GAE-lambda returns and advantages
        path_len = len(self.rewards)
        advantages = np.zeros(path_len)
        returns = np.zeros(path_len)
        last_gae_lam = 0
        for t in reversed(range(path_len)):
            if t == path_len-1:
                next_val = last_val
            else:
                next_val = self.values[t+1]
            delta = self.rewards[t] + self.gamma*next_val*(1-self.done_flags[t]) - self.values[t]
            advantages[t] = last_gae_lam = delta + self.gamma*self.lam*(1-self.done_flags[t])*last_gae_lam
        returns = advantages + np.array(self.values)

        return advantages, returns

    def update_networks(self):
        # PPO is on-policy, so update after an entire episode (done) or fixed rollout length.
        if len(self.rewards) == 0:
            return
        if self.done_flags[-1] == False:
            # If not done, bootstrap value from critic
            last_val = self.critic(torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)).item()
        else:
            last_val = 0

        advantages, returns = self.finish_path(last_val)

        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean())/(advantages_t.std() + 1e-8)

        # PPO update
        for i in range(self.update_iters):
            # Shuffle and batch
            idx = np.arange(len(states))
            np.random.shuffle(idx)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = idx[start:end]
                s_batch = states[batch_idx]
                a_batch = actions[batch_idx]
                adv_batch = advantages_t[batch_idx]
                ret_batch = returns_t[batch_idx]
                old_logp_batch = old_log_probs[batch_idx]

                mean, log_std = self.actor.forward(s_batch)
                std = torch.exp(log_std)
                normal = (a_batch - mean)/std
                logp = -0.5 * normal.pow(2) - log_std - np.log(np.sqrt(2*np.pi))
                logp = logp.sum(dim=-1)

                ratio = torch.exp(logp - old_logp_batch)
                clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv_batch
                actor_loss = -(torch.min(ratio*adv_batch, clip_adv)).mean()

                v = self.critic(s_batch)
                critic_loss = F.mse_loss(v, ret_batch.unsqueeze(-1))

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            # Could check KL divergence and early stop updates if KL > target_kl

        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.done_flags = []

    def update_networks_end_episode(self):
        # Just alias for consistency: PPO updates after entire episode
        self.update_networks()
