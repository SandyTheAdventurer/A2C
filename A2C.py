import torch
import torch.nn as nn
from gymnasium import Env
from torch import tensor
from collections import deque
import numpy as np
import random
from tqdm import trange
from line_profiler import profile
from main import Base
from torch.utils.tensorboard import SummaryWriter

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

class A2C():
    def __init__(self, env:Env, batch_size:int =32, γ:float = 0.99, ent_coeff:float = 0.01, λ:float = 0.95,lr:float = 0.001, logdir: str ="logs/A2C") -> None:
        self.env=env
        self.batch_size=batch_size
        self.γ=γ
        self.λ=λ
        self.actionlen=self.env.action_space.n
        self.Actor=Base(input_size=env.observation_space.shape[0], lr=lr, output_size=env.action_space.n, probs=True)
        self.Critic=Base(input_size=env.observation_space.shape[0], lr=lr, output_size=1)
        self.writer= SummaryWriter(logdir)
        self.buffer=deque(maxlen=100000)
        self.ent_coeff=ent_coeff

    @profile
    def advantage(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0

        for t in reversed(range(T)):
            if dones[t]:  # No bootstrap if episode ends
                next_value = 0
            else:
                next_value = values[t + 1] if t + 1 < T else 0

            # Temporal Difference residual
            delta = rewards[t] + self.γ * next_value - values[t]

            # Recursive GAE formula
            gae = delta + self.γ * self.λ * gae
            advantages[t] = gae

        return advantages
    
    @profile
    def act(self, obs: np.array):
        obs=torch.from_numpy(obs).type(torch.float32).cuda()
        dist = self.Actor(obs)
        select = torch.distributions.Categorical(dist)
        action = select.sample()
        logit = select.log_prob(action)
        entropy=select.entropy()
        return action.item(), logit, entropy
    
    @profile
    def infer(self):
        r, logit, obs, terminated, entropy= zip(*self.buffer)
        
        r=torch.stack(r)
        logit=torch.stack(logit)
        obs=torch.stack(obs)
        terminated=torch.stack(terminated)
        entropy=torch.stack(entropy)

        val=self.Critic(obs)
        advantage=self.advantage(r, val, terminated)

        ent_loss=-entropy.mean()*self.ent_coeff

        actor_loss= -logit*advantage + ent_loss
        self.Actor.step_optimizer(actor_loss.mean(), retain_graph=True)

        critic_loss= advantage**2
        self.Critic.step_optimizer(critic_loss.mean())

        return actor_loss.mean().to('cpu').detach().numpy(), critic_loss.mean().to('cpu').detach().numpy(), ent_loss.mean().to('cpu').detach().numpy()
    
    @profile
    def learn(self, timesteps: int):
        rewards=[]
        for i in trange(timesteps):
            obs, _ =self.env.reset()
            terminated=False
            total_rew=0
            steps=0
            while not terminated:
                action, logit, entropy= self.act(obs)
                obs, rew, terminated, truncated, _=self.env.step(action)
                total_rew+=rew
                steps+=1
                self.buffer.append((tensor(rew), logit, tensor(obs), tensor(terminated or truncated), entropy))
                if truncated:
                    break


            rewards.append(total_rew)

            aloss, closs, ent_loss=self.infer()
            self.buffer.clear()
            self.writer.add_scalar("Loss/Actor", closs, i)
            self.writer.add_scalar("Loss/Critic", aloss, i)
            self.writer.add_scalar("Rewards/OverallAverage", np.mean(rewards), i)
            self.writer.add_scalar("Rewards/Average1000Iters", np.mean(rewards[-1000:]), i)
            self.writer.add_scalar("Rewards/Average500Iters", np.mean(rewards[-500:]), i)
            self.writer.add_scalar("Rewards/PerEpochs", total_rew, i)
            self.writer.add_scalar("Loss/Entroy", ent_loss, i)
            self.writer.add_scalar("Steps", steps, i)
        self.writer.close()