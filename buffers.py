# -*- coding: utf-8 -*-
from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:
    """
    Standard replay buffer to hold memories for later learning. Returns
    random experiences with no consideration of their usefulness.

    When using an agent with a ROLLOUT trajectory, then instead of each
    experience holding SARS' data, it holds:
    state = state at t
    action = action at t
    reward = cumulative reward from t through t+n-1
    next_state = state at t+n

    where n=ROLLOUT.
    """

    def __init__(self, device, buffer_size=100000, gamma=0.99, rollout=5):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.gamma = gamma
        self.rollout = rollout

    def store_trajectory(self, state, action, reward, next_state):
        """
        Stores a trajectory, which may or may not be the same as an experience,
        but allows for n_step rollout.
        """

        trajectory = (state, action, reward, next_state)
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        """
        Return a sample of size BATCH_SIZE as a tuple.
        """
        batch = random.sample(self.buffer, k=batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).float().to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        return (states, actions, rewards, next_states)

    def init_n_step(self):
        """
        Creates (or recreates to zero an existing) deque to handle nstep returns.
        """
        self.n_step = deque(maxlen=self.rollout)

    def store_experience(self, experience):
        """
        Once the n_step memory holds ROLLOUT number of sars' tuples, then a full
        memory can be added to the ReplayBuffer.

        This implementation has NO functionality to deal with terminal states,
        as the Reacher environment does not have terminal states. If you would
        like to see a more mature/robust implementation, please see the MAD4PG
        implementation under Collaborate & Compete in the same repository.
        """
        self.n_step.append(experience)

        # Abort if ROLLOUT steps haven't been taken in a new episode
        if len(self.n_step) < self.rollout:
            return

        # Unpacks and stores the SARS' tuple for each actor in the environment
        # thus, each timestep actually adds K_ACTORS memories to the buffer,
        # for the Udacity environment this means 20 memories each timestep.
        for actor in zip(*self.n_step):
            states, actions, rewards, next_states = zip(*actor)
            n_steps = self.rollout

            # Calculate n-step discounted reward
            rewards = np.fromiter((self.gamma**i * rewards[i] for i in range(n_steps)), float, count=n_steps)
            rewards = rewards.sum()

            # store the current state, current action, cumulative discounted
            # reward from t -> t+n-1, and the next_state at t+n (S't+n)
            states = states[0].unsqueeze(0)
            actions = torch.from_numpy(actions[0]).unsqueeze(0).double()
            rewards = torch.tensor([rewards])
            next_states = next_states[-1].unsqueeze(0)
            self.store_trajectory(states, actions, rewards, next_states)

    def __len__(self):
        return len(self.buffer)
