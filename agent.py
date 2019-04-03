# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from buffers import ReplayBuffer
from models import ActorNet, CriticNet



class D4PG_Agent:
    """
    PyTorch Implementation of D4PG:
    "Distributed Distributional Deterministic Policy Gradients"
    (Barth-Maron, Hoffman, et al., 2018)
    As described in the paper at: https://arxiv.org/pdf/1804.08617.pdf

    Much thanks also to the original DDPG paper:
    "Continuous Control with Deep Reinforcement Learning"
    (Lillicrap, Hunt, et al., 2016)
    https://arxiv.org/pdf/1509.02971.pdf

    And to:
    "A Distributional Perspective on Reinforcement Learning"
    (Bellemare, Dabney, et al., 2017)
    https://arxiv.org/pdf/1707.06887.pdf

    D4PG utilizes distributional value estimation, n-step returns,
    prioritized experience replay (PER), distributed K-actor exploration,
    and off-policy actor-critic learning to achieve very fast and stable
    learning for continuous control tasks.

    This version of the Agent is written to interact with Udacity's
    Continuous Control robotic arm manipulation environment which provides
    20 simultaneous actors, negating the need for K-actor implementation.
    Thus, this code has no multiprocessing functionality. It could be easily
    added as part of the main.py script.

    In the original D4PG paper, it is suggested in the data that PER does
    not have significant (or perhaps any at all) effect on the speed or
    stability of learning. Thus, it too has been left out of this
    implementation but may be added as a future TODO item.
    """
    def __init__(self, env, args,
                 e_decay = 1,
                 e_min = 0.05,
                 l2_decay = 0.0001,
                 update_type = "hard"):
        """
        Initialize a D4PG Agent.
        """

        self.device = args.device
        self.framework = "D4PG"
        self.eval = args.eval
        self.agent_count = env.agent_count
        self.actor_learn_rate = args.actor_learn_rate
        self.critic_learn_rate = args.critic_learn_rate
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.action_size = env.action_size
        self.state_size = env.state_size
        self.C = args.C
        self._e = args.e
        self.e_decay = e_decay
        self.e_min = e_min
        self.gamma = args.gamma
        self.rollout = args.rollout
        self.tau = args.tau
        self.update_type = update_type

        self.num_atoms = args.num_atoms
        self.vmin = args.vmin
        self.vmax = args.vmax
        self.atoms = torch.linspace(self.vmin, self.vmax, self.num_atoms).to(self.device)

        self.t_step = 0
        self.episode = 0

        # Set up memory buffers, currently only standard replay is implemented #
        self.memory = ReplayBuffer(self.device, self.buffer_size, self.gamma, self.rollout)

        #                    Initialize ACTOR networks                         #
        self.actor = ActorNet(args.layer_sizes,
                              self.state_size,
                              self.action_size).to(self.device)
        self.actor_target = ActorNet(args.layer_sizes,
                                     self.state_size,
                                     self.action_size).to(self.device)
        self._hard_update(self.actor, self.actor_target)
        self.actor_optim = optim.Adam(self.actor.parameters(),
                                      lr=self.actor_learn_rate,
                                      weight_decay=l2_decay)

        #                   Initialize CRITIC networks                         #
        self.critic = CriticNet(args.layer_sizes,
                                self.state_size,
                                self.action_size,
                                self.num_atoms).to(self.device)
        self.critic_target = CriticNet(args.layer_sizes,
                                       self.state_size,
                                       self.action_size,
                                       self.num_atoms).to(self.device)
        self._hard_update(self.actor, self.actor_target)
        self.critic_optim = optim.Adam(self.critic.parameters(),
                                       lr=self.critic_learn_rate,
                                       weight_decay=l2_decay)

        self.new_episode()

    def act(self, states, eval=False):
        """
        Predict an action using a policy/ACTOR network π.
        Scaled noise N (gaussian distribution) is added to all actions todo
        encourage exploration.
        """

        states = states.to(self.device)
        with torch.no_grad():
            actions = self.actor(states).detach().cpu().numpy()
        if not eval:
            noise = self._gauss_noise(actions.shape)
            actions += noise
        return np.clip(actions, -1, 1)

    def step(self, states, actions, rewards, next_states, pretrain=False):
        """
        Add the current SARS' tuple into the short term memory, then learn
        """

        # Current SARS' stored in short term memory, then stacked for NStep
        experience = list(zip(states, actions, rewards, next_states))
        self.memory.store_experience(experience)
        self.t_step += 1

        # Learn after done pretraining
        if not pretrain:
            self.learn()

    def learn(self):
        """
        Performs a distributional Actor/Critic calculation and update.
        Actor πθ and πθ'
        Critic Zw and Zw' (categorical distribution)
        """

        # Sample from replay buffer, REWARDS are sum of ROLLOUT timesteps
        # Already calculated before storing in the replay buffer.
        # NEXT_STATES are ROLLOUT steps ahead of STATES
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states = batch
        atoms = self.atoms.unsqueeze(0)
        # Calculate Yᵢ from target networks using πθ' and Zw'
        # These tensors are not needed for backpropogation, so detach from the
        # calculation graph (literally doubles runtime if this is not detached)
        target_dist = self._get_targets(rewards, next_states).detach()

        # Calculate log probability DISTRIBUTION using Zw w.r.t. stored actions
        log_probs = self.critic(states, actions, log=True)

        # Calculate the critic network LOSS (Cross Entropy), CE-loss is ideal
        # for categorical value distributions as utilized in D4PG.
        # estimates distance between target and projected values
        critic_loss = -(target_dist * log_probs).sum(-1).mean()


        # Predict action for actor network loss calculation using πθ
        predicted_action = self.actor(states)

        # Predict value DISTRIBUTION using Zw w.r.t. action predicted by πθ
        probs = self.critic(states, predicted_action)

        # Multiply probabilities by atom values and sum across columns to get
        # Q-Value
        expected_reward = (probs * atoms).sum(-1)

        # Calculate the actor network LOSS (Policy Gradient)
        # Take the mean across the batch and multiply in the negative to
        # perform gradient ascent
        actor_loss = -expected_reward.mean()

        # Perform gradient ascent
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Perform gradient descent
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._update_networks()

        self.actor_loss = actor_loss.item()
        self.critic_loss = critic_loss.item()


    def initialize_memory(self, pretrain_length, env):
        """
        Fills up the ReplayBuffer memory with PRETRAIN_LENGTH number of experiences
        before training begins.
        """

        if len(self.memory) >= pretrain_length:
            print("Memory already filled, length: {}".format(len(self.memory)))
            return

        print("Initializing memory buffer.")
        states = env.states
        while len(self.memory) < pretrain_length:
            actions = np.random.uniform(-1, 1, (self.agent_count, self.action_size))
            next_states, rewards, dones = env.step(actions)
            self.step(states, actions, rewards, next_states, pretrain=True)
            if self.t_step % 10 == 0 or len(self.memory) >= pretrain_length:
                print("Taking pretrain step {}... memory filled: {}/{}\
                    ".format(self.t_step, len(self.memory), pretrain_length))

            states = next_states
        print("Done!")
        self.t_step = 0

    def _get_targets(self, rewards, next_states):
        """
        Calculate Yᵢ from target networks using πθ' and Zw'
        """

        target_actions = self.actor_target(next_states)
        target_probs = self.critic_target(next_states, target_actions)
        # Project the categorical distribution onto the supports
        projected_probs = self._categorical(rewards, target_probs)
        return projected_probs

    def _categorical(self, rewards, probs):
        """
        Returns the projected value distribution for the input state/action pair

        While there are several very similar implementations of this Categorical
        Projection methodology around github, this function owes the most
        inspiration to Zhang Shangtong and his excellent repository located at:
        https://github.com/ShangtongZhang
        """

        # Create local vars to keep code more concise
        vmin = self.vmin
        vmax = self.vmax
        atoms = self.atoms
        num_atoms = self.num_atoms
        gamma = self.gamma
        rollout = self.rollout

        rewards = rewards.unsqueeze(-1)
        delta_z = (vmax - vmin) / (num_atoms - 1)

        # Rewards were stored with 0->(N-1) summed, take Reward and add it to
        # the discounted expected reward at N (ROLLOUT) timesteps
        projected_atoms = rewards + gamma**rollout * atoms.unsqueeze(0)
        projected_atoms.clamp_(vmin, vmax)
        b = (projected_atoms - vmin) / delta_z

        # It seems that on professional level GPUs (for instance on AWS), the
        # floating point math is accurate to the degree that a tensor printing
        # as 99.00000 might in fact be 99.000000001 in the backend, perhaps due
        # to binary imprecision, but resulting in 99.00000...ceil() evaluating
        # to 100 instead of 99. Forcibly reducing the precision to the minimum
        # seems to be the only solution to this problem, and presents no issues
        # to the accuracy of calculating lower/upper_bound correctly.
        precision = 1
        b = torch.round(b * 10**precision) / 10**precision
        lower_bound = b.floor()
        upper_bound = b.ceil()

        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        projected_probs = torch.tensor(np.zeros(probs.size())).to(self.device)

        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()

    @property
    def e(self):
        """
        This property ensures that the annealing process is run every time that
        E is called.
        Anneals the epsilon rate down to a specified minimum to ensure there is
        always some noisiness to the policy actions. Returns as a property.
        """

        self._e = max(self.e_min, self._e * self.e_decay)
        return self._e

    def _gauss_noise(self, shape):
        """
        Returns the epsilon scaled noise distribution for adding to Actor
        calculated action policy.
        """

        n = np.random.normal(0, 1, shape)
        return self.e*n

    def new_episode(self):
        """
        Handle any cleanup or steps to begin a new episode of training.
        """

        self.memory.init_n_step()
        self.episode += 1

    def _update_networks(self):
        """
        Updates the network using either DDPG-style soft updates (w/ param \
        TAU), or using a DQN/D4PG style hard update every C timesteps.
        """

        if self.update_type == "soft":
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        elif self.t_step % self.C == 0:
            self._hard_update(self.actor, self.actor_target)
            self._hard_update(self.critic, self.critic_target)

    def _soft_update(self, active, target):
        """
        Slowly updated the network using every-step partial network copies
        modulated by parameter TAU.
        """

        for t_param, param in zip(target.parameters(), active.parameters()):
            t_param.data.copy_(self.tau*param.data + (1-self.tau)*t_param.data)

    def _hard_update(self, active, target):
        """
        Fully copy parameters from active network to target network. To be used
        in conjunction with a parameter "C" that modulated how many timesteps
        between these hard updates.
        """

        target.load_state_dict(active.state_dict())
