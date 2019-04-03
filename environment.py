# -*- coding: utf-8 -*-
import torch
from unityagents import UnityEnvironment
from utils import print_bracketing
import platform

class Environment:
    """
    Wrapper for the Udacity Reacher environment utilizing 20 actors. Keeps the
    main body code a bit more neat and allows for easier access to certain
    params elsewhere.
    """
    
    def __init__(self, args, id=33):

        self.train = not args.eval

        print("LOADING ON SYSTEM: {}".format(platform.system()))


        print_bracketing(do_lower=False)
        if platform.system() == 'Linux':
            unity_filename = "Reacher_Linux_NoVis/Reacher.x86_64"
        elif platform.system() == 'Darwin':
            print("MacOS not supported in this code!")
        else:
            unity_filename = 'Reacher_Windows_x86_64/Reacher.exe'

        self.env = UnityEnvironment(file_name=unity_filename, worker_id=id, no_graphics=args.nographics)
        print_bracketing(do_upper=False)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # Environment resets itself when the class is instantiated
        self.reset()

        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.states.shape[1]
        self.agent_count = len(self.env_info.agents)

    def reset(self):
        """
        Resets the environment.
        """

        self.env_info = self.env.reset(train_mode = self.train)[self.brain_name]

    def close(self):
        """
        Closes the environment when Agent is done interacting with it.
        """

        self.env.close()

    def step(self, actions):
        """
        Returns REWARDS, NEXT_STATES, DONES based on the actions provided.
        """

        self.env_info = self.env.step(actions)[self.brain_name]
        next_states = self.states
        rewards = self.env_info.rewards
        dones = self.env_info.local_done
        return next_states, rewards, dones

    @property
    def states(self):
        """
        Returns the STATES as a tensor.
        """

        states = self.env_info.vector_observations
        return torch.from_numpy(states).float()
