# -*- coding: utf-8 -*-
import numpy as np
from agent import D4PG_Agent
from environment import Environment
from data_handling import Logger, Saver, gather_args

def main():
    """
    Originall written for Udacity's Continuous Control project:
    https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control

    This environment utilizes 20 actors built into the environment for parallel
    training. This specific code therefore has no implementation of distributed
    K-Actors training, but it would be straightforward to roll it into this
    training loop as needed.
    """

    args = gather_args()

    env = Environment(args)

    agent = D4PG_Agent(env, args)

    saver = Saver(agent.framework, agent, args.save_dir, args.load_file)

    if args.eval:
        eval(agent, args, env)
    else:
        train(agent, args, env, saver)

    return



def train(agent, args, env, saver):
    """
    Train the agent.
    """

    logger = Logger(agent, args, saver.save_dir)

    # Pre-fill the Replay Buffer
    agent.initialize_memory(args.pretrain, env)

    #Begin training loop
    for episode in range(1, args.num_episodes+1):
        # Begin each episode with a clean environment
        env.reset()
        # Get initial state
        states = env.states
        # Gather experience until done or max_steps is reached
        for t in range(args.max_steps):
            actions = agent.act(states)
            next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, rewards, next_states)
            states = next_states

            logger.log(rewards, agent)
            if np.any(dones):
                break

        saver.save_checkpoint(agent, args.save_every)
        agent.new_episode()
        logger.step(episode, agent)

    env.close()
    saver.save_final(agent)
    logger.graph()
    return



def eval(agent, args, env):
    """
    Evaluate the performance of an agent using a saved weights file.
    """

    logger = Logger(agent, args)

    #Begin evaluation loop
    for episode in range(1, args.num_episodes+1):
        # Begin each episode with a clean environment
        env.reset()
        # Get initial state
        states = env.states
        # Gather experience until done or max_steps is reached
        for t in range(args.max_steps):
            actions = agent.act(states, eval=True)
            next_states, rewards, dones = env.step(actions)
            states = next_states

            logger.log(rewards, agent)
            if np.any(dones):
                break

        agent.new_episode()
        logger.step(episode)

    env.close()
    return

if __name__ == "__main__":
    main()
