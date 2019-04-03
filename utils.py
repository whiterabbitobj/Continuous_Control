# -*- coding: utf-8 -*-
import os.path
import numpy as np



def print_bracketing(info=None, do_upper=True, do_lower=True):
    """
    Formats a provided statement (INFO) for printing to the cmdline. If provided
    a list, will print each element to a new line, and if provided a string,
    will print a single line.
    """
    mult = 50
    if type(info) is not list and info is not None:
        mult = max(mult, len(info))
        info = [info]
    bracket = "#"
    upper = ("{0}\n{1}{2}{1}".format(bracket*mult, bracket, " "*(mult-2)))
    lower = ("{1}{2}{1}\n{0}".format(bracket*mult, bracket, " "*(mult-2)))
    if do_upper: print(upper)
    if info: print('\n'.join([line.center(mult) for line in info]))
    if do_lower: print(lower)
    return

def check_dir(dir):
    """
    Creates requested directory if it doesn't yet exist.
    """

    if not os.path.isdir(dir):
        os.makedirs(dir)


################################################################################
# The functions below are primarily for use with the Jupyter Notebook attached #
# to this project and can be safely disregarded if not touching the IPYNB file #
################################################################################


from data_handling import Saver, Logger
from agent import D4PG_Agent

def print_args(args):
    print('\n'.join(["{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)]))

def print_env_info(state, action, reward):
    print("The agent chooses ACTIONS that look like:\n{}\n".format(action))
    print("The environment returns STATES that look like:\n{}\n".format(state))
    print("The environment returns REWARDS that look like:\n{}".format(reward))

def notebook_eval_agent(args, env, filename, num_eps=2):
    eval_agent = D4PG_Agent(env, args)
    eval_saver = Saver(eval_agent.framework, eval_agent, args.save_dir, filename)
    args.eval = True
    logger = Logger(eval_agent, args)
    for episode in range(num_eps):
        env.reset()
        state = env.states
        for t in range(args.max_steps):
            action = eval_agent.act(state, eval=True)
            next_state, reward, done = env.step(action)
            state = next_state
            logger.log(reward, eval_agent)
            if np.any(done):
                break
        eval_agent.new_episode()
        logger.step(episode)
