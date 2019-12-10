#!/usr/bin/env python

from typing import List, Mapping, Any, Optional
import os
from glob import glob
import gym
import numpy as np
import textworld.gym

import re
from collections import defaultdict

from random_agent import RandomAgent
from neural_agent_sac_sep import NeuralAgent

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


UPDATE_FREQ=4


def play(agent, path, max_step=100, nb_episodes=10, verbose=True):
    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.
    
    gamefiles = [path]
    if os.path.isdir(path):
        gamefiles = glob(os.path.join(path, "*.ulx"))
    env_id = textworld.gym.register_games(gamefiles,
                                          request_infos=infos_to_request,
                                          max_episode_steps=max_step)
    env = gym.make(env_id)  # Create a Gym environment to play the text game.
    if verbose:
        if os.path.isdir(path):
            print(os.path.dirname(path), end="")
        else:
            print(os.path.basename(path), end="")
        
    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores, avg_norm_scores = [], [], []
    
    replay_buffer = []

    avg_score = list()
    for no_episode in range(nb_episodes):
        # obs, info, action_id, reward
        episode = []
        obs, infos = env.reset()  # Start new episode.
        score = 0
        done = False
        nb_moves = 0
        max_score = 0
        while not done: 
            command, command_id  = agent.act(obs, score, done, infos)
            episode.append({'obs':obs, 'infos':infos, 'command_id':command_id})
            last_score = score
            obs, score, done, infos = env.step(command)
            max_score = max(max_score, score)
            reward = score-last_score
            if 'won' in infos:
                reward += 100
            elif 'lost' in infos:
                reward -= 100
            episode[-1]['reward'] = reward
            nb_moves += 1
        replay_buffer.append(episode)
        avg_score.append(max_score)
        agent.act(obs, score, done, infos)  # Let the agent know the game is done.
        print('ep:{}\tlen:{}\tmax_score:{}\tavg_score:{}\tlast_score:{}'.format(no_episode, len(episode), max_score, sum(avg_score)/len(avg_score), score))
        if (no_episode+1)%UPDATE_FREQ == 0:
            agent.apply_updates(replay_buffer)
            replay_buffer = []

        if verbose:
            print(".", end="")
        avg_moves.append(nb_moves)
        avg_scores.append(score)
        avg_norm_scores.append(score / infos["max_score"])

    env.close()
    msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
    if verbose:
        if os.path.isdir(path):
            print(msg.format(np.mean(avg_moves), np.mean(avg_norm_scores), 1))
        else:
            print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))
    

os.system(' seq 1 100 | xargs -n1 -P4 tw-make tw-simple --rewards dense --goal detailed --output training_games/ --seed')

from time import time
agent = NeuralAgent()

print("Training on 100 games")
agent.train()  # Tell the agent it should update its parameters.
starttime = time()
play(agent, "./training_games/", nb_episodes=100 * 20, verbose=False)  # Each game will be seen 5 times.
print("Trained in {:.2f} secs".format(time() - starttime))


os.system(' seq 1 20 | xargs -n1 -P4 tw-make tw-simple --rewards dense --goal detailed --test --output testing_games/ --seed')
agent.test()
play(agent, "./testing_games/", nb_episodes=20 * 10)  # Averaged over 10 playthroughs for each test game.

