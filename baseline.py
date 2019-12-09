import os
from glob import glob

import gym
import textworld.gym
from neural_agent_baseline import NeuralAgent
import numpy as np

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
    avg_score = list()
    for no_episode in range(nb_episodes):
        obs, infos = env.reset()  # Start new episode.

        score = 0
        done = False
        nb_moves = 0
        max_score = 0
        while not done:
            command = agent.act(obs, score, done, infos)
            obs, score, done, infos = env.step(command)
            max_score = max(max_score, score)
            nb_moves += 1
        avg_score.append(max_score)
        agent.act(obs, score, done, infos)  # Let the agent know the game is done.
        print('ep:{}\tlen:{}\tmax_score:{}\tavg_score:{}\tlast_score:{}'.format(no_episode, nb_moves, max_score, sum(avg_score)/len(avg_score), score))
        
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


agent = NeuralAgent()
play(agent, "./games/rewardsDense_goalDetailed.ulx")


from time import time
agent = NeuralAgent()

print("Training")
agent.train()  # Tell the agent it should update its parameters.
starttime = time()
play(agent, "./games/rewardsDense_goalDetailed.ulx", nb_episodes=500, verbose=False)  # Dense rewards game.
print("Trained in {:.2f} secs".format(time() - starttime))

agent.test()
play(agent, "./games/rewardsDense_goalDetailed.ulx")

os.system("seq 1 100 | xargs -n1 -P4 tw-make tw-simple --rewards dense --goal detailed --output training_games/ --seed")

from time import time
agent = NeuralAgent()

print("Training on 100 games")
agent.train()  # Tell the agent it should update its parameters.
starttime = time()
play(agent, "./training_games/", nb_episodes=100 * 5, verbose=False)  # Each game will be seen 5 times.
print("Trained in {:.2f} secs".format(time() - starttime))

os.system("seq 1 20 | xargs -n1 -P4 tw-make tw-simple --rewards dense --goal detailed --test --output testing_games/ --seed")
agent.test()
play(agent, "./games/rewardsDense_goalDetailed.ulx")  # Averaged over 10 playthroughs.
play(agent, "./testing_games/", nb_episodes=20 * 10)  # Averaged over 10 playthroughs for each test game.
play(RandomAgent(), "./testing_games/", nb_episodes=20 * 10)