# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 00:26:47 2020

@author: Abdelhamid 
"""
import gym
from Agent import Agent
from plot import plot_epi_step

if __name__ == '__main__':
    
    env     = gym.make('CartPole-v1')
    agent   = Agent(lr=10**-4, n_actions=env.action_space.n, input_dim=env.observation_space.shape, gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_min=0.01, max_iterations=10000, lamda=0.9)
    n_games = 10000
    
    scores = []
    steps  = []
    n = 5
    for i in range(n_games):
        
        score, cont = agent.learn(env)
            
        scores.append(score)
        steps.append(cont)
        print("############## episode number = {} ######### number of steps = {} ############ score {:.4f}".format(i, cont, score))
        
    plot_epi_step(scores,steps)