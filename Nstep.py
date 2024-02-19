#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        T = len(actions)
        self.n = n
        for t in range(T):
            G = sum([self.gamma**i * rewards[t+i] for i in range(min(self.n, T-t))])
            if t+self.n < T:
                G += (self.gamma**self.n) * np.max(self.Q_sa[states[t+self.n]])
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G - self.Q_sa[states[t], actions[t]])

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your n-step Q-learning algorithm here!
    for t in range(n_timesteps):
        states, actions, rewards, done = [], [], [], False
        state = env.reset()

        # Collect an episode
        for t in range(n_timesteps):
            action = pi.select_action(state, policy, epsilon, temp)
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if done:
                break
        
        # Update Q-values using n-step Q-learning
        pi.update(states, actions, rewards, done, n)

        # Evaluate the policy at regular intervals
        if t % eval_interval == 0:
            returns = pi.evaluate(eval_env, n_eval_episodes=30, max_episode_length=100) #chose the values?
            eval_returns.append(returns)
            eval_timesteps.append(t)
        
        if plot: #comment to not plot
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
        
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()
