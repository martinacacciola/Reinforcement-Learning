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
        T = len(rewards)
        #self.n = n
        for t in range(T):
            # Number of rewards left to sum
            m = min(n, T-t)
            # If state t+m is terminal
            if t+m == T and done:
            # N-step target without bootstrap 
                G = np.sum([self.gamma**i * rewards[t+i] for i in range(m)])
            else:   # N-step target
                G = np.sum([self.gamma**i * rewards[t+i] + self.gamma**m * np.max(self.Q_sa[states[t+m]]) for i in range(m)]) # N-step target

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
    t=0
    #for t in range(n_timesteps):
    while t < n_timesteps:
    #while n_timesteps != 0:
        actions, rewards, done =  [], [], False
        states = [env.reset()]

        # Collect an episode
        for i in range(max_episode_length): # Changed t in _
            actions.append(pi.select_action(states[i], policy, epsilon, temp))
            next_state, reward, done = env.step(actions[i])
            states.append(next_state)
            rewards.append(reward)


            # Evaluate the policy at regular intervals
            if t % eval_interval == 0:
            #if n_timesteps % eval_interval == 0:
                returns = pi.evaluate(eval_env) #chose the values?, n_eval_episodes=30, max_episode_length=100
                eval_returns.append(returns)
                eval_timesteps.append(t)

            t += 1 # Update timesteps
            # n_timesteps -= 1
            # if n_timesteps == t:
            #if n_timesteps == 0:
            if done:
                break

        # Update Q-values using n-step Q-learning
        pi.update(states, actions, rewards, done, n)

        
        # eval_timesteps =eval_timesteps[::-1]
        #if plot: #comment to not plot
            #env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
    #print(eval_returns)   
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 50000
    max_episode_length = 100
    gamma = 1.0 #it was 1.0
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