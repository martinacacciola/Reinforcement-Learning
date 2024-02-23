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

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        T = len(states)
        G_t = np.zeros(T)
        for t in range(T-2,-1,-1):
            #G_t = sum((self.gamma**i)*rewards[i] for i in range(t,T))
            G_t[t] = self.gamma*G_t[t+1] + rewards[t]
            self.Q_sa[states[t],actions[t]] += self.learning_rate*(G_t[t] - self.Q_sa[states[t],actions[t]]) #indent?

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your Monte Carlo RL algorithm here!
    t=0
    #for t in range(n_timesteps):
    while t < n_timesteps:
        actions, rewards = [], []
        states = [env.reset()]

        # Collect an episode
        for i in range(max_episode_length):
            # Sample an action from the policy
            action = pi.select_action(states[i], policy, epsilon, temp)
            actions.append(action)
            # Simulate environment
            next_state, reward, done = env.step(actions[i])
            states.append(next_state)
            rewards.append(reward)
        
            # Evaluate the policy at regular intervals
            if t % eval_interval == 0:
                returns = pi.evaluate(eval_env) #chosen values?, n_eval_episodes=30, max_episode_length=100
                eval_returns.append(returns)
                eval_timesteps.append(t)

            #Update the iteration
            t += i

            if done: 
                break

        # Update the Q-value estimates
        pi.update(states, actions, rewards)

        #if plot:
            #env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

    #print(eval_returns)             
    return np.array(eval_returns), np.array(eval_timesteps) 
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    
            
if __name__ == '__main__':
    test()
