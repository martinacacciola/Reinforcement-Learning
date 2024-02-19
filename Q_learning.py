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

class QLearningAgent(BaseAgent):
        
    #def update(self,s,a,r,s_next,done):
        # TO DO: Add own code
        #if done:
          #  target = r
        #else:
            #max_next_Q = np.max(self.Q_sa[s_next]) #if not done else 0
            #target = r + self.gamma * max_next_Q
            #self.Q_sa[s,a] += self.learning_rate * (target - self.Q_sa[s,a])

    def update(self, s, a, r, s_next, done):
        max_q_next = np.max(self.Q_sa[s_next]) if not done else 0
        target = r + self.gamma * max_q_next
        self.Q_sa[s, a] += self.learning_rate * (target - self.Q_sa[s, a])
        

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    # TO DO: Write your Q-learning algorithm here!
    rewards = []
    s = env.reset()

    for timestep in range(n_timesteps):
        tot_reward = 0

        #s =s
        # Select action using the specified policy
        a = agent.select_action(s,policy,epsilon,temp)
        # Take action a, observe next state and reward
        s_next,r,done = env.step(a)
        # Update Q-value estimate
        agent.update(s,a,r,s_next,done)
        # Update reward and move to next state
        tot_reward += r
        if done:
            s = env.reset()
        else:
            s = s_next

        rewards.append(tot_reward)


        # Evaluate the agent every eval_interval timesteps
        if timestep % eval_interval == 0:
            eval_return = agent.evaluate(eval_env)
            eval_returns.append(eval_return)
            eval_timesteps.append(timestep)
            
        
    
        if plot:
            # Plot the Q-value estimates during Q-learning execution
            env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1) 

    return np.array(eval_returns), np.array(eval_timesteps)  #non sono certa dei valori timesteps 


def test():
    
    n_timesteps = 1000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()