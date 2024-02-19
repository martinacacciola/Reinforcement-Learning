#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax
#aggiungo
#import matplotlib.pyplot as plt

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
         # Greedy policy: π(s) = arg max_a Q(s, a)
        a = argmax(self.Q_sa[s])
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # Q-iteration update: Q(s, a) = Σ_s' P(s' | s, a) * [R(s, a, s') + γ * max_a' Q(s', a')]
        updated_Q = 0.0
        for s_next in range(self.n_states):
            updated_Q += p_sas[s_next] * (r_sas[s_next] + self.gamma * np.max(self.Q_sa[s_next]))
        self.Q_sa[s,a] = updated_Q
        return updated_Q
    
""" #AGGIUNGO: to show progression of Q-value iteration
def plot_q_values(Q_sa, iteration):
    plt.figure(figsize=(10, 6))
    plt.imshow(Q_sa, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f'Q-value Iteration - Iteration {iteration}')
    plt.xlabel('Action')
    plt.ylabel('State')
    plt.show() """

      
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
 
     # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
        
    # Plot current Q-value estimates & print max error
    # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    # print("Q-value iteration, iteration {}, max error {}".format(i,max_error))

    max_error = float('inf')
    iteration = 0

    while max_error > threshold:
        #Inizialize
        max_error = 0
        #Loop over all states
        for s in range(env.n_states):
            for a in range(env.n_actions):
                #Get model probabilities and rewards
                p_sas, r_sas = env.model(s,a)
                #Store current Q value for the state-action pair
                old_q_value = QIagent.Q_sa[s,a]
                #Update Q-value
                QIagent.update(s,a,p_sas,r_sas)
                #Update max_error 
                max_error = max(max_error, abs(QIagent.Q_sa[s,a] - old_q_value))
        #Plot current Q-value estimates - !uncommenta per il plot
        #env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
     
        #Print max abs error after each full sweep
        print("Iteration: {}, max_error: {}".format(iteration, max_error))
        iteration += 1
 
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # view optimal policy
    done = False
    s = env.reset()
    tot_reward = 0
    tot_timesteps = 0
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        tot_reward += r
        tot_timesteps += 1
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next

    # TO DO: Compute mean reward per timestep under the optimal policy
    avg_timesteps_to_goal=tot_timesteps/env.goal_rewards[0] #If we assume only one goal
    opt_value_start =np.max(QIagent.Q_sa[env.start_location])
    mean_reward_per_timestep = tot_reward / tot_timesteps
    print("Optimal value at the start state: {}".format(opt_value_start))
    print("Average timesteps to goal: {}".format(avg_timesteps_to_goal))
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    print("Optimal episode return: {}".format(tot_reward))
    return avg_timesteps_to_goal, opt_value_start, mean_reward_per_timestep
    
if __name__ == '__main__':
    experiment()
    
