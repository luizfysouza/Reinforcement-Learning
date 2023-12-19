#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 26, 2021
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from scipy import sparse

from agent import Agent


class PolicyEvaluation:
    """Evaluation of a policy by dynamic programming.
    
    Parameters
    ----------
    model: object of class Environment
        The model.
    policy: function
        Policy of the agent.
    player: int
        Player for games (1 or -1, default = player of the model).
    gamma: float
        Discount factor (between 0 and 1).
    n_iter: int
        Number of iterations of Bellman's equation.
    """
    
    def __init__(self, model, policy='random', player=None, gamma=1, n_iter=100):
        self.model = model
        agent = Agent(model, policy)
        self.policy = agent.policy
        if player is None:
            if model.is_game():
                self.player = model.player
            else:
                self.player = 1
        else:
            self.player = player
        self.gamma = gamma
        self.n_iter = n_iter
        self.get_states()
        self.get_rewards()
        self.get_transition_matrix()
        self.get_terminal_states()
        
    def get_states(self):
        """Index all states."""
        self.states = self.model.get_states()
        self.n_states = len(self.states)
        self.state_id = {self.model.encode(state): i for i, state in enumerate(self.states)}
        
    def get_state_id(self, state):
        return self.state_id[self.model.encode(state)]

    def get_rewards(self):
        """Get the reward of each state."""
        rewards = np.zeros(self.n_states)
        for state in self.states:    
            i = self.state_id[self.model.encode(state)]  
            rewards[i] = self.model.get_reward(state)
        self.rewards = rewards
    
    def get_transition_matrix(self):
        """Get the transition matrix (probability of moving from one state to another) in sparse format."""
        transition = sparse.lil_matrix((self.n_states, self.n_states))
        for state in self.states:    
            i = self.state_id[self.model.encode(state)]
            if not self.model.is_terminal(state):
                for prob, action in zip(*self.policy(state)):
                    probs, states = self.model.get_transition(state, action)
                    indices = np.array([self.state_id[self.model.encode(s)] for s in states])
                    transition[i, indices] += prob * np.array(probs)
        self.transition = sparse.csr_matrix(transition)
    
    def get_terminal_states(self):
        """Get terminal states as a boolean mask."""
        self.terminal = np.array([self.model.is_terminal(state) for state in self.states])
    
    def get_values(self):
        """Evaluate a policy by iteration of Bellman's equation."""
        transition = self.transition
        values = np.zeros(self.n_states)
        rewards = self.rewards
        for t in range(self.n_iter):
            values = transition.dot_max(rewards + self.gamma * values)
            values[self.terminal] = 0
        self.values = values
        return values
        
    def evaluate_policy(self):
        """Evaluate a policy by iteration of Bellman's equation. Update the transition matrix first."""
        self.get_transition_matrix()
        return self.get_values()
    
    def improve_policy(self):
        """Improve the policy based on the value function."""
        best_actions = dict()
        for state in self.states: 
            i = self.state_id[self.model.encode(state)]
            player, _ = state
            if self.model.is_game() and player != self.player:
                best_actions[i] = None
            else:
                actions = self.model.get_actions(state)
                values_actions = []
                for action in actions:
                    probs, states = self.model.get_transition(state, action)
                    indices = np.array([self.state_id[self.model.encode(s)] for s in states])
                    value = np.sum(np.array(probs) * (self.rewards + self.gamma * self.values)[indices])
                    values_actions.append(value)
                if len(values_actions):
                    values = self.player * np.array(values_actions)
                    top_actions = np.flatnonzero(values == values.max())
                    best_actions[i] = actions[np.random.choice(top_actions)]
                else:
                    best_actions[i] = None
        policy = lambda state: [[1], [best_actions[self.state_id[self.model.encode(state)]]]]
        return policy

    @staticmethod
    def get_action(policy, state):
        """Action for a deterministic policy."""
        probs, actions = policy(state)
        return actions[0]        
    
    def is_same_policy(self, policy):
        """Test if the policy has changed."""
        for state in self.states:
            if self.get_action(policy, state) != self.get_action(self.policy, state):
                return False
        return True

    
class PolicyIteration(PolicyEvaluation):
    """Policy iteration.
    
    Parameters
    ----------
    model: object of class Environment
        The model.
    player: int 
        Player for games (1 or -1, default = 1).
    gamma: float
        Discount factor (between 0 and 1).
    n_iter_eval: int
        Number of iterations of Bellman's equation for policy evaluation.
    n_iter: int
        Maximum number of policy iterations.
    """
    
    def __init__(self, model, player=1, gamma=1, n_iter_eval=100, n_iter=100, verbose=True):
        agent = Agent(model)
        policy = agent.policy
        self.n_iter = n_iter
        self.verbose = verbose
        super(PolicyIteration, self).__init__(model, policy, player, gamma, n_iter_eval)   
    
    def get_optimal_policy(self):
        """Iterate evaluation and improvement, stop if no change."""
        for t in range(self.n_iter):
            self.evaluate_policy() 
            policy = self.improve_policy()
            if self.is_same_policy(policy):
                if self.verbose:
                    print(f"Convergence after {t} iterations.")
                break
            self.policy = policy
        return policy
    

def dot_max(matrix: sparse.csr_matrix, vector: np.ndarray):
    """Get the dot-max product of a sparse matrix by a vector, replacing the sum by the max."""
    return np.maximum.reduceat(vector[matrix.indices] * matrix.data, matrix.indptr[:-1])

class ValueIteration(PolicyEvaluation):
    """Value iteration.
    
    Parameters
    ----------
    model: object of class Environment
        The model.
    player: int
        Player for games (1 or -1, default = player given by the model).
    gamma: float
        Discount factor (between 0 and 1).
    n_iter: int
        Number of value iterations.
    tol: float
        Tolerance = maximum difference between two iterations for early stopping.
    """
    
    def __init__(self, model, player=None, gamma=1, n_iter=100, tol=0, verbose=True):
        agent = Agent(model, player=player)
        policy = agent.policy
        super(ValueIteration, self).__init__(model, policy, player, gamma)  
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        
    def get_optimal_policy(self):
        """Get the optimal policy by iteration of Bellman's optimality equation."""
        if self.model.is_game():
            return self.get_optimal_policy_game()
        self.values = np.zeros(self.n_states)
        transition = self.transition.astype(bool)
        for t in range(self.n_iter):
            
            # to be modified
            # check the function dot_max above
            # ---
            values = self.values.copy()
            # ----
            
            diff = np.max(np.abs(values - self.values))
            self.values = values
            if diff <= self.tol:
                if self.verbose:
                    print(f"Convergence after {t+1} iterations.")
                break
        policy = self.improve_policy()
        return policy
    
    def get_optimal_policy_game(self):
        """Get the optimal policy for games, assuming the best response of the adversary."""
        self.values = np.zeros(self.n_states)
        transition = self.transition.astype(bool)
        mask_player = np.array([player==self.player for player, _ in self.states])
        mask_adversary = ~mask_player
        mask_player &= ~self.terminal
        mask_adversary &= ~self.terminal
        for t in range(self.n_iter):
            
            # to be modified
            # check the functions dot_max and dot_min above
            # ---
            values = self.values.copy()
            # ----            
            
            diff = np.max(np.abs(values - self.values))
            self.values = values
            if diff <= self.tol:
                if self.verbose:
                    print(f"Convergence after {t+1} iterations.")
                break
        policy = self.improve_policy()
        return policy