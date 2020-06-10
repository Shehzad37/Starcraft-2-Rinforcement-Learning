# -*- coding: utf-8 -*-

import numpy as np


float_tolerance = 10e-6

def is_state_valid(new_state, world_shape, blocked_states):
            # Is new state valid?
            return (np.all([0, 0] <= new_state)
                and np.all(np.less(new_state, world_shape))
                # Is new state blocked_state
                and not np.any([ np.array_equal(new_state, blocked_state) 
                                for blocked_state in blocked_states]))

def get_reward(new_state, final_states, final_states_rewards, default_reward):
            for final_state_i, final_state in enumerate(final_states):
                if np.array_equal(final_state, new_state):
                    return final_states_rewards[final_state_i]
            return default_reward

def update_state_values(state_values, iterations, prim_action_prob,
                        sec_action_prob, default_reward, start_state,
                        world_shape, actions, blocked_states, final_states,
                        final_states_rewards, discount_factor):
    
    new_state_values = state_values.copy()
    for iteration in range(iterations):
        print("V^{}: \n".format(iteration), state_values)
        for row in range(world_shape[0]):
            for col in range(world_shape[1]):
                action_values = []
             
                if not [row, col] in final_states + blocked_states:
                    for prim_action_i, prim_action in enumerate(actions):
                        action_value = 0
                        
                        for sec_action_i, sec_action in enumerate(actions):
                            if sec_action_i == prim_action_i:
                                trans_prob = prim_action_prob
                            else:
                                trans_prob = sec_action_prob
                            
                            action = sec_action
                            new_state = list(np.array([row, col]) + action)
                        
                            if not is_state_valid(new_state, world_shape, blocked_states):
                                # Stay at old state
                                new_state = [row, col]
                            
                            reward = get_reward(new_state, final_states, final_states_rewards, default_reward)
                            action_value += trans_prob * (reward + discount_factor * state_values[tuple(new_state)])
                        
                        action_values.append(action_value)
                    
                    new_state_values[row, col] = max(action_values)
        
        state_values = new_state_values.copy()
    
    return state_values
        
        
if __name__ == '__main__':
    
    prim_action_prob = 0.7
    sec_action_prob = 0.1
    default_reward = -1
    
    start_state = [2,0]
    world_shape = [3,3]
    actions = [[-1,0],  # north,
               [0,1],   # east,
               [1,0],   # south,
               [0,-1]]  # west
    
    blocked_states = [[1,0],]
    final_states = [[0,0], [0,2]]
    final_states_rewards = [-100, 100]
    
    # Initial state values are set to zero
    state_values = np.zeros(world_shape, dtype=float)
    discount_factor = 0.95
    
    # First run; only 5 iterations
    iterations = 100
    
    state_values_1 = update_state_values(state_values, iterations, prim_action_prob,
                        sec_action_prob, default_reward, start_state,
                        world_shape, actions, blocked_states, final_states,
                        final_states_rewards, discount_factor)
    
    
    # First run; 100 iterations
    state_values = np.zeros(world_shape, dtype=float)
    iterations = 500
    state_values_2 = update_state_values(state_values, iterations, prim_action_prob,
                    sec_action_prob, default_reward, start_state,
                    world_shape, actions, blocked_states, final_states,
                    final_states_rewards, discount_factor)

    print("V^4 =\n", state_values_1)
    print("V^99 =\n", state_values_2)