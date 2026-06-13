# A Q-Learning Implementation for Process Optimization

# Importing the libraries
import numpy as np

# Setting the parameters gamma and alpha for the Q-Learning
gamma = 0.75
alpha = 0.9

# PART 1 - DEFINING THE ENVIRONMENT

# Defining the states
location_to_state = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
                     'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}

# Defining the actions
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

# Defining the rewards matrix
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])

# PART 2 - BUILDING THE AI SOLUTION WITH Q-LEARNING

# Making a mapping from the states to the locations
state_to_location = {state: location for location, state in location_to_state.items()}

# Making a function that returns the shortest route from a starting to ending location
def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    
    # Assigning a high reward to the destination state transition
    R_new[ending_state, ending_state] = 1000
    
    # Initializing the Q-Table with zeros
    Q = np.zeros([12, 12])
    
    # Training the AI over 5000 episodes for solid convergence
    for i in range(5000):
        current_state = np.random.randint(0, 12)
        
        # Identify actions with positive reward structures in the environment
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
                
        # If no valid positive reward action exists, fallback to all physical pathways
        if len(playable_actions) == 0:
            playable_actions = [j for j in range(12) if R[current_state, j] > 0 or current_state == ending_state]
            
        next_state = np.random.choice(playable_actions)
        
        # Temporal Difference (TD) Calculation
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        
        # Updates the Q-Table
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
    
    # Reconstructing the optimal path found by the Q-Table
    route_list = [starting_location]
    current_location = starting_location
    
    # Safety iteration tracker to completely eliminate hanging infinite loops
    max_steps = 50
    steps = 0
    
    while current_location != ending_location and steps < max_steps:
        current_state = location_to_state[current_location]
        next_state = np.argmax(Q[current_state,])
        next_location = state_to_location[next_state]
        
        route_list.append(next_location)
        current_location = next_location
        steps += 1
        
    return route_list

# PART 3 - GOING INTO PRODUCTION

# Making the final function that returns the optimal route passing through an intermediary stop
def best_route(starting_location, intermediary_location, ending_location):
    first_leg = route(starting_location, intermediary_location)
    second_leg = route(intermediary_location, ending_location)
    
    # Combined legs, dropping the duplicate index at the intersection point
    return first_leg + second_leg[1:]

# Printing the final route
print('Route:')
print(best_route('E', 'K', 'G'))