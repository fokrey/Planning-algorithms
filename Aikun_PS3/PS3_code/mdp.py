import numpy as np
from utils import action_space, transition_function, probabilistic_transition_function, state_consistency_check

def implement_mdp(env, goal, gamma=0.99, max_num_of_iterations=100):
    # Initialize the value function
    V = np.zeros(env.shape)
    V[goal] = 1  # Set the goal state value to 1 as it has a reward

    for iteration in range(max_num_of_iterations):
        new_V = V.copy()

        # Iterate over all states in the grid
        for row in range(V.shape[0]):
            for column in range(V.shape[1]):
                current_state = (row, column)

                # Initialize the new value as negative infinity
                new_value = -np.inf

                # Iterate over all possible actions in the action space
                for action in action_space:
                    # Get the probabilistic transition information for the current action
                    transition_states, transition_probs = probabilistic_transition_function(
                        env, current_state, action
                    )

                    rewards = []
                    transition_states_values = []

                    # Calculate rewards and expected values for each possible transition state
                    for state in transition_states:
                        if not state_consistency_check(env, state):
                            rewards.append(-1)
                            transition_states_values.append(0)
                        elif state == goal:
                            rewards.append(1)
                            transition_states_values.append(V[state])
                        else:
                            rewards.append(0)
                            transition_states_values.append(V[state])

                    # Calculate the expected reward and expected value for the current action
                    current_reward_expectation = np.sum(
                        np.array(transition_probs) * np.array(rewards)
                    )
                    transition_value_expectation = np.sum(
                        np.array(transition_probs) * np.array(transition_states_values)
                    )

                    # Update the new value based on the current action
                    new_value = max(
                        new_value,
                        current_reward_expectation
                        + gamma * transition_value_expectation,
                    )

                # Update the new value function for the current state
                new_V[current_state] = new_value

        # Check for convergence using a small tolerance level
        if np.allclose(V, new_V, atol=1e-3):
            print(f"Converged after {iteration + 1} iterations")
            break

        # Update the value function for the next iteration
        V = new_V

    # Set the value of the goal state to the maximum value in the final value function
    V[goal] = np.max(V)

    return V

def policy_mdp(env, V):
    policy = {}
    
    # Iterate over all states in the grid
    for row in range(V.shape[0]):
        for column in range(V.shape[1]):
            current_state = (row, column)
            
            # Calculate the values of transitioning to each neighboring state
            transition_state_values = np.array([
                V[transition_function(env, current_state, action)[0]]
                for action in action_space
            ])
            
            # Choose the action with the maximum expected value
            best_action = action_space[np.argmax(transition_state_values)]
            
            # Update the policy for the current state
            policy[current_state] = best_action
    
    return policy