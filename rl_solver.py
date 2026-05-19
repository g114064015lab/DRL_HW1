import random

def evaluate_policy(n, start_id, end_id, obstacle_ids, gamma=1.0, theta=1e-4):
    """
    Computes a random policy and its state values.
    Grid cells are 1-indexed.
    """
    states = list(range(1, n * n + 1))
    
    # 0 = Up, 1 = Right, 2 = Down, 3 = Left
    actions = [0, 1, 2, 3] 
    
    # Initialize random policy
    policy = {}
    for s in states:
        if s == end_id or s in obstacle_ids:
            policy[s] = [] # No actions for terminal or obstacles
        else:
            # Random probabilities, normalized
            probs = [random.random() for _ in actions]
            total = sum(probs)
            probs = [p / total for p in probs]
            policy[s] = probs
            
    # Initialize value function
    V = {s: 0.0 for s in states}
    
    def get_next_state(s, a):
        # Calculate row and column (0-indexed)
        row = (s - 1) // n
        col = (s - 1) % n
        
        next_row, next_col = row, col
        
        if a == 0:   # Up
            next_row -= 1
        elif a == 1: # Right
            next_col += 1
        elif a == 2: # Down
            next_row += 1
        elif a == 3: # Left
            next_col -= 1
            
        # Check bounds
        if next_row < 0 or next_row >= n or next_col < 0 or next_col >= n:
            return s # Bounce off wall
            
        next_s = next_row * n + next_col + 1
        
        # Check obstacles
        if next_s in obstacle_ids:
            return s # Bounce off obstacle
            
        return next_s

    # Iterative Policy Evaluation
    while True:
        delta = 0
        new_V = V.copy()
        
        for s in states:
            if s == end_id or s in obstacle_ids:
                continue
                
            v = 0
            for a_idx, a in enumerate(actions):
                prob = policy[s][a_idx]
                next_s = get_next_state(s, a)
                
                # Reward is -1 for all transitions except when already in terminal state
                reward = -1 
                
                v += prob * (reward + gamma * V[next_s])
                
            new_V[s] = v
            delta = max(delta, abs(v - V[s]))
            
        V = new_V
        if delta < theta:
            break
            
    # Format output matrices to match the desired 2D structure
    value_matrix = []
    policy_matrix = []
    
    arrows = ['↑', '→', '↓', '←']
    
    for row in range(n):
        v_row = []
        p_row = []
        for col in range(n):
            s = row * n + col + 1
            if s == end_id:
                v_row.append(0.0)
                p_row.append('END')
            elif s in obstacle_ids:
                v_row.append(None) # None indicates obstacle
                p_row.append('OBS')
            else:
                v_row.append(round(V[s], 2))
                
                # For visualization, find the action with highest probability or just show random if it's completely uniform.
                # To make it interesting as a "random policy", we will just show the arrow with max probability, 
                # or a mix. The requirement says "random generated actions (arrows) as policy".
                # To match the image, we will pick the max probability action to display.
                best_a = max(range(4), key=lambda i: policy[s][i])
                p_row.append(arrows[best_a])
                
        value_matrix.append(v_row)
        policy_matrix.append(p_row)
        
    return {
        "value_matrix": value_matrix,
        "policy_matrix": policy_matrix
    }

def value_iteration(n, start_id, end_id, obstacle_ids, gamma=1.0, theta=1e-4):
    """
    Computes the optimal policy and state values using Value Iteration.
    Grid cells are 1-indexed.
    """
    states = list(range(1, n * n + 1))
    
    # 0 = Up, 1 = Right, 2 = Down, 3 = Left
    actions = [0, 1, 2, 3] 
    
    # Initialize value function
    V = {s: 0.0 for s in states}
    
    def get_next_state(s, a):
        # Calculate row and column (0-indexed)
        row = (s - 1) // n
        col = (s - 1) % n
        
        next_row, next_col = row, col
        
        if a == 0:   # Up
            next_row -= 1
        elif a == 1: # Right
            next_col += 1
        elif a == 2: # Down
            next_row += 1
        elif a == 3: # Left
            next_col -= 1
            
        # Check bounds
        if next_row < 0 or next_row >= n or next_col < 0 or next_col >= n:
            return s # Bounce off wall
            
        next_s = next_row * n + next_col + 1
        
        # Check obstacles
        if next_s in obstacle_ids:
            return s # Bounce off obstacle
            
        return next_s

    # Value Iteration
    while True:
        delta = 0
        new_V = V.copy()
        
        for s in states:
            if s == end_id or s in obstacle_ids:
                continue
                
            max_v = float('-inf')
            for a in actions:
                next_s = get_next_state(s, a)
                reward = -1 
                v = reward + gamma * V[next_s]
                if v > max_v:
                    max_v = v
                    
            new_V[s] = max_v
            delta = max(delta, abs(max_v - V[s]))
            
        V = new_V
        if delta < theta:
            break
            
    # Extract Optimal Policy
    policy = {}
    for s in states:
        if s == end_id or s in obstacle_ids:
            policy[s] = None
            continue
            
        max_v = float('-inf')
        best_a = actions[0]
        for a in actions:
            next_s = get_next_state(s, a)
            reward = -1
            v = reward + gamma * V[next_s]
            if v > max_v:
                max_v = v
                best_a = a
        policy[s] = best_a
        
    # Generate optimal path
    optimal_path = [start_id]
    curr_s = start_id
    max_steps = n * n # to avoid infinite loops if trapped
    steps = 0
    while curr_s != end_id and steps < max_steps:
        a = policy[curr_s]
        if a is None: 
            break
        next_s = get_next_state(curr_s, a)
        if next_s in optimal_path:
            optimal_path.append(next_s)
            break
        optimal_path.append(next_s)
        curr_s = next_s
        steps += 1
        
    # Format output matrices
    value_matrix = []
    policy_matrix = []
    
    arrows = ['↑', '→', '↓', '←']
    
    for row in range(n):
        v_row = []
        p_row = []
        for col in range(n):
            s = row * n + col + 1
            if s == end_id:
                v_row.append(0.0)
                p_row.append('END')
            elif s in obstacle_ids:
                v_row.append(None) 
                p_row.append('OBS')
            else:
                v_row.append(round(V[s], 2))
                p_row.append(arrows[policy[s]])
                
        value_matrix.append(v_row)
        policy_matrix.append(p_row)
        
    return {
        "value_matrix": value_matrix,
        "policy_matrix": policy_matrix,
        "optimal_path": optimal_path
    }
