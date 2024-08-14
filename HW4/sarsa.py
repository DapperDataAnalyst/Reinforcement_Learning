import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_actions: int,
                 num_tilings: int,
                 tile_width: np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.tile_offsets = state_low - (np.arange(num_tilings).reshape(-1, 1) / num_tilings) * tile_width
        self.num_tiles = np.ceil((state_high - state_low) / tile_width).astype(int) + 1 
           
    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions * self.num_tilings * np.prod(self.num_tiles)

    def __call__(self, s, done, a) -> np.array:
        if done:
            return np.zeros(self.feature_vector_len())

        # Initialize the feature vector
        feature_vector = np.zeros(self.feature_vector_len())

        for tiling in range(self.num_tilings):
            # Apply the tiling offset
            offset = self.tile_offsets[tiling]
            
            # Calculate tile indices for the current tiling
            # indices = np.floor((s - self.state_low + offset) / self.tile_width).astype(int)
            indices = np.floor((s - offset) / self.tile_width).astype(int)
            
            # Ensure indices are within bounds
            indices = np.clip(indices, 0, self.num_tiles - 1)
            
            # Flatten the multi-dimensional index to a single index
            flattened_index = np.ravel_multi_index(indices, self.num_tiles)
            
            # Calculate the index in the feature vector
            index = tiling * np.prod(self.num_tiles) + flattened_index + a * self.num_tilings * np.prod(self.num_tiles)
            
            # Set the corresponding entry in the feature vector to 1
            feature_vector[index] = 1

        return feature_vector

def SarsaLambda(
    env, # openai gym environment
    gamma: float, # discount factor
    lam: float, # decay rate
    alpha: float, # step size
    X: StateActionFeatureVectorWithTile,
    num_episode: int,
) -> np.array:
    """
    Implement True online Sarsa(λ)
    """

    def epsilon_greedy_policy(s, done, w, epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros(X.feature_vector_len())
    
    for episode in range(num_episode):
        s = env.reset()
        done = False
        a = epsilon_greedy_policy(s, done, w)
        x = X(s, done, a)
        z = np.zeros_like(w)
        q_old = 0

        while not done:
            s_next, reward, terminated, truncated = env.step(a)
            done = terminated or truncated  # Determine if the episode is done

            a_next = epsilon_greedy_policy(s_next, done, w)
            x_next = X(s_next, done, a_next)

            q = np.dot(w, x)
            q_next = np.dot(w, x_next) if not done else 0
            delta = reward + gamma * q_next - q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += alpha * (delta + q - q_old) * z - alpha * (q - q_old) * x
            q_old = q_next
            x = x_next
            a = a_next

    return w