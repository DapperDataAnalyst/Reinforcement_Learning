import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self, state_low: np.array, state_high: np.array, num_tilings: int, tile_width: np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.tile_offsets = state_low - (np.arange(num_tilings).reshape(-1, 1) / num_tilings) * tile_width
        self.num_tiles = np.ceil((state_high - state_low) / tile_width).astype(int) + 1
        self.weights = np.zeros((self.num_tilings, *self.num_tiles))
    
    def get_tile_indices(self, state):
        tile_indices = []
        for tiling in range(self.num_tilings):
            offset = self.tile_offsets[tiling]
            indices = np.floor((state - offset) / self.tile_width).astype(int)
            # Ensure indices are within bounds
            indices = np.clip(indices, 0, self.num_tiles - 1)
            tile_indices.append(indices)
        return tile_indices
    
    def __call__(self, s):
        indices = self.get_tile_indices(s)
        value = 0
        for tiling, index in enumerate(indices):
            index = (tiling,) + tuple(index)
            value += self.weights[index]
        return value

    def update(self, alpha, G, s_tau):
        indices = self.get_tile_indices(s_tau)
        value = self(s_tau)
        for tiling, index in enumerate(indices):
            index = (tiling,) + tuple(index)
            self.weights[index] += alpha * (G - value)
