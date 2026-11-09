import numpy as np
from numpy.typing import NDArray


class Solution:
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        def sig(arr):
            return 1 / (1 + np.exp(-arr))
        z = sig(z)
        return np.round(z, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        return np.maximum(0, z)

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        predicted_val = np.dot(x, w) + b
        activated_val = 0
        if activation == "sigmoid":
            activated_val = 1 / (1 + np.exp(-predicted_val))
        elif activation == "relu":
            activated_val = max(0.0, predicted_val)
        else:
            raise Exception("Unknown activation fn")
        return np.round(activated_val, 5)

        
