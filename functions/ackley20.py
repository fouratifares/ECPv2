import numpy as np

class Function:
    """
    High-dimensional shifted Ackley function (negated).
    Global optimum: at shift vector `s`, function value = 0.
    https://www.sfu.ca/~ssurjano/ackley.html
    """

    def __init__(self, dimensions: int = 20, shift: np.ndarray = None):
        self.dimensions = dimensions
        self.bounds = np.array([(-10, 10)] * dimensions)
        self.counter = 0

        # Define shift vector (location of the optimum)
        self.shift = shift if shift is not None else np.random.uniform(-3, 3, size=dimensions)
        self.optimal_point = self.shift
        self.optimal_value = 0  # Ackley has global minimum at 0

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        if len(x) != self.dimensions:
            raise ValueError(f"Input must have {self.dimensions} dimensions.")

        z = x - self.shift  # Apply shift

        sum_sq = np.sum(z ** 2)
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / self.dimensions))
        term2 = -np.exp(sum_cos / self.dimensions)

        value = term1 + term2 + 20 + np.e
        self.counter += 1

        return -value  # Negated for maximization
