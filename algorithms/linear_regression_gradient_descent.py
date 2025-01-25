import jax
from jax import numpy as jnp


class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the class with the hyperparameters:
        - learning_rate: the learning rate for gradient descent
        - num_iterations: the number of iterations to be performed
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def y_hat(self, X, w):
        """
        Calculates the prediction (estimated values) y_hat.
        - X: Feature matrix.
        - w: Weight vector.
        """
        return jnp.dot(w.T, X)

    def cost(self, y, y_hat):
        """
        Cost function (mean square error).
        - y : Real values (ground truth).
        - y_hat: Estimated values (predictions).
        """
        return 1 / self.m * jnp.sum(jnp.square(y - y_hat))

    def gradient(self, w, X, y, y_hat):
        """
        Calculates the gradient and updates the weights.
        - w: Weight vector.
        - X: Feature matrix.
        - y : Actual values.
        - y_hat: Predicted values.
        """
        # Calculating the gradient
        grad = (2 / self.m) * jnp.dot(X, (y_hat - y).T)

        # Updating weights
        w = w - self.learning_rate * grad
        return w

    def main(self, X, y):
        """
        Trains the model using gradient descent.
        - X: Characteristics matrix (unbiased).
        - y : Actual values.
        """
        # Add a bias line to X (column of 1).
        ones = jnp.ones((1, X.shape[1]))
        X = jnp.vstack((ones, X))  # Matrice étendue, incluant le biais.

        # Dimensions
        self.m = X.shape[1]  # Number of examples.
        self.n = X.shape[0]  # Number of characteristics (with bias).

        # Weight initialisation.
        w = jnp.zeros((X.shape[0], 1))

        # Main loop for gradient descent.
        for i in range(self.num_iterations + 1):
            y_hat = self.y_hat(X, w)  # Current predictions.
            C = self.cost(y, y_hat)  # Calculation of current cost.
            w = self.gradient(w, X, y, y_hat)  # Update weights.

            # Displays the cost every 100 iterations.
            if i % 100 == 0:
                print(f"Cost per iteration {i}: {C:.4f}")
        return w


if __name__ == "__main__":
    # Generating random data for training
    key = jax.random.PRNGKey(0)  # JAX pseudo-random key.
    X = jax.random.normal(key, (1, 100))  # 100 examples ( 1 dimension).
    y = 3 * X + 2 + jax.random.normal(key, (1, 100)) * 0.1  # y = 3X + 2 + bias.

    # Instantiating and training the model.
    regression = LinearRegressionGradientDescent(
        learning_rate=0.01, num_iterations=1000
    )
    w = regression.main(X, y)

    # Affichage du résultat final : les poids appris.
    print("Final weights (w) :")
    print(w)
