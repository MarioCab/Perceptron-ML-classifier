from sklearn.datasets import load_breast_cancer
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

scaler = StandardScaler()
df = load_breast_cancer(as_frame=True).frame
sns.heatmap(df.corr(), annot=False, cmap="viridis")
df.describe()
plt.show(block=True)


class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    random_state : int
    Random number generator seed for random weight
    initialization.

    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    b_ : Scalar
    Bias unit after fitting.
    errors_ : list
    Number of misclassifications (updates) in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the number of
        examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
        Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ("o", "s", "^", "v", "<")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f"Class {cl}",
            edgecolor="black",
        )


def run_tests():
    # plot data
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="o", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", marker="s", label="Class 1")
    plt.xlabel(df.columns[training_features[0]])
    plt.ylabel(df.columns[training_features[1]])
    plt.legend(loc="upper left")
    plt.show(block=True)

    # Track miscalculations to observe the algorithms convergence behavior
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Number of updates")
    plt.show(block=True)

    # Visualize the decision boundary
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel(df.columns[training_features[0]])
    plt.ylabel(df.columns[training_features[1]])
    plt.legend(loc="upper left")
    plt.show(block=True)

    # Calculate accuracy
    accuracy = np.mean(ppn.predict(X) == y)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")


def run_std_tests():
    # Standardize the selected features
    X_std = scaler.fit_transform(X)

    # plot standard data
    plt.scatter(
        X_std[y == 0, 0], X_std[y == 0, 1], color="red", marker="o", label="Class 0"
    )
    plt.scatter(
        X_std[y == 1, 0], X_std[y == 1, 1], color="blue", marker="s", label="Class 1"
    )
    plt.xlabel(df.columns[training_features[0]])
    plt.ylabel(df.columns[training_features[1]])
    plt.legend(loc="upper left")
    plt.show(block=True)

    # Track miscalculations to observe the algorithms convergence behavior with standard data
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X_std, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Number of updates")
    plt.show(block=True)

    # Visualize the decision boundary with standardized data
    plot_decision_regions(X_std, y, classifier=ppn)
    plt.xlabel(df.columns[training_features[0]])
    plt.ylabel(df.columns[training_features[1]])
    plt.legend(loc="upper left")
    plt.show(block=True)

    # Calculate accuracy of the standardized data
    accuracy = np.mean(ppn.predict(X_std) == y)
    print(f"Standardized Model Accuracy: {accuracy * 100:.2f}%")


################################### Example 1 ###################################
# select target value
y = df["target"].values

# select features to use in training
training_features = [24, 22]

# select training data
X = df.iloc[0:569, training_features].values

# run tests
run_tests()  # Accuracy: 37.26%
run_std_tests()  # Accuracy: 92.79%

################################### Example 2 ###################################
# select target value
y = df["target"].values

# select features to use in training
training_features = [12, 29]

# select training data
X = df.iloc[0:569, training_features].values

# run tests
run_tests()  # Accuracy: 79.61%
run_std_tests()  # Accuracy: 78.91%

################################### Example 3 ###################################
# select target value
y = df["target"].values

# select features to use in training
training_features = [4, 10]

# select training data
X = df.iloc[0:569, training_features].values

# run tests
run_tests()  # Accuracy: 81.02%
run_std_tests()  # Accuracy: 62.57%
