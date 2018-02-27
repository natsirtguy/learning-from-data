import matplotlib.pyplot as plt
import numpy as np


def chord(x):
    """Return two points defining a chord of a radius two circle given two
    points collinear to that chord.

    """
    a = np.dot(x[1] - x[0], x[1] - x[0])
    b = 2 * np.dot(x[0], x[1] - x[0])
    c = np.dot(x[0], x[0]) - 4
    s1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    s2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    p = np.array([x[0] + s1 * (x[1] - x[0]), x[0] + s2 * (x[1] - x[0])])
    return p


def y(f, x):
    """Return 1 if x is on one side of the line defined by f, -1
    otherwise.
    """
    u = f[0] - f[1]
    v = f[0] - x
    return np.sign(np.cross(u, v))


def f():
    """Return a target function f define by two points in [1,1]^2."""
    return 2. * np.random.rand(2, 2) - 1


def yf(f):
    """Return a function showing whether a point is above, below f."""
    return lambda x: y(f, x)


def wf(w):
    """Return a function computing the dot product for given weights."""
    return lambda x: np.dot(np.hstack(([1], x)), w)


def train(yf, w, p):
    """Recursively adjust w for PCA. w, p are three component vectors, yf
    is the function to compute whether the point satisfies the target
    function.

    """
    hp = np.sign(np.dot(w, p))
    yp = yf(p[1:])
    if hp != yp:
        return train(yf, w + yp * p, p)
    else:
        return w


def is_done(yf, w, ps):
    """True if the weights w work correctly."""
    for p in ps:
        hp = np.sign(np.dot(p, w))
        yp = yf(p[1:])
        if hp != yp:
            return False
    return True


def pca(yf, w, xs):
    """Return the weights for the hypothesis produced by the Perceptron
    Learning Algorithm an array of points xs with target function f. w
    should have the the negative of the threshold as the first
    argument.

    """
    ps = np.hstack((np.ones((len(xs), 1)), xs))
    while not is_done(yf, w, ps):
        for p in ps:
            w = train(yf, w, p)
    return w


def plot_points(yf, xs, t):
    """Plot and color points based on the function yf."""
    for x in xs:
        c = 'g' if yf(x) > 0 else 'r'
        plt.plot(x[0], x[1], c + 'o')
    plt.title(t)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])


exf = f()
exyf = yf(exf)
exw = np.hstack(([0], f()[0]))
exxs = 2. * np.random.rand(100, 2) - 1

ch = chord(exf)

plt.subplot(121)
plt.plot(ch[:, 0], ch[:, 1])
plot_points(yf(exf), exxs, 'Target Function')
plt.gca().set_aspect(1)

plt.subplot(122)
plt.plot(ch[:, 0], ch[:, 1])
nw = pca(exyf, exw, exxs)
plot_points(wf(nw), exxs, 'Hypothesis')
plt.gca().set_aspect(1)

plt.show()

for _ in range(100):
    f = 2. * np.random.rand(2, 2) - 1
    p = chord(f)
    plt.plot(p[:, 0], p[:, 1])

# Display
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.gca().set_aspect(1)
plt.show()
