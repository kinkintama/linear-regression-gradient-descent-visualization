import numpy as np
import matplotlib.pyplot as plt

def generate_random_points(n):
    """
    Generates random points.
    :param n: number of points
    :return: x and y values
    """
    x = np.random.randint(0, 10, n)
    y = np.random.randint(0, 10, n)
    return x, y


def plot_gradient_descent(ys, xs, ms, cs, losses):
    """
    Plots the gradient descent.
    :param ys: y values
    :param xs: x values
    :param ms: slopes of the regression line
    :param cs: intercepts of the regression line
    :param losses: mean squared errors
    :return: None
    """

    random_length = 30
    r_ms = np.outer(np.linspace(0, 1, random_length), np.ones(random_length))
    r_cs =  np.outer(np.linspace(-2, 2, random_length), np.ones(random_length)).T
    r_losses = np.zeros(r_ms.shape)

    for i in range(random_length):
        for j in range(random_length):
            r_m = r_ms[i, j]
            r_c = r_cs[i, j]
            r_losses[i, j] = mean_squared_error(ys, (r_m*xs+r_c))

    plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.title.set_text('Gradient Descent')
    ax.set_xlabel('m')
    ax.set_ylabel('c')
    ax.set_zlabel('loss')
    ax.plot_surface(r_ms, r_cs, r_losses, cmap='viridis', edgecolor='none', alpha=0.5)
    ax.plot3D(ms, cs, losses, color='red')

def plot_regression_line(xs, ys, m, c):
    """
    Plots the regression line.
    :param xs: x values
    :param ys: y values
    :param m: slope of the regression line
    :param c: intercept of the regression line
    :return: None
    """
    plt.figure(1)
    plt.plot(xs, ys, 'o')
    plt.plot(xs, m*xs+c)

def mean_squared_error(ys,y_hats):
    """
    Calculates the mean squared error.
    :param ys: y values
    :param y_hats: y predicted values
    :return: mean squared error
    """

    summation = 0
    for y, y_hat in zip(ys, y_hats):
        summation += (y-y_hat)**2
    return summation/len(ys)


def gradient_descent_with_mse(xs, ys, m=0, c=0, learning_rate=0.0001):
    """
    Performs gradient descent with mean squared error.
    :param xs: x values
    :param ys: y values
    :param m: predicted slope of the regression line
    :param c: predicted intercept of the regression line
    :param learning_rate: learning rate
    :return: updated m, c, and loss
    """
    sum_dm = 0
    sum_dc = 0
    for x, y in zip(xs, ys):
        dm = x*(y-(m*x+c))
        dc = (y-(m*x+c))
        sum_dm += dm
        sum_dc += dc
    dm = -2*sum_dm/len(xs)
    dc = -2*sum_dc/len(xs)

    m = m - learning_rate*dm
    c = c - learning_rate*dc

    loss = mean_squared_error(ys, (m*xs+c))

    return m, c, loss


def perform_gradient_descent(xs, ys, learning_rate=0.0001, iterations=100):
    """
    preforms the gradient descent.
    :param xs: x values
    :param ys: y values
    :param learning_rate: learning rate
    :param iterations: number of iterations
    :return: all predicted values of m, c, and loss
    """
    ms, cs, losses = [], [], []
    m,c = 0,0

    #loss for inital m and c
    ms.append(m)
    cs.append(c)
    losses.append(mean_squared_error(ys, (m*xs+c)))

    for _ in range(iterations):
        m, c, loss = gradient_descent_with_mse(xs, ys, m, c, learning_rate)
        ms.append(m)
        cs.append(c)
        losses.append(loss)

    return ms, cs, losses

def main():
    ##Linear Data
    xs = np.arange(100)
    delta = np.random.uniform(-10, 10, size=(100,))
    ys = 0.9 * xs + 1 + delta

    ##Random Data
    # xs = np.random.randint(0, 10, size=(100,))
    # ys = np.random.randint(0, 10, size=(100,))

    ms, cs, losses = perform_gradient_descent(xs, ys, learning_rate=0.0001, iterations=1000)
    m, c, loss = ms[-1], cs[-1], losses[-1]

    print(f'Final m: {m}, Final c: {c}, Final loss: {losses[-1]}')
    plot_regression_line(xs, ys, m, c)
    plot_gradient_descent(ys, xs, ms, cs, losses)
    plt.show()

if __name__ == "__main__":
    main()