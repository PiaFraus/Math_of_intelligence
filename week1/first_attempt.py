"""
# Math_of_intelligence
Week 1 - First order optimization - derivative, partial derivative, convexity

I've implemented the gradient descent algorithm for a generic case with polynomials of any order.
It works on ay data, as long as learning_rate parameter correctly adjusted.

Issues I've encountered and solved and other notes:
   1) For my data, the slope changed with each iteration, but y-intercept was changing not fast enough.
I also had to implement learning_rate parameter but playing with it didn't help -
it was either too big and error was bigger and bigger or too small and y-intercept almost didn't changed.
And this was my first bingo-moment - Why do we use the same constant for different spaces?
I've separated this parameter into each space and it helped a lot. Learning curve for slope coefficient
should me much slower then for the y-interception.

   2) I've draw a plot of my sample data and it was obvious linear model is not ideal for it.
It was much more parabola like
To my surprise I kinda remembered some calculus, so I've decided to generalize this solution
for any polynomial model-function. At first I did it without splitting learning_rate parameter and it was sad.
I got a parabola, but it was far from ideal - started way too low.
But splitting learning_rate into now 3 coefficients helped and after I experimentally found again, that
we need to make learning_rate smaller and smaller for higher power coefficient,
 but other  coefficients can be quite big. On my data I found the perfect values for learning_rates:
 1e-13 for applying gradient for x^2 coefficient,
 1e-10 for x
 and 1e-1 for x^0.

 3) This made me realise that actually refining learning_rates is also a task for machine learning.
 And I might implement it later.
"""

import csv
import logging
from datetime import datetime
from itertools import islice

import matplotlib.pyplot
import matplotlib.dates

logger = logging.getLogger()


def polynomial_f(x, *coeffs):
    """
    Calculates value of polynomial function with given coefficients at point x.
    E.G.
     polynomial(x, 5,3,7) = 5 * x^2 + 3 * x + 7
     polynomial(x, 6,4) = 6 * x + 4

    >>> polynomial_f(10, 2, 1)
    21
    >>> polynomial_f(5, 4, 2, 1)
    111
    """
    result = 0
    for i, c in enumerate(reversed(coeffs)):
        result += c * x ** i
    return result


def d_polynomial_f(x, *coeffs):
    """
    Returns tuple of partial derivatives of f in point x
    by each coefficient dimension.
    E.g. if f = m x + b, d_polynomial_f(x, m, b) = (x, 1)
    :param x:
    :param coeffs:
    :return:
    """
    return reversed(tuple(x ** i for i, c in enumerate(reversed(coeffs))))


def error_f(points, *coeffs):
    """
    Calculate error function from formula:
    E = 1/N * Sum((actual(x) - modeled(x))^2, for all values)
    :param points:
    :param coeffs:
    """
    result = 0
    for p in points:
        model = polynomial_f(p[0], *coeffs)
        result += 1 / len(points) * (p[1] - model) ** 2
    return result


def d_error_f(points, *coeffs):
    """
    Returns tuple of size len(coeffs) of gradients
    of error_f in coeffs space.
    E = 1/N * Sum((actual(x) - modeled(x))^2, for all values)
    de/dc = -2/N * Sum((actual(x) - modeled(x)*d modeled(x)/dc
    :param points:
    :param coeffs:
    :return:
    """
    coeffs_gradient = [0 for _ in coeffs]
    for p in points:
        model = polynomial_f(p[0], *coeffs)
        derivatives = d_polynomial_f(p[0], *coeffs)
        for i, d in enumerate(derivatives):
            coeffs_gradient[i] += 2 / len(points) * (model - p[1]) * d
    return tuple(coeffs_gradient)


def refine_coefficients(coeffs, learning_rate, data, number_of_iterations=1000):
    """
    Refines provided coefficients using gradient descent method.
    On each iteration we are adjusting coefficients in the direction of lowering the error
    :param coeffs:
    :param learning_rate:
    :param data:
    :param number_of_iterations:
    :return:
    """
    coeffs = list(coeffs)
    for i in range(number_of_iterations):
        logger.debug('Iteration #%d. Current error=%f. Current coeffs=%s',
                     i, error_f(data, *coeffs), coeffs)
        coeffs_gradient = d_error_f(data, *coeffs)
        for i, g in enumerate(coeffs_gradient):
            coeffs[i] -= g * learning_rate[i]
    return coeffs

def read_ny_home_prices():
    with open('median_price.csv', 'r') as file:
        data_iter = csv.reader(file)
        # This csv is a bit weird - all the data is in the columns
        # And we basically need just first row of data for NY area.
        headers = next(data_iter)
        ny_data = next(data_iter)
    date_strings = islice(headers, 6, None)
    days = [(datetime.strptime(d, '%Y-%m') - datetime(2010, 1, 1)).days for d in date_strings]
    values = list(map(float, islice(ny_data, 6, None)))
    points = list(zip(days, values))
    return points


def run():
    logger.info('Loading csv')
    points = read_ny_home_prices()

    # with open('data.csv', 'r') as file:
    #     points = [tuple(map(float, r)) for r in csv.reader(file)]

    coeffs = [0, 0, 0]
    learning_rate = [1e-13, 1e-10, 1e-1]
    logger.info('Before calculation. Current error=%f. Current coeffs=%s',
                error_f(points, *coeffs), coeffs)
    new_coeffs = refine_coefficients(coeffs, learning_rate, points)
    logger.info('After calculation. Current error=%f. New coeffs=%s',
                error_f(points, *new_coeffs), new_coeffs)

    model_points = [(x, polynomial_f(x, *new_coeffs)) for x, _ in points]
    matplotlib.pyplot.plot(*zip(*points))
    matplotlib.pyplot.plot(*zip(*model_points))
    matplotlib.pyplot.show()

if __name__ == '__main__':
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    run()
