#!/usr/bin/env python

from __future__ import print_function

from epsaccuratemodel import EpsAccurateModel, Params, Data
import numpy as np
import sys
import logging
from scipy.integrate import ode
import matplotlib.pyplot as plt
from sympy import symbols
from sympy.logic import boolalg
import sympy as sm
#from fillplots import plot_regions
from fillplots import Plotter, annotate_regions
import matplotlib.colors as colors

###########################################
# Built in Test Routines
###########################################

FORMAT2 = '%(levelname) -10s %(asctime)s %(module)s:\
           %(lineno)s %(funcName)s() %(message)s'
logging.basicConfig(filename='log.model', filemode='w', format=FORMAT2,
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


def sim1(x): return x + 5


def sim2(x): return 2*x + 6.7


def sim3(x):

    A = np.array([[3, -5],
                  [-2, -7]])
    b = np.array([3.56, -2.43])
    return np.dot(A, x) + b


def sim4(x): return 2*(x**2) + 0.5


def sim5(x): return 2*(np.sin(x))

test_case = {
        1: sim1,
        2: sim2,
        3: sim3,
        4: sim4,
        5: sim5,
        }


def test(tid, rng):
    # num. of data points
    N = 10
    # num. dimension of the system
    ndim = 2
    # num of constraints for each polytope
    ncons = 3
    # num. of partitions(polytopes)
    nparts = 6
    # This is required to get the types correct for later where
    # pre-fix notation has to be used
    e = np.array([1e+0]*ndim)

    params = Params(nparts, ndim, ncons)
    em = EpsAccurateModel(params)

    # generate data
    #rng = (-2, 2)
    a, b = rng
    X = (b - a) * np.random.random((N, ndim)) + a
    # round of the digits to make things simpler for the SMT solver
    decimals = 1
    X = np.around(X, decimals=decimals)
    sim = test_case[tid]
    Y = []
    for x in X:
        y = np.around(sim(x), decimals=decimals)
        em.add_data(Data(x, y, e))
        Y.append(y)
    #em.create_sym_model()
    tt = em.search_model()
    em.populate_numerical_model()
    print(em.model)
    print('time taken for model search:', tt)
    return em, (X, Y)


def test_vdp(N=1, nT=20):
    dec = 4

    def dyn(t, X):
        X = X.copy()
        X[0], X[1] = (X[1], 5.0 * (1 - X[0] ** 2) * X[1] - X[0])
        return X

    solver = ode(dyn).set_integrator('dopri5')

    def sim(dt, X0):
        solver.set_initial_value(X0, t=0.0)
        return solver.integrate(dt)

    params = Params(nparts=3, ndim=2, ncons=1)
    em = EpsAccurateModel(params)
    e = np.array([1e-1, 1e-1])

    #x1rng = (-0.4, 0.4)
    #x2rng = (-0.4, 0.4)
    # just simplify because we know x{1,2} \in [-0.4, 0.4]
    xl, xh = -0.4, 0.4
    x = np.around(xl + (xh-xl)*np.random.random((N, 2)), decimals=dec)
    dt = 0.1

    X, Y = [], []
    for xi in x:
        for n in range(nT):
            yi = np.around(sim(dt, xi), decimals=dec)
            data = Data(xi, yi, e)
            em.add_data(data)
            X.append(xi)
            Y.append(yi)
            print(xi, yi)
            xi = yi
    #em.create_sym_model()
    tt = em.search_model()
    em.populate_numerical_model()
    print(em.model)
    print('time taken for model search:', tt)
    return em, (X, Y)


def visualize_poly(m, (X, Y), xrng, tol=1e-3):
    """visualize version3 (the last versions sucked)
       Plots the regions defined by the polygonal partitions of m.
       Each submodel defines a polygonal partition.
       All unbounded polygons are bounded by a bounding box defined by
       xrng.

    Parameters
    ----------
    m : epsaccuratemodel
    (X, Y) : X, Y=sim(X) pairs to plot
    xrng : [xl, xh] range defining a bounding box for both x1, x2 dimensions
    tol : Numerical tolerance to use while plotting polygons. It is
    useful in cases when a feasible polygon is deemed infeasible due
    to numerical errors. Happens often when using an SMT solver.

    Notes
    ------
    Supports only 2-dim polygons
    """

    no_light_colors = [
            c for c in colors.cnames.keys()
            if c.find('light') != 0
            ]

    dark_colors = [
            c for c in colors.cnames.keys()
            if c.find('dark') == 0
            ]

    x1sym, x2sym = symbols('x1, x2')
    xsym = np.array([x1sym, x2sym])

    for sbmdl in m:
        C, d, ID = sbmdl.p.C, sbmdl.p.d, sbmdl.p.ID
        # num dim must be 2
        assert(C.shape[1] == 2)
        logger.debug('^'*50)
        logger.debug('analyzing submodel: {}'.format(ID))

        lines = []
        for Ci, di in zip(C, d):
            # compute the equation of the line
            li = np.dot(Ci, xsym) <= di
            logger.debug('edge: {}'.format(li))
            # if the edge is trivially True, ignore it
            if li == boolalg.true:
                continue
            # if the polyhedra's edge is infeasible, the polyhedra is
            # infeasible, hence, do not plot it.
            elif li == boolalg.false:
                # combined with the below else statement, this break
                # statement breaks acts as the continue statement for
                # the outer loop
                break
            else:
                #lines.append(li)
                lines.append(np.dot(Ci, xsym) <= di + tol)

        # if the for loop did not break, plot the polygon
        else:
            P = []
            logger.debug('plotting P{}'.format(ID))
            for l in lines:
                logger.debug('edge: {}'.format(l))
                sol = sm.solve(l, x2sym)
                logger.debug('sol: {}'.format(sol))
                if(isinstance(sol, sm.boolalg.And)):
                    exprs = sol.args
                    # can not have more than 2 constraints in 2 dim
                    # can have less?
                    assert(len(exprs) == 2)
                    f1 = sm.lambdify(x1sym, exprs[0].rhs)
                    f2 = sm.lambdify(x1sym, exprs[1].rhs)
                    assert(exprs[0].is_Relational)
                    assert(exprs[1].is_Relational)
                    if (exprs[0].lhs.is_finite and exprs[0].rhs.is_finite):
                        P.append((f1, exprs[0].rel_op.find('<') == 0))
                        print(exprs[0])
                    if (exprs[1].lhs.is_finite and exprs[1].rhs.is_finite):
                        P.append((f2, exprs[1].rel_op.find('<') == 0))
                        print(exprs[1])
                else:
                    assert(sol.is_Relational)
                    f = sm.lambdify(x1sym, sol.rhs)
                    #P.append((f, sol.rel_op == '<='))
                    P.append((f, sol.rel_op.find('<') == 0))

            plotter = Plotter([P], xlim=xrng, ylim=xrng)

            for reg in plotter.regions:
                c = np.random.choice(dark_colors)
                # Does not work as intended...why?
                #reg.config.fill_args['facecolor'] = c
                #reg.config.fill_args['edgecolor'] = c
                # Works very well, but can not use more than one color
                reg.config.fill_args['autocolor'] = False
            plotter.config.line_args['lw'] = 0
            plotter.config.fill_args['alpha'] = 0.5
            annotate_regions(plotter.regions, 'P'+str(ID))
            plotter.plot()

    # plot all the data points x
    for x in X:
        plt.plot(x[0], x[1], 'go', markersize=5, color='r')

    plt.show()
    return


def visualize_predictors(m, (X, Y)):
    fig = plt.figure()
    for sm in m:
        A, b = sm.m.A, sm.m.b
        print(A, b)
    fig.plot()


if __name__ == '__main__':
    rng = (-10, 10)
    try:
        assert(len(sys.argv) == 3)
    except AssertionError:
        print('usage:', sys.argv[0], '<test case id>', '<seed>')
        exit()
    seed = int(sys.argv[2])
    test_id = int(sys.argv[1])
    np.random.seed(seed)

    if test_id == 10:
        em, data_set = test_vdp()
    else:
        em, data_set = test(test_id, rng)

    print('='*40)
    visualize_poly(em.model, data_set, rng)
    #visualize_predictors(em.model, data_set)


# def visualize_():
#     from matplotlib import pyplot as plt
#     from matplotlib.path import Path
#     from matplotlib.patches import PathPatch

#     # use seaborn to change the default graphics to something nicer
#     # and set a nice color palette
#     #import seaborn as sns
#     #sns.set_color_palette('Set1')

#     # create the plot object
#     fig, ax = plt.subplots(figsize=(8, 8))
#     s = np.linspace(0, 100)

#     # add carpentry constraint: trains <= 80 - soldiers
#     plt.plot(s, 80 - s, lw=3, label='carpentry')
#     plt.fill_between(s, 0, 80 - s, alpha=0.1)

#     # add finishing constraint: trains <= 100 - 2*soldiers
#     plt.plot(s, 100 - 2 * s, lw=3, label='finishing')
#     plt.fill_between(s, 0, 100 - 2 * s, alpha=0.1)

#     # add demains constraint: soldiers <= 40
#     plt.plot(40 * np.ones_like(s), s, lw=3, label='demand')
#     plt.fill_betweenx(s, 0, 40, alpha=0.1)

#     # add non-negativity constraints
#     plt.plot(np.zeros_like(s), s, lw=3, label='t non-negative')
#     plt.plot(s, np.zeros_like(s), lw=3, label='s non-negative')

#     # highlight the feasible region
#     path = Path([
#         (0., 0.),
#         (0., 80.),
#         (20., 60.),
#         (40., 20.),
#         (40., 0.),
#         (0., 0.),
#     ])
#     patch = PathPatch(path, label='feasible region', alpha=0.5)
#     ax.add_patch(patch)

#     # labels and stuff
#     plt.xlabel('soldiers', fontsize=16)
#     plt.ylabel('trains', fontsize=16)
#     plt.xlim(-0.5, 100)
#     plt.ylim(-0.5, 100)
#     plt.legend(fontsize=14)
#     plt.show()


# def visualize__():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from sympy.solvers import solve
#     from sympy import Symbol

#     def f1(x):
#         return 4.0*x-2.0
#     def f2(x):
#         return 0.5*x+2.0
#     def f3(x):
#         return -0.3*x+7.0

#     x = Symbol('x')
#     x12, = solve(f1(x)-f2(x))
#     x13, = solve(f1(x)-f3(x))
#     x23, = solve(f2(x)-f3(x))

#     y12 = f1(x12)
#     y13 = f1(x13)
#     y23 = f3(x23)

#     assert(abs(f1(x12) - f2(x12)) <= 1e-10)
#     assert(abs(f1(x13) - f3(x13)) <= 1e-10)
#     assert(abs(f2(x23) - f3(x23)) <= 1e-10)

#     plt.plot(x12, y12,'go',markersize=10)
#     plt.plot(x13, y13,'go',markersize=10)
#     plt.plot(x23, y23,'go',markersize=10)

#     plt.fill([x12, x13, x23, x12],[y12, y13, y23, y12],'red',alpha=0.5)

# #     xr = np.linspace(0.5,7.5,100)
# #     y1r = f1(xr)
# #     y2r = f2(xr)
# #     y3r = f3(xr)

# #     plt.plot(xr,y1r,'k--')
# #     plt.plot(xr,y2r,'k--')
# #     plt.plot(xr,y3r,'k--')

# #     plt.xlim(0.5,7)
# #     plt.ylim(2,8)

#     plt.show()


# def visualize(m, (X, Y)):
#     #import numpy as np
#     import matplotlib.pyplot as plt
#     from sympy.solvers.solveset import linsolve
#     from sympy import Matrix, symbols

#     import itertools as it

#     x1sym, x2sym = symbols('x1, x2')

#     for sm in m:
#         C, d, ID = sm.p.C, sm.p.d, sm.p.ID
#         #xsym = np.array([Symbol('x{}'.format(i)) for i in range(ndim)])
#         Cd = np.column_stack((C, d))
#         vertices = []
#         #print(Cd)
#         for l in it.combinations(Cd, 2):
#             Ab = Matrix(l)
#             print('%'*20)
#             print(Ab)
#             sol = linsolve(Ab, x1sym, x2sym)
#             print(sol)
#             print('%'*20)
#             for (x1, x2) in sol: pass
#             try:
#                 if sm.sat(np.array([x1, x2]), 1e-2):
#                     vertices.append((x1, x2))
#                 #else:
#                  #   vertices.append((x1, x2))
#             except TypeError:
#                 print('TypeError:', sol)
#         print('v:', vertices)
#         for v in vertices:
#             plt.plot(v[0], v[1], 'go', markersize=10)

#         # close the polygon
#         vertices.append(vertices[0])
#         plt.fill([xy[0] for xy in vertices],
#                  [xy[1] for xy in vertices],
#                  'red', alpha=0.5)


#         # plot data points: only X
#     for x in X:
#         plt.plot(x[0], x[1], 'go', markersize=5, color='r')
#     plt.show()


# def visualizev2(m, (X, Y), xrng):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from sympy.solvers.solveset import linsolve
#     from sympy import Matrix, symbols
#     import sympy as sm

#     import itertools as it

#     x1sym, x2sym = symbols('x1, x2')
#     xsym = np.array([x1sym, x2sym])

#     x1l, x1h = xrng
#     x2l, x2h = xrng
#     bounding_box = np.array([
#         [1, 0, x1h],
#         [0, 1, x2h],
#         [-1, 0, -x1l],
#         [0, -1, -x2l]
#         ])

#     for sbmdl in m:
#         C, d, ID = sbmdl.p.C, sbmdl.p.d, sbmdl.p.ID
#         #xsym = np.array([Symbol('x{}'.format(i)) for i in range(ndim)])
#         Cd_ = np.column_stack((C, d))
#         # Bound the polyhedron by the bounding box
#         Cd = np.vstack((Cd_, bounding_box))

#         vertices = []
#         #print(Cd)
#         for l in it.combinations(Cd, 2):
#             Ab = Matrix(l)
#             if np.linalg.matrix_rank(l) < 2:
#                 continue
#             print('%'*20)
#             print(Ab)
#             sol = linsolve(Ab, x1sym, x2sym)
#             print(sol)
#             print('%'*20)
#             for (x1, x2) in sol: pass
#             try:
#                 if sbmdl.sat(np.array([x1, x2]), 1e-2):
#                     vertices.append((x1, x2))
#                 else:
#                     vertices.append((x1, x2))
#             except TypeError:
#                 print('TypeError:', sol)
#         print('v:', vertices)
#         for v in vertices:
#             plt.plot(v[0], v[1], 'go', markersize=10)

#         # close the polygon
#         vertices.append(vertices[0])
#         if vertices:
#             plt.fill([xy[0] for xy in vertices],
#                      [xy[1] for xy in vertices],
#                      'red', alpha=0.5)

#         X1R = np.linspace(xrng[0], xrng[1], 100)
#         X2R = np.linspace(xrng[0], xrng[1], 100)
#         for Ci, di in zip(C, d):
#             li = np.dot(Ci, xsym) - di
#             print('li:', li)
#             sol = sm.solve(li, x2sym)
#             # pick the 1st element; there should be only 1
#             print(sol)
#             assert(len(sol) <= 1)
#             if len(sol) == 1:
#                 x2_expr = sol[0]
#                 print('x2 =', x2_expr)
#                 # fi represents the edge of the polytope
#                 fi = sm.lambdify(x1sym, x2_expr)
#                 x1r = X1R
#                 x2r = fi(X1R)
#             else:
#                 sol = sm.solve(li, x1sym)
#                 if len(sol) == 1:
#                     x1_expr = sol[0]
#                     print('x1 =', x1_expr)
#                     x1r = np.tile(x1_expr, len(X2R))
#                     x2r = X2R
#                     print(x1r)
#                     print(x2r)
#                 else:
#                     raise NotImplementedError
#                     continue
#             print(x2r)
#             plt.plot(x1r, x2r, 'k--')

#     # plot data points: only X
#     for x in X:
#         plt.plot(x[0], x[1], 'go', markersize=5, color='r')

#     plt.show()
#     return

