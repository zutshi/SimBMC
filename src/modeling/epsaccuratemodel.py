
##################
# References:

# #### PySMT
# http://pysmt.readthedocs.io/en/latest/api_ref.html
# https://github.com/pysmt/pysmt/blob/2abfb4538fa93379f9b2671bce30f27967dedbcf/examples/infix_notation.py

# #### Numpy: overload operators
# http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#arithmetic-matrix-multiplication-and-comparison-operations
# http://stackoverflow.com/questions/14619449/how-can-i-override-comparisons-between-numpys-ndarray-and-my-type

# #### Matplotlib: plot polygons
# http://stackoverflow.com/questions/12881848/draw-polygons-more-efficiently-with-matplotlib
# http://stackoverflow.com/questions/26935701/ploting-filled-polygons-in-python
# http://stackoverflow.com/questions/17576508/python-matplotlib-drawing-linear-inequality-functions
##################

from __future__ import print_function

# std lib
import sys
import logging
import time

# external libs
import numpy as np
import pysmt.shortcuts as PS
import pysmt.typing as PT

# other libs
import constraints as cons
#import err

# project imports
import pwa.pwa as pwa

logger = logging.getLogger(__name__)

# Enable infix notation
PS.get_env().enable_infix_notation = True

##############
# Symbols
##############
# ## Partition
Cstr = 'C{}_{}{}' # C_<partition id>_<element id>
dstr = 'd{}_{}' # d_<partition id>_<element id>
# ## DiscreteAffineMap
Astr = 'A{}_{}{}'
bstr = 'b{}_{}'
estr = 'e{}_{}'


def Psym(partition_id, ncons, ndim, f=lambda x: x):
    Csym = [
            [f(Cstr.format(partition_id, i, j)) for j in range(ndim)]
            for i in range(ncons)
            ]
    dsym = [f(dstr.format(partition_id, i)) for i in range(ncons)]
    return np.array(Csym), np.array(dsym)


def Msym(partition_id, ndim, f=lambda x: x):
    Asym = [
            [f(Astr.format(partition_id, i, j)) for j in range(ndim)]
            for i in range(ndim)
            ]
    bsym = [f(bstr.format(partition_id, i)) for i in range(ndim)]
    #esym = [f(estr.format(partition_id, i)) for i in ndim]
    return np.array(Asym), np.array(bsym)


class Params(object):
    def __init__(self, nparts, ndim, ncons):
        self.nparts = nparts
        self.ndim = ndim
        self.ncons = ncons


class Data(object):
    def __init__(self, x, y, e):
        self.x = x
        self.y = y
        self.e = e


def model(data_set, params):
    m = EpsAccurateModel(params.nparts)
    for d in data_set:
        m.add_data(d)


class EpsAccurateModel(object):
    def __init__(self, params, solver='z3'):
        # number of partitions
        self.nparts = params.nparts
        self.ndim = params.ndim
        self.ncons = params.ncons

        self.modelsym = pwa.PWA()
        self.modelnum = pwa.PWA()
        self.create_sym_model()
        # constraints set
        self.solver = PS.Solver(name=solver)
        return

    # check sat
    def search_model(self):
        print('searching for model...')
        t0 = time.time()
        sat = self.solver.check_sat()
        tf = time.time()
        if sat:
            print('success: Model found')
            print('time taken:', tf-t0)
        else:
            print('failed')
            print('time taken:', tf-t0)
            raise NotImplementedError('No Model found with the given parameters!')
        return tf-t0

    def populate_numerical_model(self):
        get_value = np.vectorize(lambda x: float(self.solver.get_py_value(x)))
        for sub_model in self.modelsym:
            Csym, dsym, pid = sub_model.p.C, sub_model.p.d, sub_model.p.ID
            Asym, bsym = sub_model.m.A, sub_model.m.b
            C = get_value(Csym)
            d = get_value(dsym)
            A = get_value(Asym)
            b = get_value(bsym)
            e = 0
            self.modelnum.add(
                    pwa.SubModel(
                        pwa.Partition(C, d, pid),
                        pwa.DiscreteAffineMap(A, b, e)
                        )
                    )
        return

    def add_data(self, data):

        # The data must belong to a partition
        Px = self.atleast_1_partition_sats(data)
        #Px = self.unique_partition_sats(data)

        # Good prediction
        prediction = self.exPx_implies_Mx(data)
        #prediction = self.exMx(data)

        # Add assertions
        self.solver.add_assertion(Px)
        self.solver.add_assertion(prediction)
        return

    def atleast_1_partition_sats(self, data):
        # VPx
        return PS.Or([sub_model.sat(data.x) for sub_model in self.modelsym])

    def unique_partition_sats(self, data):
        # (+)Px
        sub_models = list(self.modelsym)
        l1 = []
        for i, _ in enumerate(sub_models):
            l2 = []
            for j, sbmdl in enumerate(sub_models):
                cons = sbmdl.sat(data.x)
                if i == j:
                    l2.append(cons)
                else:
                    l2.append(PS.Not(cons))
            l1.append(PS.And(l2))
        return PS.Or(l1)

    def exPx_implies_Mx(self, data):
        """ Generates constraint of the sort:
            \exists Pj. Pj(x) => |Mj(x) - y| <= e
            where
            - y = sim(x),
            - Pj is a polytope defined by: Cj, dj
            - Mj is an affine map defined by: Aj, bj
        Parameters
        ----------
        data : object of Data class

        Returns
        -------
        PySMT Constraint:   AND   (Pj(x) => |Mj(x) - y| <= e)
                          i=0...M

        """
        gd_predict = [
                PS.Implies(
                    sub_model.sat(data.x),
                    PS.And(
                        and_op(sub_model.predict(data.x) - data.y, -data.e, PS.GE),
                        and_op(sub_model.predict(data.x) - data.y, data.e, PS.LE)))
                for sub_model in self.modelsym
                ]
        return PS.And(gd_predict)

    def exMx(self, data):
        """ Generates constraint of the sort:
            \exists Mj. |Mj(x) - y| <= e
            where
            - y = sim(x),
            - Mj is an affine map defined by: Aj, bj
        Parameters
        ----------
        data : object of Data class

        Returns
        -------
        PySMT Constraint:   OR   (|Mi(x) - y| <= e)
                          i=0...M

        """
        gd_predict = [PS.And(
                        and_op(sub_model.predict(data.x) - data.y, -data.e, PS.GE),
                        and_op(sub_model.predict(data.x) - data.y, data.e, PS.LE))
                      for sub_model in self.modelsym
                      ]
        return PS.Or(gd_predict)

    def create_sym_model(self):
        def f(s): return PS.Symbol(s, PT.REAL)
        for pid in range(self.nparts):
            Csym, dsym = Psym(pid, self.ncons, self.ndim, f)
            Asym, bsym = Msym(pid, self.ndim, f)
            p = pwa.Partition(Csym, dsym, pid)
            m = pwa.DiscreteAffineMap(Asym, bsym, 0)
            self.modelsym.add(pwa.SubModel(p, m, poly_sat_sym))
        return

    def refine_along(self, trace):
        """refine along the given discrete trace: seq. of partitions

        Parameters
        ----------
        trace : seq. of partitions
        """

    @property
    def model(self):
        if self.modelnum.n:
            return self.modelnum
        else:
            raise RuntimeError('Numerical model not yet computed')

    def predict(self, X):
        Y = np.apply_along_axis(self.model.predict, 1)
        return Y

    def error_pc(self, X, Y):
        raise NotImplementedError

    def error(self, X, Y):
        """error

        Parameters
        ----------
        X : Test input
        Y : Test Output

        Returns
        -------
        Computes signed error against passed in test samples.
        Sound error interval vector (non-symmeteric).
        The interal must be added and NOT subtracted to make a sound
        model. e.g. Y_sound = A*X + e
        or,
            predict(X) + e_l <= Y_sound <= predict(X) + e_h
            Y_sound = predict(X) + [e_l, e_h]

        Notes
        ------
        Computes interval vector e = Y_true - Y_predict, such that
        the interval is sound w.r.t. to passed in test samples
        """
        Yp = self.predict(X)
        delta = Y - Yp
        max_e = np.max((delta), axis=0)
        min_e = np.min((delta), axis=0)
        return cons.IntervalCons(min_e, max_e)


def poly_sat_sym(poly, x):
    """return constraints for x being contained in poly

    Parameters
    ----------
    poly : symbolic polytope
    x : symbolic vector/array

    Returns smt constraints P(x)
    """
    #return np.all(np.dot(poly.C, x) <= poly.d)
    prod = np.dot(poly.C, x)
    cons = [r <= d for r, d in zip(prod, poly.d)]
    return PS.And(cons)


def apply_op(a1, a2, op):
    return [op(x, y) for x, y in zip(a1, a2)]


def and_op(a1, a2, op):
    return PS.And(apply_op(a1, a2, op))

###########################################
# Built in Test Routines
###########################################


def visualize_():
    from matplotlib import pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    # use seaborn to change the default graphics to something nicer
    # and set a nice color palette
    #import seaborn as sns
    #sns.set_color_palette('Set1')

    # create the plot object
    fig, ax = plt.subplots(figsize=(8, 8))
    s = np.linspace(0, 100)

    # add carpentry constraint: trains <= 80 - soldiers
    plt.plot(s, 80 - s, lw=3, label='carpentry')
    plt.fill_between(s, 0, 80 - s, alpha=0.1)

    # add finishing constraint: trains <= 100 - 2*soldiers
    plt.plot(s, 100 - 2 * s, lw=3, label='finishing')
    plt.fill_between(s, 0, 100 - 2 * s, alpha=0.1)

    # add demains constraint: soldiers <= 40
    plt.plot(40 * np.ones_like(s), s, lw=3, label='demand')
    plt.fill_betweenx(s, 0, 40, alpha=0.1)

    # add non-negativity constraints
    plt.plot(np.zeros_like(s), s, lw=3, label='t non-negative')
    plt.plot(s, np.zeros_like(s), lw=3, label='s non-negative')

    # highlight the feasible region
    path = Path([
        (0., 0.),
        (0., 80.),
        (20., 60.),
        (40., 20.),
        (40., 0.),
        (0., 0.),
    ])
    patch = PathPatch(path, label='feasible region', alpha=0.5)
    ax.add_patch(patch)

    # labels and stuff
    plt.xlabel('soldiers', fontsize=16)
    plt.ylabel('trains', fontsize=16)
    plt.xlim(-0.5, 100)
    plt.ylim(-0.5, 100)
    plt.legend(fontsize=14)
    plt.show()


def visualize__():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy.solvers import solve
    from sympy import Symbol

    def f1(x):
        return 4.0*x-2.0
    def f2(x):
        return 0.5*x+2.0
    def f3(x):
        return -0.3*x+7.0

    x = Symbol('x')
    x12, = solve(f1(x)-f2(x))
    x13, = solve(f1(x)-f3(x))
    x23, = solve(f2(x)-f3(x))

    y12 = f1(x12)
    y13 = f1(x13)
    y23 = f3(x23)

    assert(abs(f1(x12) - f2(x12)) <= 1e-10)
    assert(abs(f1(x13) - f3(x13)) <= 1e-10)
    assert(abs(f2(x23) - f3(x23)) <= 1e-10)

    plt.plot(x12, y12,'go',markersize=10)
    plt.plot(x13, y13,'go',markersize=10)
    plt.plot(x23, y23,'go',markersize=10)

    plt.fill([x12, x13, x23, x12],[y12, y13, y23, y12],'red',alpha=0.5)

#     xr = np.linspace(0.5,7.5,100)
#     y1r = f1(xr)
#     y2r = f2(xr)
#     y3r = f3(xr)

#     plt.plot(xr,y1r,'k--')
#     plt.plot(xr,y2r,'k--')
#     plt.plot(xr,y3r,'k--')

#     plt.xlim(0.5,7)
#     plt.ylim(2,8)

    plt.show()


def visualize(m, (X, Y)):
    #import numpy as np
    import matplotlib.pyplot as plt
    from sympy.solvers.solveset import linsolve
    from sympy import Matrix, symbols

    import itertools as it

    x1sym, x2sym = symbols('x1, x2')

    for sm in m:
        C, d, ID = sm.p.C, sm.p.d, sm.p.ID
        #xsym = np.array([Symbol('x{}'.format(i)) for i in range(ndim)])
        Cd = np.column_stack((C, d))
        vertices = []
        #print(Cd)
        for l in it.combinations(Cd, 2):
            Ab = Matrix(l)
            print('%'*20)
            print(Ab)
            sol = linsolve(Ab, x1sym, x2sym)
            print(sol)
            print('%'*20)
            for (x1, x2) in sol: pass
            try:
                if sm.sat(np.array([x1, x2]), 1e-2):
                    vertices.append((x1, x2))
                #else:
                 #   vertices.append((x1, x2))
            except TypeError:
                print('TypeError:', sol)
        print('v:', vertices)
        for v in vertices:
            plt.plot(v[0], v[1], 'go', markersize=10)

        # close the polygon
        vertices.append(vertices[0])
        plt.fill([xy[0] for xy in vertices],
                 [xy[1] for xy in vertices],
                 'red', alpha=0.5)


        # plot data points: only X
    for x in X:
        plt.plot(x[0], x[1], 'go', markersize=5, color='r')
    plt.show()


def visualizev2(m, (X, Y), xrng):
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy.solvers.solveset import linsolve
    from sympy import Matrix, symbols
    import sympy as sm

    import itertools as it

    x1sym, x2sym = symbols('x1, x2')
    xsym = np.array([x1sym, x2sym])

    x1l, x1h = xrng
    x2l, x2h = xrng
    bounding_box = np.array([
        [1, 0, x1h],
        [0, 1, x2h],
        [-1, 0, -x1l],
        [0, -1, -x2l]
        ])

    for sbmdl in m:
        C, d, ID = sbmdl.p.C, sbmdl.p.d, sbmdl.p.ID
        #xsym = np.array([Symbol('x{}'.format(i)) for i in range(ndim)])
        Cd_ = np.column_stack((C, d))
        # Bound the polyhedron by the bounding box
        Cd = np.vstack((Cd_, bounding_box))

        vertices = []
        #print(Cd)
        for l in it.combinations(Cd, 2):
            Ab = Matrix(l)
            if np.linalg.matrix_rank(l) < 2:
                continue
            print('%'*20)
            print(Ab)
            sol = linsolve(Ab, x1sym, x2sym)
            print(sol)
            print('%'*20)
            for (x1, x2) in sol: pass
            try:
                if sbmdl.sat(np.array([x1, x2]), 1e-2):
                    vertices.append((x1, x2))
                else:
                    vertices.append((x1, x2))
            except TypeError:
                print('TypeError:', sol)
        print('v:', vertices)
        for v in vertices:
            plt.plot(v[0], v[1], 'go', markersize=10)

        # close the polygon
        vertices.append(vertices[0])
        if vertices:
            plt.fill([xy[0] for xy in vertices],
                     [xy[1] for xy in vertices],
                     'red', alpha=0.5)

        X1R = np.linspace(xrng[0], xrng[1], 100)
        X2R = np.linspace(xrng[0], xrng[1], 100)
        for Ci, di in zip(C, d):
            li = np.dot(Ci, xsym) - di
            print('li:', li)
            sol = sm.solve(li, x2sym)
            # pick the 1st element; there should be only 1
            print(sol)
            assert(len(sol) <= 1)
            if len(sol) == 1:
                x2_expr = sol[0]
                print('x2 =', x2_expr)
                # fi represents the edge of the polytope
                fi = sm.lambdify(x1sym, x2_expr)
                x1r = X1R
                x2r = fi(X1R)
            else:
                sol = sm.solve(li, x1sym)
                if len(sol) == 1:
                    x1_expr = sol[0]
                    print('x1 =', x1_expr)
                    x1r = np.tile(x1_expr, len(X2R))
                    x2r = X2R
                    print(x1r)
                    print(x2r)
                else:
                    raise NotImplementedError
                    continue
            print(x2r)
            plt.plot(x1r, x2r, 'k--')

    # plot data points: only X
    for x in X:
        plt.plot(x[0], x[1], 'go', markersize=5, color='r')

    plt.show()
    return


def visualizev3(m, (X, Y), xrng, tol=1e-3):
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
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols
    from sympy.logic import boolalg
    import sympy as sm
    #from fillplots import plot_regions
    from fillplots import Plotter, annotate_regions
    import matplotlib.colors as colors

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
    N = 20
    # num. dimension of the system
    ndim = 2
    # num of constraints for each polytope
    ncons = 3
    # num. of partitions(polytopes)
    nparts = 10
    # This is required to get the types correct for later where
    # pre-fix notation has to be used
    e = np.array([PS.Real(1e+0)]*ndim)

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
    for x in X:
        Y = np.around(sim(x), decimals=1)
        em.add_data(Data(x, Y, e))
        print(x, Y)
    #em.create_sym_model()
    tt = em.search_model()
    em.populate_numerical_model()
    print(em.model)
    print('time taken for model search:', tt)
    return em, (X, Y)


if __name__ == '__main__':
    FORMAT2 = '%(levelname) -10s %(asctime)s %(module)s:\
               %(lineno)s %(funcName)s() %(message)s'
    logging.basicConfig(filename='log.model', filemode='w', format=FORMAT2,
                        level=logging.DEBUG)
    rng = (-10, 10)
    assert(len(sys.argv) == 3)
    np.random.seed(int(sys.argv[2]))
    em, data_set = test(int(sys.argv[1]), rng)
    print('='*40)
    visualizev3(em.model, data_set, rng)
