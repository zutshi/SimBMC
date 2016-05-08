from __future__ import print_function

# std lib
import sys

# external libs
import numpy as np
import pysmt.shortcuts as PS
import pysmt.typing as PT

# other libs
import constraints as cons
#import err

# project imports
import pwa.pwa as pwa

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
        sat = self.solver.check_sat()
        if sat:
            print('success: Model found')
        else:
            print('failed')
            raise NotImplementedError('model search failure not yet handled!')
        return

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
        VPx = PS.Or([sub_model.sat(data.x)
                    for sub_model in self.modelsym])
        self.solver.add_assertion(VPx)

        # If the data belongs to Pi, then Mi should give a 'good'
        # prediction
        gd_predict = [
                PS.Implies(
                    sub_model.sat(data.x),
                    PS.And(
                        and_op(sub_model.predict(data.x) - data.y, -data.e, PS.GE),
                        and_op(sub_model.predict(data.x) - data.y, data.e, PS.LE)))
                for sub_model in self.modelsym
                ]
        Px_implies_Mx = PS.And(gd_predict)
        self.solver.add_assertion(Px_implies_Mx)
        return

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


def visualize(sm, plt, m):
    def f1(x):
        return 4.0*x-2.0
    def f2(x):
        return 0.5*x+2.0
    def f3(x):
        return -0.3*x+7.0

    def f(p):
        return p.C

    x = sm.Symbol('x')
    x1, = sm.solvers.solve(f1(x)-f2(x))
    x2, = sm.solvers.solve(f1(x)-f3(x))
    x3, = sm.solvers.solve(f2(x)-f3(x))

    y1 = f1(x1)
    y2 = f1(x2)
    y3 = f2(x3)

    plt.plot(x1,f1(x1),'go',markersize=10)
    plt.plot(x2,f1(x2),'go',markersize=10)
    plt.plot(x3,f2(x3),'go',markersize=10)

    plt.fill([x1,x2,x3,x1],[y1,y2,y3,y1],'red',alpha=0.5)

    xr = np.linspace(0.5,7.5,100)
    y1r = f1(xr)
    y2r = f2(xr)
    y3r = f3(xr)

    plt.plot(xr,y1r,'k--')
    plt.plot(xr,y2r,'k--')
    plt.plot(xr,y3r,'k--')

    plt.xlim(0.5,7)
    plt.ylim(2,8)

    plt.show()


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


def test(tid):
    # num. of data points
    N = 100
    # num. dimension of the system
    ndim = 2
    # num of constraints for each polytope
    ncons = 4
    # num. of partitions(polytopes)
    nparts = 1
    # This is required to get the types correct for later where
    # pre-fix notation has to be used
    e = np.array([PS.Real(1e+0)]*ndim)

    params = Params(nparts, ndim, ncons)
    em = EpsAccurateModel(params)

    # generate data
    a, b = -2, 2
    X = (b - a) * np.random.random((N, ndim)) + a
    sim = test_case[tid]
    for x in X:
        em.add_data(Data(x, sim(x), e))
        print(x, sim(x))
    em.create_sym_model()
    em.search_model()
    em.populate_numerical_model()
    print(em.model)
    return em


if __name__ == '__main__':
    assert(len(sys.argv[1]) == 1)
    import matplotlib.pyplot as plt
    #from sympy.solvers import solve
    #from sympy import Symbol
    import sympy as sm
    m = test(int(sys.argv[1]))
    #visualize(sm, plt, m)
