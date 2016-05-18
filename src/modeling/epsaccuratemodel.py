
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

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)


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
        # this is solver specific...?
        #sat = self.solver.check_sat()
        # TODO: Seems OK because is_sat(f) is checked against solver's
        # state. Confirm!
        sat = self.solver.solve()
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

        fn = np.vectorize(lambda x: PS.Real(float(x)))
        data.e = fn(data.e)

        # The data must belong to a partition
        #Px = self.atleast_1_partition_sats(data)
        Px = self.unique_partition_sats(data)

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

#         for sub_model in self.modelsym:
#             c = sub_model.predict(data.x) - data.y
#             print(data)
#             print('c:')
#             for i in c:
#                 print(i)
#             print('='*80)
#             cc = and_op(c, data.e, PS.LE)
#             print('cc:', cc)
#         exit()
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
