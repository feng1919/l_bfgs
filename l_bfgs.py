###############################################################################
'''
/*
 *          Limited memory BFGS (L-BFGS) with Keras Implementation.
 *
 * Copyright (c) 2019, Feng Shi
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
'''
###############################################################################
'''                         Reference
- <a href="https://github.com/chokkan/liblbfgs">liblbfgs</a> by Naoaki Okazaki.
- <a href="http://www.ece.northwestern.edu/~nocedal/lbfgs.html">L-BFGS</a> by Jorge Nocedal.
- <a href="http://research.microsoft.com/en-us/downloads/b1eb1016-1738-4bd5-83a9-370c9d498a03/default.aspx">
  Orthant-Wise Limited-memory Quasi-Newton Optimizer for L1-regularized Objectives</a> by Galen Andrew.
- <a href="http://chasen.org/~taku/software/misc/lbfgs/">C port (via f2c)</a> by Taku Kudo.
- <a href="http://www.alglib.net/optimization/lbfgs.php">C#/C++/Delphi/VisualBasic6 port</a> in ALGLIB.
- <a href="http://cctbx.sourceforge.net/">Computational Crystallography Toolbox</a> includes
- <a href="http://cctbx.sourceforge.net/current_cvs/c_plus_plus/namespacescitbx_1_1lbfgs.html">scitbx::lbfgs</a>.
'''

###############################################################################

from l_bfgs_result import LBFGSResult as result
from l_bfgs_linesearch import LBFGSLineSearch
from l_bfgs_linesearch import supported_linesearch, backtracking, linesearch_morethuente, backtracking_owlqn
from l_bfgs_linesearch import owlqn_x1norm, owlqn_project, owlqn_pseudo_gradient
from l_bfgs_vector import *
from l_bfgs_parameters import LBFGSParameters
from l_bfgs_memory import limited_memory, iteration_data

import numpy as np
import keras.backend as K

'''
 * Callback interface to provide objective function and gradient evaluations.
 *
 * ***This function must be customized by user.***
 *
 *  The l_bfgs() function call this function to obtain the values of objective
 *  function and its gradients when needed. A client program must implement
 *  this function to evaluate the values of the objective function and its
 *  gradients, given current values of variables.
 *  
 *  @param  x           The current values of variables.
 *  @param  g           The gradient vector. The callback function must compute
 *                      the gradient values for the current variables.
 *  @param  n           The number of variables.
 *  @param  step        The current step of the line search routine.
 *  @retval x.dtype     The value of the objective function for the current 
                        variables.
'''
def l_bfgs_evaluate(x, g, n, step):
    assert False, "One should customize this function."
    pass

'''
 * Callback interface to receive the progress of the optimization process.
 *
 * ***This function could be customized by user if a process is needed.***
 *
 *  The l_bfgs() function call this function for each iteration. Implementing
 *  this function, a client program can store or display the current progress
 *  of the optimization process.
 *
 *  @param  x           The current values of variables.
 *  @param  g           The current gradient values of variables.
 *  @param  fx          The current value of the objective function.
 *  @param  xnorm       The Euclidean norm of the variables.
 *  @param  gnorm       The Euclidean norm of the gradients.
 *  @param  step        The line-search step used for this iteration.
 *  @param  k           The iteration count.
 *  @param  n           The number of variables.
 *  @param  ls          The number of evaluations called for this iteration.
 *  @retval int         Zero to continue the optimization process. Returning a
 *                      non-zero value will cancel the optimization process.
'''
def l_bfgs_progress(x, g, fx, xnorm, gnorm, step, n, k, ls):
    #assert False, "One may customize this function, if a progress is needed."
    pass

'''
 * Start a L-BFGS optimization.
 *  @param  x           The array of variables. A client program can set
 *                      default values for the optimization and receive the
 *                      optimization result through this array. 
 *  @param  ptr_fx      The pointer to the variable that receives the final
 *                      value of the objective function for the variables.
 *                      This argument can be set to None if the final
 *                      value of the objective function is unnecessary.
 *  @param  evaluate    The callback function to provide function and
 *                      gradient evaluations given a current values of
 *                      variables. A client program must implement a
 *                      callback function compatible with l_bfgs_evaluate 
 *                      and pass the pointer to the callback function.
 *  @param  progress    The callback function to receive the progress
 *                      (the number of iterations, the current value of
 *                      the objective function) of the minimization
 *                      process. This argument can be set to None if
 *                      a progress report is unnecessary.
 *  @param  param       An instance of LBFGSParameters to represent parameters 
 *                      for L-BFGS optimization. 
 *                      Call LBFGSParameters() function to initialize an
 *                      instance with the default values.
 *  @retval int         The status code. This function returns zero if the
 *                      minimization process terminates without an error. A
 *                      non-zero value indicates an error.
'''
def l_bfgs(x, evaluate, progress=None, fx=None, param=LBFGSParameters()):

    assert x is not None, "The variables to be optimized must not be None."
    assert x.dtype == np.float32 or x.dtype == np.float64, "Invalid data type of the variables."

    # Global constants.
    n = x.shape[0]
    m = param.m
    dtype = x.dtype

    # Check the input parameters for errors.
    if x.shape[0] <= 0:
        return result.ERR_INVALID_N

    if param.epsilon < 0.0:
        return result.ERR_INVALID_EPSILON

    if param.past < 0:
        return result.ERR_INVALID_TESTPERIOD

    if param.delta < 0.0:
        return result.ERR_INVALID_DELTA

    if param.min_step < 0.0:
        return result.ERR_INVALID_MINSTEP

    if param.max_step < param.min_step:
        return result.ERR_INVALID_MAXSTEP

    if param.ftol < 0.0:
        return result.ERR_INVALID_FTOL

    if param.gtol < 0.0:
        return result.ERR_INVALID_GTOL

    if param.xtol < 0.0:
        return result.ERR_INVALID_XTOL

    if param.linesearch == LBFGSLineSearch.BACKTRACKING_WOLFE or \
       param.linesearch == LBFGSLineSearch.BACKTRACKING_STRONG_WOLFE:
        if param.wolfe <= param.ftol or param.wolfe >= 1.0:
            return result.ERR_INVALID_WOLFE

    if param.max_linesearch <= 0:
        return result.ERR_INVALID_MAXLINESEARCH

    # Check for orthantwise limited-memory Quasi-Newton method parameters.
    if param.orthantwise_c < 0.0:
        return result.ERR_INVALID_ORTHANTWISE

    if param.orthantwise_start < 0 or param.orthantwise_start > n:
        return result.ERR_INVALID_ORTHANTWISE_START

    if param.orthantwise_end < 0:
        param.orthantwise_end = n
    elif param.orthantwise_end > n:
        return result.ERR_INVALID_ORTHANTWISE_END

    if not supported_linesearch(param.linesearch):
        return result.ERR_INVALID_LINESEARCH

    if param.orthantwise_c > 0.0:
        if param.linesearch == LBFGSLineSearch.BACKTRACKING:
            linesearch = backtracking_owlqn
        else:
            return result.ERR_INVALID_LINESEARCH
    elif param.linesearch == LBFGSLineSearch.MORETHUENTE:
        linesearch = linesearch_morethuente
    else:
        linesearch = backtracking

    # Allocate working space.
    xp = K.placeholder(shape=(n,), dtype=dtype, name="xp")
    g = K.placeholder(shape=(n,), dtype=dtype, name="g")
    gp = K.placeholder(shape=(n,), dtype=dtype, name="gp")
    d = K.placeholder(shape=(n,), dtype=dtype, name="d")
    w = K.placeholder(shape=(n,), dtype=dtype, name="w")

    # Allocate limited memory storage and initialize the limited memory.
    lm = limited_memory(m, n, dtype)

    if param.orthantwise_c > 0.0:
        # Allocate working space for OW-LQN.
        pg = K.placeholder(shape=(n,), dtype=dtype, name="pg")
    else:
        pg = None

    # Allocate an array for storing previous values of the objective function.
    pf = [] if param.past > 0 else None

    # Evaluate the function value and its gradient.
    fx = evaluate(x, g, n, 0.0)
    if param.orthantwise_c > 0.0:
        # Compute the L1 norm of the variable and add it to the object value.
        xnorm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end)
        fx += xnorm * param.orthantwise_c
        owlqn_pseudo_gradient(pg, x, g, n,
                              param.orthantwise_c,
                              param.orthantwise_start,
                              param.orthantwise_end)

    # Store the initial value of the objective function.
    if pf is not None:
        pf.append(fx)

    # Compute the direction, we assume the initial hessian matrix H_0 as the identity matrix.
    if param.orthantwise_c == 0.0:
        vecncpy(d, g, n)
    else:
        vecncpy(d, pg, n)

    # Make sure that the initial variables are not a minimizer.
    xnorm = vec2norm(x)
    xnorm = K.maximum(xnorm, 1.0)
    if param.orthantwise_c == 0.0:
        gnorm = vec2norm(g)
    else:
        gnorm = vec2norm(pg)
    if gnorm / xnorm <= param.epsilon:
        return result.ALREADY_MINIMIZED

    # Compute the initial step:
    #   step = 1.0 / sqrt(sum(d*d))
    step = 1.0 / gnorm

    k = 1
    end = 0
    while True:
        # Store the current position and gradient vectors.
        veccpy(xp, x, n)
        veccpy(gp, g, n)

        # Search for an optimal step.
        if param.orthantwise_c == 0.0:
            ls, fx, step = linesearch(n, x, fx, g, d, step, xp, gp, w, evaluate, param)
        else:
            ls, fx, step = linesearch(n, x, fx, g, d, step, xp, pg, w, evaluate, param)
            owlqn_pseudo_gradient(pg, x, g, n,
                                  param.orthantwise_c,
                                  param.orthantwise_start,
                                  param.orthantwise_end)
        if ls < 0:
            # failed to process.
            # Revert to the previous point.
            veccpy(x, xp, n)
            veccpy(g, gp, n)
            return ls

        xnorm = vec2norm(x)
        if param.orthantwise_c == 0.0:
            gnorm = vec2norm(g)
        else:
            gnorm = vec2norm(pg)

        # Report the progress.
        if progress is not None:
            ret = progress(x, g, fx, xnorm, gnorm, step, n, k, ls)
            if ret != 0:
                return result.USER_TERMINATE

        # Convergence test.
        # The criterion is given by the following formula:
        #     |g(x)| / max(1, |x|) < epsilon
        xnorm = K.maximum(xnorm, 1.0)
        if gnorm / xnorm <= param.epsilon:
            return result.SUCCESS

        # Test for stopping criterion.
        # The criterion is given by the following formula:
        #     |(f(past_x) - f(x))| / f(x) < delta
        if pf is not None:
            # We don't test the stopping criterion while k < past.
            if param.past <= k:
                # Compute the relative improvement from the past.
                rate = (pf[k % param.past] - fx) / fx

                # The stopping criterion.
                if K.abs(rate) < param.delta:
                    return result.STOP

            # Store the current value of the objective function.
            pf[k % param.past] = fx

        if param.max_iteration != 0 and param.max_iteration <= k:
            # Maximum number of iterations.
            return result.ERR_MAXIMUMITERATION

        # Update vectors s and y:
        #     s_{k+1} = x_{k+1} - x_{k} = sstep * d_{k}.
        #     y_{k+1} = g_{k+1} - g_{k}.
        it = lm.lm[end]
        vecdiff(it.s, x, xp, n)
        vecdiff(it.y, g, gp, n)

        # Compute scalars ys and yy. Notice that yy is used for
        # scaling the hessian matrix H_0 (Cholesky factor).
        ys = K.dot(it.y, it.s)
        yy = K.dot(it.y, it.y)
        it.ys = ys

        # Recursive formula to compute dir = -(H dot g).
        #   This is described in page 779 of:
        #   Jorge Nocedal.
        #   Updating Quasi-Newton Matrices with Limited Storage.
        #   Mathematics of Computation, Vol. 35, No. 151,
        #   pp. 773--782, 1980.
        bound = K.minimum(m, k)
        k += 1
        end = (end + 1) % m

        # Compute the steepest direction.
        if param.orthantwise_c == 0.0:
            vecncpy(d, g, n)
        else:
            vecncpy(d, pg, n)

        j = end
        for _ in range(bound):
            j = (j + m - 1) % m
            it = lm.lm[j]
            # alpha_{j} = rho_{j} s^{t}_{j} dot q_{k+1}.
            it.alpha = K.dot(it.s, d)
            it.alpha /= it.ys
            # q_{i} = q_{i+1} - alpha_{i} y_{i}.
            vecadd(d, it.y, -it.alpha, n)

        vecscale(d, ys / yy, n)

        for _ in range(bound):
            it = lm.lm[j]
            # beta_{j} = rho_{j} y ^ t_{j} dot gamma_{i}.
            beta = K.dot(it.y, d)
            beta /= it.ys
            # gamma_{i+1} = gamma_{i} + (alpha_{j} - beta_{j}) s_{j}.
            vecadd(d, it.s, it.alpha - beta, n)
            j = (j + 1) % m

        # Constrain the search direction for orthant-wise updates.
        if param.orthantwise_c != 0.0:
            for i in np.arange(param.orthantwise_start, param.orthantwise_end):
                if d[i] * pg[i] >= 0.0:
                    d[i] = 0.0

        # Now the search direction d is ready. We try step = 1 first.
        step = 1.0

