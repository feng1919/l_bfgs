
from enum import Enum
import numpy as np
from l_bfgs_result import LBFGSResult
from l_bfgs_vector import *
import keras.backend as K

'''
Line search algorithms.
'''
class LBFGSLineSearch(Enum):
    ''''''
    '''
    MoreThuente method proposd by More and Thuente.
    '''
    MORETHUENTE = 0

    '''
    * Backtracking method with the Armijo condition.
    *  The backtracking method finds the step length such that it satisfies
    *  the sufficient decrease (Armijo) condition,
    *    - f(x + a * d) <= f(x) + LBFGSParameters.ftol * a * g(x)^T d,
    *
    *  where x is the current point, d is the current search direction, and
    *  a is the step length.
    '''
    BACKTRACKING_ARMIJO = 1

    '''
    The backtracking method with the defualt (regular Wolfe) condition.
    '''
    BACKTRACKING = 2

    '''
    * Backtracking method with regular Wolfe condition.
    *  The backtracking method finds the step length such that it satisfies
    *  both the Armijo condition (BACKTRACKING_ARMIJO)
    *  and the curvature condition,
    *    - g(x + a * d)^T d >= LBFGSParameters.wolfe * g(x)^T d,
    *
    *  where x is the current point, d is the current search direction, and
    *  a is the step length.
    '''
    BACKTRACKING_WOLFE = 2

    '''
    * Backtracking method with strong Wolfe condition.
    *  The backtracking method finds the step length such that it satisfies
    *  both the Armijo condition (BACKTRACKING_ARMIJO)
    *  and the following condition,
    *    - |g(x + a * d)^T d| <= LBFGSParameters.wolfe * |g(x)^T d|,
    *
    *  where x is the current point, d is the current search direction, and
    *  a is the step length.
    '''
    BACKTRACKING_STRONG_WOLFE = 3

    '''
    The default algorithm.
    '''
    DEFAULT = BACKTRACKING


def supported_linesearch(search_type):
    return search_type >= 0 and search_type < 4


# Placeholder function.
def line_search_proc(n, x, f, g, s, step, xp, gp, wa, evaluate, param):
    pass

# The standard line search algorithm.
def backtracking(n, x, f, g, s, step, xp, gp, wp, evaluate, param):

    # Check the input parameters for errors.
    if step <= 0.0:
        return LBFGSResult.ERR_INVALIDPARAMETERS, f, step

    # Compute the initial gradient in the search direction.
    dginit = K.dot(g, s)

    # Make sure that s points to a descent direction.
    if dginit > 0.0:
        return LBFGSResult.ERR_INCREASEGRADIENT, f, step

    # The initial value of the objective function.
    finit = f
    dgtest = dginit * param.ftol
    stp = step
    count = 0
    dec = 0.5
    inc = 2.1

    while True:
        # Calculate the variables of current step.
        veccpy(x, xp, n)
        vecadd(x, s, stp, n)

        # Evaluate the function and gradient values.
        f = evaluate(x, g, n, stp)

        # Increase the number of iterations.
        count += 1

        if f > (finit+dgtest):
            # The decrease is not satisfied, do one more round.
            width = dec
        else:
            # The sufficient decrease condition (Armijo condition).
            if param.linesearch == LBFGSLineSearch.BACKTRACKING_ARMIJO:
                # Exit with the Armijo condition.
                return count, f, stp
            # Check the Wolfe condition.
            dg = K.dot(g, s)
            if dg < param.wolfe * dginit:
                width = inc
            else:
                if param.linesearch == LBFGSLineSearch.BACKTRACKING_WOLFE:
                    # Exit with the regular Wolfe condition.
                    return count, f, stp
                # Check the strong Wolfe condition.
                if dg > -param.wolfe * dginit:
                    width = dec
                else:
                    return count, f, stp

        if stp < param.min_step:
            return LBFGSResult.ERR_MINIMUMSTEP, f, stp
        if stp > param.max_step:
            return LBFGSResult.ERR_MAXIMUMSTEP, f, stp
        if param.max_linesearch <= count:
            return LBFGSResult.ERR_MAXIMUMLINESEARCH, f, stp

        stp *= width


# The MoreThuente line search method proposed by More and Thuente.
def linesearch_morethuente(n, x, f, g, s, step, xp, gp, wp, evaluate, param):

    # Check the input parameters for errors.
    if step <= 0.0:
        return LBFGSResult.ERR_INVALIDPARAMETERS, f, step

    # Compute the initial gradient in the search direction.
    dginit = K.dot(g, s)

    # Make sure that s points to a descent direction.
    if dginit > 0.0:
        return LBFGSResult.ERR_INCREASEGRADIENT, f, step

    # Initialize local variables.
    finit = f
    count = 0
    uinfo = 0
    brackt = False
    stage1 = True
    dgtest = param.ftol * dginit
    width = param.max_step - param.min_step
    prev_width = 2.0 * width


    # The variables stx, fx, dgx contain the values of the step,
    # function, and directional derivative at the best step.
    # The variables sty, fy, dgy contain the value of the step,
    # function, and derivative at the other endpoint of
    # the interval of uncertainty.
    # The variables stp, f, dg contain the values of the step,
    # function, and derivative at the current step.
    stx = sty = 0.0
    fx = fy = finit
    dgx = dgy = dginit
    stp = step

    while True:
        # Set the minimum and maximum steps to correspond to the
        # present interval of uncertainty.
        if brackt:
            stmin = K.minimum(stx, sty)
            stmax = K.maximum(stx, sty)
        else:
            stmin = stx
            stmax = stp + 4.0 * (stp - stx)

        # Clip the step in the range of [stpmin, stpmax].
        stp = K.clip(stp, param.min_step, param.max_step)

        # If an unusual termination is to occur then let
        # stp be the lowest point obtained so far.
        if brackt and (stp <= stmin or stmax <= stp or
                       param.max_linesearch < count or
                       uinfo != 0 or
                       (stmax - stmin) <= (param.xtol * stmax)):
            stp = stx

        # Compute the current value of x.
        veccpy(x, xp, n)
        vecadd(x, s, stp, n)

        # Evaluate the function and gradient values.
        f = evaluate(x, g, n, stp)
        dg = K.dot(g, s)
        ftest1 = finit + stp * dgtest

        # Increase the number of iterations.
        count += 1

        # Test for errors and convergence.
        if brackt and (stp <= stmin or stmax <= stp or uinfo != 0):
            # Rounding errors prevent further progress.
            return LBFGSResult.ERR_ROUNDING_ERROR, f, stp

        if stp == param.max_step and f <= ftest1 and dg <= dgtest:
            # The step is the maximum value.
            return LBFGSResult.ERR_MAXIMUMSTEP, f, stp

        if stp == param.min_step and (ftest1 < f or dgtest <= dg):
            # The step is the minimum value.
            return LBFGSResult.ERR_MINIMUMSTEP, f, stp

        if brackt and (stmax - stmin) <= (param.xtol * stmax):
            # Relative width of the interval of uncertainty is at most xtol.
            return LBFGSResult.ERR_WIDTHTOOSMALL, f, stp

        if param.max_linesearch <= count:
            # Maximum number of iteration. */
            return LBFGSResult.ERR_MAXIMUMLINESEARCH, f, stp

        if f <= ftest1 and K.abs(dg) <= param.gtol * (-dginit):
            # The sufficient decrease condition and the directional derivative condition hold.
            return count, f, stp

        # In the first stage we seek a step for which the modified
        # function has a nonpositive value and nonnegative derivative.
        if stage1 and f <= ftest1 and (K.minimum(param.ftol, param.gtol) * dginit) <= dg:
            stage1 = False

        # A modified function is used to predict the step only if
        # we have not obtained a step for which the modified
        # function has a nonpositive function value and nonnegative
        # derivative, and if a lower function value has been
        # obtained but the decrease is not sufficient.
        if stage1 and ftest1 < f and f <= fx:
            # Define the modified function and derivative values.
            fm = f - stp * dgtest
            fxm = fx - stx * dgtest
            fym = fy - sty * dgtest
            dgm = dg - dgtest
            dgxm = dgx - dgtest
            dgym = dgy - dgtest

            # Call update_trial_interval() to update the interval of
            # uncertainty and to compute the new step.
            (uinfo,
             stx, fxm, dgxm,
             sty, fym, dgym,
             stp, fm, dgm,
             brackt) = update_trial_interval(stx, fxm, dgxm,
                                             sty, fym, dgym,
                                             stp, fm, dgm,
                                             stmin, stmax, brackt)

            # Reset the function and gradient values for f.
            fx = fxm + stx * dgtest
            fy = fym + sty * dgtest
            dgx = dgxm + dgtest
            dgy = dgym + dgtest
        else:
            # Call update_trial_interval() to update the interval of
            # uncertainty and to compute the new step.
            (uinfo,
             stx, fx, dgx,
             sty, fy, dgy,
             stp, f, dg,
             brackt)  = update_trial_interval(stx, fx, dgx,
                                              sty, fy, dgy,
                                              stp, f, dg,
                                              stmin, stmax, brackt)

        # Force a sufficient decrease in the interval of uncertainty.
        if brackt:
            if 0.66 * prev_width <= K.abs(sty - stx):
                stp = stx + 0.5 * (sty - stx)
            prev_width = width
            width = K.abs(sty - stx)

# Find a minimizer of an interpolated cubic function.
# All of the parameters and return value are float values.
#  @param  u       The value of one point, u.
#  @param  fu      The value of f(u).
#  @param  du      The value of f'(u).
#  @param  v       The value of another point, v.
#  @param  fv      The value of f(v).
#  @param  dv      The value of f'(v).
#  @retval         The minimizer of the interpolated cubic.
def CUBIC_MINIMIZER(u, fu, du, v, fv, dv):
    d = v - u
    theta = (fu - fv) * 3 / d + du + dv
    p = K.abs(theta)
    q = K.abs(du)
    r = K.abs(dv)
    s = K.maximum(K.maximum(p, q), r)
    a = theta / s
    gamma = s * K.sqrt(a * a - (du / s) * (dv / s))
    if (v < u):
        gamma = -gamma
    p = gamma - du + theta
    q = gamma - du + gamma + dv
    r = p / q
    return u + r * d

# Find a minimizer of an interpolated cubic function.
# All of the parameters and return value are float values.
#  @param  u       The value of one point, u.
#  @param  fu      The value of f(u).
#  @param  du      The value of f'(u).
#  @param  v       The value of another point, v.
#  @param  fv      The value of f(v).
#  @param  dv      The value of f'(v).
#  @param  xmin    The minimum value.
#  @param  xmax    The maximum value.
#  @retval         The minimizer of the interpolated cubic.
def CUBIC_MINIMIZER2(u, fu, du, v, fv, dv, xmin, xmax):
    d = v - u
    theta = (fu - fv) * 3 / d + du + dv
    p = K.abs(theta)
    q = K.abs(du)
    r = K.abs(dv)
    s = K.maximum(K.maximum(p, q), r)
    a = theta / s
    gamma = s * K.sqrt(K.maximum(0, a * a - (du / s) * (dv / s)))
    if (u < v):
        gamma = -gamma
    p = gamma - dv + theta
    q = gamma - dv + gamma + du
    r = p / q
    if r < 0.0 and gamma != 0.0:
        return v - r * d
    elif a < 0.0:
        return xmax
    else:
        return xmin

# Find a minimizer of an interpolated quadratic function.
#  @param  u       The value of one point, u.
#  @param  fu      The value of f(u).
#  @param  du      The value of f'(u).
#  @param  v       The value of another point, v.
#  @param  fv      The value of f(v).
#  @retval         The minimizer of the interpolated cubic.
def QUARD_MINIMIZER(u, fu, du, v, fv):
    a = v - u
    return u + du / ((fu - fv) / a + du) / 2 * a

#Find a minimizer of an interpolated quadratic function.
# @param  u       The value of one point, u.
# @param  du      The value of f'(u).
# @param  v       The value of another point, v.
# @param  dv      The value of f'(v).
# @retval         The minimizer of the interpolated cubic.
def QUARD_MINIMIZER2(u, du, v, dv):
    a = u - v
    return v + dv / (dv - du) * a

#Update a safeguarded trial value and interval for line search.
#
# The parameter x represents the step with the least function value.
# The parameter t represents the current step. This function assumes
# that the derivative at the point of x in the direction of the step.
# If the bracket is set to true, the minimizer has been bracketed in
# an interval of uncertainty with endpoints between x and y.
#
# @param  x       The pointer to the value of one endpoint.
# @param  fx      The pointer to the value of f(x).
# @param  dx      The pointer to the value of f'(x).
# @param  y       The pointer to the value of another endpoint.
# @param  fy      The pointer to the value of f(y).
# @param  dy      The pointer to the value of f'(y).
# @param  t       The pointer to the value of the trial value, t.
# @param  ft      The pointer to the value of f(t).
# @param  dt      The pointer to the value of f'(t).
# @param  tmin    The minimum value for the trial value, t.
# @param  tmax    The maximum value for the trial value, t.
# @param  brackt  The pointer to the predicate if the trial value is
#                 bracketed.
# @retval int     Status value. Zero indicates a normal termination.
#
# @see
#     Jorge J. More and David J. Thuente. Line search algorithm with
#     guaranteed sufficient decrease. ACM Transactions on Mathematical
#     Software (TOMS), Vol 20, No 3, pp. 286-307, 1994.
def update_trial_interval(x, fx, dx, y, fy, dy, t, ft, dt, tmin, tmax, brackt):

    nx, nfx, ndx, ny, nfy, ndy, nt, nft, ndt, nbrackt = x, fx, dx, y, fy, dy, t, ft, dt, brackt

    # Check the input parameters for errors.
    if nbrackt:
        if t <= K.minimum(x, y) or K.maximum(x, y) <= t:
            # The trival value t is out of the interval.
            return LBFGSResult.ERR_OUTOFINTERVAL, nx, nfx, ndx, ny, nfy, ndy, nt, nft, ndt, nbrackt
        if 0. <= dx * (t - x):
            # The function must decrease from x.
            return LBFGSResult.ERR_INCREASEGRADIENT, nx, nfx, ndx, ny, nfy, ndy, nt, nft, ndt, nbrackt
        if tmax < tmin:
            # Incorrect tmin and tmax specified.
            return LBFGSResult.ERR_INCORRECT_TMINMAX, nx, nfx, ndx, ny, nfy, ndy, nt, nft, ndt, nbrackt

    # flag whether the value is close to upper bound.
    bound = False

    # flag whether dt and dx have the same sign.
    dsign = dt * dx > 0.0

    # minimizer of an interpolated cubic.
    mc = 0.0
    # minimizer of an interpolated quadratic.
    mq = 0.0

    # Trial value selection.
    if fx < ft:
        # Case 1: a higher function value.
        # The minimum is brackt. If the cubic minimizer is closer
        # to x than the quadratic one, the cubic one is taken, else
        # the average of the minimizers is taken.
        nbrackt = True
        bound = True
        mc = CUBIC_MINIMIZER(x, fx, dx, t, ft, dt)
        mq = QUARD_MINIMIZER(x, fx, dx, t, ft)
        if K.abs(mc - x) < K.abs(mq - x):
            nt = mc
        else:
            nt = mc + 0.5 * (mq - mc)
    elif dsign:
        # Case 2: a lower function value and derivatives of
        # opposite sign. The minimum is brackt. If the cubic
        # minimizer is closer to x than the quadratic (secant) one,
        # the cubic one is taken, else the quadratic one is taken.
        nbrackt = True
        bound = False
        mc = CUBIC_MINIMIZER(x, fx, dx, t, ft, dt)
        mq = QUARD_MINIMIZER2(x, dx, t, dt)
        if K.abs(mc - t) > K.abs(mq - t):
            nt = mc
        else:
            nt = mq
    elif K.abs(dt) < K.abs(dx):
        # Case 3: a lower function value, derivatives of the
        # same sign, and the magnitude of the derivative decreases.
        # The cubic minimizer is only used if the cubic tends to
        # infinity in the direction of the minimizer or if the minimum
        # of the cubic is beyond t. Otherwise the cubic minimizer is
        # defined to be either tmin or tmax. The quadratic (secant)
        # minimizer is also computed and if the minimum is brackt
        # then the the minimizer closest to x is taken, else the one
        # farthest away is taken.
        bound = True
        mc = CUBIC_MINIMIZER2(x, fx, dx, t, ft, dt, tmin, tmax)
        mq = QUARD_MINIMIZER2(x, dx, t, dt)
        if nbrackt:
            if K.abs(t - mc) < K.abs(t - mq):
                nt = mc
            else:
                nt = mq
        else:
            if K.abs(t - mc) > K.abs(t - mq):
                nt = mc
            else:
                nt = mq
    else:
        # Case 4: a lower function value, derivatives of the
        # same sign, and the magnitude of the derivative does
        # not decrease. If the minimum is not brackt, the step
        # is either tmin or tmax, else the cubic minimizer is taken.
        bound = False
        if nbrackt:
            nt = CUBIC_MINIMIZER(t, ft, dt, y, fy, dy)
        elif x < t:
            nt = tmax
        else:
            nt = tmin

    # Update the interval of uncertainty. This update does not
    # depend on the new step or the case analysis above.
    #
    # - Case a: if f(x) < f(t),
    #     x <- x, y <- t.
    # - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
    #     x <- t, y <- y.
    # - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0,
    #     x <- t, y <- x.
    if fx < ft:
        # Case a
        ny = t
        nfy = ft
        ndy = dt
    else:
        # Case c
        if dsign:
            ny = x
            nfy = fx
            ndy = dx
        # Cases b and c
        nx = t
        nfx = ft
        ndx = dt

    # Clip the new trial value in [tmin, tmax].
    nt = K.clip(nt, tmin, tmax)

    # Redefine the new trial value if it is close to the upper bound
    # of the interval.
    if nbrackt and bound:
        mq = x + 0.66 * (y - x)
        if x < y:
            nt = K.minimum(mq, nt)
        else:
            nt = K.maximum(mq, nt)

    return 0, nx, nfx, ndx, ny, nfy, ndy, nt, nft, ndt, nbrackt


# The Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) line search algorithm.
def backtracking_owlqn(instance, finit, g, s, step, xp, gp, wp, evaluate, param):
    # TODO
    pass

def owlqn_x1norm(x, start, n):
    s = 0.0
    for i in np.arange(start, n):
        s += K.abs(x[i])
    return s

def owlqn_project(d, sign, start, end):
    for i in np.arange(start, end):
        if d[i] * sign[i] <= 0.0:
            d[i] = 0.0

def owlqn_pseudo_gradient(pg, x, g, n, c, start, end):

    assert start >= 0 and start <= end, "Invalid 'start' parameter."
    assert end <= n, "Invalid 'end' parameter."

    # Compute the negative of gradients.
    pg[:start] = g[:start]
    pg[end:] = g[end:]

    # Compute the psuedo-gradients.
    for i in np.arange(start, end):
        if x[i] < 0.0:
            # Differentiable.
            pg[i] = g[i] - c
        elif x[i] > 0.0:
            # Differentiable.
            pg[i] = g[i] + c
        else:
            if g[i] < -c:
                # Take the right partial derivative.
                pg[i] = g[i] + c
            elif g[i] > c:
                # Take the left partial derivative.
                pg[i] = g[i] - c
            else:
                pg[i] = 0.0
