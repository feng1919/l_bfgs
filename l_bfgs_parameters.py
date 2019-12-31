
from l_bfgs_linesearch import LBFGSLineSearch
import numpy as np

'''
The parameters of L_BFGS().
'''
class LBFGSParameters():
    def __init__(self):
        ''''''
        ''' m
        The number of corrections to approximate the inverse hessian 
        matrix.
        An int value. The default value is 6, values less than 3 are 
        not recommended, large values will result in excessive 
        computing time.
        The L-BFGS routine stores the computation results of previous 
        'm' iterations to approximate the inverse hessian matrix of 
        the current iteration. This parameter controls the size of the 
        limited memories (corrections). 
        '''
        self.m = 6

        ''' epsilon
        Epsilon for convergence test.
        A float value. The default value is 1e-5.
        This parameter determines the accuracy with which the solution is 
        to be found. A minimization terminates when 
        ||g|| < epsilon * max(1, ||x||), where ||.|| denotes the Euclidean 
        (L2) norm. 
        '''
        self.epsilon = 1e-5

        ''' past
        Distance for delta-based convergence test.
        An int value. The default value is 0.
        This parameter determines the distance, in iterations, to compute 
        the rate of decrease of the objective function. If the value of 
        this parameter is zero, the library does not perform the delta-
        based convergence test.
        '''
        self.past = 0

        ''' delta
        Delta for convergence test.
        A float value. The default value is 1e-5.
        This parameter determines the minimum rate of decrease of the 
        objective function. The library stops iterations when the following 
        condition is met: (f' - f) / f < delta, where f' is the objective 
        value of 'past' iterations ago, and f is the objective value of the 
        current iteration.
        '''
        self.delta = 1e-5

        ''' max_iteration
        The maximum number of iterations.
        An int value. The default value is 0.
        The L_BFGS() function terminates an optimization process with 
        ERR_MAXIMUMITERATION status code when the iteration count exceedes 
        this parameter. Setting this parameter to zero continues an 
        optimization process until a convergence or error. 
        '''
        self.max_iteration = 0

        ''' linesearch
        The line search algorithm.
        This parameter specifies a line search algorithm to be used by the 
        L-BFGS routine.
        '''
        self.linesearch = LBFGSLineSearch.DEFAULT

        ''' max_linesearch
        The maximum number of trials for the line search.
        An int value. The default value is 40.
        This parameter controls the number of function and gradients 
        evaluations per iteration for the line search routine.
        '''
        self.max_linesearch = 40

        ''' min_step
        The minimum step of the line search routine.
        A float value. The default value is 1e-20. 
        This value need not be modified unless the exponents are too large 
        for the machine being used, or unless the problem is extremely badly 
        scaled (in which  the exponents should be increased).
        '''
        self.min_step = 1e-20

        ''' max_step
        The maximum step of the line search.
        A float value. The default value is 1e+20. 
        This value need not be modified unless the exponents are too large 
        for the machine being used, or unless the problem is extremely badly 
        scaled (in which  the exponents should be increased).
        '''
        self.max_step = 1e20

        ''' ftol
        A parameter to control the accuracy of the line search routine.
        A float value. The default value is 1e-4. 
        This parameter should be greater than zero and smaller than 0.5.
        '''
        self.ftol = 1e-4

        ''' wolfe
        A coefficient for the Wolfe condition.
        A float value. The default value is 0.9. This parameter should be 
        greater the ftol parameter and smaller than 1.0.
        This parameter is valid only when the backtracking line-search 
        algorithm is used with the Wolfe condition, BACKTRACKING_STRONG_WOLFE 
        or BACKTRACKING_WOLFE.
        '''
        self.wolfe = 0.9

        ''' gtol
        A parameter to control the accuracy of the line search routine.
        A float value. The default value is 0.9. 
        If the function and gradient evaluations are inexpensive with respect 
        to the cost of the iteration (which is sometimes the  when solving 
        very large problems) it may be advantageous to set this parameter to 
        a small value. A typical small value is 0.1. This parameter shuold be
        greater than the ftol parameter (1e-4) and smaller than 1.0.
        '''
        self.gtol = 0.9

        ''' xtol
        The machine precision for floating-point values.
        A float value. The default value is 1e-16.
        This parameter must be a positive value set by a client program to 
        estimate the machine precision. The line search routine will terminate 
        with the status code (ERR_ROUNDING_ERROR) if the relative width of 
        the interval of uncertainty is less than this parameter.
        '''
        self.xtol = 1.0e-16

        ''' orthantwise_c
        Coeefficient for the L1 norm of variables.
        A float value. The default value is 0.0.
        This parameter should be set to zero for standard minimization
        problems. Setting this parameter to a positive value activates
        Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method, which
        minimizes the objective function F(x) combined with the L1 norm |x|
        of the variables, {F(x) + C |x|}. This parameter is the coeefficient
        for the |x|, i.e., C. As the L1 norm |x| is not differentiable at
        zero, the library modifies function and gradient evaluations from
        a client program suitably, a client program thus have only to return
        the function value F(x) and gradients G(x) as usual.
        '''
        self.orthantwise_c = 0.0

        '''
        Start index for computing L1 norm of the variables.
        An int value. The default value is 0.
        This parameter is valid only for OWL-QN method (i.e.,orthantwise_c != 0). 
        This parameter b (0 <= b < N) specifies the index number from which the 
        library computes the L1 norm of the variables x,
            |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| .
        In other words, variables x_1, ..., x_{b-1} are not used for computing the 
        L1 norm. Setting b (0 < b < N), one can protect variables, x_1, ..., x_{b-1} 
        (e.g., a bias term of logistic regression) from being regularized.
        '''
        self.orthantwise_start = 0

        '''
        End index for computing L1 norm of the variables.
        An int value. The default value is -1.
        This parameter is valid only for OWL-QN method (i.e.,orthantwise_c != 0). 
        This parameter e (0 < e <= N) specifies the index number at which the 
        library stops computing the L1 norm of the variables x.
        '''
        self.orthantwise_end = -1
