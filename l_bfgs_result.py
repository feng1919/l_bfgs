
from enum import Enum

'''
Return values of l_bfgs().
'''
class LBFGSResult(Enum):
    # L-BFGS reaches convergence.
    SUCCESS = 0
    CONVERGENCE = 0
    STOP = 0
    #The initial variables already minimize the objective function.
    ALREADY_MINIMIZED = 0
    #Terminated by use.
    USER_TERMINATE = 1

    #Unknown error.
    ERR_UNKNOWNERROR = -1024
    #Logic error.
    ERR_LOGICERROR = -1023
    #Insufficient memory, aka 'OOM'.
    ERR_OUTOFMEMORY = -1022
    #The minimization process has been canceled.
    ERR_CANCELED = -1021
    #Invalid number of variables specified.
    ERR_INVALID_N = -1020
    '''
    #Invalid number of variables (for SSE) specified. 
    ERR_INVALID_N_SSE,
    #The array x must be aligned to 16 (for SSE). 
    ERR_INVALID_X_SSE,
    '''
    #Invalid parameter LBFGSParameters.epsilon specified.
    ERR_INVALID_EPSILON = -512
    #Invalid parameter LBFGSParameters.past specified.
    ERR_INVALID_TESTPERIOD = -511
    #Invalid parameter LBFGSParameters.delta specified.
    ERR_INVALID_DELTA = -510
    #Invalid parameter LBFGSParameters.linesearch specified.
    ERR_INVALID_LINESEARCH = -509
    #Invalid parameter LBFGSParameters.max_step specified.
    ERR_INVALID_MINSTEP = -508
    #Invalid parameter LBFGSParameters.max_step specified.
    ERR_INVALID_MAXSTEP = -507
    #Invalid parameter LBFGSParameters.ftol specified.
    ERR_INVALID_FTOL = -506
    #Invalid parameter LBFGSParameters.wolfe specified.
    ERR_INVALID_WOLFE = -505
    #Invalid parameter LBFGSParameters.gtol specified.
    ERR_INVALID_GTOL = -504
    #Invalid parameter LBFGSParameters.xtol specified.
    ERR_INVALID_XTOL = -503
    #Invalid parameter LBFGSParameters.max_linesearch specified.
    ERR_INVALID_MAXLINESEARCH = -502
    #Invalid parameter LBFGSParameters.orthantwise_c specified.
    ERR_INVALID_ORTHANTWISE = -501
    #Invalid parameter LBFGSParameters.orthantwise_start specified.
    ERR_INVALID_ORTHANTWISE_START = -500
    #Invalid parameter LBFGSParameters.orthantwise_end specified.
    ERR_INVALID_ORTHANTWISE_END = -499
    #The line-search step went out of the interval of uncertainty.
    ERR_OUTOFINTERVAL = -498
    #A logic error occurred, alternatively, the interval of uncertainty became too small.
    ERR_INCORRECT_TMINMAX = -497
    #A rounding error occurred, alternatively, no line-search step
    #satisfies the sufficient decrease and curvature conditions.
    ERR_ROUNDING_ERROR = -496
    #The line-search step became smaller than LBFGSParameters.min_step.
    ERR_MINIMUMSTEP = -495
    #The line-search step became larger than LBFGSParameters.max_step.
    ERR_MAXIMUMSTEP = -494
    #The line-search routine reaches the maximum number of evaluations.
    ERR_MAXIMUMLINESEARCH = -493
    #The algorithm routine reaches the maximum number of iterations.
    ERR_MAXIMUMITERATION = -492
    #Relative width of the interval of uncertainty is at most LBFGSParameters.xtol.
    ERR_WIDTHTOOSMALL = -491
    #A logic error (negative line-search step) occurred.
    ERR_INVALIDPARAMETERS = -490
    #The current search direction increases the objective function value.
    ERR_INCREASEGRADIENT = -489

    _result_description = {
        # Also handles CONVERGENCE.
        SUCCESS:"Success: reached convergence (gtol).",
        CONVERGENCE:"Success: reached convergence (gtol).",
        STOP:"Success: met stopping criteria (ftol).",
        ALREADY_MINIMIZED:"The initial variables already minimize the objective function.",

        ERR_UNKNOWNERROR:"Unknown error.",
        ERR_LOGICERROR:"Logic error.",
        ERR_OUTOFMEMORY:"Insufficient memory.",
        ERR_CANCELED:"The minimization process has been canceled.",
        ERR_INVALID_N:"Invalid number of variables specified.",
        # ERR_INVALID_N_SSE:"Invalid number of variables (for SSE) specified.",
        # ERR_INVALID_X_SSE:"The array x must be aligned to 16 (for SSE).",
        ERR_INVALID_EPSILON:"Invalid parameter LBFGSParameters.epsilon specified.",
        ERR_INVALID_TESTPERIOD:"Invalid parameter LBFGSParameters.past specified.",
        ERR_INVALID_DELTA:"Invalid parameter LBFGSParameters.delta specified.",
        ERR_INVALID_LINESEARCH:"Invalid parameter LBFGSParameters.linesearch specified.",
        ERR_INVALID_MINSTEP:"Invalid parameter LBFGSParameters.max_step specified.",
        ERR_INVALID_MAXSTEP:"Invalid parameter LBFGSParameters.max_step specified.",
        ERR_INVALID_FTOL:"Invalid parameter LBFGSParameters.ftol specified.",
        ERR_INVALID_WOLFE:"Invalid parameter LBFGSParameters.wolfe specified.",
        ERR_INVALID_GTOL:"Invalid parameter LBFGSParameters.gtol specified.",
        ERR_INVALID_XTOL:"Invalid parameter LBFGSParameters.xtol specified.",
        ERR_INVALID_MAXLINESEARCH:"Invalid parameter LBFGSParameters.max_linesearch specified.",
        ERR_INVALID_ORTHANTWISE:"Invalid parameter LBFGSParameters.orthantwise_c specified.",
        ERR_INVALID_ORTHANTWISE_START:"Invalid parameter LBFGSParameters.orthantwise_start specified.",
        ERR_INVALID_ORTHANTWISE_END:"Invalid parameter LBFGSParameters.orthantwise_end specified.",
        ERR_OUTOFINTERVAL:"The line-search step went out of the interval of uncertainty.",
        ERR_INCORRECT_TMINMAX:"A logic error occurred, alternatively, the interval of \
                               uncertainty became too small.",
        ERR_ROUNDING_ERROR:"A rounding error occurred, alternatively, no line-search step\
                            satisfies the sufficient decrease and curvature conditions.",
        ERR_MINIMUMSTEP:"The line-search step became smaller than LBFGSParameters.min_step.",
        ERR_MAXIMUMSTEP:"The line-search step became larger than LBFGSParameters.max_step.",
        ERR_MAXIMUMLINESEARCH:"The line-search routine reaches the maximum number of evaluations.",
        ERR_MAXIMUMITERATION:"The algorithm routine reaches the maximum number of iterations.",
        ERR_WIDTHTOOSMALL:"Relative width of the interval of uncertainty is at most LBFGSParameters.xtol.",
        ERR_INVALIDPARAMETERS:"A logic error (negative line-search step) occurred.",
        ERR_INCREASEGRADIENT:"The current search direction increases the objective function value.",
    }

    def code_to_string(code):
        assert code in LBFGSResult._result_description, "Invalid code: "+str(code)
        return LBFGSResult._result_description[code]
