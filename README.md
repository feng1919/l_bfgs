# l_bfgs
A l_bfgs algorithm with keras implementation

This library is a keras implementation of Limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method.
The original FORTRAN source code is available at:
http://www.ece.northwestern.edu/~nocedal/lbfgs.html

Reference
- <a href="https://github.com/chokkan/liblbfgs">liblbfgs</a> by Naoaki Okazaki.
- <a href="http://www.ece.northwestern.edu/~nocedal/lbfgs.html">L-BFGS</a> by Jorge Nocedal.
- <a href="http://research.microsoft.com/en-us/downloads/b1eb1016-1738-4bd5-83a9-370c9d498a03/default.aspx">
  Orthant-Wise Limited-memory Quasi-Newton Optimizer for L1-regularized Objectives</a> by Galen Andrew.
- <a href="http://chasen.org/~taku/software/misc/lbfgs/">C port (via f2c)</a> by Taku Kudo.
- <a href="http://www.alglib.net/optimization/lbfgs.php">C#/C++/Delphi/VisualBasic6 port</a> in ALGLIB.
- <a href="http://cctbx.sourceforge.net/">Computational Crystallography Toolbox</a> includes
- <a href="http://cctbx.sourceforge.net/current_cvs/c_plus_plus/namespacescitbx_1_1lbfgs.html">scitbx::lbfgs</a>.
