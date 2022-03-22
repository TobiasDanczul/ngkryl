# General Info

Ngmor is a Python module that provides routines for applying rational Krylov approximations in the Python interface of NGSolve and should be understood as complementary material of the author's doctoral thesis.

# Technologies

The package is implemented in pure Python (version 3.8.10). To run the code, make sure that you have installed
- numpy (version 1.22.2 or higher),
- baryrat[^2] (version 1.3.0 or higher),
- scipy (version 1.7.0 or higher),
- and NGSolve[^1] (version v6.2.2105 or higher).

[^1]: https://ngsolve.org/ 
[^2]: https://github.com/c-f-h/baryrat
 

# Examples of Use

Here is an example of how to use our module to aproximate solutions to the fractional poisson problem. First, we make use of the Finite Element library NGSolve to construct the desired finite element space, the mass matrix, the stiffness matrix, and the desired vector.
```python
from ngsolve import *
from netgen.geom2d import unit_square

mesh = Mesh(unit_square.GenerateMesh(maxh = 0.01))
fes = H1(mesh, order = 1, dirichlet = "left|right|top|bottom")

u,v = fes.TnT()

M = BilinearForm(fes, symmetric = True)
M +=  SymbolicBFI(u * v)
A = BilinearForm(fes, symmetric = True)
A +=  SymbolicBFI(grad(u) * grad(v))
rhs = LinearForm(fes)
rhs += x * (1-x) * y * (1-y) * v * dx

M.Assemble()
A.Assemble()
rhs.Assemble()

u = GridFunction(fes)
u.vec.data = m.mat.Inverse(fes.FreeDofs(), inverse = "sparsecholesky") * rhs.vec
```
After importing our module, we establish the matrix L = M^-1A whose (negative) fractional power we want to compute.
```python
import ngmor

L = ngmor.FiniteElementMatrix(A.mat, M.mat, fes.FreeDofs())
```
The Krylov operator for approximating matrix vector products of f(L) and u is instantiated as follows.
```python
krylov_operator = ngmor.KrylovOperator(u.vec, L)
```
Before computing the reduced basis approximation, the krylov space needs to be established. In this example, we choose a Krylov space of dimension 10.
```python
krylov_operator.compute_krylov_space(10)
```
To approximate f(L)u, the following function is called 
```python
f = [lambda x: 1 / x**0.5]
krylov_approximation = krylov_operator.apply(f)[0]
```
The vector encoded in `krylov_approximation` represents the approximation of the solution to a fractional Poisson problem with exponent s = 0.5. Note that the list `f` may contain more than one lambda function, in which case a list of rational krylov surrogates is returned, at negligible extra costs. 

By default, the `KrylovOperator` class usese the Zolotarev poles on the negative real line for building the rational Krylov space. Several other pole choices are possible by setting the flag `poletype` in the class' constructor to
- zolo: The negative spectral interval of L,
- eds: The EDS poles on the negative spectral interval of L,
- edshat: The EDS poles on the negative real line,
- automatic: The automatic poles on the negative spectral interval of L,
- fully_automatic: The fully automatic poles on the negative spectral interal of L.

Provided a sufficiently fine training set, assigned to the operator using the `.set_trainset` method
```python
lam_min = krylov_operator.get_lam_min()
lam_max = krylov_operator.get_lam_max()
trainset = np.geomspace(lam_min, lam_max, 10000)
krylov_operator.set_trainset(trainset)
```
one can additionally choose between
- spectral: The spectral poles,
- greedy: The weak greedy poles using a residual based error estimator.

To implement the BURA poles, the desired function needs to be assigned to the Krylov operator in terms of
```python
function = f[0]
krylov_operator.set_function(function)
```
Using the flag `poletype = bura`, the rational Krylov space is cumputed using the BURA poles, which typically yield the best performance among all the pole configurations listed above. Unlike the others, however, the BURA poles are not suitable for multi-querring the surrogate for multiple different functions.

# Project Status

Further improvements to be implemented:

- Implementing the weak greedy poles in a computationally feasible manner as suggested by the literature; see e.g., 
  - A. Buhr, C. Engwer, M. Ohlberger, and S. Rave. A numerically stable a posteriori error estimator for reduced basis approximations of elliptic equations. Proceedings of the 11th World Congress on Computational Mechanics, pages 4094–4102. CIMNE, Barcelona, 2014.
  - C. Yanlai, J. Jiang, and A. Narayan. A robust error estimator and a residual-free error indicator for reduced basis methods. Computers & Mathematics  with Applications, 77(7):1963–1979, 2019.

# References
For more information on model order reduction schemes for fractional diffusion problems, we refer to

- T. Danczul and J. Schöberl. A reduced basis method for fractional diffusion operators II. Journal of Numerical Mathematics, 2021.
- T. Danczul and C. Hofreither. On rational Krylov and reduced basis methods for fractional diffusion. Journal of Numerical Mathematics, 2021.
- C. Hofreither, An algorithm for best rational approximation based on barycentric rational interpolation. Numerical Algorithms, 88(1): 365-388, 2021.
- T. Danczul, C. Hofreither, and J. Schöberl. A unified rational Krylov method for elliptic and parabolic fractional diffusion problems. arXiv preprint, 2021.
- T. Danczul. Model Order Reduction for Fractional Diffusion Problems, PhD Thesis, 2021
