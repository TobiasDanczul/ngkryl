from ngsolve import *
import numpy as np
from netgen.geom2d import unit_square
import ngkryl
import matplotlib.pyplot as plt

mesh = Mesh(unit_square.GenerateMesh(maxh = 0.04))
fes = H1(mesh, order = 1, dirichlet = "left|right|top|bottom")

u, v = fes.TnT()

# mass matrix
M = BilinearForm(fes, symmetric = True)
M +=  SymbolicBFI(u * v)

# stiffness matrix
A = BilinearForm(fes, symmetric = True)
A +=  SymbolicBFI(grad(u) * grad(v))

# right-hand side
rhs = LinearForm(fes)
rhs += x * y * (1-x) * (1-y) * v * dx

M.Assemble()
A.Assemble()
rhs.Assemble()

# L2 orthogonal projection onto fes
u = GridFunction(fes)
u.vec.data = M.mat.Inverse(fes.FreeDofs(), inverse = "sparsecholesky") * rhs.vec

L = ngkryl.FiniteElementMatrix(A.mat, M.mat, fes.FreeDofs())
lam_min = L.get_lam_min()
lam_max = L.get_lam_max()

# compute reference solution
operator = ngkryl.KrylovOperator(u.vec, L, "zolohat")
operator.compute_krylov_space(100)
reference = operator.apply(fun)

fun = [lambda x: 1 / x**0.5]

poletype = "zolohat"
operator_zolo = ngkryl.KrylovOperator(u.vec, L, poletype = "zolo")
operator_fully_automatic = ngkryl.KrylovOperator(u.vec, L, 
                                                 poletype = "fully_automatic")

operator_bura = ngkryl.KrylovOperator(u.vec, L, poletype = "bura")
operator_bura.set_function(fun[0])

operator_spectral = ngkryl.KrylovOperator(u.vec, L, poletype = "spectral")
trainset = np.geomspace(lam_min, lam_max, 100000)
operator_spectral.set_trainset(trainset)

# compute expected convergence rate of zolotarev poles on -[lam_min, lam_max]
cstar = operator_zolo.get_expected_convergence_rate()

# for plotting
dimension = []
error_zolo = []
error_fully_automatic = []
error_bura = []
error_spectral = []
yref = []

diff = u.vec.CreateVector()
for i in range(1,16):
    
    print('Krylov space dimension = ', i)
    dimension.append(i)
    
    operator_zolo.compute_krylov_space(i)
    operator_fully_automatic.compute_krylov_space(i)
    operator_bura.compute_krylov_space(i)
    operator_spectral.compute_krylov_space(i)
    
    zolo_approximation = operator_zolo.apply(fun)
    diff.data = reference[0] - zolo_approximation[0]
    error = sqrt(InnerProduct(diff, M.mat * diff)) 
    error_zolo.append(error)
    
    fully_automatic_approximation = operator_fully_automatic.apply(fun)
    diff.data = reference[0] - fully_automatic_approximation[0]
    error = sqrt(InnerProduct(diff, M.mat * diff)) 
    error_fully_automatic.append(error)
    
    bura_approximation = operator_bura.apply(fun)
    diff.data = reference[0] - bura_approximation[0]
    error = sqrt(InnerProduct(diff, M.mat * diff)) 
    error_bura.append(error)
    
    spectral_approximation = operator_spectral.apply(fun)
    diff.data = reference[0] - spectral_approximation[0]
    error = sqrt(InnerProduct(diff, M.mat * diff)) 
    error_spectral.append(error)
    
    yref.append(exp(-i*cstar))
    

plt.yscale("log")
plt.plot(dimension, error_zolo, 'r', label = "zolo")
plt.plot(dimension, error_fully_automatic, 'g--', label = "fully automatic")
plt.plot(dimension, error_bura, 'b-*', label = "bura")
plt.plot(dimension, error_spectral, '-o', label = "spectral")
plt.plot(dimension, yref, 'c:')
plt.show()