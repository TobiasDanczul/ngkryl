from ngsolve import *
import config
import numpy as np
from numpy import pi
import scipy
from scipy.special import ellipk, ellipkm1, ellipj
import scipy.integrate as integrate
from scipy.linalg import eigh
import baryrat


"""----------------------------------Helper Functions-----------------------"""
dn = lambda x, m: ellipj(x, m)[2]
cosh = lambda x: (exp(x) + exp(-x)) / 2
sech = lambda x: 1 / cosh(x)
sinh = lambda x:  (exp(x) - exp(-x)) / 2
tanh = lambda x: sinh(x) / cosh(x)


def compute_lam_max(stiffmat, massmat, freedofs, tol = 1e-8, maxit = 100000):
    """
    Computes an approximation of the maximal eigenvalue of the generalized
    eigenvalue problem stiffmat * v = lam * massmat * v using the rational
    Arnoldi algorithm.
    
    Parameters
    ----------
    stiffmat : ngsolve.la.SparseMatrixd
        The stiffness matrix.
    massmat : ngsolve.la.SparseMatrixd
        The mass matrix.
    freedofs : pyngcore.BitArray
        The degrees of freedome of the finite element space.
    tol : float, optional
        Determines when to stop the iteration. The default is 1e-8.
    maxit : int, optional
        The maximal iteration number. The default is 100000.

    Returns
    -------
    lam_max : float
        An approximation of the maximal eigenvalue.
    """
    N = stiffmat.width
    tmp1 = BaseVector(N)
    tmp2 = tmp1.CreateVector()
    eigenvec_max = tmp1.CreateVector()
    eigenvec_max.FV().NumPy()[:] = np.random.rand(N)

    lam_max_old = 0
    diff = 100
    counter = 0
    with TaskManager():
        massmat_inv = massmat.Inverse(freedofs, inverse = "sparsecholesky")
        while (diff > tol and counter < maxit):
            tmp1.data = stiffmat * eigenvec_max
            tmp2.data = massmat_inv * tmp1

            eigenvec_max.data = 1/sqrt(InnerProduct(tmp2,tmp2)) * tmp2
            lam_max = InnerProduct(eigenvec_max, tmp2) / InnerProduct(eigenvec_max, eigenvec_max)

            diff = abs(lam_max - lam_max_old)
            lam_max_old = lam_max
            counter += 1

        if (counter == maxit):
            print("RuntimeWarning: Power Iteration did not converge!")
            return lam_max

        return lam_max


def compute_lam_min(stiffmat, massmat, freedofs, dim = 20):
    """
    Computes an approximation of the minimal eigenvalue of the generalized
    eigenvalue problem stiffmat * v = lam * massmat * v using the rational
    Arnoldi algorithm.
    
    Parameters
    ----------
    stiffmat : ngsolve.la.SparseMatrixd
        The stiffness matrix.
    massmat : ngsolve.la.SparseMatrixd
        The mass matrix.
    freedofs : pyngcore.BitArray
        The degrees of freedome of the finite element space.
    dim : int, optional
        The subspace dimension for the rational Arnoldi. The default is 20.

    Returns
    -------
    lam_min : float
        The approximation of the minimal eigenvalue.
    """
    N = stiffmat.width
    eigenvec = [BaseVector(N) for i in range(dim)]
    lam = ArnoldiSolver(stiffmat, massmat, freedofs, eigenvec, 0)
    lam_min = np.real(lam[0])

    return lam_min


def dn1(x, m1):
    """
    Provides a numerically stable implementation of the Jacobi elliptic 
    function whenever m = 1-m1 is close to 1
    
    Parameters
    ----------
    x : float
        Argument of the Jacobi elliptic function.
    m1 : float
        The complimentary modulus.

    Returns
    -------
    val : float
        The value of the Jacobi elliptic function with argument x and elliptic
        modulus m = 1-m1
    """
    if (m1 < 1e-8):
            val = sech(x) + 0.25 * m1 * (sinh(x) * cosh(x) + x) * tanh(x) * sech(x)
    else:
            val = dn(x, 1-m1)
    
    return val                


def compute_zpoint(delta, j, order):
    """
    Computes the jth Zolotarev point for the given order on [delta, 1]. 

    Parameters
    ----------
    delta : float
        delta < 1 determines the interval on which the points are computed.
    j : int
        The index of the Zolotarev point
    order : int
        The total number of the Zolotarev points.

    Returns
    -------
    zpoint : float
        The jth Zolotarev point for the given order on [delta, 1]. 
    """
    assert(delta < 1)
    
    m1 = delta**2
    K = ellipkm1(m1)
    zpoint = dn1( ( 2 * (order - j) + 1 ) / (2 * order) * K, m1)
    
    return zpoint

def compute_integral(delta, t):
    """
    Computes the integral of 1 / sqrt((x - delta**2) * x * (1-x)) from delta²
    to t.
    
    Parameters
    ----------
    delta : float
        The lower bound of the integral.
    t : float
        The upper bound of the integral

    Returns
    -------
    integral : float
        The value of the integral
    """
    integrand = lambda x: 1 / sqrt((x - delta**2) * x * (1-x))
    integral = integrate.quad(integrand, delta**2 + 1e-15, t, limit = 10000)[0]
    
    return integral


def compute_zero(delta, shift, tol, maxiter):
    """
    Computes an approximation of the root of the function that maps each t 
    to the integral of 1/(2*K) * 1 / sqrt((x - delta**2) * x * (1-x)) from 
    delta² to t minus the shift using a damped Newton method.
    
    Parameters
    ----------
    delta : float
        Lower bound of the zero.
    shift : float
    tol : float
        The allowable error of the zero value.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    tn : float
        Estimated location where the function is zero.
    """
    K = ellipkm1(delta**2)
    f = lambda t: 1 / (2*K) * compute_integral(delta, t) - shift
    fprime = lambda t: 1 / (2*K) * 1 / sqrt((t - delta**2) * t * (1-t))
     
    # initial guess for newtin
    t0 = shift * (delta-delta**2) / compute_integral(delta, delta) + delta**2        
    tn = t0
    
    # damping parameter
    alpha = 1
    for n in range(maxiter):
        f_tn = f(tn)
        if abs(f_tn) < tol:
            return tn
        
        fprime_tn = fprime(tn)
        if fprime_tn == 0:
            raise Exception("Zero derivative. No solution found.")
        
        tmp = tn - f_tn/fprime_tn
        # reduce stepsize if iterate is outside of admissible interval
        while tmp >= 1 or tmp <= delta**2:
            tmp = tn - alpha * f_tn/fprime_tn
            alpha = 0.5*alpha  
        
        tn = tn - alpha * f_tn/fprime_tn
        alpha = 1
    
    raise Exception("Exceeded maximum iterations. No solution found.")


def rat_fun(x, poles):
    """
    Computes the value of a rational function with poles in poles and roots in
    -poles; see Definition 8.29 from the doctoral thesis "Model Order Reduction
    for Fractional Diffusion Problems" for a precise definition.

    Parameters
    ----------
    x : float
        The argument of the rational function.
    poles : list
        A list of poles.

    Returns
    -------
    val : float
        The value of the rational function at x.
    """
    val = 1
    for pole in poles:
        val *= (x + pole) / (x - pole)
        
    return val

def rat_fun_p(x, poles):
    """
    Computes the value of the derivative of rat_fun in x.

    Parameters
    ----------
    x : float
        Argument of the rational function.
    poles : list
        A list of poles.

    Returns
    -------
    val : float
        The value of the derivative of the rational function in x.
    """
    val = 0
    for j, pole in enumerate(poles):
        poles_j = np.copy(poles).tolist()
        del poles_j[j]
        val += pole / (x - pole)**2 * rat_fun(x, poles_j)
    
    val *= -2
    
    return val

def rat_fun_pp(x, poles):
    """
    Computes the value of the second derivative of rat_fun in x.

    Parameters
    ----------
    x : float
        The argument of the function.
    poles : list
        A list of poles.

    Returns
    -------
    val : float
        The value of the second derivative of the rational function in x.
    """
    val = 0
    for j, pole in enumerate(poles):
        poles_j = np.copy(poles).tolist()
        del poles_j[j]
        val += pole / (x - pole)**2 * (rat_fun_p(x, poles_j) - 2 * rat_fun(x, poles_j) / (x - pole))
   
    val *= -2
       
    return val

def approximate_extremum(poles, trainset):
    """
    Computes the extremum of the rational function rat_fun on the training set
    trainset.

    Parameters
    ----------
    poles : list
        a list of poles.
    trainset : list
        The discrepte trainset on whose convex hull the extremum is sought.

    Returns
    -------
    extremum : float
        The extremum of rat_fun on the training set.
    """
    val = rat_fun(trainset, poles)
    j_max = np.argmax(abs(val))
    
    extremum = trainset[j_max]
    
    return extremum

"""-------------------------------------------------------------------------"""
class FiniteElementMatrix:
    """
    A class for representing a finite elment matrix of the form L = M⁻¹A.
    
    Attributes
    ----------
    A : ngsolve.la.SparseMatrixd
        The stiffness matrix.
    M : ngsolve.la.SparseMatrixd
        The mass matrix.
    freedofs : pyngcore.BitArray
        The degrees of freedome of the finite element space.
    lam_min : float
        The minimal eigenvalue of A * v = lam * M * v.
    lam_max : float
        maximal eigenvalue
        The maximal eigenvalue of A * v = lam * M * v.
    """
    def __init__(self, A, M, freedofs,
                 lam_min = None, lam_max = None):
        """
        Constructs all the necessary attributes for the FiniteElementMatrix
        object.

        Parameters
        ----------
        A : ngsolve.la.SparseMatrixd
            The stiffness matrix.
        M : ngsolve.la.SparseMatrixd
            The mass matrix.
        freedofs : pyngcore.BitArray
            The degrees of freedome of the finite element space.
        lam_min : float, optional
            The minimal eigenvalue of A * v = lam * M * v. The default is None.
        lam_max : float, optional
            The maximal eigenvalue of A * v = lam * M * v. The default is None.
        """
        self.A = A
        self.M = M
        self.freedofs = freedofs

        self.lam_min = lam_min if lam_min else compute_lam_min(
            self.A, self.M, self.freedofs)
        self.lam_max = lam_max if lam_max else compute_lam_max(
            self.A, self.M, self.freedofs)
    
    def get_lam_max(self):
        return self.lam_max

    def get_lam_min(self):
        return self.lam_min

    def get_condition(self):
        """returns the ratio of the largest and the smalles eigenvalue"""
        return self.lam_max / self.lam_min

    def get_freedofs(self):
        return self.freedofs
    
    def get_A(self):
        return self.A
    
    def get_M(self):
        return self.M

"""-------------------------------------------------------------------------"""
class KrylovOperator:
    """
    A class for representing rational Krylov approximations of the matrix
    vector product f(L) * v, where
        -) L is a FiniteElementMatrix,
        -) v is a vector,
        -) and f is a function defined on  the spectrum of L.
    
    Attributes
    ----------
    vector : ngsolve.la.BaseVector
        The vector whose matrix vector approximation is sought.
    L : FiniteElementMatrix
        The respective matrix L = M⁻¹A.
    freedofs : pyngcore.BitArray
        The degrees of freedome of the finite element space associated to L.
    lam_min : float
        The minimal eigenvalue of L.
    lam_max : float, optional
        The maximal eigenvalue of L.
    """
    def __init__(self, vector, L, poletype = 'zpoles'):
        
        self.vector = vector
        
        self.L = L
        self.poletype = poletype
        self.poles = []
         
        # for later use
        self.M_vector = vector.CreateVector()
        self.M_vector.data = L.get_M() * vector
        
        # temporary help vectors/matrices
        self.tmp1 = vector.CreateVector()
        self.tmp2 = vector.CreateVector()
        self.mat = L.get_A().CreateMatrix()
        self.mat_inverse = None
        
        if poletype == "gpoles":
            self.projector = Projector(L.get_freedofs(), True)
        
        # Create reduced basis
        self.basis = MultiVector(vector, 1)
        self.basis[0].data = 1 / sqrt(InnerProduct(vector, L.get_M() * vector)) \
            * vector
        self.update()
            
    def get_ritzvalues(self):
        return self.ritzvalues        
    
    def get_M_vector(self):
        return self.M_vector
    
    def get_vector(self):
        return self.vector
    
    def get_L(self):
        return self.L
    
    def get_M(self):
        return self.get_L().get_M()
    
    def get_A(self):
        return self.get_L().get_A()
    
    def dim(self):
        """returns the dimension of the Krylov space"""
        return len(self.basis)
    
    def get_poles(self):
        """returns the list of poles"""
        return self.poles
    
    def get_poletype(self):
        """returns the type of poles"""
        return self.poletype
    
    def get_lam_min(self):
        """returns the lower spectral bound of the FiniteElementMatrix"""
        return self.L.get_lam_min()

    def get_lam_max(self):
        """returns the lower spectral bound of the FiniteElementMatrix"""
        return self.L.get_lam_max()
    
    def is_nested(self):
        """returns True if pole sequence is nested; False otherwise"""
        return config.nested[self.poletype]
    
    def set_trainset(self, trainset):
        """assign trainset"""
        self.trainset = trainset
      
    def get_trainset(self):
        return self.trainset
        
    def set_function(self, function):
        """assign funcion for computing the bura poles"""
        self.function = function
        
    def get_expected_convergence_rate(self):
        """
        Computes the expected convergence rate encoded in the constant cstar 
        so that error <= C exp(-cstar * k), where C is some constant and k the
        the amount of poles.

        Returns
        -------
        cstar : float
            The constant in the exponential convergence rate.
        """
        condition = self.get_lam_max() / self.get_lam_min()
        try:
            range_trainest = self.get_trainset()[-1] - self.get_trainset()[0]
            range_spectrum = self.get_lam_max() - self.get_lam_min()
            if range_spectrum < range_trainset:
                cstar = pi**2 / (2 * log(4*condition))
            else:
                cstar = pi**2 / (log(16*condition))
        
        except:
            if self.poletype in config.poles_on_spectrum:
                cstar = pi**2 / (2 * log(4*condition)) 
            else:
                cstar = pi**2 / (log(16*condition))
            
        return cstar     
    
    def append(self, basis_vector):
        """
        Adds the orthonormaization of the basis vector to the basis 
        of the Krylov space and executes self.update().

        Parameters
        ----------
        basis_vector : ngsolve.la.BaseVector
            The new basis vector.

        Returns
        -------
        None.
        """
        self.basis.AppendOrthogonalize(basis_vector, self.get_M())
        
        # reorthogonalize twice for numerical purposes
        self.basis.Orthogonalize(self.get_M())
        self.basis.Orthogonalize(self.get_M())
        
        # normalize orthogonalized vector
        norm = sqrt(InnerProduct(self.basis[-1], self.get_M() * self.basis[-1]))
        self.basis[-1].data *= 1/norm
        self.update()
        
    def project(self, v):
        """
        Computes the coordinate vector (of dimension self.dim()) of the 
        orthogonal projection of v onto the Krylov space

        Parameters
        ----------
        v : ngsolve.bla.VectorD
            The vector whose projection is sought.

        Returns
        -------
        vs : ngsolve.bla.VectorD
            The coordinate vector of the projection.
        """
        vs = Vector(self.dim())
        self.tmp1.data = self.get_M() * v
        for i, bi in enumerate(self.basis):
                vs[i] = InnerProduct(bi, self.tmp1)
                
        return vs
    
    def prolongate(self, vs):
        """
        Prolongates the coordinate to the dimension of L.

        Parameters
        ----------
        vs : ngsolve.bla.VectorD
            The coordinate vector whose prolongation is sought.

        Returns
        -------
        ngsolve.la.BaseVector
            The prolongated vector.
        """
        self.tmp1[:] = 0.0
        for i, val in enumerate(vs):
            self.tmp1.data += val * self.basis[i]
        
        return self.tmp1

    def clear(self):
        """Discards all basis vectors but the first one."""
        self.tmp1.data = self.basis[0]
        self.basis = MultiVector(self.tmp1, 1)
        self.basis[0].data = self.tmp1
        self.poles = []
            
    def compute_mat_inverse(self):
        """Efficiently computes the inverse of the matrix self.mat."""
        if self.mat_inverse is None:
            self.mat_inverse = self.mat.Inverse(self.L.get_freedofs(),
                                           inverse="sparsecholesky")
        else:
            self.mat_inverse.Update()
      
    def compress(self, matrix):
        """Computes the compression of the matrix onto the Krylov space"""
        return InnerProduct(self.basis, matrix * self.basis)
    
    def update(self):
        """
        Updates the coordinate vector, the compression, and its eigensystem for
        the present basis.
        """
        self.vector_projected = self.project(self.vector)
        self.compression =self.compress(self.get_A())
        self.ritzvalues, self.ritzvectors = eigh(self.compression)
    
    def compute_basis_vector(self, pole):
        """
        Computes the basis vector of the Krylov space associated with a given
        pole. 

        Parameters
        ----------
        pole : float
            A pole of the Krylov space.

        Returns
        -------
        ngsolve.la.BaseVector
            The basis vector of the Krylov space associated with the pole.
        """
        if np.isinf(pole):
            self.tmp1.data = self.get_A() * self.basis[-1]
            self.tmp2.data =\
                self.get_M().Inverse(self.L.freedofs,\
                                       inverse = 'sparsecholesky') * self.tmp1 
            
            return self.tmp2
        
        else:
            self.tmp1.data = self.basis[-1]
            self.mat.AsVector().data = \
                self.get_A().AsVector() - pole * self.get_M().AsVector()
            self.compute_mat_inverse()
            self.tmp2.data = self.get_A() * self.tmp1
            self.tmp1.data = self.mat_inverse * self.tmp2
        
            return self.tmp1
    
    def compute_krylov_space(self, k, dispsolves = True):
        """
        Computes the k-dimensional orthonormal basis of the rational Krylov
        space.

        Parameters
        ----------
        k : int
            The dimension of the resulting Krylov space.
        dispsolves : bool, optional
            Determines wether or not to print the progress. The default is True.

        Returns
        -------
        None.
        """
        # Compute number of new basis vectors
        if self.is_nested():
            n = k - self.dim()
        else:
            self.clear()
            n = k - 1
        
        # For efficiency, bura poles need to be computed in advance
        if self.poletype == "bpoles":
            self.compute_bpoles(n)  
            
        with TaskManager():
            for i in range(1, n+1):
                if dispsolves:
                    print('Progress: %d/%d\r'%(i, n), end="")
                pole = self.compute_pole(i, n)
                self.tmp1.data = self.compute_basis_vector(pole)
                self.append(self.tmp1)
                self.poles.append(pole)            
            if dispsolves: print("")
              
    def compute_coordinates(self, f):
        """
        Computes the coordinate vector of the Krylov surrogate in the 
        orthonormal basis.

        Parameters
        ----------
        f : function
            The function for which f(L) * v is approximated.
 
        Returns
        -------
        coordinates : numpy.ndarray
            The coordinate vector of the Krylov surrogate of f(L) * v

        """
        D = np.diag([f(abs(lam)) for lam in self.ritzvalues])
        coordinates = self.ritzvectors @ D \
            @ np.transpose(self.ritzvectors) @  self.vector_projected
        
        return coordinates

    def apply(self, fun):
        """
        Computes the Krylov surrogates of f(L) * v for each f in fun.  

        Parameters
        ----------
        fun : list
            The list of functions.

        Returns
        -------
        surrogates : list
            The list of Krylov surrogates.

        """
        surrogates = [self.tmp1.CreateVector() for i in range(len(fun))]
        for i, f in enumerate(fun):
            coordinates = self.compute_coordinates(f)
            surrogates[i].data = self.prolongate(coordinates)
        
        return surrogates
            
    def compute_pole(self, j, order):
        """
        Computes the jth pole of the given order of type self.poletype.

        Parameters
        ----------
        j : int
            If the pole seqence is not nested, j determines the index of the
            desired pole.
        order: int
            If the pole sequence is not nested, order determines the total 
            number of desired poles.
            
        Returns
        -------
        pole : float
            The jth pole (of the given order).
        """
        if self.poletype == "zpoles":
            pole = self.compute_zpole(j, order)
            
        if self.poletype == "zhatpoles":
            pole = self.compute_zhatpole(j, order)
        
        if self.poletype == "epoles":
            pole = self.compute_epole()
            
        if self.poletype == "ehatpoles":
            pole = self.compute_ehatpole()
            
        if self.poletype == "apoles":
            pole = self.compute_apole()
            
        if self.poletype == "fpoles":
            pole = self.compute_fpole()
      
        if self.poletype == "spoles":
            pole = self.compute_spole()
            
        if self.poletype == "gpoles":
            pole = self.compute_gpole()
        
        if self.poletype == "bpoles":
            pole = self.bpoles[j-1]
            
        if self.poletype == "polynomial":
            pole = np.inf
            
        return pole
    
    def compute_zpole(self, j, order):
        """
        Computes the jth Zolotarev pole on -[lam_min, lam_max] of 
        the given order.

        Parameters
        ----------
        j : int
            The index of the desired Zolotarev pole.
        order : int
            The total number of Zolotarev poles.

        Returns
        -------
        zpole : float
            The jth Zolotarev pole on -[self.lam_min, self.lam_max] of 
            the desired order.
        """
        assert(j <= order)
        
        delta = self.get_lam_min() / self.get_lam_max()
        zpoint = compute_zpoint(delta, j, order)
        zpole = -self.get_lam_max() * zpoint
        
        return zpole

    def compute_zhatpole(self, j, order):
        """
        Computes the jth Zolotarev pole on the negative real line of the given 
        order.

        Parameters
        ----------
        j : int
            The index of the desired Zolotarev pole.
        order : int
            The total number of Zolotarev poles.

        Returns
        -------
        zpole : float
            The jth Zolotarev pole on -[lam_min, lam_max] of the
            desired order.
        """
        assert(j <= order)
        
        lam_min = self.get_lam_min()
        lam_max = self.get_lam_max()
        
        tmp = sqrt( lam_max**2 - lam_min * lam_max)
        deltahat =  (lam_max - tmp) / (lam_max + tmp)
        zpoint = compute_zpoint(deltahat, j, order)
        
        const = 2 * lam_max / (deltahat + 1)
        zhatpole = -const * (zpoint + deltahat) / (zpoint + 1)
            
        return zhatpole   


    def compute_eds_element(self):
        """
        Computes the next element of the equidistributed sequence
        generated by config.EDS_PARAM.

        Parameters
        ----------
        None
     
        Returns
        -------
        epoint : list
            The (self.dim()+1)th element of the equidistributed sequence.

        """
        zeta = config.EDS_PARAM
        j = len(self.get_poles()) + 1
        epoint = j * zeta - np.floor(j * zeta)
        
        return epoint
    
    def compute_epoint(self, delta):
        """
        Computes the next EDS point on [delta, 1].

        Parameters
        ----------
        delta : float
            A parameter that must be smaller than 1.
        
        Returns
        -------
        edspoint : float
            The (self.dim()+1)th EDS point on [delta, 1] 
        """
        assert(delta < 1)
        
        eds_element = self.compute_eds_element()
        
        zero = compute_zero(delta, eds_element, config.newton_tolerance, config.newton_maxit)
        edspoint = np.sqrt(zero)

        return edspoint

    def compute_epole(self):
        """
        Computes the next EDS pole on -[lam_min, lam_max].

        Parameters
        ----------
        None

        Returns
        -------
        epole : float
            The (self.dim()+1)th EDS poles on -[lam_min, lam_max].
        """
        delta = self.get_lam_min() / self.get_lam_max()
        epoint = self.compute_epoint(delta)
        epole = -self.get_lam_max() * epoint
        
        return epole

    def compute_ehatpole(self):
        """
        Computes the next EDS pole on negative real line.

        Parameters
        ----------
        None

        Returns
        -------
        ehatpole : float
            (self.dim()+1)th EDS poles on negative real line.
        """
        
        lam_min = self.get_lam_min()
        lam_max = self.get_lam_max()
        
        tmp = sqrt( lam_max**2 - lam_min * lam_max )
        deltahat =  (lam_max - tmp) / (lam_max + tmp)
        epoint = self.compute_epoint(deltahat)
        
        const = 2 * lam_max / (deltahat + 1)
        ehatpole = -const * (epoint + deltahat) / (epoint + 1)
            
        return ehatpole  
    
    def compute_extremum(self):
        """
        Computes a global extremum of rat_fun in (lam_min, lam_max).

        Raises
        ------
        ValueError
            If the extremum is not in the expected interval.

        Returns
        -------
        extremum : float
            The global extremum of rat_fun.
        """
        poles = np.sort(self.get_poles())
        extrema = np.zeros(len(poles)-1)
        for i in range(len(extrema)):
            refinement = config.initial_refinement
            trainset = np.linspace(-poles[i+1], -poles[i], refinement)
            x0 = approximate_extremum(poles, trainset)
            while True:
                try:
                    extremum = scipy.optimize.newton(rat_fun_p, x0, rat_fun_pp, args = (poles,))
                    if -extremum < poles[i] or -extremum > poles[i+1]:
                        raise ValueError
                    
                    extrema[i] = extremum
                    break
                
                except RuntimeError or ValueError:
                    print("Newton did not converge: Improve initial value")
                    refinement *= 2
                    trainset = np.linspace(-poles[i+1], -poles[i], refinement)
                    x0 = approximate_extremum(poles, trainset)
        
        vals = rat_fun(extrema, poles)
        j_max = np.argmax(abs(vals))
        extremum = extrema[j_max]
        
        return extremum
    
    def compute_apole(self):
        """
        Computes the next automatic pole.      

        Raises
        ------
        ValueError
            If the extremum is not in the expected interval.

        Returns
        -------
        apole : float
            The (self.dim()+1)th automatic pole.
        """
        if len(self.get_poles()) == 0:
            apole = -self.get_lam_min()
        
        elif len(self.get_poles()) == 1:
            apole = -self.get_lam_max()
            
        else:
            extremum = self.compute_extremum()
            apole = -extremum 
            
        return apole
    
    def compute_fpole(self):
        """
        Computes the next fully automatic pole.      

        Raises
        ------
        ValueError
            If the extremum is not in the expected interval.

        Returns
        -------
        fpole : float
            The (self.dim()+1)th fully automatic pole.
        """
        if len(self.get_poles()) == 0:
            polynomial_basis = MultiVector(self.get_vector(), 1)
            polynomial_basis[0].data = self.basis[0]
            self.tmp1.data = self.compute_basis_vector(np.inf)
            
            polynomial_basis.AppendOrthogonalize(self.tmp1, self.get_M())
            norm = InnerProduct(polynomial_basis[1], self.get_M() * polynomial_basis[1])
            polynomial_basis[1] /= norm
            
            compression = InnerProduct(polynomial_basis, self.get_A() * polynomial_basis)
            
            ritzvalues = eigh(compression, eigvals_only = True)
            fpole = -min(ritzvalues)
            
        elif len(self.get_poles()) == 1:
            polynomial_basis = MultiVector(self.get_vector(), 1)
            polynomial_basis[0].data = self.basis[0]
            self.tmp1.data = self.compute_basis_vector(np.inf)
            
            polynomial_basis.AppendOrthogonalize(self.tmp1, self.get_M())
            norm = InnerProduct(polynomial_basis[1], self.get_M() * polynomial_basis[1])
            polynomial_basis[1] /= norm
            
            compression = InnerProduct(polynomial_basis, self.get_A() * polynomial_basis)
            
            ritzvalues = eigh(compression, eigvals_only = True)
            fpole = -max(ritzvalues)
            
        else:
            local_extremum = self.compute_extremum()
            ritzvalue_min = min(self.get_ritzvalues())
            ritzvalue_max = max(self.get_ritzvalues())
            
            critical_values = np.array([ritzvalue_min, ritzvalue_max, local_extremum])
            j_max = np.argmax(abs(rat_fun(critical_values, self.get_poles())))
            
            fpole = -critical_values[j_max]
            
        return fpole

    def compute_spole(self):
        """
        Computes the next spectral pole on -self.trainset.

        Parameters
        ----------
        None
          
        Returns
        -------
        spole : float
            The (self.dim()+1)th spectral pole.
        """
        try: 
            self.get_trainset()
        except:
            raise Exception("No training set assigned. Use set_trainset mehtod!")
        
        product = np.ones_like(self.trainset)
        for mu in self.get_ritzvalues():
            product *= (self.trainset - mu)
    
        for pole in self.get_poles():
            product /= (self.trainset - pole)
    
    
        # find new pole as minimizer of |product|
        j_min = np.argmin(abs(product))
        spole = -self.trainset[j_min]
    
        return spole
    
    def compute_residual_error(self, coordinates, shift):
        """
        Computes the Euclidean norm of the residual error. Note that the
        computation of the residual can be made more efficient by an
        online/offline decomposition.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Coordinate Vector of the Krylov approximation.
        shift : float
            The shift parameter.

        Returns
        -------
        residual_error : float
            Euclidean norm of the residual error.
        """
        self.tmp1.data = self.prolongate(coordinates)
        self.tmp2.data = float(shift) * self.get_M() * self.tmp1
        self.tmp2.data += self.get_A() * self.tmp1
        self.tmp1.data = self.projector * self.tmp2
        self.tmp1.data -= self.projector * self.M_vector
        
        residual_error = sqrt(InnerProduct(self.tmp1, self.tmp1))
        
        return residual_error
    
    def compute_gpole(self):
        """
        Computes the next weak greedy pole on -self.trainset using a residual
        based error estimator.
        """
        try: 
            self.trainset
        except:
            raise Exception("No training set assigned. Use set_trainset mehtod!")

        max_error = 0
        maximizer = self.trainset[0]
        for j, val in enumerate(self.trainset):
            coordinates = self.compute_coordinates(lambda x: 1 / (x + val))
            error = self.compute_residual_error(coordinates, val)
            if error > max_error:
                maximizer = val
                max_error = error
        
        gpole = -maximizer
        return gpole
    
    def compute_bpoles(self, k):
       """ 
       Computes the BURA poles of self.function. 
       """ 
       try: 
           self.function
       except:
           raise Exception("No function assigned. Use set_function method!")
            
       bura = baryrat.brasil(self.function, 
                             (self.get_lam_min(), self.get_lam_max()), k)
       bpoles = bura.poles(use_mp = True).real
       self.bpoles = bpoles