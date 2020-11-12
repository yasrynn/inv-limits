import numpy as np
from operator import le as less_eq
from itertools import tee

def adj_pairs(it):
    itr = iter(it)
    a = next(itr)
    for b in itr:
        yield a,b
        a = b

def issorted(it, compare = less_eq):
    a,b = tee(it)
    return all(map(compare,a,b))

def rev_enumerate(seq):
    return zip(range(len(seq)-1,-1,-1),reversed(seq))

class BondingMap(object):
    def __init__(self, domains, ms, bs):
        """
        domains: Domain partition for the piecewise defined function.  
        
        ms: Slopes for each subdomain.

        bs: Constants for each subdomain.
        """
        assert issorted(domains) and domains[0] == 0.0 and domains[-1] == 1.0, \
            "Invalid domain partition.  Must begin with 0.0 and end with 1.0 and be in increasing order."
        assert len(domains)-1 == len(ms) == len(bs), \
            "Parameter and partition lengths don't match."
        for x,(end_m,start_m),(end_b,start_b) in zip(domains[1:-1],adj_pairs(ms),adj_pairs(bs)):
            assert end_m*x+end_b == start_m*x+start_b, \
                f"Function is discontinuous at {x}," + \
                f" {end_m}*{x}+{end_b} == {end_m*x+end_b} != {start_m*x+start_b} == {start_m}*{x}+{start_b}."
        
        self.domains,self.ms,self.bs = domains,ms,bs
    
    def __call__(self,x):
        """ Apply the bonding map to x. """
        assert 0.0 <= x <= 1.0, f"Function undefined at {x}."
        
        m,b = self.params(x)

        return m*x+b
    
    def params(self,x):
        """ Returns slope and constant parameters for the domain associated with x. """
        return next(((m,b) for end,m,b in zip(self.domains[1:],self.ms,self.bs) if x<=end))

class Limit(object):
    """
    Limit class for visualization of inverse limits on the Hilbert cube.  Candidates
    must be obtainable using finitely piecewise linear bonding maps.
    """
    
    # Some constants defining different scaling rates.
    GEOMETRIC = 'geometric'
    HARMONIC = 'harmonic'
    
    def __init__(self, fs, repeats=True, iterations=5, pr=10, coord_met_dec=HARMONIC):
        """
        fs: A list of BondingMap objects representing the bonding maps of the limit.
        
        repeats: Flag indicating whether a single bonding map should be used repeatedly.
        If false, it is assumed all the bonding maps were given individually.
        
        iterations: In the case repeats == True, the number of iterations to apply to
        the given bonding maps.  If several bonding maps were given, they will all be 
        repeated <iterations> many times.  If repeats == False, this is ignored.
        
        prec: Precision to be used when comparing numbers to domain values.  This is only
        really necessary when we are deciding whether to split domain partitions.  Set to
        None to use machine precision.  Precision too low will cause domains to fail to 
        split when they should, but precision too high could cause splits due to 
        rounding errors.  Probably as long as it's not excessively low (like 2 or 3) it
        won't make much difference to the final plot and splits due to rounding error
        won't affect the plots at all.  On the other hand each split is potentially an 
        exponential increase in computation, so splits due to rounding errors could get 
        troublesome at high depths.
        
        coord_met_dec: Specifies the rate at which the Hilbert cube metric evaluation
        decreases as coordinates increase.  There's not a whole lot of qualitative or
        computational complexity difference between options, but the plots do change
        noticeably.  I've made harmonic the default because I think the plots are more
        clear.
        """
        for f in fs:
            assert isinstance(f,BondingMap)
        
        if coord_met_dec == Limit.GEOMETRIC:
            self.scale_f = lambda i:2**i
        elif coord_met_dec == Limit.HARMONIC:
            self.scale_f = lambda i:i+1
        else: 
            raise NameError(f"Scaling rate {coord_met_dec} is not recognized.")
        
        self.fs = fs*iterations if repeats else fs
        self.pr = pr
        self.n = n = len(self.fs) # length of the sequence for our approximation.
        
        # We need a pair of orthogonal vectors that will give us a 2-d view of
        # n+1 space.  Some views are better than others, but I chose a view
        # that keeps information about all coordinates and favors the first.
        # The criteria I used are:
        # - all coordinates of both vectors are non-zero
        # - orthanogality, i.e. np.dot(p,q) == 0 (up to rounding error)
        # - leading coordinate of each has largest magnitude (except in the edge case n=1).
        self.p = np.ones(n+1)
        self.p[0] = 2.0
        self.p[-1] = -1.0
        
        self.q = (3.0/(n+4.0)) * np.ones(n+1)
        self.q[0] = (-2.0 * n - 2.0)/(n + 4.0)
        self.q[1] = (n + 7.0)/(n + 4.0)
        self.q[-1] = -self.q[-1]
        
        # Getting the vectors can be computationally intensive, so we'll
        # generate them when make_vects() is called.
        self.vects_computed = False
        self.vects = None
    
    def make_vects(self):
        """
        Generate the vectors needed to plot the visualization.
        """
        partition = np.array([0.0,1.0])
        for f in self.fs:
            # The idea here is, for every partition divider num, to find every number in the *next* 
            # coordinate that maps to it and make it a partition divider if it isn't already.
            starts = np.array(f.domains[:-1])[..., None]
            ends = np.array(f.domains[1:])[..., None]
            ms = np.array(f.ms)[..., None]
            bs = np.array(f.bs)[..., None]
            inv_images = (partition - bs) / ms
            partition = np.unique(inv_images[(starts <= inv_images) & (inv_images <= ends)])
    
    def isbetween(self,x,a,b):
        """
        Working with floating point, there is the danger that due to rounding error,
        two numbers that should be the same aren't.  So we'll use this to approximate
        whether a number is between two others.
        
        Checks whether x is strictly between a and b looking at the first self.pr
        decimal places.  
        
        Alternatively, if self.pr is set to None, machine precision is used.
        """
        return (round(a-x,self.pr) < 0 < round(b-x,self.pr)) if self.pr is not None else a<x<b
    
    def scale(self,v,i):
        """ Scale in place the ith coordinate of vector v."""
        v[i] /= self.scale_f(i)
        
        return None
