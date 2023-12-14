import os
import site

def get_site_packages_dir():
        return [p for p  in site.getsitepackages()
                if p.endswith(("site-packages", "dist-packages"))][0]

def search_file_in_site_package(directory, package):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(package):
                return os.path.join(root, file)
    return None


class MapCombinatorics:
    
    def __init__(self, nradii, order_max):
        self.nradii = nradii
        self.order_max = order_max
        self.psummem = None
        self.nindices = self.psumtot(order_max+1, nradii)
        
    def psumtot(self, n, m):
        """ Calls to (n-1)-fold nested loop over m indicices
        where i1 <= i2 <= ... <= in. This is equivalent to the
        number of independent Map^i components over a range of
        m radii (0<i<=n) as well as to the size of the multivariate 
        power sum set generating those multivariate cumulants.

        Example:
        psumtot(m=10,n=4) gives the same result as the code
        >>> res = 0
        >>> for i1 in range(10):
        >>>     for i2 in range(i1,10):
        >>>         for i3 in range(i2,10):
        >>>             res += 1
        >>> print(res)

        Notes:
        * The recursion reads as follows:
          s(m,0) = 1
          s(m,n) = \sum_{i=1}^{m-1} s(m-1,n-1)
          [Have not formally proved that but checked with pen and paper
          up until n=3 on examples and the underlying geometry does make 
          sense. Testing against nested loops also works as long as the
          loops can be computed in a sensible amount of time]
        * As the solution is recusive and therefore might take long to
          compute we use a memoization technique to get rid of all of
          the unneccessary nested calls.
        """
        
        assert(m<=self.nradii)
        assert(n<=self.order_max+1)
            
        # For initial call allocate memo
        if self.psummem is None:
            self.psummem = np.zeros((n,m))#, dtype=np.int)
            self.psummem[0] = np.ones(m)
        # Base case
        if m<=0 or n<=0:
            return self.psummem[n,m]
        # Recover from memo
        if self.psummem[n-1,m-1] != 0:
            return self.psummem[n-1,m-1]
        # Add to memo
        else:
            res = 0
            for i in range(m):
                res += self.psumtot(n-1,m-i)
            self.psummem[n-1,m-1] = res
            return int(self.psummem[n-1,m-1])
        
    def sel2ind(self, sel):
        """  
        Assignes unique index to given selection in powr sum set
        Note that sel[0] <= sel[1] <= ... <= sel[self.nradii-1] is required!
        """
        # Check validity
        #assert(len(sel)==n-1)
        #for el in range(len(sel)-1):
        #    assert(sel[el+1] >= sel[el])
        #assert(sel[-1] <= m)

        i = 0
        ind = 0
        ind_sel = 0
        lsel = len(sel)
        while True:
            while i >= sel[ind_sel]:
                #print(i,ind_sel)
                ind_sel += 1
                if ind_sel >= lsel:
                    return int(ind)
            #ind += psumtot(m=m-i, n=n-1-ind_sel, mem=psummem)
            ind += self.psummem[self.order_max-1-ind_sel, self.nradii-1-i]
            i += 1

        return int(ind)
    
    
    def ind2sel(self, ind):
        """ Inverse of sel2ind """
        
        sel = np.zeros(self.order_max)#, dtype=np.int)
        # Edge cases
        if ind==0:
            return sel.astype(np.int)
        if ind==1:
            sel[-1] = 1
            return sel.astype(np.int)
        if ind==self.nindices-1:
            return (self.nradii-1)*np.zeros(self.order_max, dtype=np.int)
        
        tmpind = ind # Remainder of index in psum
        nextind_ax0 = self.order_max-1 # Value of i_k
        nextind_ax1 = self.nradii-1 # Helper
        tmpsel = 0 # Value of i_k
        indsel = 0 # Index in selection
        while True:
            nextsubs = 0
            while True:
                tmpsubs = self.psummem[nextind_ax0, nextind_ax1]
                #print(tmpind, nextsubs, tmpsubs, sel)
                if tmpind > nextsubs + tmpsubs:
                    nextind_ax1 -= 1
                    tmpsel += 1
                    nextsubs += tmpsubs
                elif tmpind < nextsubs + tmpsubs:
                    nextind_ax0 -= 1
                    tmpind -= nextsubs
                    sel[indsel] = tmpsel
                    indsel += 1
                    break
                else:
                    sel[indsel:] = tmpsel + 1
                    return sel.astype(np.int)
            if sel[-2] != 0:
                sel[-1] = sel[-2] + tmpind
            if sel[-1] != 0:
                return sel.astype(np.int)