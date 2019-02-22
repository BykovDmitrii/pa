import gf
import numpy as np
import time

class BCH:
    def __init__(self, n, t):
        dc = 2 * t + 1
        q = 1
        while (2 ** q < n + 1):
            q += 1
        f = open('primpoly.txt')
        poly = 0
        c = 0
        for s in f:
            for i in s.split(', '):
                a = int(i)
                if (a > 2**q):
                    poly = a
                    break
        self.pm = gf.gen_pow_matrix(poly)
        self.g, self.R = gf.minpoly(self.pm.T[1][:dc-1], self.pm)
        self.k = n - self.g.shape[0] + 1
        self.t = t
    
    def encode(self, U):
        res = np.zeros((U.shape[0], U.shape[1] + self.g.shape[0] - 1))
        for j in range(U.shape[0]):
            for i in range(U.shape[1]):
                res[j][i] = U[j][i]
            q, r = gf.polydiv(res[j], self.g, self.pm)
            for i in range(r.shape[0]):
                res[j][res.shape[1] - i - 1] = r[r.shape[0] - i - 1]
        return res

    def dist(self):
        m = np.zeros((2 ** self.k, self.k))
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i][j] = int(i / (2**(self.k - j - 1))) % 2
        codes = self.encode(m).astype(int)
        minDist = 2 ** (self.k + self.t)
        for i in range(m.shape[0]):
            for j in range(m.shape[0]):
                if (i != j):
                    a = (0 + np.bitwise_xor(codes[i], codes[j])).sum()
                    if (a < minDist):
                        minDist = a
        return minDist

    def PGZ(self, s):
        matrix = np.zeros((self.t, self.t))
        nu = self.t
        answer = np.zeros(self.t)
        r = np.zeros(self.t)
        while nu != 0:
            for i in range(nu):
                for j in range(nu):
                    matrix[i][j] = s[i + j]
                answer[i] = s[nu + i]
            m = np.array(matrix[:nu, :nu]).astype(int)
            ans = answer[:nu].astype(int)
            r = gf.linsolve(m, ans, self.pm)
            if not(r is np.nan):
                n = np.ones(r.shape[0] + 1)
                for i in range(r.shape[0]):
                    n[i] = r[i]  
                return n
            nu = nu - 1
        return np.nan

    def EVCL(self, s):
        arr = np.ones(s.shape[0] + 1)
        for i in range(s.shape[0]):
            arr[i] = s[s.shape[0] - i - 1]
        arr2 = np.zeros(self.t * 2 + 2)
        arr2[0] = 1
        #print(arr, arr2, self.pm, self.t)
        n = gf.euclid(arr, arr2, self.pm, self.t+1)
        #print(n)
        n = n[1]
        return n

    def decode(self, W, method='euclid'):
        d = self.g.shape[0] - 1
        ans = np.zeros((W.shape[0], self.k))
        for j in range(W.shape[0]):
            w = W[j]
            s = np.array(gf.polyval(w, self.pm.T[1][:2*self.t], self.pm))
            b = 0
            for i in s:
                if(i != 0):
                    b = 1
            if (b == 0):
                ans[j] = w[:self.k]
            else:
                n = np.array([0])
                r = w.copy()
                self.time = 0
                b = 0
                if (method == 'PGZ'):
                    start_time = time.time()
                    n = np.array(self.PGZ(s))
                    self.time += time.time() - start_time
                if (method == 'euclid'):
                    start_time = time.time()
                    n = self.EVCL(s)
                    self.time += time.time() - start_time
                if (n is np.nan):
                    b = 1
                    ans[j] = np.nan
                    continue
                else:
                    n = n.astype(int) 
                a = gf.polyval(n, self.pm.T[1], self.pm)
                an = []
                for i in range(a.shape[0]):
                    if a[i] == 0:
                        an = an + [i]
                rr = np.array(an)
                for i in rr:
                    r[i] = (r[i] + 1) % 2
                s = np.array(gf.polyval(r, self.pm.T[1][:2*self.t], self.pm))
                for i in s:
                    if(i != 0):
                        b = 1
                if (b == 1):
                    ans[j] = np.nan
                else:
                    ans[j] = r[:self.k]
        return ans
