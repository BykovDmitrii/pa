import numpy as np

def gen_pow_matrix(primpoly):
    q2 = 0
    while((2 ** q2) <= primpoly):
        q2 += 1
    q2 -= 1
    a = 1 
    j = 1
    mat = np.zeros((2, (2 ** q2) - 1))
    b = True
    while b:
        a = a * 2
        if not((2 ** q2) > a):
            a = a ^ primpoly
        mat[0][(a) - 1] = j
        mat[1][j-1] = (a)
        j += 1
        b = a != 1
    return mat.T.astype(int)

def add(X, Y):
    res = X ^ Y
    return res

def sum(X, axis=0):
    s = list(X.shape)
    s[axis] = 1
    return np.bitwise_xor.reduce(X, axis=axis).reshape(*s)

def prod(X, Y, pm):
    def prodel(x, y):
        if (x == 0) or (y == 0):
            return 0
        xi = pm[int(x)-1][0]
        yi = pm[int(y)-1][0]
        return pm[(xi + yi) % pm.shape[0] - 1][1]
    g = np.vectorize(prodel)
    return g(X, Y)

def divide(X, Y, pm):
    def divel(x, y):
        if (x == 0) or (y == 0):
            return 0
        xi = pm[x-1][0]
        yi = pm[y-1][0]
        return pm[(pm.shape[0] + xi - yi - 1) % pm.shape[0]][1]
    g = np.vectorize(divel)
    return g(X.astype(int), Y.astype(int))

def linsolve(A, b, pm):
    X = np.hstack((A, b.reshape(-1, 1))).astype(int)
    try:
        for i in range(X.shape[0]):
            index = [st.nonzero()[0][0] for st in X[:, :-1]]
            X = X[np.argsort(index)]
            for j in range(1, X.shape[0] - i):
                mn = divide(X[i + j][i], X[i][i], pm)
                X[i + j] = add(prod(X[i], mn, pm), X[i+j])
    except:
        return np.nan
    for i in range(X.shape[0]):
        for j in range(1, i+1):
            mn = divide(X[i - j][i], X[i][i], pm)
            X[i-j] = add(prod(X[i], mn, pm), X[i-j])
    ans = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        ans[i] = divide(X[i][-1] , X[i][i], pm)
    return ans

def minpoly(x, pm):
    s = set()
    res = np.array([1])
    for i in x:
        el = i
        while (el not in s):
            if(el not in s):
                res = polyprod(res, np.array([1, el]), pm)
                s.add(el)
            el = prod(np.array([el]), np.array([el]), pm)[0]
    return (res, np.array(list(s)))

def polyval(p, x, pm):
    def powv(x1, y1):
        a = 1
        for i in range(y1):
            a = prod(a, x1, pm)
        return a
    def polv(x):
        i = p.shape[0] - 1
        res = 0
        for j in p:
            res = add(np.array([res]), prod(powv(np.array([x]), i), j, pm))[0]
            i = i - 1
        return res
    g = np.vectorize(polv)
    return g(x.astype(int)).astype(int)

def polyprod(p1, p2, pm):
    res = np.zeros(p1.shape[0] + p2.shape[0] - 1).astype(int)
    for i in range(p1.shape[0]):
        for j in range(p2.shape[0]):
            res[i+j] = add(res[i+j], prod(p1[i], p2[j], pm))
    return res

def polydiv(p1, p2, pm):
    try:
        nnull = p1.nonzero()[0][0]
        p1 = p1[nnull:]
    except:
        return (np.array([0]), np.array([0]))
    try:
        nnull = p2.nonzero()[0][0]
        p2 = p2[nnull:]
    except:
        return (np.nan, np.nan)
    if (p1.shape[0] < p2.shape[0]):
        res = np.array([0])
        return (res, p1[nnull:])
    res = np.zeros(p1.shape[0]-p2.shape[0]+1).astype(int)
    pc1 = p1.copy().astype(int)
    for i in range(p1.shape[0]-p2.shape[0]+1):
        res[i] = divide(pc1[i], p2[0], pm)
        for j in range(p2.shape[0]):
            pc1[i+j] = add(prod(p2[j], res[i], pm), pc1[i+j])
    try:
        nnull = pc1.nonzero()[0][0]
        return (res, pc1[nnull:])
    except:
        return (res, np.array([0]))

def polysum(p1, p2):
    n = max(p1.shape[0], p2.shape[0])
    res = np.zeros(n).astype(int)
    p1 = p1.astype(int)
    p2 = p2.astype(int)
    for i in range(n):
        z = np.array([0]).astype(int)
        if (i < p1.shape[0]):
            z = np.array([p1[-i-1]])
        if (i < p2.shape[0]):
            z = add(z, p2[-i-1])
        res[-i-1] = z
    try:
        nnull = res.nonzero()[0][0]
        return res[nnull:]
    except:
        return np.array([0])

def euclid(p1, p2, pm, max_deg=0):
    x1 = np.array([0]).astype(int)
    x0 = np.array([1]).astype(int)
    y1 = np.array([1]).astype(int)
    y0 = np.array([0]).astype(int)
    pc1 = p1.copy().astype(int)
    pc2 = p2.copy().astype(int)
    r = np.zeros(max_deg+10)
    while ((max_deg != 0) and (r.shape[0] - 1 >= max_deg)) or not((r.shape[0] == 1) and (r[0] == 0)) and (max_deg ==0):
        q, r = polydiv(pc1, pc2, pm)
        x1, x0 = (polysum(x0, polyprod(q, x1, pm)), x1)
        y1, y0 = (polysum(y0, polyprod(q, y1, pm)), y1)
        pc1, pc2 = pc2, r
    if (max_deg == 0):
        return (pc1, x0, y0)
    else:
        return (pc2, x1, y1)
