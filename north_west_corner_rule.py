import numpy as np

def NW_corner_rule(a, b):
    """
    North-West Corner rule for discrete optimal transport:
        Finding the vertex of the polytope of the feasible set U(a, b),
        where the minimum is attained.

    Reference: P44, Sec 3.4.2

    @input a, b: np.ndarray, of shape (n, ) and (m, ), the source histogram and the target histogram
    @return P: np.ndarray, of shape (n, m), one of the vertex of the polytope, P[i, j]: the flow from a[i] to b[j]

    >>> a = np.array([0.2, 0.5, 0.3])
    >>> b = np.array([0.5, 0.1, 0.4])
    >>> NW_corner_rule(a, b)
    array([[0.2, 0. , 0. ],
           [0.3, 0.1, 0.1],
           [0. , 0. , 0.3]])
    """
    P = np.zeros((a.shape[0], b.shape[0]))
    i = j = 0
    r = a[0]
    c = b[0]
    while i < a.shape[0] and j < b.shape[0]:
        P[i, j] = min(r, c)
        r -= P[i, j]
        c -= P[i, j]
        if r == 0:
            i += 1
            if i < a.shape[0]:
                r = a[i]
        if c == 0:
            j += 1
            if j < b.shape[0]:
                c = b[j]
    return P

def compute_NW_solutions(a, b, sigma1, sigma2):
    """
    compute an arbitrary NW solution by permutation

    Reference: Page 45, Section 3.4.2

    @param a, b: the source histogram and the target histogram
    @param sigma1, sigma2: the permutation vector for a and b respectively
    @return P_permutation: the permutated NW solution by
        P_permutation = sigma1_inv(sigma2_inv(P, column), row)
        where P is the transport matrix of sigma1(a) and sigma2(b)

    >>> a = np.array([0.2, 0.5, 0.3])
    >>> b = np.array([0.5, 0.1, 0.4])
    >>> sigma1 = np.array([2, 0, 1])
    >>> sigma2 = np.array([2, 1, 0])
    >>> compute_NW_solutions(a, b, sigma1, sigma2)
    array([[0., 0.1, 0.1],
        [0.5, 0., 0.],
        [0., 0., 0.3]])
    """
    a = a[sigma1]
    b = b[sigma2]
    P = NW_corner_rule(a, b)
    print(P)
    sigma1_inv = np.argsort(sigma1)
    sigma2_inv = np.argsort(sigma2)
    P = P[:, sigma2_inv]
    P = P[sigma1_inv, :]
    return P


if __name__ == "__main__":
    import doctest
    doctest.testmod()
