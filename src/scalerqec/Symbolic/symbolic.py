from sympy import symbols, binomial, Rational, simplify, latex

# ----------------------------------------------------------------------
# Physical-error model
p = symbols('p')
q = 1 - p                     #   probability of "no error"
Px = Py = Pz = p / 3          #   probabilities of  X  Y  Z

# Syndrome-propagation vectors (two bits, XOR addition)
# index encoding  D0 + 2*O0   (00→0, 01→1, 10→2, 11→3)
def vec_to_idx(vec):
    return vec[1] + 2 * vec[0]

def idx_to_vec(idx):
    return ((idx >> 1) & 1,idx & 1)

PROP_X = [(1,1),(0,1),(1,0),(0,1)]
PROP_Y = [(1,1),(0,1),(1,0),(0,1)]
PROP_Z  =[(0, 0),(0, 0),(0, 0),(0, 0)]
PROPAGATORS = (PROP_X, PROP_Y, PROP_Z)


MAX_degree=200


def xor_vec(a, b):
    return (a[0] ^ b[0], a[1] ^ b[1])


# ----------------------------------------------------------------------
# Check if all row elements sum up to 1
def verity_table(i):
    sum=0
    for j in range(0,i+1):
        for vec_index in range(4):
            sum+=dp[i][j][vec_index]
        #print(dp[i][j][vec_index])
    sum=simplify(sum)
    print(i,sum)
    assert sum==1


# ----------------------------------------------------------------------
# DP tables
MAX_I = 4               # change here if you need longer arrays
dp = [ [ [0]*4 for _ in range(MAX_I+1) ]          # dp[i][j][vecIdx]
      for _ in range(MAX_I+1) ]

# Base:  i = 0, j = 0 ⇒ probability 1 at vec = (0,0)
dp[0][0][ vec_to_idx((0,0)) ] = 1



# ----------------------------------------------------------------------
# Fill   dp[i][j][·]   using the recurrence in Eq. (1)
for i in range(1, MAX_I+1):
    dp[i][0][0] = (1-p)**i

    for j in range(1, i+1):           # j ≤ i
        for vec_idx in range(4):

            vec = idx_to_vec(vec_idx)

            # 1) “no error’’ branch
            acc = q * dp[i-1][j][vec_idx]

            # 2) X, Y, Z branches  (need j-1 ≥ 0)
            if j >= 1:
                for (prob, prop) in ((Px, PROP_X[i-1]),
                                     (Py, PROP_Y[i-1]),
                                     (Pz, PROP_Z[i-1])):
                    prev_vec = xor_vec(prop, vec)
                    acc += prob * dp[i-1][j-1][ vec_to_idx(prev_vec) ]

            dp[i][j][vec_idx] = simplify(acc)

            #print(f"dp[{i}][{j}][{vec_idx}] = {dp[i][j][vec_idx].series(p, 0, MAX_degree).removeO()}")
    verity_table(i)




def binom(k):
    binomweight=binomial(4,k)*p**k*q**(4-k)
    return simplify(binomweight)


# ----------------------------------------------------------------------
# Calculate logical error rate
# The input is a list of rows with logical errors
def calculate_LER(error_row_indices):
    LER=0
    for weight in range(1,5):
        subLER=0
        for rowindex in error_row_indices:
            subLER+=dp[MAX_I][weight][rowindex]
        print("Weight: {}".format(weight))
        print(subLER.expand())
        LER+=simplify(subLER)

    #LER=LER.series(p, 0, MAX_degree).removeO()    # no .expand()
    return simplify(LER).expand()




# ----------------------------------------------------------------------
# Pretty-print the tables
def print_table(i, j):
    header = rf"\begin{{table}}[h!]\centering\caption{{Table of $dp[{i}][{j}]$ for all $\vec{{D}}$}}\n"
    header += r"\resizebox{\columnwidth}{!}{%" "\n"
    header += r"\begin{tabular}{|l|l|l|l|l|}" "\n   \hline " \
              r"\n Index  & $D_0$ & $O_0$  & Probability\\\n  \hline"
    print(header)

    for idx in range(4):
        D0, O0 = idx_to_vec(idx)
        prob_latex = latex(dp[i][j][idx])
        error_tag = " ({\\color{red} Error})" if O0 == 1 else ""

        print(f" {idx} {error_tag}  & {D0} & {O0}  & ${prob_latex}$\\\\")
        print("  \\hline")

    footer = r"\end{tabular}" "\n}\n\end{table}\n"
    print(footer)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    '''
    for i in range(1, MAX_I+1):
        for j in range(1, i+1):
            print_table(i, j)
    '''
    
    print("------------------LER----------------------")
    sum=calculate_LER([1,3])
    print(sum)
    print(sum.evalf(subs={p:0.1}))

