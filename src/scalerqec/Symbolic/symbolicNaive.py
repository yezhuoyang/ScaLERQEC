from sympy import symbols, binomial, Rational, simplify, latex,Poly

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

def xor_vec(a, b):
    return (a[0] ^ b[0], a[1] ^ b[1])


def pos_int_to_vec(vecidx):
    return [vecidx>>(3-i) & 1 for i in range(4)]

PROP_X = [(1,1),(0,1),(1,0),(0,1)]
PROP_Y = [(1,1),(0,1),(1,0),(0,1)]
PROP_Z  =[(0, 0),(0, 0),(0, 0),(0, 0)]
PROPAGATORS = (PROP_X, PROP_Y, PROP_Z)



MAX_degree=100


LER=p*0
#We use an 4bit integer to represent the position of the pauli noise
for n0 in range(0,4):
    n0count=(n0!=0)
    for n1 in range(0,4):
        n1count=(n1!=0)
        for n2 in range(0,4):
            n2count=(n2!=0)
            for n3 in range(0,4):     

                n3count=(n3!=0)  
                count=n0count+n1count+n2count+n3count

                print("{},{},{},{}".format(n0,n1,n2,n3))             

                init_vec=(0,0)
                if(n0==1):
                    init_vec=xor_vec(init_vec, PROP_X[0])
                elif(n0==2):
                    init_vec=xor_vec(init_vec, PROP_Y[0])                    
                elif(n0==3):
                    init_vec=xor_vec(init_vec, PROP_Z[0])    

                if(n1==1):
                    init_vec=xor_vec(init_vec, PROP_X[1])
                elif(n1==2):
                    init_vec=xor_vec(init_vec, PROP_Y[1])                    
                elif(n1==3):
                    init_vec=xor_vec(init_vec, PROP_Z[1])    

                if(n2==1):
                    init_vec=xor_vec(init_vec, PROP_X[2])
                elif(n2==2):
                    init_vec=xor_vec(init_vec, PROP_Y[2])                    
                elif(n2==3):
                    init_vec=xor_vec(init_vec, PROP_Z[2])    

                if(n3==1):
                    init_vec=xor_vec(init_vec, PROP_X[3])
                elif(n3==2):
                    init_vec=xor_vec(init_vec, PROP_Y[3])                    
                elif(n3==3):
                    init_vec=xor_vec(init_vec, PROP_Z[3])  

                print(init_vec)

                if(init_vec[1]==1):
                    LER+=simplify((p/3)**count*(1-p)**(4-count))
            # keep only terms with deg(p) < MAX_degree
    LER=LER.series(p, 0, MAX_degree).removeO()    # no .expand()


print(LER)


                  