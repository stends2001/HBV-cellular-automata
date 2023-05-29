import numpy as np
import matplotlib.pyplot as plt
########################
# Arbitrary parameters #
########################
L_glob              = 100
generations_glob    = 5000
Ini_size_glob       = 10
Ini_v_glob          = 0.1
Ini_s_glob          = 0.015
I_glob           = 0.23
K_glob           = 1

###############################
###### Death parameters #######
# (only for epsilon_max != 0) #
###############################
epsilon_max_glob = 0
a_glob           = 50
lamda_glob       = 0.75

####################
# Vital parameters #
####################
alpha_glob       = 1
beta_glob        = 2
gamma_glob       = 1.5
etha_s_glob      = 0.9
epsilon_glob     = 3*10**-2

mu_glob          = 0.2
etha_D_glob      = 1
#####################
# Output simulation #
#####################
v = np.genfromtxt('relative_v.csv', delimiter=',')
s = np.genfromtxt('relative_s.csv', delimiter=',')
D = np.genfromtxt('relative_D.csv', delimiter=',')
dead = np.genfromtxt('relative_dead.csv', delimiter=',')
time_axis = np.arange(0,generations_glob-1)

plt.figure()
plt.grid()
plt.plot(time_axis[:100],v[:100],label="[v]",color='blue')
plt.plot(time_axis[:100],s[:100],label="[s]",color='brown')
plt.plot(time_axis[:100],D[:100],label="[D]",color='green')
plt.legend()
plt.title("Normalized concentration of virus species over time")
plt.savefig("total loads beta smaller than gamma.png")
plt.show()


'''
# requirements
equilibrium = np.zeros((4,1))
equilibrium[0,0] = v[generations_glob-2]
equilibrium[1,0] = s[generations_glob-2]
equilibrium[2,0] = D[generations_glob-2]
equilibrium[3,0] = dead[generations_glob-2]

sigma1 = equilibrium[0,0]     + etha_s_glob * equilibrium[1,0] + etha_D_glob * equilibrium[2,0]
SIGMA  = 1 - equilibrium[0,0] - etha_s_glob * equilibrium[1,0] - etha_D_glob * equilibrium[2,0]
ETHA   = 1 - equilibrium[0,0] -               equilibrium[1,0] -               equilibrium[2,0]

# Requirement 1:
print("SIGMA(x)",SIGMA," needs to be equal to epsilon/(alpha*(1-mu))",(epsilon_glob/(alpha_glob*(1-mu_glob))))

D2_numerical = alpha_glob*mu_glob/(beta_glob-gamma_glob)

#plt.plot(time_axis,v,label="v")
#plt.plot(time_axis,s,label="s")
plt.plot(time_axis,D,label="D")
plt.axhline(D2_numerical,color='r',label="numerical D")
plt.grid()
plt.legend()
plt.show()
'''

'''
# for P points --> when s = 0
if equilibrium[1,0] == 0:
    v1 = sigma1-etha_D_glob*equilibrium[2,0]
    print("simulation v at P point:",round(equilibrium[0,0],2))
    print("numerical v at P point:",round(v1,2))
    D1 = (sigma1-equilibrium[0,0])/etha_D_glob
    print("simulation D at P point:",round(equilibrium[2,0],2))
    print("numerical D at P point:",round(D1,2))
# for Q points
if equilibrium[1,0] != 0:
    D2     = alpha_glob*mu_glob/(beta_glob-gamma_glob)
    s2     = 1 - equilibrium[0,0]-equilibrium[2,0]-epsilon_glob/(beta_glob*equilibrium[0,0])
    print("D numerically:",D2)
    print("D simulation",equilibrium[2,0])
    v_m    = epsilon_glob * etha_s_glob/(beta_glob*(etha_s_glob-1))
    v_M1    = (sigma1-equilibrium[2,0]*(etha_D_glob-etha_s_glob)-etha_s_glob)/(etha_s_glob-1)
    v_M2 = (sigma2-equilibrium[2,0]*(etha_D_glob-etha_s_glob)-etha_s_glob)/(etha_s_glob-1)
    inequal1 = v_M1**2 - 4*v_m                                                                           
    inequal2 = v_M2**2 - 4*v_m   
    print(inequal2)
    v2_pos = (-v_M2+inequal2**0.5)/2
    v2_neg = (-v_M2-inequal2**0.5)/2
    print("v simulation",equilibrium[0,0])
    print("v pos numericallyl", v2_pos)
    print("v neg numerically", v2_neg)
'''