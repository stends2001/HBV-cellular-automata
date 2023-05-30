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
beta_glob        = 2     # 1
gamma_glob       = 1.5   # 1.5
etha_s_glob      = 1.3
epsilon_glob     = 3*10**-2

mu_glob          = 0.2
etha_D_glob      = 1.1
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
#plt.savefig("total loads beta smaller than gamma.png")
#plt.show()

equilibrium = np.zeros((4,1))
equilibrium[0,0] = v[generations_glob-2]
equilibrium[1,0] = s[generations_glob-2]
equilibrium[2,0] = D[generations_glob-2]
equilibrium[3,0] = dead[generations_glob-2]



sigma  = equilibrium[0,0]+etha_s_glob*equilibrium[1,0]+etha_D_glob*equilibrium[2,0]
sigma_small = 1 - epsilon_glob/(alpha_glob*(1-mu_glob))
OMEGA  = 1 - equilibrium[0,0] - etha_s_glob * equilibrium[1,0] - etha_D_glob * equilibrium[2,0]
ETHA   = 1 - equilibrium[0,0] -               equilibrium[1,0] -               equilibrium[2,0]


# P point
'''
v_guess_P = sigma_small-etha_D_glob*equilibrium[2,0]
print("simulated v:",round(equilibrium[0,0],2))
print("estimated v:",round(v_guess_P,2))
print("simulated D",round(equilibrium[2,0],2))
'''
# Q point
#'''
D_guess = alpha_glob*mu_glob/(beta_glob-gamma_glob)

# req 1
req1 = etha_D_glob*(alpha_glob*mu_glob)/(beta_glob-gamma_glob) + epsilon_glob/(alpha_glob*(1-mu_glob))
#print("req1",req1,"needs to be smaller than",1)

# req 2
req2 = sigma_small/etha_D_glob
#print("D",D_guess,"has to be smaller than",req2)

# req3 --> V guess
M1 = 1/(etha_s_glob-1) * (sigma_small-(etha_D_glob-etha_s_glob)*D_guess-etha_s_glob)
m  = etha_s_glob*epsilon_glob/(beta_glob*(etha_s_glob-1))

v_guess11 = 0.5*(-M1+(M1**2-4*m)**0.5)
v_guess12 = 0.5*(-M1-(M1**2-4*m)**0.5)



# S guess
s_guess= 1-v_guess12-D_guess-epsilon_glob/(beta_glob*equilibrium[0,0])

print("simulation v:",round(equilibrium[0,0],2))
print("estimated  v1:",round(v_guess11,2))
print("estimated  v2:",round(v_guess12,2))
print("simulation s:",round(equilibrium[1,0],2))
print("estimated  s:",round(s_guess,2))
print("simulation D:",round(equilibrium[2,0],2))
print("estimated  D:",D_guess)
#'''