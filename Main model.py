import time
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import matplotlib

def Reproduction(loads):
    G_matrix    = np.zeros((len(loads),1))
    Repr_matrix = np.zeros((len(loads),3))
    Normalized_inflow    = np.zeros((len(loads),3))
    S_scaled = etha_s*loads[:,1]
    D_scaled = etha_D*loads[:,2]
    G_matrix[:,0] = 1/(1+S_scaled[:]+D_scaled[:])
    Repr_matrix[:,0]=G_matrix[:,0]*alpha*(1-mu)                   *loads[:,0]
    Repr_matrix[:,1]=beta*loads[:,1]                              *loads[:,0]
    Repr_matrix[:,2]=(1+gamma*loads[:,0])*(loads[:,2]+G_matrix[:,0]*alpha*mu*loads[:,0])-loads[:,2]
    Available = 1-np.sum(loads[:,:],axis=1)
    Absolute_inflow=np.sum(Repr_matrix[:,:],axis=1)
    tonormalize = Absolute_inflow>Available
    Normalized_inflow           = Repr_matrix
    Normalized_inflow[tonormalize,0]    = Repr_matrix[tonormalize,0]*(Available[tonormalize]/Absolute_inflow[tonormalize])
    Normalized_inflow[tonormalize,1]    = Repr_matrix[tonormalize,1]*(Available[tonormalize]/Absolute_inflow[tonormalize])
    Normalized_inflow[tonormalize,2]    = Repr_matrix[tonormalize,2]*(Available[tonormalize]/Absolute_inflow[tonormalize])
    toosmall = Normalized_inflow<0
    Normalized_inflow[toosmall]=0
    return Normalized_inflow

def infection(inputt,x,y):
    Intermed_flows               = np.zeros((L,L,3))
    Outflow_per_species_at_xy    = np.zeros((L,L,3))
    multiplication_factors       = np.zeros((L,L,3))
    multiplication_factors[x,y,0]=1
    multiplication_factors[x,y,1]=1#inputt[x,y,0]
    multiplication_factors[x,y,2]=1#inputt[x,y,0]
    Outflow_per_species_at_xy[:,:,0]    = I* inputt[:,:,0]*multiplication_factors[:,:,0]
    Outflow_per_species_at_xy[:,:,1]    = I* inputt[:,:,1]*multiplication_factors[:,:,1]
    Outflow_per_species_at_xy[:,:,2]    = I* inputt[:,:,2]*multiplication_factors[:,:,2]
    x_N = x+np.random.randint(low=-1, high=2, size=(len(x)))
    y_N = y+np.random.randint(low=-1, high=2, size=(len(x)))

    x_reroll = np.where((x_N<0) | (x_N>L-1))
    y_reroll = np.where((y_N<0) | (y_N>L-1))
    while (len(x_reroll[0])!=0):
        sz = len(x_reroll)
        x_N[np.where((x_N<0) | (x_N>L-1))] = x[np.where((x_N<0) | (x_N>L-1))]+np.random.randint(low=-1, high=2, size=(sz,1))
        x_reroll = np.where((x_N<0) | (x_N>L-1))
    while (len(y_reroll[0])!=0):
        sz = len(y_reroll)
        y_N[np.where((y_N<0) | (y_N>L-1))] = y[np.where((y_N<0) | (y_N>L-1))]+np.random.randint(low=-1, high=2, size=(sz,1))
        y_reroll = np.where((y_N<0) | (y_N>L-1))

    Intermed_flows[x_N,y_N,:] = Outflow_per_species_at_xy[x,y,:]
    Intermed_flows[x,y,:]=Intermed_flows[x,y,:]-Outflow_per_species_at_xy[x,y,:]
    avail= 1 -np.sum(inputt,axis=2)
    Absolutes = np.sum(Intermed_flows,axis=2)
    norm_ind=np.where(Absolutes[:,:]>avail)
    return Intermed_flows

def degradation(loads):
    outflow = loads[:,:]*epsilon 
    return outflow

def death(loads):
    Gamma_fac=  np.array([[lamda],                                                              # Capital gamma is defined as lambda([v]+[s]) + [D](1-lambda)
                          [lamda],
                          [1-lamda]])
    Gamma_death = np.dot(loads[:,:],Gamma_fac)
    max_value = np.max(Gamma_death)
    max_index = np.where(Gamma_death==max_value)
    Probability_death = epsilon_max*(1/(a-1) * ((a**(Gamma_death))-1))
    random_numbers = np.random.rand(len(loads),1)
    comp =  random_numbers<Probability_death
    Dead_cells_ind = np.where(comp == True)
    return Dead_cells_ind

def simulation(L, generations, Ini_size, Ini_factor, alpha, epsilon, mu, etha_s, etha_D, beta, gamma, epsilon_max, a, lamda, I, K, save_plotter, save_directory, updater,plot_every):
# Initialization
    Default_matrix = np.zeros((L,L))
    Dead           = np.zeros((L,L,generations))                                                            # Logical matrix
    Viral_loads    = np.zeros((L,L,3,generations))                                              # v=((x,y,0,t)), s=((x,y,1,t)), D=((x,y,2,t))
    Ini_min = int((L-Ini_size)/2)                                                               # Initial grid of nonzero values, smalles coordinate (both x and y)
    Ini_max = int((L+Ini_size)/2)                                                               # Initial grid of nonzero values, largest coordinate (both x and y)
    rand_cluster = np.random.rand(Ini_size,Ini_size)                                            # Initial grid has value between 0 and 1 for concentration of D
    Viral_loads[Ini_min:Ini_max,Ini_min:Ini_max,0,0] = 0.5*rand_cluster                  # Initial grid can have value between 0 and 1, or adjusted, depending on this Ini_factor
    Viral_loads[Ini_min:Ini_max,Ini_min:Ini_max,0,1] = 0.15*rand_cluster
    occupancies       = Default_matrix
    occupancies_previous= Default_matrix
    availables        = Default_matrix
    Intermediate_flows = np.zeros((L,L,3))
    Intermediate_repr = np.zeros((L,L,3))
    Intermediate_degr = np.zeros((L,L,3))
    Intermediate_inf  = np.zeros((L,L,3))


    #NORMALIZE THIS#
    choices = np.random.randint(4, size=(L,L,generations))                           # Randomized order of choices for each cell at each time
    for tt in range(generations-1):
        reproducing_cells    = np.where(choices[:,:,tt]==0)
        degrading_cells      = np.where(choices[:,:,tt]==1)
        dying_cells          = np.where(choices[:,:,tt]==2)
        infecting_cells      = np.where(choices[:,:,tt]==3)

        # Reproduction - DONE !
        Intermediate_repr[reproducing_cells[0],reproducing_cells[1],:]= Reproduction(Viral_loads[reproducing_cells[0],reproducing_cells[1],:,tt])
        # Degredation - DONE !
        Intermediate_degr[degrading_cells[0],degrading_cells[1],:]    = -degradation(Viral_loads[degrading_cells[0],degrading_cells[1],:,tt])
        # Infection
        Intermediate_inf= infection(Viral_loads[:,:,:,tt],infecting_cells[0],infecting_cells[1])
        Intermediate_flows[:,:,:] = Intermediate_repr+Intermediate_inf+Intermediate_inf #NEEDS TO BE ADDED TO IPYNB
        #Death-function - DONE!
        x_possible_death = dying_cells[0];        y_possible_death = dying_cells[1]
        cells_to_die                            = death(Viral_loads[dying_cells[0],dying_cells[1],:,tt])
        x_dead=x_possible_death[cells_to_die[0]]; y_dead=y_possible_death[cells_to_die[0]]
        Dead[:,:,tt+1]=Dead[:,:,tt]
        Dead[x_dead,y_dead,tt+1]                      =1
        x_all_dead,y_all_dead = np.where(Dead[:,:,tt+1]==1)

        Viral_loads[:,:,:,tt+1] = Viral_loads[:,:,:,tt]+Intermediate_flows
        toosmall = np.where(Viral_loads[:,:,:,tt+1]<0)
        x_s,y_s,z_s=toosmall[0],toosmall[1],toosmall[2]
        Viral_loads[x_s,y_s,z_s,tt+1]=0

        occupancies[:,:]     = np.sum(Viral_loads[:,:,:,tt+1],axis=2)
        to_normalize_x, to_normalize_y = np.where(occupancies>1)
        if len(to_normalize_x!=0):
            nmz_loads = Viral_loads[to_normalize_x,to_normalize_y,:,tt+1]
            Viral_loads[to_normalize_x,to_normalize_y,0,tt+1]=nmz_loads[:,0]/occupancies[to_normalize_x,to_normalize_y]
            Viral_loads[to_normalize_x,to_normalize_y,1,tt+1]=nmz_loads[:,1]/occupancies[to_normalize_x,to_normalize_y]
            Viral_loads[to_normalize_x,to_normalize_y,2,tt+1]=nmz_loads[:,2]/occupancies[to_normalize_x,to_normalize_y]
        occupancies[:,:]     = np.sum(Viral_loads[:,:,:,tt+1],axis=2)

        Viral_loads[x_all_dead,y_all_dead,:,tt+1]=0
        if tt%updater==0:
            print('generation', tt)

        if (save_plotter):
            if (tt%plot_every==0):
                plt.clf()
                plt.imshow(Viral_loads[:,:,0,tt], cmap='hot_r', interpolation='nearest') # In this case only plotting v!
                plt.colorbar()
                plt.clim(0,1)
                title_string = "Heatmap of [v] for model at full func t =" + str(tt)
                plt.title(title_string)
                directory_and_name = save_directory + "[v]/"+title_string+".png"
                plt.savefig(directory_and_name)

                plt.clf()
                plt.imshow(Viral_loads[:,:,2,tt], cmap='hot_r', interpolation='nearest') # In this case only plotting v!
                plt.colorbar()
                plt.clim(0,1)
                title_string = "Heatmap of [D] for model at full func t =" + str(tt)
                plt.title(title_string)
                directory_and_name = save_directory + "[D]/" + title_string+".png"
                plt.savefig(directory_and_name)
    return Viral_loads, Dead

def extra_plotter(loads,tt):
    N_plots = len(tt)
    for n in range(N_plots):
        plt.figure(n)
        plt.imshow(loads[:,:,0,tt[n]], cmap='hot_r', interpolation='nearest')
        plt.colorbar()
        title = "v at time =" + str(tt[n])
        plt.title(title)
        plt.clim(0,1)
        plt.show()

def dead_plotter(Dead,tt):
    plt.figure()
    cmap = matplotlib.colors.ListedColormap(['white', 'k'])
    bounds = [0.,0.5 ,1.]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(Dead[:,:,tt], interpolation='none', cmap=cmap, norm=norm)
    plt.colorbar()
    title = "Dead cells"
    plt.title(title)
    plt.show()





if __name__ == '__main__':
    # default parameters
    L              = 100
    generations    = 2000
    Ini_size       = 10
    Ini_factor     = 0.4

    alpha       = 0.4
    epsilon     = 0.01
    mu          = 0.4# [0,1]
    etha_s      = 1.5 # [1.5, 15]
    etha_D      = 2.0 # [2, 20]
    beta        = 0.6
    gamma       = 0.6
    epsilon_max = 0#.1
    a           = 50
    lamda       = 0.75
    I           = 0.23 #not 0.01!
    K           = 1

    save_plotter  = False
    plot_every    = 10
    save_directory= "Full model/"
    updater       = generations/10
    tic = time.time()
    loads,Dead         = simulation(L, generations, Ini_size, Ini_factor,alpha, epsilon, mu, etha_s, etha_D, beta, gamma, epsilon_max, a, lamda, I, K, save_plotter, save_directory, updater, plot_every)
    toc = time.time()
    print("execution time =",toc-tic)
    extra_plotter(loads,[0,generations-1])
    dead_plotter(Dead,[generations-1])
    #exec_t=3.06 for L=100 and gen = 1000
    loads_v = loads[:,:,0,:-1]
    loads_vx  = np.sum(loads_v[:,:,:],axis=0)
    loads_vxy = np.sum(loads_vx[:,:],axis=0)
    loads_D = loads[:,:,2,:-1]
    loads_Dx  = np.sum(loads_D[:,:,:],axis=0)
    loads_Dxy = np.sum(loads_Dx[:,:],axis=0)
    Dead = Dead[:,:,:-1]
    Number_deathsx = np.sum(Dead[:,:,:],axis=0)
    Number_deathsxy=np.sum(Number_deathsx[:,:],axis=0)


    loads_vxy[np.isnan(loads_vxy)]=0
    loads_Dxy[np.isnan(loads_Dxy)]=0
    time_axis = np.arange(0,generations-1)
    plt.figure()
    plt.clf()
    plt.plot(time_axis,loads_vxy/L**2,label="[v]")
    plt.plot(time_axis,loads_Dxy/L**2,label="[D]")
    plt.plot(time_axis,Number_deathsxy/L**2,label="Dead")
    plt.legend()
    plt.title("Normalized concentration of virus species over time")
    plt.savefig("total loads.png")
    plt.show()
