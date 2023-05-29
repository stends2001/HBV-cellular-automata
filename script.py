import time
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import matplotlib

def Reproduction(loads, etha_s, etha_D, alpha, beta, gamma, mu):
    G_matrix    = np.zeros((len(loads),1))
    Repr_matrix = np.zeros((len(loads),3))
    Normalized_inflow    = np.zeros((len(loads),3))
    S_scaled = etha_s*loads[:,1]
    D_scaled = etha_D*loads[:,2]
    G_matrix[:,0] = 1/(1+S_scaled[:]+D_scaled[:])                                                                               #Interference values per cell
    Repr_matrix[:,0]=G_matrix[:,0]*alpha*(1-mu)                   *loads[:,0]                                                   #Inflow v: interference * alpha * (1-mu) * [v]
    Repr_matrix[:,1]=beta*loads[:,1]                              *loads[:,0]                                                   #Inflow s: beta * [v]
    Repr_matrix[:,2]=(1+gamma*loads[:,0])*(loads[:,2]+G_matrix[:,0]*alpha*mu*loads[:,0])-loads[:,2]                             #Inflow D: (1+gamma*[v])([D]+interference*alpha*mu*[v])-[D]
    Available = 1-np.sum(loads[:,:],axis=1)
    Absolute_inflow=np.sum(Repr_matrix[:,:],axis=1)
    tonormalize = Absolute_inflow>Available                                                                                     #Cells to normalize = every cell where threshold of 1 is exceeded
    Normalized_inflow           = Repr_matrix                                                                                   #Normalize towards an added occupancy of 1
    toosmall = Normalized_inflow<0
    Normalized_inflow[toosmall]=0
    return Normalized_inflow

def infection(inputt,x,y,L,I):
    Intermed_flows               = np.zeros((L,L,3))
    Outflow_per_species_at_xy    = np.zeros((L,L,3))
    multiplication_factors       = np.zeros((L,L,3))
    multiplication_factors[x,y,0]=1
    multiplication_factors[x,y,1]=inputt[x,y,0]
    multiplication_factors[x,y,2]=inputt[x,y,0]
    Outflow_per_species_at_xy[:,:,0]    = I* inputt[:,:,0]*multiplication_factors[:,:,0]                                        #Outflow [v]: I*[v]
    Outflow_per_species_at_xy[:,:,1]    = I* inputt[:,:,1]*multiplication_factors[:,:,1]                                        #Outflow [s]: I*[v]*[s]
    Outflow_per_species_at_xy[:,:,2]    = I* inputt[:,:,2]*multiplication_factors[:,:,2]                                        #Outflow [D]: I*[v]*[D]
    #############################################
    # Todo: Not allow x_N and y_N to be x and y #
    #############################################
    x_N = x+np.random.randint(low=-1, high=2, size=(len(x)))                                                                    # x and y get a randomn number by Moore's neighborhood
    y_N = y+np.random.randint(low=-1, high=2, size=(len(x)))
    x_reroll = np.where((x_N<0) | (x_N>L-1))                                                                                    #reroll if the neighbor is outside boundaries (x)
    y_reroll = np.where((y_N<0) | (y_N>L-1))                                                                                    #reroll if the neighbor is outside boundaries (y)
    while (len(x_reroll[0])!=0):                                                                                                #while there are x coordinates to reroll for a new neighbor, determine the new x
        sz = len(x_reroll)
        x_N[np.where((x_N<0) | (x_N>L-1))] = x[np.where((x_N<0) | (x_N>L-1))]+np.random.randint(low=-1, high=2, size=(sz,1))
        x_reroll = np.where((x_N<0) | (x_N>L-1))
    while (len(y_reroll[0])!=0):                                                                                                #while there are y coordinates to reroll for a new neighbor, determine the new y
        sz = len(y_reroll)
        y_N[np.where((y_N<0) | (y_N>L-1))] = y[np.where((y_N<0) | (y_N>L-1))]+np.random.randint(low=-1, high=2, size=(sz,1))
        y_reroll = np.where((y_N<0) | (y_N>L-1))

    Intermed_flows[x_N,y_N,:] = Outflow_per_species_at_xy[x,y,:]
    Intermed_flows[x,y,:]=Intermed_flows[x,y,:]-Outflow_per_species_at_xy[x,y,:]
    avail= 1 -np.sum(inputt,axis=2)
    Absolutes = np.sum(Intermed_flows,axis=2)
    return Intermed_flows

def degradation(loads, epsilon):
    outflow = loads[:,:]*epsilon
    return outflow

def death(loads, lamda, epsilon_max, a):
    Gamma_fac=  np.array([[lamda],                                                              #Capital gamma is defined as lambda([v]+[s]) + [D](1-lambda)
                          [lamda],
                          [1-lamda]])
    Gamma_death = np.dot(loads[:,:],Gamma_fac)
    Probability_death = epsilon_max*(1/(a-1) * ((a**(Gamma_death))-1))                          #Pareto distribution, calculated per cell
    random_numbers = np.random.rand(len(loads),1)                                               #Every cell potentially dying gets a random number
    comp =  random_numbers<Probability_death                                                    #If the random number is smaller than the pareto probability, then the cell dies
    Dead_cells_ind = np.where(comp == True)
    return Dead_cells_ind                                                                       #Returns the coordinates of the cells that die

def simulation(L, generations, Ini_size, Ini_v, Ini_s, alpha, epsilon, mu, etha_s, etha_D, beta, gamma, epsilon_max, a, lamda, I, K, save_plotter, save_directory, updater,plot_every):
# Initialization
    Default_matrix = np.zeros((L,L))
    Dead           = np.zeros((L,L,generations))                                                #Logical matrix
    Viral_loads    = np.zeros((L,L,3,generations))                                              #v=((x,y,0,t)), s=((x,y,1,t)), D=((x,y,2,t))
    Ini_min = int((L-Ini_size)/2)                                                               #Initial grid of nonzero values, smalles coordinate (both x and y)
    Ini_max = int((L+Ini_size)/2)                                                               #Initial grid of nonzero values, largest coordinate (both x and y)
    rand_cluster = np.random.rand(Ini_size,Ini_size)                                            #Initial grid has value between 0 and 1 for concentration of D
    Viral_loads[Ini_min:Ini_max,Ini_min:Ini_max,0,0] = Ini_v*rand_cluster                        #Initial grid for v
    Viral_loads[Ini_min:Ini_max,Ini_min:Ini_max,1,0] = Ini_s*rand_cluster                        #Initial gid for s
    occupancies         = Default_matrix
    occupancies_previous= Default_matrix
    availables          = Default_matrix
    Intermediate_flows  = np.zeros((L,L,3))
    Intermediate_repr   = np.zeros((L,L,3))
    Intermediate_degr   = np.zeros((L,L,3))
    Intermediate_inf    = np.zeros((L,L,3))
    choices = np.random.randint(4, size=(L,L,generations))                                      #Randomized order of choices for each cell at each time
    counter=0
    update_every = generations/updater
    for tt in range(generations-1):                                                                                                                    #loop of generation
        reproducing_cells    = np.where(choices[:,:,tt]==0)                                                                                            #identify reproducing cells
        degrading_cells      = np.where(choices[:,:,tt]==1)                                                                                            #identify degrading cells
        dying_cells          = np.where(choices[:,:,tt]==2)                                                                                            #identify dying cells
        infecting_cells      = np.where(choices[:,:,tt]==3)                                                                                            #identify infecting cells (not the infectED cells)

        # Flow of reproduction
        #input: 2D matrix, rows are cells with columns of [v], [s] and [D]
        #output:2D matrix, rows are cells with columns of [v], [s] and [D] of the inflow
        Intermediate_repr[reproducing_cells[0],reproducing_cells[1],:]= Reproduction(Viral_loads[reproducing_cells[0],reproducing_cells[1],:,tt],etha_s, etha_D, alpha, beta, gamma, mu)

        # Flow of degredation
        #input: 2D matrix, rows are cells with columns of [v], [s] and [D]
        #output:2D matrix, rows are cells with columns of [v], [s] and [D] of the (positive) outflow
        Intermediate_degr[degrading_cells[0],degrading_cells[1],:]    = -degradation(Viral_loads[degrading_cells[0],degrading_cells[1],:,tt],epsilon)

        # Flow of infection
        #input: - 3D grid of cells [x,y] and viral concentration of [v], [s] and [D] [third dimension] at timepoint
        #       - list of x coordinates of infecting cells
        #       - list of y coordinates of infecting cells
        #output:- 3D grid of cells [x,y] and viral concentration of [v], [s] and [D] [third dimension] at timepoint
        Intermediate_inf                                              = infection(Viral_loads[:,:,:,tt],infecting_cells[0],infecting_cells[1], L, I)

        Intermediate_flows[:,:,:]                                     = Intermediate_repr[:,:,:]+Intermediate_degr[:,:,:]+Intermediate_inf[:,:,:]       #3D grid of inflows and outflows

        Viral_loads[:,:,:,tt+1] = Viral_loads[:,:,:,tt]+Intermediate_flows

        ################################
        # Todo: make this more elegant #
        ################################
        x_possible_death        = dying_cells[0];  y_possible_death = dying_cells[1]                                                                    #identify cells with function of dying
        cells_to_die            = death(Viral_loads[dying_cells[0],dying_cells[1],:,tt], lamda, epsilon_max, a)                                                                #from these cells, determine the ones that actually die (output of function!)
        x_dead                  = x_possible_death[cells_to_die[0]]                                                                                     #definite dead cells x
        y_dead                  = y_possible_death[cells_to_die[0]]                                                                                     #definite dead cells y
        Dead[:,:,tt+1]          =Dead[:,:,tt]                                                                                                           #all dead cells, updated per generation
        Dead[x_dead,y_dead,tt+1]=1
        x_all_dead,y_all_dead   = np.where(Dead[:,:,tt+1]==1)
        Viral_loads[x_all_dead,y_all_dead,:,tt+1]=0

        #Protective mechanism agains negative values
        toosmall = np.where(Viral_loads[:,:,:,tt+1]<0)
        x_s,y_s,z_s=toosmall[0],toosmall[1],toosmall[2]
        Viral_loads[x_s,y_s,z_s,tt+1]=0
        
        #Normalize to not exceed threshold of 1
        occupancies[:,:]     = np.sum(Viral_loads[:,:,:,tt+1],axis=2)
        to_normalize_x, to_normalize_y = np.where(occupancies>1)
        if len(to_normalize_x!=0):
            nmz_loads = Viral_loads[to_normalize_x,to_normalize_y,:,tt+1]
            Viral_loads[to_normalize_x,to_normalize_y,0,tt+1]=nmz_loads[:,0]/occupancies[to_normalize_x,to_normalize_y]
            Viral_loads[to_normalize_x,to_normalize_y,1,tt+1]=nmz_loads[:,1]/occupancies[to_normalize_x,to_normalize_y]
            Viral_loads[to_normalize_x,to_normalize_y,2,tt+1]=nmz_loads[:,2]/occupancies[to_normalize_x,to_normalize_y]
        occupancies[:,:]     = np.sum(Viral_loads[:,:,:,tt+1],axis=2)
        #Update every `updater` generations
        if (tt+1)%update_every==0:
            counter=counter+1
            per=counter*100/updater
            print('Generation: {0}; Simulation completed for {1}%'.format(tt+1,per))

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
                plt.imshow(Viral_loads[:,:,1,tt], cmap='hot_r', interpolation='nearest') # In this case only plotting s!
                plt.colorbar()
                plt.clim(0,1)
                title_string = "Heatmap of [s] for model at full func t =" + str(tt)
                plt.title(title_string)
                directory_and_name = save_directory + "[s]/" + title_string+".png"
                plt.savefig(directory_and_name)

                plt.clf()
                plt.imshow(Viral_loads[:,:,2,tt], cmap='hot_r', interpolation='nearest') # In this case only plotting D!
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

def simulation_loads(loads,dead,L,save_loads):
    loads_v = loads[:,:,0,:-1]
    loads_vx  = np.sum(loads_v[:,:,:],axis=0)
    loads_vxy = np.sum(loads_vx[:,:],axis=0)/L**2

    loads_s = loads[:,:,1,:-1]
    loads_sx= np.sum(loads_s[:,:,:],axis=0)
    loads_sxy=np.sum(loads_sx[:,:], axis=0)/L**2

    loads_D = loads[:,:,2,:-1]
    loads_Dx  = np.sum(loads_D[:,:,:],axis=0)
    loads_Dxy = np.sum(loads_Dx[:,:],axis=0)/L**2

    dead = dead[:,:,:-1]
    Number_deathsx = np.sum(dead[:,:,:],axis=0)
    Number_deathsxy=np.sum(Number_deathsx[:,:],axis=0)/L**2

    if save_loads == True:
        np.savetxt("relative_v.csv",loads_vxy , delimiter=",")
        np.savetxt("relative_s.csv",loads_sxy , delimiter=",")
        np.savetxt("relative_D.csv",loads_Dxy , delimiter=",")
        np.savetxt("relative_dead.csv",Number_deathsxy , delimiter=",")

    return loads_vxy, loads_sxy, loads_Dxy, Number_deathsxy


def load_plotter(loads_v, loads_s, loads_D, Number_death, generations):
    time_axis = np.arange(0,generations-1)
    plt.figure()
    plt.clf()
    plt.grid()
    plt.plot(time_axis,loads_v,label="[v]",color='blue')
    plt.plot(time_axis,loads_s,label="[s]",color='brown')
    plt.plot(time_axis,loads_D,label="[D]",color='green')
    plt.plot(time_axis,Number_death,label="Dead",color='black')
    plt.legend()
    plt.title("Normalized concentration of virus species over time")
    plt.savefig("total loads.png")
    plt.show()





if __name__ == '__main__':
    ########################
    # Arbitrary parameters #
    ########################
    L_glob              = 100
    generations_glob    = 5000
    Ini_size_glob       = 10
    Ini_v_glob          = 0.1
    Ini_s_glob          = 0.9 # 0.015
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
    beta_glob        = 1.5      # 2
    gamma_glob       = 2   # 1.5
    etha_s_glob      = 1.1
    epsilon_glob     = 3*10**-2

    mu_glob          = 0.2
    etha_D_glob      = 1.3
    
    ###############
    # Data/output #
    ###############
    save_plotter    = False
    plot_every      = 5
    save_directory  = ""
    updater         = 10
    calc_rel        = True
    save_lds        = True

    tic = time.time()
    loads,Dead         = simulation(L_glob, generations_glob, Ini_size_glob, Ini_v_glob, Ini_s_glob, alpha_glob, epsilon_glob, mu_glob, etha_s_glob, etha_D_glob, beta_glob, gamma_glob, epsilon_max_glob, a_glob, lamda_glob, I_glob, K_glob, save_plotter, save_directory, updater, plot_every)
    if calc_rel == True:
        rel_v, rel_s, rel_D, rel_dead = simulation_loads(loads,Dead,L_glob,save_lds)
        load_plotter(rel_v, rel_s, rel_D,rel_dead, generations_glob)
    toc = time.time()