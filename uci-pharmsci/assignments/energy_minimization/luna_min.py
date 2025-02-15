import numpy as np
import emlib
import pickle

import time 
def LineSearch(Pos, Dir, dx, EFracTol, Accel = 1.5, MaxInc = 10.,
               MaxIter = 10000):
    """Performs a line search along direction Dir.
    Input:
       Pos: starting positions, (N,3) array
       Dir: (N,3) array of gradient direction
       dx: initial step amount, a float
       EFracTol: fractional energy tolerance
       Accel: acceleration factor
       MaxInc: the maximum increase in energy for bracketing
       MaxIter: maximum number of iteration steps
Output:
       PEnergy: value of potential energy at minimum along Dir
       PosMin: minimum energy (N,3) position array along Dir
"""
    #start the iteration counter
    Iter = 0

    #find the normalized direction
    NormDir = Dir / np.sqrt(np.sum(Dir * Dir))

    #take the first two steps and compute energies
    Dists = [0., dx]
    PEs = [emlib.calcenergy(Pos + NormDir * x) for x in Dists]

    #if the second point is not downhill in energy, back
    #off and take a shorter step until we find one
    while PEs[1] > PEs[0]:
        Iter += 1
        dx = dx * 0.5
        Dists[1] = dx
        PEs[1] = emlib.calcenergy(Pos + NormDir * dx)

    #find a third point
    Dists.append( 2. * dx )
    PEs.append( emlib.calcenergy(Pos + NormDir * 2. * dx) )

    #keep stepping forward until the third point is higher
    #in energy; then we have bracketed a minimum
    while PEs[2] < PEs[1]:
        Iter += 1

        #find a fourth point and evaluate energy
        Dists.append( Dists[-1] + dx )
        PEs.append( emlib.calcenergy(Pos + NormDir * Dists[-1]) )

        #check if we increased too much in energy; if so, back off
        if (PEs[3] - PEs[0]) > MaxInc * (PEs[0] - PEs[2]):
            PEs = PEs[:3]
            Dists = Dists[:3]
            dx = dx * 0.5
        else:
            #shift all of the points over
            PEs = PEs[-3:]
            Dists = Dists[-3:]
            dx = dx * Accel    
                
    #we've bracketed a minimum; now we want to find it to high
    #accuracy
    OldPE3 = 1.e300
    while True:
        Iter += 1
        if Iter > MaxIter:
            print("Warning: maximum number of iterations reached in line search.")
            break

        #store distances for ease of code-reading
        d0, d1, d2 = Dists
        PE0, PE1, PE2 = PEs

        #use a parobolic approximation to estimate the location
        #of the minimum
        d10 = d0 - d1
        d12 = d2 - d1
        Num = d12*d12*(PE0-PE1) - d10*d10*(PE2-PE1)
        Dem = d12*(PE0-PE1) - d10*(PE2-PE1)
        if Dem == 0:
            #parabolic extrapolation won't work; set new dist = 0
            d3 = 0
        else:
            #location of parabolic minimum
            d3 = d1 + 0.5 * Num / Dem

        #compute the new potential energy
        PE3 = emlib.calcenergy(Pos + NormDir * d3)

        #sometimes the parabolic approximation can fail;
        #check if d3 is out of range < d0 or > d2 or the new energy is higher
        if d3 < d0 or d3 > d2 or PE3 > PE0 or PE3 > PE1 or PE3 > PE2:
            #instead, just compute the new distance by bisecting two
            #of the existing points along the line search
            if abs(d2 - d1) > abs(d0 - d1):
                d3 = 0.5 * (d2 + d1)
            else:
                d3 = 0.5 * (d0 + d1)
            PE3 = emlib.calcenergy(Pos + NormDir * d3)

        #decide which three points to keep; we want to keep
        #the three that are closest to the minimum
        if d3 < d1:
            if PE3 < PE1:
                #get rid of point 2
                Dists, PEs = [d0, d3, d1], [PE0, PE3, PE1]
            else:
                #get rid of point 0
                Dists, PEs = [d3, d1, d2], [PE3, PE1, PE2]
        else:
            if PE3 < PE1:
                #get rid of point 0
                Dists, PEs = [d1, d3, d2], [PE1, PE3, PE2]
            else:
                #get rid of point 2
                Dists, PEs = [d0, d1, d3], [PE0, PE1, PE3]

        #check how much we've changed
        if abs(OldPE3 - PE3) < EFracTol * abs(PE3):
            #the fractional change is less than the tolerance,
            #so we are done and can exit the loop
            break
        OldPE3 = PE3

    #return the position array at the minimum (point 1)
    PosMin = Pos + NormDir * Dists[1]
    PEMin = PEs[1]

    return PEMin, PosMin

def InitPositions(N, L):
    """Returns an array of initial positions of each atom,
placed randomly within a box of dimensions L.
Input:
    N: number of atoms
    L: box width
Output:
    Pos: (N,3) array of positions
"""
    #### WRITE YOUR CODE HERE ####
    ## In my code, I can accomplish this function in 1 line 
    ## using a numpy function.
    ## Yours can be longer if you want. It's more important that it be right than that it be short.
    Pos = np.random.uniform(0,L,size=(N,3)) #generate random uniformly distributed numbers in the interval of [0,L)
    return Pos

def ConjugateGradient(Pos, dx, EFracTolLS, EFracTolCG):
    """Performs a conjugate gradient search.
Input:
    Pos: starting positions, (N,3) array
    dx: initial step amount
    EFracTolLS: fractional energy tolerance for line search
    EFracTolCG: fractional energy tolerance for conjugate gradient
Output:
    PEnergy: value of potential energy at minimum
    Pos: minimum energy (N,3) position array
"""
    #### WRITE YOUR CODE HERE ####
    ## In my code, I can accomplish this function in 10 lines ###
         
    PEnergy, Forces = emlib.calcenergyforces(Pos) #energy and forces of the inital position
    Dir = Forces #initial search direction is the direction of forces 
    while True: #this while loop is True until the energy of recent iteration is equal to the energy of the previous iteration 
        #do LineSearch to find potential energy at minimum and minimum energy position array along Dir
        new_PEnergy, new_Pos = LineSearch(Pos, Dir, dx, EFracTolLS, Accel = 1.5, MaxInc = 10.,MaxIter = 10000)
        #find a new forces using emlib
        new_Forces =  emlib.calcforces(new_Pos)

        #check convergence with EFracTolCG
        if abs(new_PEnergy - PEnergy)/abs(PEnergy) < EFracTolCG:
            break
        
        gamma = np.sum((new_Forces - Forces)*new_Forces)/np.sum(Forces*Forces)
        Dir = new_Forces+gamma*Dir #update the search direction matrix
        
        Pos = new_Pos
        Forces = new_Forces
        PEnergy = new_PEnergy
    #A return statement you may/will use to finish things off    
    return PEnergy, Pos

#Your energy minimization code here
#This will be the longest code you write in this assignment

import multiprocessing

def run_conjugate_gradeint(args):
    Pos, dx, EFracTolLS, EFracTolCG = args 

    Penergy, _ = ConjugateGradient(Pos,dx,EFracTolLS,EFracTolCG)

    return Penergy

def min_cluster(k_iter):
    dx = 0.001
    EFracTolCG = 1.0e-10
    EFracTolLS = 1.0e-8

    data = {
        'N': [],
        'min_energy':[],
        'average': [],
        'std_dev': [],
        'energies':[]
    }
    for N in range(25,26):
        print(f'Number of Particles: {N}')
        L = (N/0.001)**(1/3)
        positions = [InitPositions(N,L) for _ in range(k_iter)]
        tasks = [(pos, dx, EFracTolLS, EFracTolCG) for pos in positions]
        
        with multiprocessing.Pool(processes=11) as pool:
            energy_values = pool.map(run_conjugate_gradeint, tasks)

   
        data['N'].append(N)
        data['min_energy'].append(np.min(energy_values))
        data['average'].append(np.mean(energy_values))
        data['std_dev'].append(np.std(energy_values))
        data['energies'].append(energy_values)
        # print('minimum energies', minimum)
        # if N not in data:
        #     data[N] = {}
        # data[N][K] = {'minimum_energy': minimum, 'average_energy': average, 'standard_deviation': std_dev}

    with open(f'test_N_25_energies_k_{k_iter}.pickle','wb') as file:
        pickle.dump(data,file)

if __name__ == "__main__":
    start = time.time()
    min_cluster(1000)
    end = time.time()
    diff = end - start
    print(f'total time: {diff}')