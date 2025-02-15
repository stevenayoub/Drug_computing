import numpy as np
from pos_to_pdb import *  #This would allow you to export coordinates if you want, later
import py_emlib
import multiprocessing
import pickle
import time
import os 
import argparse

# This will use the Python version of emlib 
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
    #print('line search')
    
    #start the iteration counter
    Iter = 0

    #find the normalized direction
    NormDir = Dir / np.sqrt(np.sum(Dir * Dir))

    #take the first two steps and compute energies
    Dists = [0., dx]
    PEs = [py_emlib.calc_energy(Pos + NormDir * x) for x in Dists]

    #print('PEs', PEs)
    #if the second point is not downhill in energy, back
    #off and take a shorter step until we find one
    while PEs[1] > PEs[0]:
        Iter += 1
        dx = dx * 0.5
        Dists[1] = dx
        PEs[1] = py_emlib.calc_energy(Pos + NormDir * dx)
        #print('second point is not downhill in energy')
        #print('PES 2nd point: ', PEs)
    #find a third point
    Dists.append( 2. * dx )
    PEs.append( py_emlib.calc_energy(Pos + NormDir * 2. * dx) )
    #print('found a 3rd point PEs:', PEs)
    #keep stepping forward until the third point is higher
    #in energy; then we have bracketed a minimum
    while PEs[2] < PEs[1]:
        #print('PEs while loop')
        Iter += 1

        #find a fourth point and evaluate energy
        Dists.append( Dists[-1] + dx )
        PEs.append( py_emlib.calc_energy(Pos + NormDir * Dists[-1]) )
        #print('4th point in while loop', PEs)
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
       # print('End while PEs in while loop', PEs)
    #print('PEs were updated',PEs )            
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
        PE3 = py_emlib.calc_energy(Pos + NormDir * d3)

        #sometimes the parabolic approximation can fail;
        #check if d3 is out of range < d0 or > d2 or the new energy is higher
        if d3 < d0 or d3 > d2 or PE3 > PE0 or PE3 > PE1 or PE3 > PE2:
            #instead, just compute the new distance by bisecting two
            #of the existing points along the line search
            if abs(d2 - d1) > abs(d0 - d1):
                d3 = 0.5 * (d2 + d1)
            else:
                d3 = 0.5 * (d0 + d1)
            PE3 = py_emlib.calc_energy(Pos + NormDir * d3)

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
    Parameters
    ----------
        N:int 
            Number of atoms
        L: float
            box width
    Returns
    ------
        Pos: np.array
            (N,3) array of positions
    """
    #### WRITE YOUR CODE HERE ####
    ## In my code, I can accomplish this function in 1 line 
    ## using a numpy function.
    ## Yours can be longer if you want. It's more important that it be right than that it be short.
    Pos = np.random.uniform(0,L,size=(N,3))
    return Pos



# use calculate force from python version 
def ConjugateGradient(Pos, dx, EFracTolLS, EFracTolCG):
    """Performs a conjugate gradient search.
"""
    #### WRITE YOUR CODE HERE ####
    ## In my code, I can accomplish this function in 10 lines ###
    #A return statement you may/will use to finish things off    
    
    # Calculate the initial Forces 
    intial_energy, Forces = py_emlib.calc_energy_forces(Pos)
    #print('initial energy python: ', intial_energy)
    Dir = Forces
    previous_energy = intial_energy 
    count = 0 
    while True:
        # Perform line search along the current gradient 
        PEnergy, Pos = LineSearch(Pos, Dir, dx, EFracTolLS)
        #print('new energy:', PEnergy)
        # check if energies have converged?
        if abs(previous_energy - PEnergy) < EFracTolCG * abs(PEnergy):
            break
        count +=1
        # If not converged then calculate the new forces and gamma
        new_forces = py_emlib.calc_forces(Pos)
        gamma = np.sum((new_forces - Forces)*new_forces)/np.sum(Forces*Forces)
        
        # Update the new direction vector 
        Dir = new_forces + gamma * Dir
        
        # Update parameters for next iteration
        Forces = new_forces
        previous_energy = PEnergy
        
        
    return PEnergy, Pos



def run_ConjugateGradient(args):
    Pos, dx, EFracTolCG, EFracTolLS = args
    #print(f'current pos: {Pos}')
    PEnergy, _ = ConjugateGradient(Pos, dx, EFracTolLS, EFracTolCG)
    return PEnergy



cpu_count = os.cpu_count()
def minimize_clusters(num_iterations=1, N_min=2, N_max=25, density=0.001, dx=0.001, EFracTolCG=1.0e-10, EFracTolLS=1.0e-8):
    """
    Perform energy minimization on clusters of various sizes and store energies.
    
    Parameters 
    -----------
    num_iterations: int 
        The number of minimizations to be performed per N number of particles 
    N_min: int
        Minumum number of particles to start
    N_max: int 
        Maximum number of particles 
    density: float 
        
    """
    results = {
        'N': [],
        'min_energies': [],
        'avg_energies': [],
        'std_energies': []
    }
    for N in range(N_min, N_max + 1):
        print(f"Minimizing clusters for N={N}...")
        # Calculate L such that density = N / L^3
        L = (N / density) ** (1/3)
        
        # Store energies 
        energies = []
        # # Perform num_iterations of minimzations
        # for _ in range(num_iterations):
        
        # Generate a random initial configuration of particles within a box of size L
        Positions = [np.random.uniform(0, L, size=(N, 3)) for _ in range(num_iterations)]
        tasks = [(pos, dx, EFracTolCG, EFracTolLS) for pos in Positions]
        # Perform energy minimization in parallel got no time to wait! lol 
        with multiprocessing.Pool(processes=cpu_count) as pool:
            energies = pool.map(run_ConjugateGradient, tasks)
        
        # PEnergy, _ = ConjugateGradient(Pos, dx, EFracTolLS, EFracTolCG)
        # energies.append(PEnergy)
        
        # # Analyze and store results
        results['N'].append(N)
        results['min_energies'].append(np.min(energies))
        results['avg_energies'].append(np.mean(energies))
        results['std_energies'].append(np.std(energies))
        

    
    # Save results to a file for later reuse
    with open(f'cluster_minimization_results_k_{num_iterations}.pkl', 'wb') as f:
        pickle.dump(results, f)
    #print(results['min_energies'])
    return results


    
if __name__ == "__main__":
    start_time = time.time()
    minimize_clusters(num_iterations=100)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")