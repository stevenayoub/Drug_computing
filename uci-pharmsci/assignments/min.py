# def ConjugateGradient(Pos, dx, EFracTolLS, EFracTolCG):
#     """Performs a conjugate gradient search.
# Input:
#     Pos: starting positions, (N,3) array
#     dx: initial step amount
#     EFracTolLS: fractional energy tolerance for line search
#     EFracTolCG: fractional energy tolerance for conjugate gradient
# Output:
#     PEnergy: value of potential energy at minimum
#     Pos: minimum energy (N,3) position array
# """
#     #### WRITE YOUR CODE HERE ####
#     ## In my code, I can accomplish this function in 10 lines ###
#     #A return statement you may/will use to finish things off    
    
#     # Calculate the initial Forces 
#     PEnergy, Forces = emlib.calcenergyforces(Pos)
#     Dir = Forces
#     previous_energy = PEnergy 
    
#     # print(f'Initial position: {Pos}')
#     # print(f'Initial Energy: {PEnergy}')
#     while True:

#         # Perform line search along the current gradient 
#         PEnergy, Pos = LineSearch(Pos, Dir, dx, EFracTolLS)
        
#         # check if energies have converged?
#         if abs(previous_energy - PEnergy) < EFracTolCG * abs(PEnergy):
#             # print('Energy has converge')
#             # print(f'Final Position: {Pos}')
#             # print(f'Final Energy: {PEnergy}')
#             break
            
#         # If not converged then calculate the new forces and gamma
#         new_forces = emlib.calcforces(Pos)
#         delta_forces = new_forces - Forces
#         gamma = np.sum(delta_forces * new_forces) / np.sum(Forces * Forces)
        
#         # Update the new direction vector 
#         Dir = new_forces + gamma * Dir
        
#         # Update parameters for next iteration
#         Forces = new_forces
#         previous_energy = PEnergy
        
        
#     return PEnergy, Pos