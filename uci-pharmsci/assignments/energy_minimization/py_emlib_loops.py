
import numpy as np 

def calc_energy(Pos):
    """
    Calculate the potential energy of a system of atoms.
    
    Parameters
    ----------
    Pos: numpy.array 
        Position matrix of shape (NAtom, Dim)
    Dim: int
        Dimensions of space 
    NAtom: int 
        Number of atoms
    
    Returns 
    ------
    PEnergy: float 
        Calculated potential energy
    """
    NAtom = Pos.shape[0]
    alpha = 0.0001 * NAtom**(-2./3.)
    PEnergy = 0.0
    # iterate through all atoms 
    for i in range(NAtom):
        position_i = Pos[i]
        for j in range(i + 1, NAtom):  
            rij = Pos[j] - position_i 
            d2 = np.sum(rij * rij)
            id2 = 1.0/d2
            id6 = id2 * id2 * id2    
            id12 = id6 * id6
            PEnergy += 4.0 * (id12 - id6)
        
        PEnergy += alpha * np.sum(position_i * position_i)
        
    return PEnergy



def calc_forces(Pos):
    """
    Calculate the Forces based on atom positions.

    Parameters
    ----------
    Pos : numpy.ndarray
        Position matrix of shape (NAtom, Dim).

    Returns
    -------
    Forces : numpy.ndarray
        The calculated forces, shape (NAtom, Dim).
    """
    Pos = np.array(Pos, dtype=np.float64)
    Dim = Pos.shape[1]
    NAtom = Pos.shape[0]
    alpha = 0.0001 * NAtom**(-2./3.)
    
    # Initialize Forces array with zeros
    Forces = np.zeros((NAtom, Dim), dtype=np.float64)
    
    # Loop over all pairs (i, j)
    for i in range(NAtom):
        for j in range(i + 1, NAtom):
            rij = Pos[j, :] - Pos[i, :]
            d2 = np.sum(rij**2)
            if d2 > 0:
                id2 = 1.0 / d2
                id6 = id2**3
                id12 = id6**2
                Fij = rij * ((-48.0 * id12 + 24.0 * id6) * id2)
                Forces[i, :] += Fij
                Forces[j, :] -= Fij
    
    # Apply single-body term
    Forces -= 2.0 * alpha * Pos
    
    return Forces



import numpy as np

# def calc_energy_forces(Pos):
#     """
#     Calculate the potential energy and forces acting on particles.

#     Parameters:
#         Pos (numpy.ndarray): Position matrix of shape (NAtom, Dim).

#     Returns:
#         float: Total potential energy.
#         numpy.ndarray: Forces array of shape (NAtom, Dim).
#     """
#     Dim = Pos.shape[1]
#     NAtom = Pos.shape[0]
#     alpha = 0.0001 * NAtom**(-2.0 / 3.0)
    
#     # Initialize Forces array with zeros
#     Forces = np.zeros((NAtom, Dim))
    
#     # Compute pairwise displacement vectors
#     rij = Pos[:, np.newaxis, :] - Pos[np.newaxis, :, :]  # Shape: (NAtom, NAtom, Dim)
    
#     # Compute squared distances
#     d2 = np.sum(rij**2, axis=2)  # Shape: (NAtom, NAtom)
#     np.fill_diagonal(d2, np.inf)  # Avoid division by zero for self-interaction
    
#     # Compute inverse distance terms
#     id2 = 1.0 / d2
#     id6 = id2**3
#     id12 = id6**2
    
#     # Compute pairwise potential energy
#     potential_energies = 4.0 * (id12 - id6)  # Shape: (NAtom, NAtom)
#     PEnergy = np.sum(np.triu(potential_energies))
    
#     # Compute pairwise forces
#     force_magnitude = (-48.0 * id12 + 24.0 * id6) * id2  # Shape: (NAtom, NAtom)
#     pairwise_forces = rij * force_magnitude[:, :, np.newaxis]  # Shape: (NAtom, NAtom, Dim)
    
#     # Update forces for each pair
#     for i in range(NAtom):
#          for j in range(i + 1, NAtom):
#              Forces[i, :] += pairwise_forces[i, j, :]
#              Forces[j, :] -= pairwise_forces[i, j, :]
             
    
#     # Add single-body term to energy and forces
#     single_body_energy = alpha * np.sum(Pos**2)
#     PEnergy += single_body_energy
#     Forces -= 2.0 * alpha * Pos
    
#     return PEnergy, Forces



def calc_energy_forces(Pos):
    """
    Calculate potential energy and forces based on atom positions.

    Parameters
    ----------
    Pos : numpy.ndarray
        Position matrix of shape (NAtom, Dim).

    Returns
    -------
    PEnergy : float
        The calculated potential energy.
    Forces : numpy.ndarray
        The calculated forces, shape (NAtom, Dim).
    """
    Pos = np.array(Pos, dtype=np.float64)
    NAtom, Dim = Pos.shape
    alpha = 0.0001 * NAtom**(-2. / 3.)  # alpha varies with system size

    # Initialize energy and forces
    PEnergy = 0.0
    Forces = np.zeros((NAtom, Dim), dtype=np.float64)

    # Loop over atoms to compute forces and potential energy
    for i in range(NAtom):
        # Store Pos(i,:) in a temporary array for faster access in the inner loop
        Posi = Pos[i, :]
        for j in range(i + 1, NAtom):
            rij = Pos[j, :] - Posi
            d2 = np.sum(rij**2)  # Squared distance
            if d2 > 0:
                id2 = 1.0 / d2             # Inverse squared distance
                id6 = id2**3               # Inverse sixth distance
                id12 = id6**2              # Inverse twelfth distance
                PEnergy += 4.0 * (id12 - id6)  # Lennard-Jones potential
                Fij = rij * ((-48.0 * id12 + 24.0 * id6) * id2)  # Force magnitude
                Forces[i, :] += Fij
                Forces[j, :] -= Fij

        # Compute the single-body term for potential energy and forces
        PEnergy += alpha * np.sum(Posi**2)
        Forces[i, :] -= 2.0 * alpha * Posi

    return PEnergy, Forces







if __name__ == "__main__":
    # Input parameters
    N = 25
    density = 0.001
    dx = 0.001
    L = (N / density) ** (1 / 3)

    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    Pos = np.random.uniform(0, L, size=(N, 3))

    # Call the Python function
    Forces_python = calc_energy_forces_parallel(Pos)

    # Print Python results
    print("Fortran Forces:")
    print(Forces_python)