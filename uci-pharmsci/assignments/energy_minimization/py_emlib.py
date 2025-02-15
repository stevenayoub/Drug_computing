
import numpy as np 
import os 
from multiprocessing import Pool
from numba import jit
import numba
#from numba import vectorize, float64
from numba import guvectorize, float64
def calc_energy(Pos):
    """
    Calculate the potential energy of a system of atoms using vectorized operations.
    
    Parameters
    ----------
    Pos: numpy.array 
        Position matrix of shape (NAtom, Dim)
    
    Returns 
    ------
    PEnergy: float 
        Calculated potential energy
    """
    NAtom = Pos.shape[0]
    alpha = 0.0001 * NAtom**(-2./3.)

    # Calculate pairwise distances
    diff = Pos[:, np.newaxis, :] - Pos[np.newaxis, :, :]  # Shape (NAtom, NAtom, Dim)
    # Squared distances, Shape (NAtom, NAtom)
    d2 = np.sum(diff**2, axis=-1)  

    # Avoid self-interactions by setting diagonal to infinity
    np.fill_diagonal(d2, np.inf)

    # Compute energy terms
    id2 = 1.0 / d2
    id6 = id2**3
    id12 = id6**2
    pairwise_energy = 4.0 * (id12 - id6)
    # Avoid double-counting pairs
    total_pairwise_energy = np.sum(pairwise_energy) / 2  

    # Add position-dependent term
    position_energy = alpha * np.sum(np.sum(Pos**2, axis=-1))

    # Total energy
    PEnergy = total_pairwise_energy + position_energy
    return PEnergy

@guvectorize([(float64[:])])
def calc_forces(Pos):
    Pos = np.array(Pos, dtype=np.float64)
    NAtom, Dim = Pos.shape
    alpha = 0.0001 * NAtom**(-2. / 3.)

    Forces = np.zeros((NAtom, Dim), dtype=np.float64)

    # Pairwise differences and squared distances
    diff = Pos[:, np.newaxis, :] - Pos[np.newaxis, :, :] 
    # Squared distances, Shape (NAtom, NAtom)
    d2 = np.sum(diff**2, axis=-1, dtype=np.float64) 
    # Avoid self-interactions by setting diagonal to infinity
    np.fill_diagonal(d2, np.inf)

    # Compute inverse distance powers
    id2 = 1.0 / d2
    id6 = id2**3
    id12 = id6**2

    # Compute pairwise force magnitudes
    force_magnitudes = (-48.0 * id12 + 24.0 * id6) * id2

    # Compute forces and update symmetrically 
    # I was unable to vectorized this due to floating point error rounding...
    for i in range(NAtom):
        for j in range(i + 1, NAtom):
            rij = Pos[j, :] - Pos[i, :]
            Fij = rij * force_magnitudes[i, j]
            Forces[i, :] += Fij
            Forces[j, :] -= Fij
    
    # Apply single-body term
    Forces -= 2.0 * alpha * Pos

    return Forces

@guvectorize([(float64[:])])
def calc_energy_forces(Pos):
    """
    Calculate potential energy and forces based on atom positions using vectorized operations with symmetric updates in a loop.

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

    # Initialize forces and energy
    Forces = np.zeros((NAtom, Dim), dtype=np.float64)
    PEnergy = 0.0

    # Pairwise differences and squared distances
    diff = Pos[:, np.newaxis, :] - Pos[np.newaxis, :, :]  # Shape (NAtom, NAtom, Dim)
    # Squared distances, Shape (NAtom, NAtom)
    d2 = np.sum(diff**2, axis=-1) 

    # Avoid self-interactions by setting diagonal to infinity
    np.fill_diagonal(d2, np.inf)

    # Compute inverse distance powers
    id2 = 1.0 / d2
    id6 = id2**3
    id12 = id6**2

    for i in range(NAtom):
        for j in range(i + 1, NAtom):
            rij = Pos[j, :] - Pos[i, :]  
            d2_ij = d2[i, j] 
            if d2_ij > 0:
                id2_ij = 1.0 / d2_ij
                id6_ij = id2_ij**3
                id12_ij = id6_ij**2
                # Increment energy
                PEnergy += 4.0 * (id12_ij - id6_ij)
                # Compute force contribution
                Fij = rij * ((-48.0 * id12_ij + 24.0 * id6_ij) * id2_ij)
                Forces[i, :] += Fij
                Forces[j, :] -= Fij  
    # Add single-body term contributions to energy and forces
    PEnergy += alpha * np.sum(np.sum(Pos**2, axis=-1))
    Forces -= 2.0 * alpha * Pos

    return PEnergy, Forces

