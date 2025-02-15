from multiprocessing import Pool
import os 
import numpy as np

def compute_energy_chunk(start, end, Pos, NAtom):
    """
    Compute potential energy for a chunk of atom pairs.
    
    Parameters
    ----------
    start : int
        Start index of the atom range.
    end : int
        End index of the atom range.
    Pos : numpy.ndarray
        Position matrix of shape (NAtom, Dim).
    NAtom : int
        Total number of atoms.

    Returns
    -------
    PEnergy_local : float
        Partial potential energy calculated for the chunk.
    """
    PEnergy_local = 0.0
    for i in range(start, end):
        for j in range(i + 1, NAtom):
            rij = Pos[j, :] - Pos[i, :]
            d2 = np.sum(rij**2)
            if d2 > 0:
                id2 = 1.0 / d2
                id6 = id2**3
                id12 = id6**2
                PEnergy_local += 4.0 * (id12 - id6)
    return PEnergy_local

def calc_energy_parallel(Pos, num_processes=None):
    """
    Parallelized computation of potential energy based on atom positions.
    
    Parameters
    ----------
    Pos : numpy.ndarray
        Position matrix of shape (NAtom, Dim).
    num_processes : int, optional
        Number of parallel processes to use (default: all available cores).

    Returns
    -------
    PEnergy : float
        The calculated potential energy.
    """
    Pos = np.array(Pos, dtype=np.float64)
    NAtom = Pos.shape[0]
    alpha = 0.0001 * NAtom**(-2. / 3.)

    # Default to using all available CPU cores if num_processes is not specified
    if num_processes is None:
        num_processes = os.cpu_count()

    # Divide the atom indices into chunks for parallel processing
    chunk_size = (NAtom + num_processes - 1) // num_processes
    chunks = [(i, min(i + chunk_size, NAtom), Pos, NAtom) for i in range(0, NAtom, chunk_size)]

    # Use multiprocessing Pool to compute energy in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(compute_energy_chunk, chunks)

    # Combine the results from all processes
    total_pairwise_energy = sum(results)

    # Add position-dependent term
    position_energy = alpha * np.sum(np.sum(Pos**2, axis=-1))

    # Total energy
    PEnergy = total_pairwise_energy + position_energy
    return PEnergy

def compute_force_chunk(start, end, Pos, NAtom, Dim):
    """
    Compute forces for a chunk of atom pairs.
    
    Parameters
    ----------
    start : int
        Start index of the atom range.
    end : int
        End index of the atom range.
    Pos : numpy.ndarray
        Position matrix of shape (NAtom, Dim).
    NAtom : int
        Total number of atoms.
    Dim : int
        Number of dimensions.

    Returns
    -------
    Forces_local : numpy.ndarray
        Partial forces calculated for the chunk, shape (NAtom, Dim).
    """
    Forces_local = np.zeros((NAtom, Dim), dtype=np.float64)
    for i in range(start, end):
        for j in range(i + 1, NAtom):
            rij = Pos[j, :] - Pos[i, :]
            d2 = np.sum(rij**2)
            if d2 > 0:
                id2 = 1.0 / d2
                id6 = id2**3
                id12 = id6**2
                Fij = rij * ((-48.0 * id12 + 24.0 * id6) * id2)
                Forces_local[i, :] += Fij
                Forces_local[j, :] -= Fij
    return Forces_local

def calc_forces_parallel(Pos, num_processes=None):
    """
    Calculate forces with parallelized pairwise computation.
    
    Parameters
    ----------
    Pos : numpy.ndarray
        Position matrix of shape (NAtom, Dim).
    num_processes : int, optional
        Number of parallel processes to use (default: all available cores).

    Returns
    -------
    Forces : numpy.ndarray
        The calculated forces, shape (NAtom, Dim).
    """
    Pos = np.array(Pos, dtype=np.float64)
    NAtom, Dim = Pos.shape
    alpha = 0.0001 * NAtom**(-2. / 3.)
    
    # Default to using all available CPU cores if num_processes is not specified
    if num_processes is None:
        num_processes = os.cpu_count()  # Use all available cores by default

    # Divide the atom indices into chunks for parallel processing
    chunk_size = (NAtom + num_processes - 1) // num_processes  # Divide evenly
    chunks = [(i, min(i + chunk_size, NAtom), Pos, NAtom, Dim) for i in range(0, NAtom, chunk_size)]

    # Use multiprocessing Pool to compute forces in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(compute_force_chunk, chunks)

    # Combine the results from all processes
    Forces = np.sum(results, axis=0)

    # Apply single-body term
    Forces -= 2.0 * alpha * Pos

    return Forces


def compute_energy_force_chunk(start, end, Pos, NAtom, Dim, alpha):
    """
    Compute potential energy and forces for a chunk of atom pairs.
    
    Parameters
    ----------
    start : int
        Start index of the atom range.
    end : int
        End index of the atom range.
    Pos : numpy.ndarray
        Position matrix of shape (NAtom, Dim).
    NAtom : int
        Total number of atoms.
    Dim : int
        Number of dimensions.
    alpha : float
        Coefficient for single-body term.

    Returns
    -------
    PEnergy_local : float
        Partial potential energy calculated for the chunk.
    Forces_local : numpy.ndarray
        Partial forces calculated for the chunk, shape (NAtom, Dim).
    """
    PEnergy_local = 0.0
    Forces_local = np.zeros((NAtom, Dim), dtype=np.float64)
    for i in range(start, end):
        for j in range(i + 1, NAtom):
            rij = Pos[j, :] - Pos[i, :]
            d2 = np.sum(rij**2)
            if d2 > 0:
                id2 = 1.0 / d2
                id6 = id2**3
                id12 = id6**2
                PEnergy_local += 4.0 * (id12 - id6)
                Fij = rij * ((-48.0 * id12 + 24.0 * id6) * id2)
                Forces_local[i, :] += Fij
                Forces_local[j, :] -= Fij

    return PEnergy_local, Forces_local

def calc_energy_forces_parallel(Pos, num_processes=None):
    """
    Parallelized computation of potential energy and forces based on atom positions.

    Parameters
    ----------
    Pos : numpy.ndarray
        Position matrix of shape (NAtom, Dim).
    num_processes : int, optional
        Number of parallel processes to use (default: all available cores).

    Returns
    -------
    PEnergy : float
        The calculated potential energy.
    Forces : numpy.ndarray
        The calculated forces, shape (NAtom, Dim).
    """
    Pos = np.array(Pos, dtype=np.float64)
    NAtom, Dim = Pos.shape
    alpha = 0.0001 * NAtom**(-2. / 3.)

    # Default to using all available CPU cores if num_processes is not specified
    if num_processes is None:
        num_processes = os.cpu_count()

    # Divide the atom indices into chunks for parallel processing
    chunk_size = (NAtom + num_processes - 1) // num_processes
    chunks = [(i, min(i + chunk_size, NAtom), Pos, NAtom, Dim, alpha)
              for i in range(0, NAtom, chunk_size)]

    # Use multiprocessing Pool to compute forces and energy in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(compute_energy_force_chunk, chunks)

    # Combine the results from all processes
    PEnergy = sum(res[0] for res in results)  # Sum partial energies
    Forces = np.sum([res[1] for res in results], axis=0)  # Sum partial forces

    # Add single-body term contributions to energy and forces
    PEnergy += alpha * np.sum(np.sum(Pos**2, axis=-1))
    Forces -= 2.0 * alpha * Pos

    return PEnergy, Forces
