from scipy import sparse

def generate_1_particle_hamiltonian(nsite, V):
    """
    Generates the Hamiltonian for one particle Holstein calculations for a linear chain
    with nsite sites and inter-site coupling V.

    Parameters
    ----------
    1. nsite: int
              Number of sites in the system.
    2. V: float
          Inter-site coupling strength [units: cm^-1].

    Returns
    -------
    1. H2_sys_hamiltonian: sparse.coo_matrix
                           The Hamiltonian in COO format.
    """
    list_row = list(range(1, nsite))
    list_row += list(range(0, nsite-1))
    list_col = list(range(0, nsite-1))
    list_col += list(range(1, nsite))
    list_val = [V]*2*(nsite-1)

    # Create the Hamiltonian matrix in COO format
    H2_sys_hamiltonian = sparse.coo_matrix((list_val, (list_row, list_col)),
                                        shape=(nsite, nsite))
    return H2_sys_hamiltonian

def generate_2_particle_hamiltonian(nsite, V):
    """
    Generates the Hamiltonian for two particle Holstein calculations for a linear chain
    with nsite sites and inter-site coupling V.

    Parameters
    ----------
    1. nsite: int
              Number of sites in the system.
    2. V: float
          Inter-site coupling strength [units: cm^-1].

    Returns
    -------
    1. H2_sys_hamiltonian: sparse.coo_matrix
                           The Hamiltonian in COO format.
    """
    list_row = []
    list_col = []
    list_states = []
    state_ind = 0

    for i in range(nsite):
        for j in range(i+1):
            list_states.append((i, j))
            if i > 0 and i != j:
                list_row.append(state_ind - i)
                list_col.append(state_ind)
                list_row.append(state_ind)
                list_col.append(state_ind - i)
            if j > 0:
                list_row.append(state_ind - 1)
                list_col.append(state_ind)
                list_row.append(state_ind)
                list_col.append(state_ind - 1)
            state_ind += 1

    list_data = [V]*len(list_row)
    nstate = len(list_states)

    # Create the Hamiltonian matrix in COO format
    H2_sys_hamiltonian = sparse.coo_matrix((list_data, (list_col, list_row)),
                                        shape=(nstate, nstate))
    return H2_sys_hamiltonian

def generate_spectroscopy_hamiltonian(nsite, V):
    """
    Generates the Hamiltonian for spectroscopy calculations (containing global ground
    state) for a linear chain with nsite sites and inter-site coupling V.

    Parameters
    ----------
    1. nsite: int
              Number of sites in the system.
    2. V: float
          Inter-site coupling strength [units: cm^-1].

    Returns
    -------
    1. H2_sys_hamiltonian: sparse.coo_matrix
                           The Hamiltonian in COO format.
    """
    list_row = list(range(2, nsite+1))
    list_row += list(range(1, nsite))
    list_col = list(range(1, nsite))
    list_col += list(range(2, nsite+1))
    list_val = [V]*2*(nsite-1)

    # Create the Hamiltonian matrix in COO format
    H2_sys_hamiltonian = sparse.coo_matrix((list_val, (list_row, list_col)),
                                        shape=(nsite + 1, nsite + 1))
    return H2_sys_hamiltonian

