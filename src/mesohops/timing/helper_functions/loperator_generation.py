from scipy import sparse

def generate_holstein_1_particle_loperators(nsite):
    """
    Generates the L-operators for one particle Holstein calculations for a linear chain
    with nsite sites.

    Parameters
    ----------
    1. nsite: int
              Number of sites in the system.

    Returns
    -------
    1. list_loperators: list(sparse.coo_matrix)
                        List of L-operators in COO format.
    """
    list_loperators = [sparse.coo_matrix(([1], ([site_ind], [site_ind])),
                       shape=(nsite, nsite)) for site_ind in range(nsite)]
    return list_loperators


def generate_peierls_1_particle_loperators(nsite):
    """
    Generates the L-operators for one particle Peierls calculations for a linear chain
    with nsite sites.

    Parameters
    ----------
    1. nsite: int
              Number of sites in the system.

    Returns
    -------
    1. list_loperators: list(sparse.coo_matrix)
                        List of L-operators in COO format.
    """
    list_loperators = [sparse.coo_matrix(
        ([1, 1], ([site_ind+1, site_ind], [site_ind, site_ind+1])),
        shape=(nsite, nsite)) for site_ind in range(nsite-1)]
    return list_loperators


def check_state_occupies_site(site, state):
    """
    Checks if a given state occupies a specific site in a two-particle linear chain.

    Parameters
    ----------
    1. site: int
             The site to check.
    2. state: tuple(int)
              The state represented by a tuple of two integers, where each integer
              corresponds to the site occupied by a particle.

    Returns
    -------
    1. occupation: int
                   Returns 1 if the site is occupied by either particle, otherwise
                   returns 0.

    """
    return int(state[0] == site) + int(state[1] == site)


def generate_holstein_2_particle_loperators(nsite):
    """
    Generates the L-operators for two particle Holstein calculations for a linear chain
    with nsite sites.

    Parameters
    ----------
    1. nsite: int
              Number of sites in the system.

    Returns
    -------
    1. list_loperators: list(sparse.coo_matrix)
                        List of L-operators in COO format.
    """
    nstate = (nsite * (nsite + 1)) // 2
    list_loperators = []
    list_states = []

    # Generate all possible states for two particles in a linear chain
    for i in range(nsite):
        for j in range(i+1):
            list_states.append((i, j))

    # Generate L-operators for each site
    for site in range(nsite):
        list_l_data = []
        list_l_diag = []
        for state_ind in range(nstate):
            occupation = check_state_occupies_site(site, list_states[state_ind])
            if occupation:
                list_l_data.append(occupation)
                list_l_diag.append(state_ind)
        list_loperators.append(sparse.coo_matrix((list_l_data,
                                                  (list_l_diag, list_l_diag)),
                                                 shape=(nstate, nstate)))
    return list_loperators

def generate_spectroscopy_loperators(nsite):
    """
    Generates the L-operators for spectroscopy calculations (containing global ground
    state) for a linear chain with nsite sites.

    Parameters
    ----------
    1. nsite: int
              Number of sites in the system.

    Returns
    -------
    1. list_loperators: list(sparse.coo_matrix)
                        List of L-operators in COO format.
    """
    list_loperators = [sparse.coo_matrix(([1], ([site_ind + 1], [site_ind + 1])),
                       shape=(nsite + 1, nsite + 1)) for site_ind in range(nsite)]
    return list_loperators

