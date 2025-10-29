from __future__ import annotations

from collections import Counter
from typing import Any, Dict

import numpy as np
from scipy import sparse
from mesohops.util.helper_functions import get_states_from_L2, array_to_tuple

def initialize_system_dict(system_param: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extends the user input to the complete set of parameters defined above.

    Parameters
    ----------
    1. system_param : dict
                      Dictionary with the system and system-bath coupling
                      parameters defined.

    Returns
    -------
    1. param_dict : dict
                    Dictionary containing the user input and the derived parameters.
    """
    param_dict = system_param
    if(sparse.issparse(system_param["HAMILTONIAN"])):
        param_dict["NSTATES"] = sparse.coo_matrix.get_shape(system_param["HAMILTONIAN"])[0]
    else:
        param_dict["NSTATES"] = len(system_param["HAMILTONIAN"][0])
    param_dict["N_HMODES"] = len(system_param["GW_SYSBATH"])
    param_dict["G"] = np.array([g for (g, w) in system_param["GW_SYSBATH"]])
    param_dict["W"] = np.array([w for (g, w) in system_param["GW_SYSBATH"]])
    param_dict["LIST_STATE_INDICES_BY_HMODE"] = [
        get_states_from_L2(L2) for L2 in param_dict["L_HIER"]
    ]
    param_dict["LIST_HMODE_INDICES_BY_STATE"] = [[] for i in range(param_dict["NSTATES"])]
    for (hmode ,state_indices) in enumerate(param_dict["LIST_STATE_INDICES_BY_HMODE"]):
        for state in state_indices:
            param_dict["LIST_HMODE_INDICES_BY_STATE"][state].append(hmode)


    param_dict["SPARSE_HAMILTONIAN"] = sparse.csc_array(param_dict["HAMILTONIAN"])
    param_dict["SPARSE_HAMILTONIAN"].eliminate_zeros()

    sparse_ham = sparse.coo_matrix(system_param["HAMILTONIAN"])
    param_dict["COUPLED_STATES"] = [[] for state in range(param_dict["NSTATES"])]
    for i in range(len(sparse_ham.row)):
        param_dict["COUPLED_STATES"][sparse_ham.row[i]].append(sparse_ham.col[i])
    for i in range(param_dict["NSTATES"]):
        param_dict["COUPLED_STATES"][i] = sorted(param_dict["COUPLED_STATES"][i])

    # Checks for low-temperature correction terms - if there are none, initialize
    # empty lists as placeholders:
    if not "L_LT_CORR" in param_dict.keys():
        param_dict["L_LT_CORR"] = []
        param_dict["PARAM_LT_CORR"] = []

    # Define the Hierarchy Operator Values
    # ------------------------------------
    # Since arrays and lists are not hashable, we will turn our operators
    # into tuples in order to conveniently define a number of indexing
    # parameters.

    # Creates list of unique l2 tuples in order they appear in "L_HIER"
    l2_as_tuples = [array_to_tuple(L2) for L2 in param_dict["L_HIER"]]
    list_unique_l2_as_tuples = list(Counter(l2_as_tuples))
    param_dict["N_L2"] = len(set(list_unique_l2_as_tuples))
    # Generates a list of destination state indices linked to each state index by
    # one of the L-operators.
    param_dict["LIST_DESTINATION_STATES_BY_STATE_INDEX"] = [[] for s in range(
        param_dict["NSTATES"])]
    for l in list_unique_l2_as_tuples:
        for j in range(len(l[0])):
            param_dict["LIST_DESTINATION_STATES_BY_STATE_INDEX"][l[0][j]].append(
                l[1][j])
    param_dict["LIST_DESTINATION_STATES_BY_STATE_INDEX"] = [list(set(
        list_dest_state)) for list_dest_state in
        param_dict["LIST_DESTINATION_STATES_BY_STATE_INDEX"]]

    # Creates L2 indexing parameters
    param_dict["LIST_INDEX_L2_BY_HMODE"] = [
        None for i in range(param_dict["N_HMODES"])
    ]
    param_dict["LIST_L2_COO"] = [0 ] *param_dict["N_L2"]
    param_dict["LIST_LT_PARAM"] = [0 ] *param_dict["N_L2"]
    param_dict["LIST_HMODE_BY_INDEX_L2"] = [[] ] *param_dict["N_L2"]

    param_dict["LIST_STATE_INDICES_BY_INDEX_L2"] = []
    dict_unique_l2 = {}
    unique_l2_insertion = 0
    for (i ,l_2) in enumerate(l2_as_tuples):
        try:
            index_l2_unique = dict_unique_l2[l_2]
            param_dict["LIST_INDEX_L2_BY_HMODE"][i] = index_l2_unique
            param_dict["LIST_HMODE_BY_INDEX_L2"][index_l2_unique].append(i)

        except:
            dict_unique_l2[l_2] = unique_l2_insertion
            param_dict["LIST_INDEX_L2_BY_HMODE"][i] = unique_l2_insertion
            param_dict["LIST_HMODE_BY_INDEX_L2"][unique_l2_insertion].append(i)
            tmp = sparse.coo_matrix(param_dict["L_HIER"][i])
            tmp.eliminate_zeros()
            param_dict["LIST_L2_COO"][unique_l2_insertion] = tmp
            param_dict["LIST_STATE_INDICES_BY_INDEX_L2"].append(
                param_dict["LIST_STATE_INDICES_BY_HMODE"][i]
            )
            unique_l2_insertion += 1

    param_dict["list_L2_off_diag"] = np.array([not np.allclose(L2.col, L2.row)
                                               for L2 in param_dict["LIST_L2_COO"]])

    param_dict["LIST_INDEX_L2_BY_STATE_INDICES"] = [[] for i in range(param_dict["NSTATES"])]
    for (index_L2 ,state_indices) in enumerate(param_dict["LIST_STATE_INDICES_BY_INDEX_L2"]):
        for state in state_indices:
            param_dict["LIST_INDEX_L2_BY_STATE_INDICES"][state].append(index_L2)

    l2_LT_CORR_as_tuples = [array_to_tuple(l) for l in
                            param_dict["L_LT_CORR"]]
    dict_index_unique_l2_as_tuples = {}
    for (i ,l) in enumerate(list_unique_l2_as_tuples):
        dict_index_unique_l2_as_tuples[l] = i

        # Build a list of low-temperature coefficients guaranteed to be in the same
    # order as the associated unique sparse L2 operators.
    for (i ,l) in enumerate(l2_LT_CORR_as_tuples):
        try:
            mister_mc_index = dict_index_unique_l2_as_tuples[l]
            param_dict["LIST_LT_PARAM"][mister_mc_index] += param_dict["PARAM_LT_CORR"][i]
        except:
            print("WARNING: the list of low-temperature correction "
                  "L-operators contains an L-operator not associated with any "
                  "existing thermal environment. This low-temperature "
                  "correction factor will be discarded!")

    # Define the Noise1 Operator Values
    # ---------------------------------
    param_dict["LIST_NMODE1_BY_INDEX_L2"] = [[] for i in range(len(l2_as_tuples))]
    param_dict["LIST_INDEX_L2_BY_NMODE1"] = [
        None for i in range(len(param_dict["PARAM_NOISE1"]))
    ]
    l2_as_tuples = [array_to_tuple(l) for l in param_dict["L_NOISE1"]]
    for (i, l_2) in enumerate(l2_as_tuples):
        try:
            index_l2_unique = dict_unique_l2[l_2]
            param_dict["LIST_INDEX_L2_BY_NMODE1"][i] = index_l2_unique
            param_dict["LIST_NMODE1_BY_INDEX_L2"][index_l2_unique].append(i)
        except:
            print("WARNING: the list of noise 1 L-operators contains an L-operator "
                  "not associated with any existing thermal environment. The noise "
                  "associated with this L-operator will be discarded!")

    # Define the Noise2 Operator Values
    # ---------------------------------
    if "L_NOISE2" in param_dict.keys():
        param_dict["LIST_NMODE2_BY_INDEX_L2"] = [[] for i in range(len(l2_as_tuples))]
        param_dict["LIST_INDEX_L2_BY_NMODE2"] = [
            None for i in range(len(param_dict["PARAM_NOISE2"]))
        ]
        l2_as_tuples = [array_to_tuple(l) for l in param_dict["L_NOISE2"]]

        for (i, l_2) in enumerate(l2_as_tuples):

            try:
                index_l2_unique = dict_unique_l2[l_2]
                param_dict["LIST_INDEX_L2_BY_NMODE2"][i] = index_l2_unique
                param_dict["LIST_NMODE2_BY_INDEX_L2"][index_l2_unique].append(i)
            except:
                print("WARNING: the list of noise 2 L-operators contains an L-operator "
                      "not associated with any existing thermal environment. The noise "
                      "associated with this L-operator will be discarded!")

    return param_dict