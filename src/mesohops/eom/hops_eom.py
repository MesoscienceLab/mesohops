import numpy as np

from mesohops.eom.eom_functions import (
    calc_LT_corr,
    calc_LT_corr_linear,
    calc_LT_corr_to_norm_corr,
    calc_delta_zmem,
    calc_norm_corr,
    compress_zmem,
    operator_expectation
)
from mesohops.eom.eom_hops_ksuper import calculate_ksuper, update_ksuper
from mesohops.util.dynamic_dict import Dict_wDefaults
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.physical_constants import hbar

__title__ = "Equations of Motion"
__author__ = "D. I. G. B. Raccah, B. Citty"
__version__ = "1.6"

EOM_DICT_DEFAULT = {
    "TIME_DEPENDENCE": False,
    "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR",
    "ADAPTIVE_H": False,
    "ADAPTIVE_S": False,
    "DELTA_A": 0,
    "DELTA_S": 0,
    "UPDATE_STEP": None,
    "F_DISCARD": 0
}

EOM_DICT_TYPES = {
    "TIME_DEPENDENCE": [type(False)],
    "EQUATION_OF_MOTION": [str],
    "ADAPTIVE": [type(False)],
    "ADAPTIVE_H": [type(False)],
    "ADAPTIVE_S": [type(False)],
    "DELTA_A": [type(1.0), type(1)],
    "DELTA_S": [type(1.0), type(1)],
    "UPDATE_STEP": [type(1.0), type(False), None, type(1)],
    "F_DISCARD": [type(0.0)]
}


class HopsEOM(Dict_wDefaults):
    """
    Defines the equation of motion for time-evolving the hops trajectory, primarily the
    derivative of the system state. Also contains the parameters that determine what
    kind of adaptive-hops strategies are used.
    """

    __slots__ = (
        # --- Parameter management ---
        'normalized',      # Normalization flag for wavefunction
        '_default_param',  # Default parameter dictionary
        '_param_types',    # Parameter type dictionary

        # --- Time derivative ---
        'dsystem_dt',         # System derivative (time evolution)
        'K2_k',               # K super-operator (self-interaction)
        'K2_kp1',             # K+1 super-operator (upward coupling)
        'Z2_kp1',             # Z+1 super-operator (noise coupling)
        'K2_km1',             # K-1 super-operator (downward coupling)
        'list_hier_mask_Zp1', # Hierarchy mask for Z+1 operator
        '_hier_timescale'     # The estimated fastest timescale of the hierarchy
    )

    def __init__(self, eom_params):
        """
        Inputs
        ------
        1. eom_params : dict
                        Dictionary of user-defined parameters.
            a. TIME_DEPENDENCE : bool
                                 True indicates a time-dependence of system Hamiltonian
                                 while False indicates otherwise (options: False).
            b. EQUATION_OF_MOTION : str
                                    Hops equation being solved (options: NORMALIZED
                                    NONLINEAR, LINEAR, NONLINEAR, NONLINEAR
                                    ABSORPTION).
            c. ADAPTIVE_H : bool  
                            True indicates that the hierarchy should be adaptively updated
                            while False indicates otherwise.             
            d. ADAPTIVE_S : bool           
                            True indicates that the state basis should be adaptively
                            updated while False indicates otherwise.
            e. DELTA_A : float
                         Threshold value for the adaptive hierarchy (options: >= 0).
            f. DELTA_S : float
                         Threshold value for the adaptive state basis (options: >= 0).
    
        Returns
        -------
        None
        """
        self._default_param = EOM_DICT_DEFAULT
        self._param_types = EOM_DICT_TYPES
        self.param = eom_params
        # Defines normalization condition
        # ------------------------------
        if self.param["EQUATION_OF_MOTION"] == "NORMALIZED NONLINEAR":
            self.normalized = True
        elif self.param["EQUATION_OF_MOTION"] == "NONLINEAR":
            self.normalized = False
        elif self.param["EQUATION_OF_MOTION"] == "LINEAR":
            self.normalized = False
        else:
            raise UnsupportedRequest(
                "EQUATION_OF_MOTION =" + self.param["EQUATION_OF_MOTION"],
                type(self).__name__,
            )

        # Checks adaptive definition
        # -------------------------
        if self.param["ADAPTIVE_H"] or self.param["ADAPTIVE_S"]:
            self.param["ADAPTIVE"] = True
        else:
            self.param["ADAPTIVE"] = False

    def _prepare_derivative(
        self,
        system,
        hierarchy,
        mode,
        zmem,
        permute_index=None,
        update=False,
        skip_ksuper=False,
    ):
        """
        Prepares a new derivative function that performs an update
        on previous super-operators.

        Parameters
        ----------
        1. system : instance(HopsSystem)

        2. hierarchy : instance(HopsHierarchy)

        3. mode : instance(HopsMode)
        
        4. zmem : instance(HopsZmem)

        5. permute_index : list(int)
                           List of rows and columns of non-zero entries that define a
                           permutation matrix.

        6. update : bool
                    True indicates an adaptive calculation while False indicates a
                    non-adaptive calculation.

        7. skip_ksuper : bool
                         If True, skip recalculating the Krylov super-operators.

        Returns
        -------
        1. dsystem_dt : function
                        Function that returns the derivative of phi and z_mem based
                        on whether the calculation is linear or nonlinear.
        """
        # Prepares super-operators
        # -----------------------
        if skip_ksuper:
            pass
        elif not update:
            self.K2_k, self.K2_kp1, self.Z2_kp1, self.K2_km1, self.list_hier_mask_Zp1 = calculate_ksuper(
                system,
                hierarchy,
                mode
            )
        else:
            self.K2_k, self.K2_kp1, self.Z2_kp1, self.K2_km1, self.list_hier_mask_Zp1 = update_ksuper(
                self.K2_k,
                self.K2_kp1,
                self.Z2_kp1,
                self.K2_km1,
                system,
                hierarchy,
                mode,
                permute_index,
            )

        # Combines sparse matrices
        # -----------------------
        K2_stable = self.K2_kp1 + self.K2_km1
        min_K2_k = np.min(self.K2_k)
        if min_K2_k == 0:
            self._hier_timescale = np.inf
        else:
            self._hier_timescale = hbar / np.abs(min_K2_k)
        list_L2 = mode.list_L2_coo
        # Map absolute L2 indices to relative indices in the current mode basis
        list_activel2idx_rel = [list(mode.list_l2idx_abs).index(absindex)
                                      for absindex in system.list_activel2idx_abs]
        if (self.param["EQUATION_OF_MOTION"] == "NORMALIZED NONLINEAR"
                or self.param["EQUATION_OF_MOTION"] == "NONLINEAR"):
            nmode = len(hierarchy.auxiliary_list[0])

            # Constructs tuple containing:
            # 1. aux_index for the phi_1
            # 2. associated L2 index
            # 3. associated absindex mode
            list_tuple_index_phi1_L2_mode = []
            aux0 = hierarchy.auxiliary_list[0]
            for absmode in aux0.dict_aux_p1.keys():
                index_aux = aux0.dict_aux_p1[absmode]._index
                relmode = list(mode.list_modeidx_abs).index(absmode)
                index_l2 = mode.list_index_L2_by_hmode[relmode]
                if index_l2 in list_activel2idx_rel:
                    actindex_l2 = list_activel2idx_rel.index(index_l2)
                    list_tuple_index_phi1_L2_mode.append([index_aux, actindex_l2, relmode])

            def dsystem_dt(
                Φ,
                z_mem1_tmp,
                z_rnd1_tmp,
                z_rnd2_tmp,
                K2_stable=K2_stable,
                Z2_kp1=self.Z2_kp1,
                list_hier_mask_Zp1 = self.list_hier_mask_Zp1,
                list_L2=list_L2,
                list_L2_masks = mode.list_L2_masks,
                list_index_L2_by_hmode=mode.list_index_L2_by_hmode,
                nsys=system.size,
                list_l2idx_abs=mode.list_l2idx_abs,
                list_modeidx_abs=mode.list_modeidx_abs,
                list_zmemmodeidx_abs=zmem.list_zmemmodeidx_abs,
                list_zmemactivemodeidx_rel=zmem.list_zmemactivemodeidx_rel,
                list_activel2idx_rel=list_activel2idx_rel,
                list_g=mode.list_g,
                list_w=mode.list_w,
                list_zmemg_abs =zmem.list_zmemg_abs,
                list_zmemw_abs = zmem.list_zmemw_abs,
                list_lt_corr_param=system.list_lt_corr_param,
                list_L2_csr = mode.list_L2_csr,
                list_l2_nz_csr = mode.list_l2_nz_csr,
                list_L2_sq_csr = mode.list_L2_sq_csr,
                list_tuple_index_phi1_L2_mode=list_tuple_index_phi1_L2_mode,
            ):

                """
                Calculates the time-evolution of the wave function. The logic here
                becomes slightly complicated because we need use both the relative and
                absolute indices at different points.

                z_hat1_tmp : relative
                z_rnd1_tmp : absolute
                z_rnd2_tmp: absolute
                z_mem1_tmp : absolute
                list_avg_L2 : relative
                Z2_k, Z2_kp1 : relative
                Φ_deriv : relative
                z_mem1_deriv : absolute

                The nonlinear evolution equation used to perform this calculation
                takes the following form:
                ~
                hbar dΨ̇_t^(k)/dt = (-iH-kw+(z~_t)L)ψ_t^(k) + κwLψ_t^(k-1) - (g/w)(L†-〈L†〉_t)ψ_t^(k+1)
                with z~ = z^* + ∫ds(a^*)(t-s)〈L†〉
                presented here for a single mode m: w = w_m, g = g_m, L = L_m, z_t = z_{m,t}.
                A super operator notation is implemented in this code.

                Parameters
                ----------
                1. Φ : np.array(complex)
                       Full hierarchy.

                2. z_mem1_tmp : np.array(complex)
                                Array of memory values for each mode.

                3. z_rnd1_tmp : np.array(complex)
                                Array of random noise corresponding to NOISE1 for the
                                set of time points required in the integration.

                4. z_rnd2_tmp : np.array(complex)
                                Array of random noise corresponding to NOISE2 for the
                                set of time points required in the integration.

                5. K2_stable : np.array(complex)
                               Component of the super operator that does not depend
                               on noise.

                6. Z2_kp1 : np.array(complex)
                            Component of the super operator that is multiplied by
                            noise z and maps the (K+1) hierarchy to the kth hierarchy.

                7. list_hier_mask_Zp1 : list(tuple(np.array, np.array, np.array))
                                        Precomputed masks and indices used to apply
                                        Z2_kp1 on the (k+1) hierarchy for each active
                                        L-operator.

                8. list_L2 : np.array(sparse matrix)
                             List of L operators.

                9. list_L2_masks : list(tuple(np.array, np.array))
                                   Precomputed masks used to apply each sparse
                                   L-operator to the flattened hierarchy.

                10. list_index_L2_by_hmode : list(int)
                                             List of length equal to the number of modes
                                             in the current hierarchy basis and each
                                             entry is an index for the relative list_L2.

                11. nsys : int
                            Current dimension (size) of the system basis.

                12. list_l2idx_abs : list(int)
                                      List of length equal to the number of L-operators
                                      in the current system basis where each element
                                      is the index for the absolute list_L2.

                13. list_modeidx_abs : list(int)
                                        List of length equal to the number of modes in
                                        the current mode basis that corresponds to
                                        the absolute index of the modes.

                14. list_zmemmodeidx_abs : list(int)
                                            List of length equal to the number of modes in
                                            z_mem that corresponds to the absolute index of
                                            the modes.

                15. list_zmemactivemodeidx_rel : list(int)
                                                 List of length equal to the number of modes in the
                                                 current mode basis corresponding to relative index
                                                 in z_mem.

                16. list_activel2idx_rel : list(int)
                                            List of relative indices of L-operators that have any
                                            non-zero values.

                17. list_g : list(complex)
                             List of pre exponential factors for bath correlation
                             functions.

                18. list_w : list(complex)
                             List of exponents for bath correlation functions (w =
                             γ+iΩ).

                19. list_zmemg_abs : list(complex)
                                      List of pre exponential factors for bath
                                      correlation functions used in z_mem.
                                      Indexed over list_zmemmodeidx_abs.

                20. list_zmemw_abs : list(complex)
                                      List of exponents for bath correlation
                                      functions used in z_mem.
                                      Indexed over list_zmemmodeidx_abs.

                21. list_lt_corr_param : list(complex)
                                         List of low-temperature correction factors.

                22. list_L2_csr : np.array(sparse matrix)
                                  L-operators in csr format in the current basis.

                23. list_l2_nz_csr : np.array(sparse matrix)
                                      L-operators in csr format, truncated to only nonzero
                                      entries.

                24. list_L2_sq_csr : np.array(sparse matrix)
                                      Squared L-operators in csr format in the current
                                      basis.

                25. list_tuple_index_phi1_L2_mode : list(tuple)
                                                    List of tuples with each tuple
                                                    containing the index of the first
                                                    auxiliary mode (phi1) in the
                                                    hierarchy and the index of the
                                                    corresponding L operator.

                Returns
                -------
                1. Φ_deriv : np.array(complex)
                             Derivative of phi with respect to time.

                2. z_mem1_deriv : np.array(complex)
                                  Derivative of z_mem with respect to time.
                """
                
                # Construct noise terms
                # ---------------------
                z_hat1_tmp = (np.conj(z_rnd1_tmp) + compress_zmem(
                    z_mem1_tmp, list_index_L2_by_hmode, list_zmemactivemodeidx_rel
                ))[list_activel2idx_rel]
                z_tmp2 = z_rnd2_tmp[list_activel2idx_rel]

                # Construct other fluctuating terms
                # ---------------------------------
                list_avg_L2 = [operator_expectation(list_L2[index], Φ[:nsys])
                               for index in list_activel2idx_rel] # <L>

                norm_corr = 0
                if self.normalized:
                    norm_corr = calc_norm_corr(
                        Φ,
                        z_hat1_tmp,
                        list_avg_L2,
                        list_L2[list_activel2idx_rel],
                        nsys,
                        list_tuple_index_phi1_L2_mode,
                        list_g,
                        list_w,
                    )
                    
                # Check for a low-temperature correction stemming from flux from
                # Markovian auxiliaries
                C2_gamma_LT_corr_to_norm_corr = 0
                if any(np.array(list_lt_corr_param)):
                    # Find <L^2>
                    list_avg_L2_sq = [operator_expectation(list_L2_sq_csr[index],
                                                           Φ[:nsys])
                                      for index in list_activel2idx_rel]  # <L^2>
                    
                    # Gets LT correction to the physical wavefunction stemming from
                    # the terminator approximation to the Markovian auxiliaries and
                    # to the full hierarchy stemming from delta-function
                    # approximation of noise memory drift
                    C2_LT_corr_physical, C2_LT_corr_hier = calc_LT_corr(
                        np.array(list_lt_corr_param),
                        list_L2_csr[list_activel2idx_rel],
                        list_avg_L2,
                        list_L2_sq_csr[list_activel2idx_rel]
                    )

                    if self.normalized:
                        # Get LT correction to the normalization correction factor
                        C2_gamma_LT_corr_to_norm_corr = calc_LT_corr_to_norm_corr(
                            np.array(list_lt_corr_param),
                            list_avg_L2,
                            list_avg_L2_sq
                        )

                
                # Calculates dphi/dt
                # -----------------
                
                Φ_view_F = np.asarray(Φ).reshape([system.size,hierarchy.size],order="F")
                Φ_view_C = np.asarray(Φ).reshape([hierarchy.size,system.size],order="C")
                
                Φ_deriv = K2_stable @ Φ
                Φ_deriv += (self.K2_k @ Φ_view_C).reshape([hierarchy.size * system.size],order="C")
                Φ_deriv += ((-1j * system.hamiltonian) @ Φ_view_F).reshape([system.size * hierarchy.size],order="F")

                Φ_deriv_view_F = np.asarray(Φ_deriv).reshape([system.size,hierarchy.size],order="F")
                Φ_deriv_view_C = np.asarray(Φ_deriv).reshape([hierarchy.size,system.size],order="C")
                
                # Implement the low-temperature correction
                if any(np.array(list_lt_corr_param)):
                    Φ_deriv += (C2_LT_corr_hier @ Φ_view_F).reshape([system.size * hierarchy.size],order="F")
                    Φ_deriv[:system.size] += C2_LT_corr_physical @ np.asarray(
                        Φ[:system.size])
                    norm_corr += C2_gamma_LT_corr_to_norm_corr
                
                if self.normalized:
                    Φ_deriv -= norm_corr * Φ
                
                
                
                for j in range(len(list_avg_L2)):
                    rel_index = list_activel2idx_rel[j]
                    # ASSUMING: L = L^*
                    
                    Φ_view_red = Φ_view_F[list_L2_masks[rel_index][1],:]
                    Φ_deriv_view_F[list_L2_masks[rel_index][0],:] += (
                            (z_hat1_tmp[j] - 1.0j * z_tmp2[j]) *
                            (list_l2_nz_csr[rel_index] @ Φ_view_red)
                    )
                    
                    Φ_view_red = Φ_view_C[list_hier_mask_Zp1[rel_index][1],:]
                    Z2_kp1_red = Z2_kp1[rel_index][list_hier_mask_Zp1[rel_index][2]]
                    
                    Φ_deriv_view_C[list_hier_mask_Zp1[rel_index][0],:] += np.conj(list_avg_L2[j]) * (Z2_kp1_red @ Φ_view_red)


                    
                # Calculates dz/dt
                # ---------------
                z_mem1_deriv = calc_delta_zmem(
                    z_mem1_tmp,
                    list_avg_L2,
                    list_zmemg_abs,
                    list_zmemw_abs,
                    list_index_L2_by_hmode,
                    list_modeidx_abs,
                    list_zmemmodeidx_abs,
                    list_l2idx_abs,
                    system.list_activel2idx_abs
                )

                return Φ_deriv, z_mem1_deriv

        elif self.param["EQUATION_OF_MOTION"] == "LINEAR":

            def dsystem_dt(
                Φ,
                z_mem1_tmp,
                z_rnd1_tmp,
                z_rnd2_tmp,
                list_L2_csr = mode.list_L2_csr,
                list_L2_sq_csr = mode.list_L2_sq_csr,
                K2_stable=K2_stable,
                Z2_kp1=self.Z2_kp1,
                list_lt_param = system.list_lt_corr_param
            ):
                """
                NOTE: Unlike in the non-linear case, this equation of motion does
                      NOT support an adaptive solution at the moment. Implementing
                      adaptive integration would require updating the adaptive
                      routine to handle non-normalized wave-functions.

                      A USER CALLING LINEAR AND ADAPTIVE SHOULD GET AN INFORMATIVE
                      ERROR MESSAGE, BUT THAT SHOULD BE HANDLED IN HopsTrajectory
                      CLASS.

                The linear evolution equation used to perform this calculation
                takes the following form:

                hbar*dΨ_̇t^(k)/dt=(-iH-kw+((z^*)_t)L)ψ_t^(k) + κw Lψ_t^(k-1) - (g/w)(L^†)ψ_t^(k+1)
                presented here for a single mode m: w = w_m, g = g_m, L = L_m, z_t = z_{m,t}.
                A super operator notation is implemented in this code.

                Parameters
                ----------
                1. Φ : np.array(complex)
                       Current full hierarchy.

                2. z_mem1_tmp : np.array(complex)
                                Array of memory values for each mode.

                3. z_rnd1_tmp : np.array(complex)
                                Array of random noise corresponding to NOISE1 for the
                                set of time points required in the integration.

                4. z_rnd2_tmp : np.array(complex)
                                Array of random noise corresponding to NOISE2 for the
                                set of time points required in the integration.

                5. list_L2_csr : np.array(sparse matrix)
                                 L-operators in csr format in the full basis.

                6. list_L2_sq_csr : np.array(sparse matrix)
                                    Squared L-operators in csr format in the full basis.

                7. K2_stable : np.array(complex)
                               Component of the super operator that does not depend
                               on noise.

                8. Z2_kp1 : np.array(complex)
                            Component of the super operator that is multiplied by
                            noise z and maps the (K+1)th hierarchy to the Kth hierarchy.

                9. list_lt_corr_param : list(complex)
                                        List of low-temperature correction factors.

                Returns
                -------
                1. Φ_deriv : np.array(complex)
                             Derivative of phi with respect to time.

                2. z_mem1_deriv : np.array(complex)
                                  Derivative of z_mem with respect to time.
                """

                Φ_view_F = np.asarray(Φ).reshape([system.size,hierarchy.size],order="F")
                Φ_view_C = np.asarray(Φ).reshape([hierarchy.size,system.size],order="C")

                # Prepares noise
                # -------------
                z_hat1_tmp = np.conj(z_rnd1_tmp)

                # Calculates dphi/dt
                # -----------------
                Φ_deriv = K2_stable @ Φ
                Φ_deriv += (self.K2_k @ Φ_view_C).reshape([hierarchy.size * system.size],order="C")
                Φ_deriv += ((-1j * system.hamiltonian) @ Φ_view_F).reshape([system.size * hierarchy.size],order="F")
                if any(list_lt_param):
                    Φ_deriv[:system.size] += calc_LT_corr_linear(list_lt_param,
                                                                 list_L2_sq_csr) @ \
                                             np.asarray(Φ[:system.size])

                Φ_deriv_view = np.asarray(Φ_deriv).reshape([system.size,hierarchy.size],order="F")
                Φ_view = np.asarray(Φ).reshape([system.size,hierarchy.size],order="F")
                for j in range(len(list_L2_csr)):
                    Φ_deriv_view += (z_hat1_tmp[j] - 1.0j * z_rnd2_tmp[j]) * (
                                list_L2_csr[j] @ Φ_view)

                # Calculates dz/dt
                # ---------------
                z_mem1_deriv = 0 * z_mem1_tmp

                return Φ_deriv, z_mem1_deriv

        else:
            raise UnsupportedRequest(
                "EQUATION_OF_MOTION =" + self.param["EQUATION_OF_MOTION"],
                type(self).__name__,
            )

        self.dsystem_dt = dsystem_dt

        return dsystem_dt

    @property
    def hier_timescale(self) -> float:
        return self._hier_timescale
