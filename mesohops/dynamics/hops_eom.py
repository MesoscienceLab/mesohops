import numpy as np
from scipy import sparse
from mesohops.util.dynamic_dict import Dict_wDefaults
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.dynamics.eom_hops_ksuper import calculate_ksuper, update_ksuper
from mesohops.dynamics.eom_functions import (
    calc_norm_corr,
    calc_delta_zmem,
    operator_expectation,
    compress_zmem,
)

__title__ = "Equations of Motion"
__author__ = "D. I. G. Bennett"
__version__ = "1.0"

EOM_DICT_DEFAULT = {
    "TIME_DEPENDENCE": False,
    "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR",
    "ADAPTIVE_H": False,
    "ADAPTIVE_S": False,
    "DELTA_H": 0,
    "DELTA_S": 0,
}

EOM_DICT_TYPES = {
    "TIME_DEPENDENCE": [type(False)],
    "EQUATION_OF_MOTION": [type(str())],
    "ADAPTIVE": [type(False)],
    "DELTA_H": [type(1.0)],
    "DELTA_S": [type(1.0)],
}


class HopsEOM(Dict_wDefaults):
    """
    HopsEOM is the class that defines the equation of motion for time evolving
    the hops trajectory. Its primary responsibility is to define the derivative
    of the system state. It also contains the parameters that determine what
    kind of adaptive-hops strategies are used.
    """

    def __init__(self, eom_params):
        """
        A self initializing function that defines the normalization condition and
        checks the adaptive definition.

        PARAMETERS
        ----------
        1. eom_params: a dictionary of user-defined parameters
            a. TIME_DEPENDENCE : boolean          [Allowed: False]
                                 defining time-dependence of system Hamiltonian
            b. EQUATION_OF_MOTION : str           [Allowed: NORMALIZED NONLINEAR, LINEAR]
                                    The hops equation that is being solved
            c. ADAPTIVE_H : boolean               [Allowed: True, False]
                            Boolean that defines if the hierarchy should be
                           adaptively updated
            d. ADAPTIVE_S : boolean               [Allowed: True, False]
                            Boolean that defines if the system should be
                            adaptively updated.
            e. DELTA_H : float                    [Allowed: >0]
                         The threshold value for the adaptive hierarchy
            f. DELTA_S : float                    [Allowed: >0]
                         The threshold value for the adaptive system

        RETURNS
        -------
        None
        """
        self._default_param = EOM_DICT_DEFAULT
        self._param_types = EOM_DICT_TYPES
        self.param = eom_params

        # Define Normalization condition
        # ------------------------------
        if self.param["EQUATION_OF_MOTION"] == "NORMALIZED NONLINEAR":
            self.normalized = True
        elif self.param["EQUATION_OF_MOTION"] == "LINEAR":
            self.normalized = False
        else:
            raise UnsupportedRequest(
                "EQUATION_OF_MOTION =" + self.param["EQUATION_OF_MOTION"],
                type(self).__name__,
            )

        # Check Adaptive Definition
        # -------------------------
        if self.param["ADAPTIVE_H"] or self.param["ADAPTIVE_S"]:
            self.param["ADAPTIVE"] = True
        else:
            self.param["ADAPTIVE"] = False

    def _prepare_derivative(
        self,
        system,
        hierarchy,
        list_stable_aux=None,
        list_add_aux=None,
        list_stable_state=None,
        list_old_absindex_L2=None,
        n_old=None,
        permute_index=None,
        update=False,
    ):
        """
        This function will prepare a new derivative function that performs an update
        on previous super-operators.

        PARAMETERS
        ----------
        1. system : HopsSystems object
                    an instance of HopsSystem
        2. hierarchy : HopsHierarchy object
                       an instance of HopsHierarchy
        3. list_stable_aux : list
                             list of current auxiliaries that were also present in
                             the previous basis
        4. list_add_aux : list
                          list of additional auxiliaries needed for the current basis
        5. list_stable_state : list
                               list of current states that were also present in the
                               previous basis
        6. list_old_absindex_L2 : list
                                  list of the absolute indices of L operators in
                                  the last basis
        7. n_old : int
                   multiplication of the previous hierarchy and system dimensions
        8. permute_index : list
                           list of rows and columns of non zero entries that define
                           a permutation matrix
        9. update : boolean
                    True = updating adaptive calculation,
                    False = non-adaptive calculation

        RETURNS
        -------
        1. dsystem_dt : function
                        a function that returns the derivative of phi and z_mem based
                        on if the calculation is linear or nonlinear
        """

        # Prepare Super-Operators
        # -----------------------
        if not update:
            self.K2_k, self.Z2_k, self.K2_kp1, self.Z2_kp1, self.K2_km1 = calculate_ksuper(
                system, hierarchy
            )
        else:
            self.K2_k, self.Z2_k, self.K2_kp1, self.Z2_kp1, self.K2_km1 = update_ksuper(
                self.K2_k,
                self.Z2_k,
                self.K2_kp1,
                self.Z2_kp1,
                self.K2_km1,
                list_stable_aux,
                list_add_aux,
                list_stable_state,
                list_old_absindex_L2,
                system,
                hierarchy,
                n_old,
                permute_index,
            )

        # Combine Sparse Matrices
        # -----------------------
        K2_stable = self.K2_k + self.K2_kp1 + self.K2_km1
        list_L2 = system.list_L2_coo  # list_L2
        if self.param["EQUATION_OF_MOTION"] == "NORMALIZED NONLINEAR":
            nmode = len(hierarchy.auxiliary_list[0])
            list_tuple_index_phi1_L2_mode = [
                (
                    hierarchy._aux_index(
                        hierarchy._const_aux_edge(
                            system.list_absindex_mode[i], 1, nmode
                        )
                    ),
                    index_L2,
                    i,
                )
                for (i, index_L2) in enumerate(system.list_index_L2_by_hmode)
                if (
                    hierarchy._const_aux_edge(system.list_absindex_mode[i], 1, nmode)
                    in hierarchy.auxiliary_list
                )
            ]

            def dsystem_dt(
                Φ,
                z_mem1_tmp,
                z_rnd1_tmp,
                z_rnd2_tmp,
                K2_stable=K2_stable,
                Z2_k=self.Z2_k,
                Z2_kp1=self.Z2_kp1,
                list_L2=list_L2,
                list_index_L2_by_hmode=system.list_index_L2_by_hmode,
                list_mode_absindex_L2=system.param["LIST_INDEX_L2_BY_HMODE"],
                nsys=system.size,
                list_absindex_L2=system.list_absindex_L2,
                list_absindex_mode=system.list_absindex_mode,
                list_g=system.param["G"],
                list_w=system.param["W"],
                list_tuple_index_phi1_L2_mode=list_tuple_index_phi1_L2_mode,
            ):

                """
                This is the core function for calculating the time-evolution of the
                wave function. The logic here becomes slightly complicated because
                we need use both the relative and absolute indices at different points.

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
                Ψ̇_t^(k)=(-iH-kw+(z~_t)L)ψ_t^(k) + κα(0)Lψ_t^(k-1) - (L†-〈L†〉_t)ψ_t^(k+1)
                with z~ = z^* + ∫ds(a^*)(t-s)〈L†〉
                A super operator notation is implemented in this code.

                PARAMETERS
                ----------
                1. Φ : np.array
                       current full hierarchy
                2. z_mem1_tmp : np.array
                                array of memory values for each mode
                3. z_rnd1_tmp : np.array
                                array of random noise corresponding to NOISE1 for the
                                set of time points required in the integration
                4. z_rnd2_tmp : np.array
                                array of random noise corresponding to NOISE2 for the
                                set of time points required in the integration
                5. K2_stable : np.array
                               the component of the super operator that does not depend
                               on noise
                6. Z2_k : np.array
                          the component of the super operator that is multiplied by
                          noise z and maps the Kth hierarchy to the Kth hierarchy
                7. Z2_kp1 : np.array
                            the component of the super operator that is multiplied by
                            noise z and maps the (K+1) hierarchy to the kth hierarchy
                8. list_L2 : list
                             list of L operators
                9. list_index_L2_by_hmode : list
                                             list of length equal to the number of modes
                                             in the current hierarchy basis and each
                                             entry is an index for the relative list_L2.
                10. list_mode_absindex_L2 : list
                                            list of length equal to the number of
                                            'modes' in the current hierarchy basis and
                                            each entry is an index for the absolute
                                            list_L2.
                11. nsys : int
                          t he current dimension (size) of the system basis
                12. list_absindex_L2 : list
                                       list of length equal to the number of L-operators
                                       in the current system basis where each element
                                       is the index for the absolute list_L2
                13. list_absindex_mode : list
                                         list of length equal to the number of modes in
                                         the current system basis that corresponds to
                                         the absolute index of the modes
                14. list_g : list
                             list of pre exponential factors for bath correlation
                             functions
                15. list_w : list
                             list of exponents for bath correlation functions (w = γ+iΩ)
                16. list_tuple_index_phi1_index_L2 : list
                                                     list of tuples with each tuple
                                                     containing the index of the first
                                                     auxiliary mode (phi1) in the
                                                     hierarchy and the index of the
                                                     corresponding L operator

                RETURNS
                -------
                1. Φ_deriv : np.array
                             the derivative of phi with respect to time
                2. z_mem1_deriv : np.array
                                  the derivative of z_mem with respect to time
                """

                # Construct Noise Terms
                # ---------------------
                z_hat1_tmp = np.conj(z_rnd1_tmp[list_absindex_L2]) + compress_zmem(
                    z_mem1_tmp, list_index_L2_by_hmode, list_absindex_mode
                )
                z_tmp2 = z_rnd2_tmp[list_absindex_L2]

                # Construct other fluctuating terms
                # ---------------------------------
                list_avg_L2 = [
                    operator_expectation(L, Φ[:nsys]) for L in list_L2
                ]  # <L>
                norm_corr = calc_norm_corr(
                    Φ,
                    z_hat1_tmp,
                    list_avg_L2,
                    list_L2,
                    nsys,
                    list_tuple_index_phi1_L2_mode,
                    np.array(list_g)[np.array(list_absindex_mode)],
                    np.array(list_w)[np.array(list_absindex_mode)],
                )

                # calculate dphi/dt
                # -----------------
                Φ_deriv = K2_stable @ Φ
                Φ_deriv -= norm_corr * Φ
                for j in range(len(list_avg_L2)):
                    # ASSUMING: L = L^*
                    Φ_deriv += (z_hat1_tmp[j] + 2*np.real(z_tmp2[j])) * (Z2_k[j] @ Φ)
                    Φ_deriv += np.conj(list_avg_L2[j]) * (Z2_kp1[j] @ Φ)

                # calculate dz/dt
                # ---------------
                z_mem1_deriv = calc_delta_zmem(
                    z_mem1_tmp,
                    list_avg_L2,
                    list_g,
                    list_w,
                    list_mode_absindex_L2,
                    list_absindex_mode,
                )

                return Φ_deriv, z_mem1_deriv

        elif self.param["EQUATION_OF_MOTION"] == "LINEAR":

            def dsystem_dt(
                Φ,
                z_mem1_tmp,
                z_rnd1_tmp,
                z_rnd2_tmp,
                K2_stable=K2_stable,
                Z2_k=self.Z2_k,
                Z2_kp1=self.Z2_kp1,
            ):
                """
                NOTE: unlike in the non-linear case, this equation of motion does
                      NOT support an adaptive solution at the moment. Implementing
                      adaptive integration would require updating the adaptive
                      routine to handle non-normalized wave-functions.

                      A USER CALLING LINEAR AND ADAPTIVE SHOULD GET AN INFORMATIVE
                      ERROR MESSAGE, BUT THAT SHOULD BE HANDLED IN HopsTrajectory
                      CLASS.

                The linear evolution equation used to perform this calculation
                takes the following form:

                Ψ_̇t^(k)=(-iH-kw+((z^*)_t)L)ψ_t^(k) + κα(0)Lψ_t^(k-1) - (L^†)ψ_t^(k+1)
                A super operator notation is implemented in this code.

                PARAMETERS
                ----------
                1. Φ : np.array
                       current full hierarchy
                2. z_mem1_tmp : np.array
                                array of memory values for each mode
                3. z_rnd1_tmp : np.array
                                array of random noise corresponding to NOISE1 for the
                                set of time points required in the integration
                4. z_rnd2_tmp : np.array
                                array of random noise corresponding to NOISE2 for the
                                set of time points required in the integration
                5. K2_stable : np.array
                               the component of the super operator that does not depend
                               on noise
                6. Z2_k : np.array
                          the component of the super operator that is multiplied by
                          noise z and maps the Kth hierarchy to the Kth hierarchy
                7. Z2_kp1 : np.array
                            the component of the super operator that is multiplied by
                            noise z and maps the (K+1) hierarchy to the kth hierarchy

                RETURNS
                -------
                1. Φ_deriv : np.array
                             the derivative of phi with respect to time
                2. z_mem1_deriv : np.array
                                  the derivative of z_men with respect to time (=0)
                """

                # prepare noise
                # -------------
                z_hat1_tmp = np.conj(z_rnd1_tmp)

                # calculate dphi/dt
                # -----------------
                Φ_deriv = K2_stable @ Φ
                for j in range(len(Z2_k)):
                    Φ_deriv += (z_hat1_tmp[j] + 2*np.real(z_rnd2_tmp[j])) * (Z2_k[j]@Φ)

                # calculate dz/dt
                # ---------------
                z_mem1_deriv = 0 * z_hat1_tmp

                return Φ_deriv, z_mem1_deriv

        else:
            raise UnsupportedRequest(
                "EQUATION_OF_MOTION =" + self.param["EQUATION_OF_MOTION"],
                type(self).__name__,
            )

        self.dsystem_dt = dsystem_dt

        return dsystem_dt
