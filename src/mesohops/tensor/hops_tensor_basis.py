from __future__ import annotations

import numpy as np

from mesohops.basis.hops_modes import HopsModes  # noqa: F401 (type hint)
from mesohops.basis.hops_noise_memory import HopsNoiseMemory  # noqa: F401 (type hint)
from mesohops.basis.hops_system import HopsSystem  # noqa: F401 (type hint)
from mesohops.tensor.hops_tensor_eom import HopsTensorEOM  # noqa: F401 (type hint)
from mesohops.tensor.hops_tensor_wavefunction import (
    HopsTensorWavefunction,  # noqa: F401 (type hint)
)
from mesohops.tensor.tensor_functions_adaptive import (
    tensor_state_adaptive_check_add_state,
    tensor_state_adaptive_check_remove_state,
)

__title__ = 'TadHOPS Basis'
__author__ = 'B. Z. Citty'
__maintainer__ = 'B. Z. Citty'


class HopsTensorBasis:
    """
    Holds shared references to the physical basis objects (system, mode,
    noise_memory) owned by HopsBasis, plus the tensor-specific EOM, and
    manages adaptive-basis logic for a tensor-network HOPS calculation.

    Hierarchy concepts (n_hier, n_hmodes, adaptive_h) do not apply —
    tensor HOPS encodes hierarchy depth in MPS core dimensions (k_max + 1
    per mode), not through explicit auxiliary vector enumeration.

    HopsTensorWavefunction manages its own MPS core restructuring via
    add_state_cores/remove_state_cores. HopsTensorBasis decides which
    states to add/remove and updates the basis bookkeeping (system, mode).
    """

    __slots__ = (
        'system',  # HopsSystem instance
        'mode',  # HopsModes instance
        'noise_memory',  # HopsNoiseMemory instance
        'eom',  # HopsTensorEOM instance (set by trajectory)
        'adaptive',  # bool: True when state adaptivity is active
        'delta_s',  # float: adaptive threshold (stored in initialize)
    )

    def __init__(
        self,
        system: HopsSystem,
        mode: HopsModes,
        noise_memory: HopsNoiseMemory,
    ) -> None:
        """
        Stores references to the shared basis objects owned by HopsBasis.
        Does not construct HopsSystem, HopsModes, or HopsNoiseMemory —
        those are owned by HopsBasis and passed in by HopsTensorTrajectory.

        Parameters
        ----------
        1. system : HopsSystem
                    Shared system instance (owned by HopsBasis).

        2. mode : HopsModes
                  Shared mode instance (owned by HopsBasis).

        3. noise_memory : HopsNoiseMemory
                          Shared noise memory instance (owned by HopsBasis).

        Returns
        -------
        None
        """
        self.system = system
        self.mode = mode
        self.noise_memory = noise_memory
        self.eom = None
        self.adaptive = False
        self.delta_s = 0

    def initialize(self, delta_s: float) -> None:
        """
        Stores tensor-specific adaptive configuration. The shared objects
        (system, mode, noise_memory) are already initialized by
        HopsTensorTrajectory.initialize before this call.

        Parameters
        ----------
        1. delta_s : float
                     Adaptive threshold. delta_s > 0 enables state adaptivity.

        Returns
        -------
        None
        """
        self.delta_s = delta_s
        self.adaptive = delta_s > 0

    def define_basis(
        self,
        wavefunction: HopsTensorWavefunction,
        z_step: np.ndarray,
    ) -> tuple[list[int], list[int]]:
        """
        Performs state adaptivity: computes errors for removing boundary states
        and adding new ones.

        Receives wavefunction explicitly rather than reaching through self.eom.
        The parent HopsBasis.define_basis takes (phi, tau, z_step) — tensor
        omits tau because tensor HOPS does not do hierarchy adaptivity.

        Parameters
        ----------
        1. wavefunction : HopsTensorWavefunction
                        Tensor wavefunction to evaluate for adaptivity.

        2. z_step : np.array(complex)
                    Noise values for the current time step.

        Returns
        -------
        1. list_states_old : list(int)
                             Absolute state indices to remove.

        2. list_states_new : list(int)
                             Absolute state indices to add.
        """
        if not self.adaptive:
            return [], []

        tensor_eom = self.eom

        def calc_deriv_cores(z_mem, z_rnd, z_rnd2):
            # build_generator stores MPO in eom.mpo_cores (side effect).
            # derivative() now stores its peak-size proxy on
            # eom.last_matvec_complexity; the adaptive-basis path
            # doesn't track max-across-substeps, so we just leave that
            # attribute set and return the cores.
            tensor_eom.build_generator(z_mem, z_rnd, z_rnd2)
            return tensor_eom.derivative()

        old_state_indices = tensor_state_adaptive_check_remove_state(
            wavefunction.list_cores_phi,
            self.system.param['HAMILTONIAN'],
            z_step,
            self.system.param['NSTATES'],
            len(self.system.state_list),
            self.delta_s,
            self.system.state_list,
            wavefunction.method,
            wavefunction.M1_modes_per_state,
            calc_deriv_cores,
        )
        list_states_old = [self.system.state_list[i] for i in old_state_indices]
        list_states_new = tensor_state_adaptive_check_add_state(
            wavefunction.list_cores_phi,
            self.system.param['HAMILTONIAN'],
            list_states_old,
            self.system.param['NSTATES'],
            len(self.system.state_list),
            self.delta_s,
            self.system.state_list,
            wavefunction.method,
            wavefunction.M1_modes_per_state,
        )
        return list_states_old, list_states_new

    def update_basis(
        self,
        wavefunction: HopsTensorWavefunction,
        z_mem: np.ndarray,
        list_states_old: list[int],
        list_states_new: list[int],
    ) -> tuple[HopsTensorWavefunction, np.ndarray]:
        """
        Removes and adds states in both the tensor cores and the basis objects,
        then inflates bonds if TDVP is in use.

        wavefunction is mutated in place and returned for call-site visibility —
        the return value is the same object as the argument.

        z_mem passes through unchanged today.

        TODO: z_mem remapping bug — when states are added/removed,
        mode.list_modeidx_abs changes but z_mem is not remapped, causing
        noise memory values to be read at wrong array positions on the next
        step. The parent calls noise_memory.update_zmem_indexing(z_mem) and
        constructs a new z_mem array — the tensor version must do the same.

        The parent returns (phi, z_mem, dsystem_dt) — tensor omits dsystem_dt
        because tensor EOM rebuilds the MPO on the fly. After restructuring,
        refreshes the persistent MpoBuilder's state-dependent attributes via
        eom.refresh_builder.

        Parameters
        ----------
        1. wavefunction : HopsTensorWavefunction
                        Tensor wavefunction to update in-place.

        2. z_mem : np.ndarray(complex)
                   Noise memory drift terms. Passed through unchanged (see
                   TODO above for the known remapping bug).

        3. list_states_old : list(int)
                             Absolute state indices to remove.

        4. list_states_new : list(int)
                             Absolute state indices to add.

        Returns
        -------
        1. wavefunction : HopsTensorWavefunction
                        The same object passed in, mutated in place.

        2. z_mem : np.ndarray(complex)
                   Noise memory drift terms, passed through unchanged.
        """
        # Remove old states
        wavefunction.remove_state_cores(list_states_old, self.system.state_list)
        self.system.state_list = sorted(
            set(self.system.state_list) - set(list_states_old)
        )
        self.mode.list_modeidx_abs = sorted(self.system.list_statemodeidx_abs)

        # Add new states
        wavefunction.add_state_cores(list_states_new, self.system.state_list)
        self.system.state_list = sorted(
            set(self.system.state_list) | set(list_states_new)
        )
        self.mode.list_modeidx_abs = sorted(self.system.list_statemodeidx_abs)

        # Refresh the persistent MpoBuilder with the new state data,
        # matching the parent pattern where update_basis rebuilds dsystem_dt.
        self.eom.refresh_builder()

        return wavefunction, z_mem
