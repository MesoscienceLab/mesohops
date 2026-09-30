"""
Time-integration routines for tensor-based HOPS.

Each integrator operates on a ``HopsTensorEOM`` instance (which wraps the MPS)
used by ``HopsTensorTrajectory``.

Functions
---------
runge_kutta_step_tensor(eom, z_mem, z_rnd, z_rnd2, tau)
    RK4 step for the MPS wavefunction, using tensor_add at each stage.

runge_kutta_variables(z_mem, t, noise, noise2, tau, ...)
    Gathers noise samples at the three RK4 time points and returns a dict
    ready to unpack into runge_kutta_step_tensor.

single_point_variables(z_mem, t, noise, noise2, tau, ...)
    Gathers noise at time t and returns a dict ready to unpack into
    a single-point-noise integrator (used by TDVP steps).

tdvp_step_tensor(eom, z_mem, z_rnd, z_rnd2, tau, method, **kwargs)
    TDVP step (1-site or 2-site) for the MPS wavefunction.
"""
from __future__ import annotations

import numpy as np

from mesohops.noise.hops_noise import HopsNoise
from mesohops.tensor import tdvp
from mesohops.tensor.hops_tensor_eom import HopsTensorEOM
from mesohops.util.physical_constants import hbar
from mesohops.util.tensor_operations import scale_mps, tensor_add

__title__ = 'Tensor Integrators'
__author__ = 'D. I. G. Bennett, B. Z. Citty'
__maintainer__ = 'B. Z. Citty'


def runge_kutta_step_tensor(
    eom: HopsTensorEOM,
    z_mem: np.ndarray,
    z_rnd: np.ndarray,
    z_rnd2: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, int]:
    """
    Performs a single Runge-Kutta step for the MPS wavefunction.

    Parameters
    ----------
    1. eom: HopsTensorEOM
             Equation of motion object wrapping the MPS and MPO.
    2. z_mem: np.ndarray(complex)
              Current memory term values.
    3. z_rnd: np.ndarray(complex)
              Primary noise at three time points.
    4. z_rnd2: np.ndarray(complex)
               Secondary noise at three time points.
    5. tau: float
            Integration time step [units: fs].

    Returns
    -------
    1. z_mem: np.ndarray(complex)
              Updated memory terms.

    Side effects
    ------------
    Sets `eom.max_complexity_step` to the largest per-call tensor
    complexity observed across the four RK4 matvec-then-compress
    evaluations of this step. Read via the side-channel by propagate();
    avoids plumbing the scalar through every return value.
    """
    wavefunction = eom.wavefunction
    list_k_phi = [[] for _ in range(4)]  # RK4 stages for dphi/dt
    list_k_zmem = [[] for _ in range(4)]  # RK4 stages for dz_mem/dt
    # RK4 stage time-offsets: evaluate at t, t+tau/2, t+tau/2, t+tau
    list_rk_coeff = [0.0, 0.5, 0.5, 1.0]
    # Map each RK4 stage to a noise time-point index: 0=t, 1=t+tau/2, 2=t+tau
    list_noise_idx = [0, 1, 1, 2]
    # Peak uncompressed-core complexity across the 4 RK4 matvec calls;
    # tracked locally and published on eom.max_complexity_step at the end.
    max_complexity = 0

    # Deep-copy the initial MPS state as the RK4 checkpoint.
    # Each group is copied element-wise for statenumber (nested list-of-lists);
    # a single list comprehension handles both nested and flat structures.
    list_cores_phi_checkpoint = [
        [c.copy() for c in g] if isinstance(g, list) else g.copy()
        for g in wavefunction.list_cores_phi
    ]

    for i in range(4):
        if i == 0:
            z_mem_tmp = z_mem
        else:
            z_mem_tmp = z_mem + list_rk_coeff[i] * list_k_zmem[i - 1] * tau / hbar
            # Scale k[i-1] by the RK4 prefactor, then unscale after.
            # Scaling only the first core is equivalent to scaling the whole
            # MPS; tensor_add distributes the weight through the boundary core.
            scale_mps(list_k_phi[i - 1], list_rk_coeff[i] * tau / hbar)
            wavefunction.list_cores_phi = tensor_add(
                list_cores_phi_checkpoint,
                list_k_phi[i - 1],
                wavefunction.mps_epsilon,
                wavefunction.bond_dim_max,
            )
            # Restore k[i-1] to its unscaled derivative form so the
            # final RK4 weighted sum (below) starts from raw k values.
            scale_mps(list_k_phi[i - 1], hbar / (list_rk_coeff[i] * tau))
        # Build MPO and get dz; then get d(phi)/dt cores
        list_k_zmem[i] = eom.build_generator(
            z_mem_tmp, z_rnd[:, list_noise_idx[i]], z_rnd2[:, list_noise_idx[i]]
        )
        list_k_phi[i] = eom.derivative()
        # eom.last_matvec_complexity was set by the derivative() call.
        if eom.last_matvec_complexity > max_complexity:
            max_complexity = eom.last_matvec_complexity

    wavefunction.list_cores_phi = list_cores_phi_checkpoint

    # RK4 weighted sum: scale each derivative by its RK4 weight and
    # the tau/hbar factor. Scaling only the first core is equivalent
    # to scaling the whole MPS; tensor_add absorbs the weight into the
    # boundary core.
    scale_mps(list_k_phi[0], tau / (6 * hbar))
    scale_mps(list_k_phi[1], tau / (3 * hbar))
    scale_mps(list_k_phi[2], tau / (3 * hbar))
    scale_mps(list_k_phi[3], tau / (6 * hbar))

    for k in list_k_phi:
        wavefunction.list_cores_phi = tensor_add(
            wavefunction.list_cores_phi,
            k,
            wavefunction.mps_epsilon,
            wavefunction.bond_dim_max,
        )
    z_mem = (
        z_mem
        + tau
        / hbar
        * (
            list_k_zmem[0]
            + 2.0 * list_k_zmem[1]
            + 2.0 * list_k_zmem[2]
            + list_k_zmem[3]
        )
        / 6.0
    )

    eom.max_complexity_step = max_complexity
    return z_mem


def runge_kutta_variables(
    z_mem: np.ndarray,
    t: float,
    noise: HopsNoise,
    noise2: HopsNoise,
    tau: float,
    list_l2idx_abs: list[int],
    effective_noise_integration: bool = False,
) -> dict:
    """
    Accepts noise objects and returns the variables needed for a
    Runge-Kutta integration step.

    Parameters
    ----------
    1. z_mem: np.ndarray(complex)
              Memory terms [units: cm^-1].
    2. t: float
          Integration time point [units: fs].
    3. noise: instance(HopsNoise)
              Primary noise generator.
    4. noise2: instance(HopsNoise)
               Secondary noise generator.
    5. tau: float
            Integration time step [units: fs].
    6. list_l2idx_abs: list(int)
                       Absolute L2 operator indices for noise
                       sampling.
    7. effective_noise_integration: bool
                                    True uses moving average over
                                    noise; False uses point samples.

    Returns
    -------
    1. variables: dict
                  Dictionary of variables needed for Runge-Kutta.
    """
    if effective_noise_integration:
        # Number of fine noise sub-steps per integration step
        tau_ratio = round(tau / noise.param['TAU'])
        tau_ratio2 = round(tau / noise2.param['TAU'])
        # Sample noise at fine resolution over [t, t + 1.5*tau)
        z_rnd_raw = noise.get_noise(
            [t + (i / tau_ratio) * tau for i in range(round(tau_ratio * 1.5))],
            list_l2idx_abs,
        )
        z_rnd2_raw = noise2.get_noise(
            [t + (i / tau_ratio2) * tau for i in range(round(tau_ratio2 * 1.5))],
            list_l2idx_abs,
        )
        # Average fine noise into 3 bins matching RK4 time-points:
        #   bin 0: [t, t+tau/2)         -> noise at t
        #   bin 1: [t+tau/2, t+tau)     -> noise at t+tau/2
        #   bin 2: [t+tau, t+1.5*tau)   -> noise at t+tau
        z_rnd = np.array(
            [
                np.mean(z_rnd_raw[:, : round(tau_ratio / 2)], axis=1),
                np.mean(z_rnd_raw[:, round(tau_ratio / 2) : tau_ratio], axis=1),
                np.mean(z_rnd_raw[:, tau_ratio:], axis=1),
            ]
        ).T
        z_rnd2 = np.array(
            [
                np.mean(z_rnd2_raw[:, : round(tau_ratio2 / 2)], axis=1),
                np.mean(z_rnd2_raw[:, round(tau_ratio2 / 2) : tau_ratio2], axis=1),
                np.mean(z_rnd2_raw[:, tau_ratio2:], axis=1),
            ]
        ).T

    else:
        z_rnd = noise.get_noise([t, t + tau * 0.5, t + tau], list_l2idx_abs)
        z_rnd2 = noise2.get_noise([t, t + tau * 0.5, t + tau], list_l2idx_abs)

    return {'z_mem': z_mem, 'z_rnd': z_rnd, 'z_rnd2': z_rnd2, 'tau': tau}


def single_point_variables(
    z_mem: np.ndarray,
    t: float,
    noise: HopsNoise,
    noise2: HopsNoise,
    tau: float,
    list_l2idx_abs: list[int] | None = None,
    effective_noise_integration: bool = False,
) -> dict:
    """
    Accepts noise objects and returns the variables needed for a
    single-point-noise integration step (used by TDVP).

    Parameters
    ----------
    1. z_mem: np.ndarray(complex)
              Memory terms [units: cm^-1].
    2. t: float
          Integration time point [units: fs].
    3. noise: instance(HopsNoise)
              Primary noise generator.
    4. noise2: instance(HopsNoise)
               Secondary noise generator.
    5. tau: float
            Integration time step [units: fs].
    6. list_l2idx_abs: list(int)
                       Absolute L2 operator indices passed to
                       noise.get_noise for adaptive filtering.
    7. effective_noise_integration: bool
                                    True uses moving average over
                                    noise; False uses point samples.
                                    **Not yet implemented; raises
                                    NotImplementedError.**

    Returns
    -------
    1. variables: dict
                  Dictionary of variables needed for a
                  single-point-noise integration step.
    """
    if effective_noise_integration:
        raise NotImplementedError(
            'effective_noise_integration is not implemented for single_point_variables.'
        )
    z_rnd = noise.get_noise([t], list_l2idx_abs)
    z_rnd2 = noise2.get_noise([t], list_l2idx_abs)
    return {'z_mem': z_mem, 'z_rnd': z_rnd, 'z_rnd2': z_rnd2, 'tau': tau}


def _build_tdvp_solver_kwargs(update_type, krylov_conv_tol, **kwargs):
    """
    Map trajectory-level TDVP parameters to tdvp.timestep kwargs.

    Parameters
    ----------
    1. update_type: str
                    TDVP solver type ('arnoldi', 'lanczos', 'krylov', or 'ivp').
    2. krylov_conv_tol: float
                        Relative convergence tolerance for arnoldi/lanczos solvers.
    3. **kwargs: dict
                 Additional parameters (ivp_method, ivp_rtol, ivp_atol,
                 ivp_max_step) forwarded when update_type is 'ivp'.

    Returns
    -------
    1. solver_kwargs: dict
                      Keyword arguments for tdvp.timestep.
    """
    # 'krylov' is a convenience alias for 'arnoldi'
    solver = 'arnoldi' if update_type == 'krylov' else update_type
    solver_kwargs = {'solver': solver}
    if solver in ('arnoldi', 'lanczos'):
        solver_kwargs['conv_tol'] = krylov_conv_tol
    elif solver == 'ivp':
        solver_kwargs['method'] = kwargs.get('ivp_method', 'BDF')
        solver_kwargs['rtol'] = kwargs.get('ivp_rtol', 1e-7)
        solver_kwargs['atol'] = kwargs.get('ivp_atol', 1e-9)
        max_step = kwargs.get('ivp_max_step', None)
        if max_step is not None:
            solver_kwargs['max_step'] = max_step
    return solver_kwargs


def tdvp_step_tensor(
    eom, z_mem, z_rnd, z_rnd2, tau, method='1tdvp',
    krylov_conv_tol=1e-6, update_type='krylov', **kwargs
):
    """
    Performs a single TDVP step (1-site or 2-site) for the MPS wavefunction.

    Builds the MPO via eom.build_generator, then evolves the MPS using the
    Strang-split TDVP sweep from tdvp.timestep. The memory terms z_mem are
    updated with a first-order step.

    For 2TDVP, bond dimension is controlled by wavefunction.bond_dim_max and
    wavefunction.mps_epsilon.

    Parameters
    ----------
    1. eom: HopsTensorEOM
             Equation of motion object wrapping the MPS and MPO.
    2. z_mem: np.ndarray(complex)
              Current memory term values.
    3. z_rnd: np.ndarray(complex)
              Primary noise at this time point.
    4. z_rnd2: np.ndarray(complex)
               Secondary noise at this time point.
    5. tau: float
            Integration time step [units: fs].
    6. method: str
               TDVP variant ('1tdvp' or '2tdvp').
    7. krylov_conv_tol: float
                        Relative convergence tolerance for the local exponential
                        solver.  Iteration stops when the Hochbruck-Lubich
                        residual estimate drops below this value.
    8. update_type: str
                    Solver type ('arnoldi', 'lanczos', 'krylov', or 'ivp').
    9. **kwargs: dict
                 Forwarded to the solver (e.g. ivp_method, ivp_rtol, ivp_atol).

    Returns
    -------
    1. z_mem: np.ndarray(complex)
              Updated memory terms.

    Side effects
    ------------
    Sets `eom.max_complexity_step = 0`. TDVP does not evaluate
    tensor_matvec_prod, so the peak-size metric tracked by the RK4
    path does not apply; zero is published on the side-channel only
    to keep the attribute populated for propagate's storage write.
    """
    wavefunction = eom.wavefunction

    # (1) Build MPO and compute dz/dt.
    dz_dt = eom.build_generator(z_mem, z_rnd[:, 0], z_rnd2[:, 0])

    # (2) Initialize TDVP: right-normalize MPS and build environments.
    # (initialize recenters the MPS to site 0 internally for stability)
    core_M, list_cores_B, L0, list_envs_R = tdvp.initialize(
        wavefunction.flat_cores, eom.mpo_cores
    )

    # (3) Strang-split TDVP sweep.
    solver_kwargs = _build_tdvp_solver_kwargs(update_type, krylov_conv_tol, **kwargs)
    if method == '2tdvp':
        solver_kwargs['chi_max'] = wavefunction.bond_dim_max
        solver_kwargs['eps'] = wavefunction.mps_epsilon
    core_M, list_cores_B, L0, list_envs_R = tdvp.timestep(
        tau / hbar,
        L0,
        list_envs_R,
        eom.mpo_cores,
        core_M,
        list_cores_B,
        method=method,
        **solver_kwargs,
    )

    # (4) Write updated cores back into wavefunction.
    wavefunction.update_phi_from_flat([core_M] + list(list_cores_B))

    # (5) First-order update for memory terms.
    z_mem = z_mem + tau / hbar * dz_dt

    # TDVP does not evaluate tensor_matvec_prod, so the peak-size metric
    # tracked by runge_kutta_step_tensor does not apply.
    eom.max_complexity_step = 0
    return z_mem


# Convenience aliases for backwards compatibility with tests
tdvp1_step_tensor = tdvp_step_tensor


def tdvp2_step_tensor(
    eom, z_mem, z_rnd, z_rnd2, tau, krylov_conv_tol=1e-6, update_type='krylov', **kwargs
):
    """
    2TDVP step. Convenience wrapper around tdvp_step_tensor with method='2tdvp'.
    """
    return tdvp_step_tensor(
        eom, z_mem, z_rnd, z_rnd2, tau, method='2tdvp',
        krylov_conv_tol=krylov_conv_tol, update_type=update_type, **kwargs
    )
