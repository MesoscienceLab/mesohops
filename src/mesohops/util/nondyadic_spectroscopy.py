from typing import NamedTuple

import numpy as np
from scipy import sparse

from mesohops.util.exceptions import UnsupportedRequest

__title__ = 'nondyadic_spectroscopy'
__author__ = 'A. Hartzell'
__maintainer__ = 'A. Hartzell'


class SpectroscopyDispatch(NamedTuple):
    traj_kind: str
    tensor_method: str | None
    hilbert_tag: str
    eom_tag: str


def _spectroscopy_key(traj, n_site):
    """
    Canonical dispatch tuple for the spectroscopy code path.

    Encodes (in order) the trajectory type, tensor method (if tensor),
    Hilbert-space convention, and EOM tag. The Hilbert-space convention
    means:
        - embedded: the ground state is an explicit basis slot in the
          Hilbert space, so the trajectory dimension is n_site + 1.
        - vacuum: the ground state is tracked implicitly as the all-zero
          MPS configuration, so the trajectory dimension is n_site and
          the tensor representation must be ``number``.
        - excited_only: only the excited manifold is represented; this
          is used for the LINEAR EOM for vector and fullstate
          trajectories.

    Validates that the trajectory is uninitialized, the EOM is in the
    supported set, and the Hilbert dimension is either n_site or
    n_site + 1.

    Example return values:
        SpectroscopyDispatch('vector', None, 'embedded', 'NL')
        SpectroscopyDispatch('tensor', 'fullstate', 'embedded', 'LINEAR')
        SpectroscopyDispatch('tensor', 'number', 'vacuum', 'NL')

    Parameters
    ----------
    1. traj : uninitialized trajectory object
    2. n_site : int
                Number of physical sites (from len(list_transition_dipoles)).

    Returns
    -------
    1. spectroscopy_key : SpectroscopyDispatch
                          Canonical dispatch tuple.

    Raises
    ------
    ValueError
        If the trajectory is already initialized or the Hilbert
        dimension is not in {n_site, n_site + 1}.

    UnsupportedRequest
        On unsupported EOM or tensor method / convention combinations.
    """
    if traj.__initialized__:
        raise ValueError('initialized trajectory is not valid for spectroscopy dispatch')
    eom = traj.basis.eom.param['EQUATION_OF_MOTION']
    if eom == 'NONLINEAR':
        eom_tag = 'NL'
    elif eom == 'LINEAR':
        eom_tag = 'LINEAR'
    else:
        raise UnsupportedRequest(eom, 'nondyadic_spectroscopy')

    n_state_traj = traj.basis.system.param['NSTATES']
    if n_state_traj == n_site + 1:
        hilbert_tag = 'embedded'
    elif n_state_traj == n_site:
        hilbert_tag = 'excited_only'
    else:
        raise ValueError(
            f'trajectory dim {n_state_traj} != n_site ({n_site}) '
            f'or n_site+1 ({n_site + 1})',
        )

    wf = getattr(traj, 'wavefunction', None)
    if wf is None:
        return SpectroscopyDispatch('vector', None, hilbert_tag, eom_tag)

    method = traj.tensor_param['METHOD']
    if method == 'number':
        # Number representation always uses the vacuum convention (ground is
        # the all-zeros config, dim n_site); the embedded layout (n_site+1)
        # is not supported.
        if hilbert_tag == 'excited_only':
            wf.flag_gs_vacuum = True
            return SpectroscopyDispatch('tensor', 'number', 'vacuum', eom_tag)
        raise UnsupportedRequest(
            _format_dispatch(
                SpectroscopyDispatch('tensor', 'number', hilbert_tag, eom_tag)
            ),
            'nondyadic_spectroscopy',
        )

    if method != 'fullstate':
        raise UnsupportedRequest(method, 'nondyadic_spectroscopy')

    if hilbert_tag in ('embedded', 'excited_only'):
        return SpectroscopyDispatch('tensor', 'fullstate', hilbert_tag, eom_tag)

    raise UnsupportedRequest(
        _format_dispatch(
            SpectroscopyDispatch('tensor', 'fullstate', hilbert_tag, eom_tag)
        ),
        'nondyadic_spectroscopy',
    )


def _format_dispatch(dispatch):
    if dispatch.tensor_method is None:
        return f'{dispatch.traj_kind}_{dispatch.hilbert_tag}_{dispatch.eom_tag}'
    return (
        f'{dispatch.traj_kind}_{dispatch.tensor_method}_'
        f'{dispatch.hilbert_tag}_{dispatch.eom_tag}'
    )


def _readout_prefactor(traj, norm_ratio, list_norm_sq):
    """
    Returns the prefactor that multiplies <psi|F|psi> in the
    spectroscopy readout, branching on the EOM.

    NORMALIZED NONLINEAR (the validated dyadic reference) and plain
    NONLINEAR are rescale-equivalent: operator_expectation divides
    by <psi|psi>, so <L> is invariant under psi -> c(t) psi, and
    plain NONLINEAR is just NORMALIZED NONLINEAR multiplied by a
    complex scalar c(t) with |c(t)|^2 = norm_ratio / ||psi(t)||^2. The
    dyadic readout norm_ratio * <psi|F|psi> / ||psi||^2 is therefore
    identical per-realization between the two EOMs.

    LINEAR uses raw noise (no Girsanov shift, no z_mem evolution)
    and is NOT a rescaling of NORMALIZED NONLINEAR — different
    measure entirely. The (N+1)-embedded operators decouple |g>
    from the bath in H and L under all EOMs, and neither LINEAR
    nor plain NONLINEAR has a rescaling term, so psi[0]=1 holds
    deterministically under both — the difference is in the noise
    measure, not psi[0] dynamics. Under LINEAR's raw measure the
    unbiased per-trajectory estimator of C(t) is <psi|F|psi>
    directly; the dyadic norm_ratio/||psi||^2 prefactor has no scaling
    identity to support it under LINEAR and would distort the
    readout (||psi(t)||^2 random-walks freely without the Girsanov
    shift).

    Parameters
    ----------
    1. traj : trajectory object
              An initialized trajectory.

    2. norm_ratio : float
               Product of operator-norm changes from each
               raise/lower step (>= 1 for the operators built here).

    3. list_norm_sq : np.ndarray(float)
                    ||psi(t)||^2 at each detection-phase timestep.

    Returns
    -------
    1. prefactor : float or np.ndarray(float)
                   Scalar 1.0 under LINEAR; per-timestep
                   norm_ratio / list_norm_sq otherwise.
    """
    if traj.basis.eom.param['EQUATION_OF_MOTION'] == 'LINEAR':
        return 1.0
    return norm_ratio / list_norm_sq


def _build_operators_abs(list_transition_dipoles, E_1):
    """
    Builds the raising and response operators for absorption.

    Parameters
    ----------
    1. list_transition_dipoles : np.ndarray, shape (n_site, 3)
                  Transition dipole moments per chromophore.

    2. E_1 : np.ndarray, shape (3,) or (3, 1)
             Field polarization (E_sig = E_1 for absorption).

    Returns
    -------
    1. O2_raise : sparse.coo_matrix, shape (n_state, n_state)
                  Excitation operator: maps |g> to sum_i (mu_i . E)|e_i>, plus
                  ground-state identity to preserve |g>.

    2. F2_dense : np.ndarray, shape (n_state, n_state)
                  Response operator for the expectation value
                  <psi|F|psi> / ||psi||^2.
                  Only row 0 is nonzero: F[0, i+1] = mu_i . E.
    """
    # Convention: in the wavefunction picture, this operator excites:
    # O|g> = sum_i (mu_i . E)|e_i>. In the density matrix picture,
    # |e><g| is conventionally called a lowering operator.
    E_1 = np.asarray(E_1).ravel()
    n_site = len(list_transition_dipoles)
    n_state = n_site + 1
    list_mu_dot_E = list_transition_dipoles @ E_1

    # Raising + ground identity: (i+1, 0) maps |g> -> |e_i>, (0, 0) keeps |g>
    raise_data = np.concatenate([list_mu_dot_E, [1.0]])
    raise_row = np.concatenate([np.arange(1, n_state), [0]])
    raise_col = np.zeros(n_state, dtype=int)
    O2_raise = sparse.coo_matrix(
        (raise_data, (raise_row, raise_col)),
        shape=(n_state, n_state),
    )

    # Response operator: only row 0 nonzero, F[0, i+1] = mu_i . E
    F2_dense = np.zeros((n_state, n_state))
    F2_dense[0, 1:] = list_mu_dot_E

    return O2_raise, F2_dense


def _build_operators_fluor(list_transition_dipoles, E_1, E_sig):
    """
    Builds the raising, lowering+identity, and response operators for
    fluorescence.

    Parameters
    ----------
    1. list_transition_dipoles : np.ndarray, shape (n_site, 3)
                  Transition dipole moments per chromophore.

    2. E_1 : np.ndarray, shape (3,) or (3, 1)
             Field polarization for excitation.

    3. E_sig : np.ndarray, shape (3,) or (3, 1)
               Signal field polarization (also E_3 for lowering).

    Returns
    -------
    1. O2_raise : sparse.coo_matrix, shape (n_state, n_state)
                  Excitation operator: maps |g> to sum_i (mu_i . E_1)|e_i>.

    2. O2_lower_ident : sparse.coo_matrix, shape (n_state, n_state)
                        Lowering |g><e_i| weighted by mu_i . E_sig, plus
                        identity in the excited block.

    3. F2_dense : np.ndarray, shape (n_state, n_state)
                  Response operator for the expectation value <psi|F|psi>.
                  Only row 0 is nonzero: F[0, i+1] = mu_i . E_sig.
    """
    # Convention: see _build_operators_abs for wavefunction vs density
    # matrix naming convention for raising/lowering operators.
    E_1 = np.asarray(E_1).ravel()
    E_sig = np.asarray(E_sig).ravel()
    n_site = len(list_transition_dipoles)
    n_state = n_site + 1
    list_mu_dot_E1 = list_transition_dipoles @ E_1
    list_mu_dot_Esig = list_transition_dipoles @ E_sig

    # Raising operator: nonzero at (i+1, 0), maps |g> -> |e_i>
    O2_raise = sparse.coo_matrix(
        (list_mu_dot_E1, (np.arange(1, n_state), np.zeros(n_site, dtype=int))),
        shape=(n_state, n_state),
    )

    # Lowering operator: row 0 has |g><e_i| entries weighted by mu_i . E_sig.
    O2_lower = sparse.coo_matrix(
        (
            list_mu_dot_Esig,
            (np.zeros(n_site, dtype=int), np.arange(1, n_state)),
        ),
        shape=(n_state, n_state),
    )

    # Add identity only on the excited block so the coherences survive the
    # lower+ident step while the ground row remains the pure lowering term.
    O2_excited_ident = sparse.coo_matrix(
        (
            np.ones(n_site, dtype=np.complex128),
            (np.arange(1, n_state), np.arange(1, n_state)),
        ),
        shape=(n_state, n_state),
    )
    O2_lower_ident = O2_lower + O2_excited_ident

    # Response operator: only row 0 nonzero, F[0, i+1] = mu_i . E_sig
    F2_dense = np.zeros((n_state, n_state))
    F2_dense[0, 1:] = list_mu_dot_Esig

    return O2_raise, O2_lower_ident, F2_dense


def _apply_op_and_track_norm(traj, apply_operator, spectroscopy_key):
    """
    Applies an operator to the trajectory in place and returns the
    norm-squared coefficient ||O psi||^2 / ||psi||^2.  Needed because
    the nonlinear EOM renormalizes the wavefunction, so the physical
    norm change must be tracked separately.

    Branches on the trajectory convention encoded in ``spectroscopy_key``:
    * Vacuum tensor (tensor + number + vacuum): norm uses the GS +
      single-excitation manifold norm via manifold_norm_sq.  The full
      MPS norm sees auxiliary hierarchy content and any multi-excitation
      leakage; manifold_norm_sq excludes both by construction.
    * Embedded (vector or fullstate tensor): norm uses <psi|psi> of
      traj.psi, which already carries the GS amplitude at slot 0.

    Parameters
    ----------
    1. traj : trajectory object
              An initialized trajectory.

    2. apply_operator : callable
                        No-arg callable that performs the operator
                        application on traj in place, e.g.
                  ``lambda: traj._operator(O2.toarray())`` for the
                  embedded key or
                  ``lambda: traj.apply_dipole_raise(list_mu)`` for the
                  vacuum key.

    3. spectroscopy_key : SpectroscopyDispatch
                          Spectroscopy dispatch tuple from
                          _spectroscopy_key.

    Returns
    -------
    1. norm_ratio : float
                    Norm-squared ratio ||O psi||^2 / ||psi||^2.
    """
    if spectroscopy_key.tensor_method == 'number' and spectroscopy_key.hilbert_tag == 'vacuum':
        norm_sq_pre = traj.wavefunction.manifold_norm_sq
        apply_operator()
        norm_sq_post = traj.wavefunction.manifold_norm_sq
    else:
        psi_pre = traj.psi
        norm_sq_pre = np.dot(np.conj(psi_pre), psi_pre).real
        apply_operator()
        psi_post = traj.psi
        norm_sq_post = np.dot(np.conj(psi_post), psi_post).real
    return norm_sq_post / norm_sq_pre


def calc_absorption_response(traj, list_transition_dipoles, E_1, t_max, t_step):
    """
    Non-dyadic linear absorption C(t) for a single trajectory.

    Parameters
    ----------
    1. traj : uninitialized trajectory object
              Any type supporting .initialize(), ._operator(),
              .propagate(), and .storage['psi_traj']. Hilbert
              dimension must equal len(list_transition_dipoles) (vacuum tensor or
              excited-only LINEAR) or len(list_transition_dipoles) + 1 (embedded).
              Conventions:
                - embedded: an explicit ground-state basis slot is
                  present in the Hilbert space.
                - vacuum: the ground state is implicit in the number
                  representation and corresponds to the all-zero MPS
                  configuration.
                - excited_only: only the excited manifold is
                  propagated, and only for the LINEAR shortcut.

    2. list_transition_dipoles : np.ndarray, shape (n_site, 3)
                  Transition dipole moments per chromophore.

    3. E_1 : np.ndarray, shape (3,)
             Field polarization (also used as E_sig for absorption).

    4. t_max : float [units: fs]
               Propagation time after excitation.

    5. t_step : float [units: fs]
                Time step.

    Returns
    -------
    1. C1_corr_t : np.ndarray(complex)
                   Absorption correlation function C(t) sampled at
                   t = t_step, 2*t_step, ....
    """
    n_site = len(list_transition_dipoles)
    spectroscopy_key = _spectroscopy_key(traj, n_site)
    traj_kind, tensor_method, hilbert_tag, eom_tag = spectroscopy_key

    if hilbert_tag == 'embedded':
        # vector + fullstate tensor + number tensor (GS embedded in
        # the n_site+1 dim Hilbert space).
        O2_raise, F2_dense = _build_operators_abs(list_transition_dipoles, E_1)
        P1_psi_0 = np.zeros(n_site + 1, dtype=np.complex128)
        P1_psi_0[0] = 1.0
        traj.initialize(P1_psi_0)
        norm_ratio = _apply_op_and_track_norm(
            traj, lambda: traj._operator(O2_raise.toarray()), spectroscopy_key,
        )
        traj.propagate(t_max, t_step)

        # Skip psi_traj[0]: the ground state stored by initialize()
        # before the raise.  C(t) starts at t = t_step.
        psi_traj_slice = np.asarray(traj.storage['psi_traj'])[1:]
        list_norm_sq = np.sum(np.conj(psi_traj_slice) * psi_traj_slice, axis=1)
        # psi @ F.T computes F @ psi[t] for all timesteps simultaneously.
        psi_f_traj = psi_traj_slice @ F2_dense.T
        list_expectation = np.sum(np.conj(psi_traj_slice) * psi_f_traj, axis=1)
        return 2 * _readout_prefactor(traj, norm_ratio, list_norm_sq) * list_expectation

    if tensor_method == 'number' and hilbert_tag == 'vacuum':
        # Vacuum tensor: |g> is the all-zeros MPS configuration;
        # raise is applied as a bond-dim MPO.
        E_1 = np.asarray(E_1).ravel()
        list_mu_dot_E = list_transition_dipoles @ E_1
        P1_psi_0 = np.zeros(n_site, dtype=np.complex128)
        traj.initialize(P1_psi_0)
        # |psi_g|^2 = 1 by absorption's decoupled-GS assumption, so
        # manifold_norm_sq mirrors the embedded key's
        # ||O psi||^2 / ||psi||^2.
        GS_AMP_SQ = 1.0
        norm_ratio = _apply_op_and_track_norm(
            traj,
            lambda: traj.apply_dipole_raise_plus_ground_ident(list_mu_dot_E),
            spectroscopy_key,
        )
        traj.propagate(t_max, t_step)
        psi_traj_slice = np.asarray(traj.storage['psi_traj'])[1:]
        list_norm_sq = np.sum(
            np.conj(psi_traj_slice) * psi_traj_slice, axis=1,
        ).real
        list_response = psi_traj_slice @ list_mu_dot_E
        return 2 * _readout_prefactor(traj, norm_ratio, list_norm_sq + GS_AMP_SQ) * list_response

    if eom_tag == 'LINEAR' and hilbert_tag == 'excited_only':
        # Excited-only LINEAR optimization: raise is absorbed into the
        # init via psi-linearity.  psi_unnorm(t) = sqrt(norm_ratio) * psi(t),
        # so C(t) = 2 * <V | psi_unnorm(t)> = 2 * sqrt(norm_ratio) * (psi(t) . V*).
        E_1 = np.asarray(E_1).ravel()
        list_mu_dot_E = list_transition_dipoles @ E_1
        # This is the norm of the initial excited-only seed vector, not the
        # nonlinear readout prefactor used in the other branches.
        mu_dot_e_norm_sq = float(np.dot(np.conj(list_mu_dot_E), list_mu_dot_E).real)
        if mu_dot_e_norm_sq == 0.0:
            raise ValueError(
                'mu . E has zero norm; cannot initialize excited-only '
                'LINEAR absorption trajectory'
            )
        P1_psi_0 = np.zeros(n_site, dtype=np.complex128)
        P1_psi_0[:] = list_mu_dot_E / np.sqrt(mu_dot_e_norm_sq)
        traj.initialize(P1_psi_0)
        traj.propagate(t_max, t_step)
        # Skip psi_traj[0] so the output time grid matches the
        # embedded key: C(t) starts at t = t_step.
        psi_traj_slice = np.asarray(traj.storage['psi_traj'])[1:]
        return 2 * np.sqrt(mu_dot_e_norm_sq) * (
            psi_traj_slice @ np.conj(list_mu_dot_E)
        )

    raise UnsupportedRequest(
        f'unsupported key {_format_dispatch(spectroscopy_key)}',
        'calc_absorption_response',
    )


def calc_fluorescence_response(traj, list_transition_dipoles, E_1, E_sig, t2, t3_max, t_step):
    """
    Non-dyadic fluorescence C(t) for a single trajectory.

    Excites from the ground state with mu.E_1, waits for t2, applies
    the (lower + I_excited) operator weighted by mu.E_sig, propagates
    for t3_max, and reads out the fluorescence correlation function
    from the bilinear <psi|F|psi> / <psi|psi>. The returned array is
    sampled at t = t_step, 2*t_step, ...

    Parameters
    ----------
    1. traj : uninitialized trajectory object
              Any type supporting .initialize(), ._operator(),
              .propagate(), and .storage['psi_traj'].  Hilbert
              dimension must equal len(list_transition_dipoles) + 1 (embedded) or
              len(list_transition_dipoles) (vacuum tensor).  Conventions:
                - embedded: an explicit ground-state basis slot is
                  present in the Hilbert space.
                - vacuum: the ground state is implicit in the number
                  representation and corresponds to the all-zero MPS
                  configuration.
                - excited_only: only the excited manifold is
                  propagated, and only for the LINEAR shortcut.

    2. list_transition_dipoles : np.ndarray, shape (n_site, 3)
                  Transition dipole moments per chromophore.

    3. E_1 : np.ndarray, shape (3,)
             Field polarization for excitation.

    4. E_sig : np.ndarray, shape (3,)
               Signal field polarization (also used as E_3 for lowering).

    5. t2 : float [units: fs]
            Waiting time in the excited manifold before detection.

    6. t3_max : float [units: fs]
                Detection time after de-excitation.

    7. t_step : float [units: fs]
                Time step.

    Returns
    -------
    1. C1_corr_t : np.ndarray(complex)
                   Fluorescence correlation function C(t) sampled at
                   t = t_step, 2*t_step, ....
    """
    n_site = len(list_transition_dipoles)
    spectroscopy_key = _spectroscopy_key(traj, n_site)
    traj_kind, tensor_method, hilbert_tag, eom_tag = spectroscopy_key

    if t2 % t_step > 1e-10 * t_step:
        raise ValueError(
            f't2={t2} is not an integer multiple of t_step={t_step}.'
        )
    idx_t2 = round(t2 / t_step)

    if hilbert_tag == 'embedded':
        # vector + fullstate tensor + number tensor (GS embedded in
        # the n_site+1 dim Hilbert space).
        O2_raise, O2_lower_ident, F2_dense = _build_operators_fluor(
            list_transition_dipoles, E_1, E_sig
        )
        P1_psi_0 = np.zeros(n_site + 1, dtype=np.complex128)
        P1_psi_0[0] = 1.0
        traj.initialize(P1_psi_0)

        norm_ratio_1 = _apply_op_and_track_norm(
            traj, lambda: traj._operator(O2_raise.toarray()), spectroscopy_key,
        )
        traj.propagate(t2, t_step)
        norm_ratio_2 = _apply_op_and_track_norm(
            traj, lambda: traj._operator(O2_lower_ident.toarray()), spectroscopy_key,
        )
        traj.propagate(t3_max, t_step)

        psi_traj_slice = np.asarray(traj.storage['psi_traj'])[idx_t2 + 1:]
        list_norm_sq = np.sum(np.conj(psi_traj_slice) * psi_traj_slice, axis=1)
        # psi @ F.T computes F @ psi[t] for all timesteps simultaneously.
        psi_f_traj = psi_traj_slice @ F2_dense.T
        list_expectation = np.sum(np.conj(psi_traj_slice) * psi_f_traj, axis=1)
        return 4 * _readout_prefactor(
            traj, norm_ratio_1 * norm_ratio_2, list_norm_sq,
        ) * list_expectation

    if tensor_method == 'number' and hilbert_tag == 'vacuum':
        # Vacuum tensor: |g> is the all-zeros MPS configuration;
        # raise and (lower + I) are bond-dim MPOs.
        if t3_max % t_step > 1e-10 * t_step:
            raise ValueError(
                f't3_max={t3_max} is not an integer multiple of '
                f't_step={t_step}.'
            )
        # Capture the per-step GS amplitude for the detection-phase readout.
        if not traj.storage.storage_dic.get('psi_g_traj', False):
            traj.storage.storage_dic['psi_g_traj'] = True
            traj.storage.adaptive = traj.storage.adaptive

        list_mu_dot_E1 = list_transition_dipoles @ np.asarray(E_1).ravel()
        list_mu_dot_Esig = list_transition_dipoles @ np.asarray(E_sig).ravel()
        # Response operator: only F[0, k+1] = mu_k . E_sig is nonzero.
        F2_dense = np.zeros((n_site + 1, n_site + 1))
        F2_dense[0, 1:] = list_mu_dot_Esig

        P1_psi_0 = np.zeros(n_site, dtype=np.complex128)
        traj.initialize(P1_psi_0)

        # raise(E_1): |g> -> sum_k mu_k(E_1) |e_k>.
        norm_ratio_1 = _apply_op_and_track_norm(
            traj, lambda: traj.apply_dipole_raise(list_mu_dot_E1), spectroscopy_key,
        )
        traj.propagate(t2, t_step)
        # (I + mu^-) with E_sig: +I preserves the single-ex content
        # of psi(t2); the mu^- part injects the GS amplitude that
        # creates the G/E coherence the detection-phase response
        # measures.
        norm_ratio_2 = _apply_op_and_track_norm(
            traj,
            lambda: traj.apply_dipole_lower_plus_ident(list_mu_dot_Esig),
            spectroscopy_key,
        )
        traj.propagate(t3_max, t_step)

        # Stitch psi_g (slot 0) and the single-excitation amplitudes
        # (slots 1..n_site) into a length-(n_site+1) state vector per
        # timestep.
        psi_traj_slice = np.asarray(traj.storage['psi_traj'])[idx_t2 + 1:]
        gs_amp_traj = np.asarray(traj.storage['psi_g_traj'])[idx_t2 + 1:]
        psi_full_traj = np.column_stack([gs_amp_traj, psi_traj_slice])

        list_norm_sq = np.sum(
            np.conj(psi_full_traj) * psi_full_traj, axis=1,
        ).real
        # psi @ F.T computes F @ psi[t] for all timesteps simultaneously.
        psi_f_traj = psi_full_traj @ F2_dense.T
        list_expectation = np.sum(np.conj(psi_full_traj) * psi_f_traj, axis=1)
        return 4 * _readout_prefactor(
            traj, norm_ratio_1 * norm_ratio_2, list_norm_sq,
        ) * list_expectation

    raise UnsupportedRequest(
        f'unsupported key {_format_dispatch(spectroscopy_key)}',
        'calc_fluorescence_response',
    )
