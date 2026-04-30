import numpy as np

from mesohops.util.physical_constants import precision


class HopsNoiseMemory:
    """
    This class manages the indexing information of the noise memory z_mem.
    For O(1) calculations, z_mem grows to accommodate new modes in the basis.
    When a mode is removed, the mode remains in z_mem until it has decayed below
    precision.

    Key variables:

    1. list_zmemmodeidx_abs:  List of absolute modes in z_mem.  This will include all of the
                              modes in the current mode basis (HOPS.mode) plus modes that are
                              not in the current basis but have not yet decayed below precision.
                              This list is used in HOPS.eom_functions, 'calc_delta_zmem' and 'compress_zmem'.

    2. list_zmemactivemodeidx_rel:  List of mode indices in 'list_zmemmodeidx_abs' corresponding to the
                                    list of modes in the HOPS.mode mode basis 'mode.list_modeidx_abs'.  The indexing
                                    lists 'zmem.list_zmemmodeidx_abs' and 'mode.list_modeidx_abs' are generally not the
                                    same because of modes which are removed from HOPS.mode but not yet decayed to zero.
                                    e.g., If mode.list_modeidx_abs = [4,5,7,8], and zmem.list_zmemmodeidx_abs = [3,4,5,6,7,8], then
                                    list_zmemactivemodeidx_rel = [1,2,4,5].

    Key method:

    1. update_zmem_indexing:  This method adjusts the z_mem indexing arrays based on changes to HOPS.mode.  It is assumed that HOPS.mode has
                              been updated before this method is called.  z_mem itself is passed into this method, because we need to check if
                              there are decayed modes which need to be removed.  A tuple of index lists is also calculated to help reshape z_mem
                              in HOPS.basis where the wave function phi is reshaped.
    """

    __slots__ = (
        # --- Core Basis Components ---
        'mode',                            # HopsMode
        'system',                          # HopsSystem
        # --- Global indexing lists ---
        '__list_g',                        # List of g values (global)
        '__list_w',                        # List of w values (global)
        # --- Zmem indexing lists  ---
        '_list_zmemmodeidx_abs',           # List of absolute Zmem modes
        '_list_zmemactivemodeidx_rel',    # List of relative indices corresponding to mode basis
        '_list_zmemg_abs',                    # List of g values for Zmem
        '_list_zmemw_abs',                 # List of w values for Zmem
    )

    def __init__(self, system, mode):
        """
        Initialize the noise memory indexing manager.

        Parameters
        ----------
        1. system: instance(HopsSystem)

        2. mode: instance(HopsMode)

        Returns
        -------
        None
        """

        self.mode = mode
        self.system = system
        # Absolute indices of z_mem basis modes.
        self._list_zmemmodeidx_abs = []

        # Relative indices of active modes in the z_mem basis.
        # These are indices in `self._list_zmemmodeidx_abs` corresponding to modes that
        # also appear in `mode.list_modeidx_abs`.
        self._list_zmemactivemodeidx_rel = []

    def initialize(self):
        """
        Initialize z_mem indexing and parameter lists from the current mode basis.

        Returns
        -------
        None
        """

        # At initialization, zmem basis is identical to the mode basis,
        # so relative indices are trivially sequential [0, 1, ..., n-1].
        self._list_zmemmodeidx_abs = self.mode.list_modeidx_abs
        self._list_zmemactivemodeidx_rel = list(np.arange(len(self.mode.list_modeidx_abs)))
        self.__list_g = self.system.param['G']
        self.__list_w = self.system.param['W']
        self._list_zmemg_abs = self.mode.list_g
        self._list_zmemw_abs = self.mode.list_w

    def update_zmem_indexing(self, z_mem):
        """
        Update z_mem indexing after changes to the mode basis.

        Modes that are present in the previous z_mem basis but not in the
        current mode basis are removed once their amplitudes have decayed
        below ``precision``. Any newly active modes in ``mode.list_modeidx_abs``
        are added. A mapping between old and new z_mem indices is returned so
        that the caller can remap the underlying z_mem array.

        Parameters
        ----------
        1. z_mem: np.ndarray | list[complex]
                  Noise memory array whose entries correspond
                  to modes indexed by the previous
                  ``list_zmemmodeidx_abs`` list.

        Returns
        -------
        1. map_zmem: tuple[list[int], list[int]]
                     Tuple of index lists
                     ``(list_zmemstblmodeidx_prevrel, list_zmemstblmodeidx_rel)``.
                     The first element contains indices in the
                     old z_mem basis; the second contains the
                     corresponding indices in the new z_mem
                     basis, used to remap z_mem as
                     ``z_mem_new[list_zmemstblmodeidx_rel]
                     = z_mem_old[list_zmemstblmodeidx_prevrel]``.
        """

        # Validate z_mem length matches zmem basis
        if len(z_mem) != len(self._list_zmemmodeidx_abs):
            raise ValueError(
                'HopsNoiseMemory.update_zmem_indexing: '
                f'len(z_mem)={len(z_mem)} != '
                f'len(list_zmemmodeidx_abs)='
                f'{len(self._list_zmemmodeidx_abs)}.'
            )

        # Previous mode indices are stored so that old z_mem can be mapped to new z_mem
        list_zmemmodeidx_prevabs = self._list_zmemmodeidx_abs.copy()
        list_modeidx_abs = self.mode.list_modeidx_abs
        # Identify modes to remove: a mode is truncated only if it has decayed
        # below precision AND is no longer in the active mode basis. Modes that
        # have decayed but are still active must be kept.
        list_truncated_modes = [list_zmemmodeidx_prevabs[i] for i in range(len(z_mem)) if np.abs(z_mem[i]) < precision \
                               and list_zmemmodeidx_prevabs[i] not in list_modeidx_abs]

        # New modes are added to zmem modes
        list_zmemmodeidx_abs = sorted((set(list_zmemmodeidx_prevabs) | set(list_modeidx_abs)) - set(list_truncated_modes))

        # Create mapping between common z_mem modes
        # For example, if
        # old = [1,3,4,5,6,8,9], and
        # new = [0,1,4,6,7,8,9,10], then
        # common_modes = [1,4,6,8,9]. So,
        # indices_old = [0,2,4,5,6], and
        # indices_new = [1,2,3,5,6].  Thus, we can update z_mem via
        # Z1_newzmem[indices_new] = old_zmem[indices_old]
        list_zmemstblmodeidx_abs = sorted(set(list_zmemmodeidx_prevabs) & set(list_zmemmodeidx_abs))
        list_zmemstblmodeidx_prevrel = [list(list_zmemmodeidx_prevabs).index(mode) for mode in list_zmemstblmodeidx_abs]
        list_zmemstblmodeidx_rel = [list(list_zmemmodeidx_abs).index(mode) for mode in list_zmemstblmodeidx_abs]
        map_zmem = (list_zmemstblmodeidx_prevrel, list_zmemstblmodeidx_rel)

        # Map currently active absolute indices to their relative positions in the new basis.
        self._list_zmemactivemodeidx_rel = [list(list_zmemmodeidx_abs).index(mode) for mode in list_modeidx_abs]

        # List of (g,w) pairs for each mode in z_mem
        self._list_zmemg_abs = np.array([self.__list_g[m] for m in list_zmemmodeidx_abs])
        self._list_zmemw_abs = np.array([self.__list_w[m] for m in list_zmemmodeidx_abs])

        self._list_zmemmodeidx_abs = list_zmemmodeidx_abs
        return map_zmem

    def set_zmem_indexing(self, list_zmemmodeidx_abs):
        """
        Set the z_mem mode list and recompute all derived indexing arrays.

        Parameters
        ----------
        1. list_zmemmodeidx_abs : list[int]
                                  Absolute mode indices for the z_mem basis.

        Returns
        -------
        None
        """

        self._list_zmemmodeidx_abs = list(list_zmemmodeidx_abs)
        # Map active mode absolute indices to their relative positions in zmem
        try:
            self._list_zmemactivemodeidx_rel = [
                self._list_zmemmodeidx_abs.index(mode)
                for mode in self.mode.list_modeidx_abs
            ]
        except ValueError as exc:
            missing = set(self.mode.list_modeidx_abs) - set(self._list_zmemmodeidx_abs)
            raise ValueError(
                f'HopsNoiseMemory.set_zmem_indexing: active modes {sorted(missing)} '
                f'are not present in list_zmemmodeidx_abs. Every mode in '
                f'HopsModes.list_modeidx_abs must appear in the z_mem mode list.'
            ) from exc
        self._list_zmemg_abs = np.array(
            [self.__list_g[m] for m in self._list_zmemmodeidx_abs]
        )
        self._list_zmemw_abs = np.array(
            [self.__list_w[m] for m in self._list_zmemmodeidx_abs]
        )

    @property
    def list_zmemmodeidx_abs(self):
        """Absolute mode indices in z_mem."""
        return self._list_zmemmodeidx_abs

    @property
    def list_zmemactivemodeidx_rel(self):
        """Relative indices of active modes in z_mem."""
        return self._list_zmemactivemodeidx_rel

    @property
    def list_zmemg_abs(self):
        """Coupling strengths for z_mem modes. Indexed over list_zmemmodeidx_abs."""
        return self._list_zmemg_abs

    @property
    def list_zmemw_abs(self):
        """Frequencies for z_mem modes. Indexed over list_zmemmodeidx_abs."""
        return self._list_zmemw_abs
