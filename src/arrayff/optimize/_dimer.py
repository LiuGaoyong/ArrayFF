from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.constraints import FixAtoms
from ase.mep import DimerControl, MinModeAtoms
from ase.mep.dimer import MinModeTranslate

from ._helper import _ERR_MSG, OptimizationResult, _trajectory_record


class DimerResult(OptimizationResult):
    pass


def call_dimer_displace(
    atoms: Atoms,
    calc: Calculator,
    *,
    parse_mask_from_atoms: bool = True,
    mask: list[bool] | np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Call dimer method to search transition state."""
    res = call_dimer(
        atoms=atoms,
        calc=calc,
        mask=mask,
        parse_mask_from_atoms=parse_mask_from_atoms,
        max_steps=2,
        **kwargs,
    )
    return res.lst[1].positions - res.lst[0].positions


def call_dimer(
    atoms: Atoms,
    calc: Calculator,
    *,
    displacement: np.ndarray | None = None,
    mask: list[bool] | np.ndarray | None = None,
    parse_mask_from_atoms: bool = True,
    max_steps: int = 1000,
    fmax: float = 0.02,
    **kwargs,
) -> DimerResult:
    """Call dimer method to search transition state.

    Parameters
    ----------
    atoms : Atoms
        The atoms to optimize.
    calc : Calculator
        The calculator to use.
    displacement : np.ndarray | None, optional
        The displacement vector. Defaults to None.
    mask : list[bool] | np.ndarray | None, optional
        Which atoms will be moved during displacement. Defaults to None.
    parse_mask_from_atoms : bool, optional
        Whether to parse mask from atoms. Defaults to True.
    max_steps : int, optional
        The maximum number of steps. Defaults to 1000.
    fmax : float, optional
        The maximum force. Defaults to 0.01.
    debug : bool, optional
        Whether to print debug information. Defaults to False.
        If True, the optimization will print log information to stdout.

    Returns:
    -------
    DimerResult
        The optimized atoms trajectory and whether the optimization converged.

    """
    if mask is None and parse_mask_from_atoms:
        fixed_indices = []
        for constr in atoms.constraints:
            if isinstance(constr, FixAtoms):
                fixed_indices.extend(constr.get_indices())
        fixed_indices = list(set(fixed_indices))
        mask = [i not in fixed_indices for i in range(len(atoms))]
    else:
        mask = np.asarray(mask, dtype=bool).tolist()
    if calc is not None:
        atoms.calc = calc
    assert atoms.calc is not None, "Please set calculator."

    param: dict[str, Any] = kwargs.copy()
    param.setdefault("logfile", "-")
    param.update({"mask": mask})
    if displacement is not None:
        param["initial_eigenmode_method"] = "displacement"
        param["displacement_method"] = "vector"
        displacement = np.asarray(displacement, copy=True)

    # record initial energy
    atoms.calc.reset()
    atoms.get_potential_energy()
    target = atoms.copy()
    target.calc = SPC(target, **atoms.calc.results)
    target.info["dimer_curvature"] = np.inf
    data: list[Atoms] = [target]
    converged = False

    # run dimer
    with DimerControl(**param) as d_control:
        d_atoms = MinModeAtoms(atoms, d_control)
        d_atoms.displace(displacement_vector=displacement)
        with MinModeTranslate(d_atoms, logfile=param["logfile"]) as dim_rlx:
            for _ in range(max_steps):
                _trajectory_record(container=data, object=dim_rlx)
                converged = dim_rlx.run(fmax=fmax, steps=1)
                if np.any(np.isinf(data[-1].positions)):
                    raise RuntimeError(_ERR_MSG)
                if np.any(np.isnan(atoms.positions)):
                    raise RuntimeError(_ERR_MSG)
                if converged:
                    _trajectory_record(container=data, object=dim_rlx)
                    break
    return DimerResult(lst=data, converged=converged)
