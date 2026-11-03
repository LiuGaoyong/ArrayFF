import dataclasses as dc
from tempfile import TemporaryDirectory

import ase.optimize
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.mep import MinModeAtoms
from ase.mep.dimer import DimerTranslate, MinModeTranslate
from ase.mep.neb import NEB, BaseNEB, DyNEB
from ase.optimize.optimize import Dynamics, Optimizer
from ase.units import invcm
from ase.vibrations import Vibrations, VibrationsData

_ERR_MSG = "Optimization failed because of infinite position."
OPTIMIZE_METHODS: dict[str, type[Optimizer]] = {}
for k in ase.optimize.__all__:
    v = getattr(ase.optimize, k)
    if isinstance(v, type) and issubclass(v, Optimizer):
        OPTIMIZE_METHODS[k] = v


@dc.dataclass(frozen=True)
class OptimizationResult:
    lst: list[Atoms]
    converged: bool

    @property
    def latest(self) -> Atoms:
        return self.lst[-1]


def call_vib(
    atoms: Atoms,
    calc: Calculator,
    *,
    ignore_fqmin: float = 10.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Vibration atoms (frequencies in cm^-1 & harmonic modes).

    Parameters
    ----------
    atoms : Atoms
        The atoms to analyze.
    calc : Calculator
        The calculator to use.
    ignore_fqmin : float, optional
        The minimum frequency to ignore. Defaults to 10.0.
        Effect: the element frequency will be set to 1e-5 if the
          absolute value of element frequency is smaller than this value.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The frequencies in cm^-1 and the harmonic modes.
    """
    if calc is not None:
        atoms.calc = calc
    assert atoms.calc is not None, "Please set calculator."
    atoms.calc.reset()

    with TemporaryDirectory() as tmpdir:
        vib = Vibrations(atoms, name=tmpdir)
        vib.run()
        vibdata: VibrationsData = vib.get_vibrations()
    eng, modes = vibdata.get_energies_and_modes(all_atoms=True)
    freq = np.asarray(eng / invcm, dtype=complex)
    freq = np.real(freq) - np.imag(freq)  # complex to real
    freq[np.abs(freq) < abs(ignore_fqmin)] = 1e-5
    return freq, modes


def call_optimization(
    atoms: Atoms,
    calc: Calculator,
    *,
    method: str = "LBFGS",
    max_steps: int = 100,
    fmax: float = 0.01,
) -> OptimizationResult:
    """Optimize the `ase.Atoms`.

    Parameters
    ----------
    atoms : Atoms
        The atoms to optimize.
    calc : Calculator
        The calculator to use.
    method : str, optional
        The method to use. Defaults to "LBFGS".
    max_steps : int, optional
        The maximum number of steps. Defaults to 100.
    fmax : float, optional
        The maximum force. Defaults to 1e-4.

    Returns:
    -------
    OptimizationResult
        The optimized atoms trajectory and whether the optimization converged.
    """
    assert method in OPTIMIZE_METHODS, (
        f"Method {method} not found in `ase.optimize`."
    )
    OPT_CLS: type[Optimizer] = OPTIMIZE_METHODS[method]
    result_lst: list[Atoms] = []
    converged = False

    atoms.calc = calc
    atoms.calc.reset()
    optimizer: Optimizer = OPT_CLS(atoms)
    optimizer.attach(
        _trajectory_record,
        container=result_lst,
        object=optimizer,
        interval=1,
    )
    try:
        for _ in range(max_steps):
            converged = optimizer.run(steps=1, fmax=fmax)
            # Check for infinite/NaN positions (handle both Atoms and NEB)
            if hasattr(atoms, "images"):
                arr = [img.positions for img in atoms.images]  # type: ignore
                positions = np.concatenate(arr)
            else:
                positions = atoms.positions
            if np.any(np.isinf(positions)):
                raise RuntimeError(_ERR_MSG)
            if np.any(np.isnan(positions)):
                raise RuntimeError(_ERR_MSG)
            if converged:
                break
    except RuntimeError as e:
        converged = False
        raise e
    return OptimizationResult(
        converged=converged,
        lst=result_lst,
    )


def _trajectory_record(container: list[Atoms], object: Dynamics) -> None:
    """Record the trajectory of the dynamics object.

    Parameters
    ----------
    container : list[Atoms]
        The container to store the trajectory.
    object : Dynamics
        The dynamics object to record.

    Returns: None
    -------------
    """
    if isinstance(object, (DimerTranslate, MinModeTranslate)):
        d_atoms = object.atoms
        assert isinstance(d_atoms, MinModeAtoms)
        atoms0: Atoms = d_atoms.get_atoms()
        assert isinstance(atoms0.calc, Calculator)
        d_atoms.get_potential_energy()
        target = atoms0.copy()
        target.calc = SPC(target, **atoms0.calc.results)
        target.info["dimer_curvature"] = d_atoms.get_curvature()
        container.append(target)
    elif isinstance(object, Dynamics):
        if isinstance(object.atoms, (NEB, DyNEB, BaseNEB)):
            for atoms in object.atoms.images:
                target = atoms.copy()
                assert isinstance(atoms.calc, Calculator)
                target.calc = SPC(target, **atoms.calc.results)
                container.append(target)
        else:
            target = object.atoms.copy()
            assert isinstance(object.atoms.calc, Calculator)
            target.calc = SPC(target, **object.atoms.calc.results)
            container.append(target)
    else:
        raise ValueError(f"Unknown type {type(object)}.")
