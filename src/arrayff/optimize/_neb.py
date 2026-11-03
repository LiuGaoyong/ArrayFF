from copy import deepcopy

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.mep.neb import DyNEB

from ._helper import OptimizationResult, call_optimization


class NEBResult(OptimizationResult):
    nimages: int

    @property
    def latest(self) -> list[Atoms]:  # type: ignore
        """The latest images list."""
        assert self.converged, "Optimization did not converge."
        return self.lst[-abs(self.nimages) :]


def call_neb_raw(
    images: list[Atoms],
    calc: Calculator,
    *,
    climb: bool = True,
    k4spring: float = 0.1,
    dynamic_relaxation: bool = True,
    method4spring: str = "improvedtangent",
    method4interpolate: str | None = None,
    method4opt: str = "FIRE",
    max_steps: int = 100,
    fmax: float = 0.05,
) -> NEBResult:
    """Call NEB method to search transition state.

    Parameters
    ----------
    images : list[Atoms]
        The images to optimize.
    calc : Calculator
        The calculator to use.
    climb : bool, optional
        Whether to climb. Defaults to True.
    k4spring : float, optional
        The spring constant. Defaults to 0.1.
    dynamic_relaxation : bool, optional
        Whether to use dynamic relaxation. Defaults to True.
        True skips images with forces below the convergence criterion.
        This is updated after each force call; if a previously converged
        image goes out of tolerance (due to spring adjustments between
        the image and its neighbors), it will be optimized again.
        False reverts to the default NEB implementation.
    method4spring : str | None, optional
        The method to use for spring optimization. Defaults to None.
        Choice betweeen five methods:

            * aseneb: legacy ase NEB implementation
            * improvedtangent: Paper I NEB implementation (default)
            * eb: Paper III full spring force implementation
            * spline: Paper IV spline interpolation (supports precon)
            * string: Paper IV string method (supports precon)

    method4interpolate : str, optional
        The method to use for interpolation. Defaults to "FIRE".
        Method by which to interpolate: 'linear' or 'idpp'.
            linear provides a standard straight-line interpolation, while
            idpp uses an image-dependent pair potential.


    Paper I:
        G. Henkelman and H. Jonsson, Chem. Phys, 113, 9978 (2000).
        :doi:`10.1063/1.1323224`
    Paper II:
        G. Henkelman, B. P. Uberuaga, and H. Jonsson, Chem. Phys,
        113, 9901 (2000).
        :doi:`10.1063/1.1329672`
    Paper III:
        E. L. Kolsbjerg, M. N. Groves, and B. Hammer, J. Chem. Phys,
        145, 094107 (2016)
        :doi:`10.1063/1.4961868`
    Paper IV:
        S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
        150, 094109 (2019)
        https://dx.doi.org/10.1063/1.5064465


    Returns:
    -------
    NEBResult
        The optimized images trajectory and whether the optimization converged.
    """
    for at in images:
        at.calc = deepcopy(calc)
        at.calc.reset()
        at.get_potential_energy()
    neb = DyNEB(
        images,
        method=method4spring,
        dynamic_relaxation=dynamic_relaxation,
        climb=climb,
        k=k4spring,
    )
    if method4interpolate is not None:
        neb.interpolate(method=method4interpolate)
    return call_optimization(
        neb,  # type: ignore
        calc,
        method=method4opt,
        max_steps=max_steps,
        fmax=fmax,
    )


def call_neb(
    atoms: Atoms,
    calc: Calculator,
    final_atoms: Atoms,
    *,
    nimages: int = 5,
    climb: bool = True,
    k4spring: float = 0.1,
    dynamic_relaxation: bool = True,
    method4spring: str = "improvedtangent",
    method4interpolate: str = "linear",
    method4opt: str = "FIRE",
    max_steps: int = 100,
    fmax: float = 0.05,
) -> NEBResult:
    """Call NEB method to search transition state.

    Parameters
    ----------
    atoms : Atoms
        The atoms to optimize.
    calc : Calculator
        The calculator to use.
    final_atoms : Atoms
        The final atoms to optimize.
    nimages : int, optional
        The number of images. Defaults to 5.
    climb : bool, optional
        Whether to climb. Defaults to True.
    k4spring : float, optional
        The spring constant. Defaults to 0.1.
    dynamic_relaxation : bool, optional
        Whether to use dynamic relaxation. Defaults to True.
        True skips images with forces below the convergence criterion.
        This is updated after each force call; if a previously converged
        image goes out of tolerance (due to spring adjustments between
        the image and its neighbors), it will be optimized again.
        False reverts to the default NEB implementation.
    method4spring : str | None, optional
        The method to use for spring optimization. Defaults to None.
        Choice betweeen five methods:

            * aseneb: legacy ase NEB implementation
            * improvedtangent: Paper I NEB implementation (default)
            * eb: Paper III full spring force implementation
            * spline: Paper IV spline interpolation (supports precon)
            * string: Paper IV string method (supports precon)

    method4interpolate : str, optional
        The method to use for interpolation. Defaults to "FIRE".
        Method by which to interpolate: 'linear' or 'idpp'.
            linear provides a standard straight-line interpolation, while
            idpp uses an image-dependent pair potential.


    Paper I:
        G. Henkelman and H. Jonsson, Chem. Phys, 113, 9978 (2000).
        :doi:`10.1063/1.1323224`

    Paper II:
        G. Henkelman, B. P. Uberuaga, and H. Jonsson, Chem. Phys,
        113, 9901 (2000).
        :doi:`10.1063/1.1329672`

    Paper III:
        E. L. Kolsbjerg, M. N. Groves, and B. Hammer, J. Chem. Phys,
        145, 094107 (2016)
        :doi:`10.1063/1.4961868`

    Paper IV:
        S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
        150, 094109 (2019)
        https://dx.doi.org/10.1063/1.5064465


    Returns:
    -------
    tuple[list[Atoms], bool]
        The optimized atoms trajectory and whether the optimization converged.
    """
    images: list[Atoms] = []
    assert nimages >= 3, "nimages must be at least 3."
    for k, at in zip(["first", "final"], [atoms, final_atoms]):
        res = call_optimization(
            at,
            calc,
            method=method4opt,
            max_steps=max_steps,
            fmax=fmax,
        )
        if not res.converged:
            raise RuntimeError(f"Optimization did not converge for {k} image.")
        images.append(res.latest)
    assert len(images) == 2
    for _ in range(nimages - 2):
        target = images[0].copy()
        images.insert(0, target)
    assert len(images) == nimages
    return call_neb_raw(
        images=images,
        calc=calc,
        climb=climb,
        method4spring=method4spring,
        dynamic_relaxation=dynamic_relaxation,
        method4interpolate=method4interpolate,
        method4opt=method4opt,
        max_steps=max_steps,
        k4spring=k4spring,
        fmax=fmax,
    )
