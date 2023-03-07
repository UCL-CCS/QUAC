from typing import Optional, Union
import warnings
from pathlib import Path

class BaseChemDriver():
    """ Base chemistry driver function

    Args:
        geometry (str): Path to .xyz file containing molecular geometry or raw xyz string.
        basis_set (str): The name of an atomic orbital basis set to use for chemistry calculations.
        mo_basis (str): what molecular orbital basis to use (Hartree-Fock, Lowdwin, DFT ...etc)
        convergence (float): The convergence tolerance for energy calculations.
        charge (int): Charge of molecular species
        max_ram_memory (int): Amount of RAM memery in MB available for PySCF calculation
        pyscf_print_level (int): Amount of information PySCF prints
        unit (str): molecular geometry unit 'Angstrom' or 'Bohr'
        max_hf_cycles (int): max number of Hartree-Fock iterations allowed
        spin (int): 2S value
        symmetry (str, bool): Point-group symmetry of molecular system (see pyscf for details)
        run_mp2 (bool): Whether to run mp2 calc.
        run_cisd (bool): Whether to run cisd calc.
        run_ccsd (bool): Whether to run ccsd calc.
        run_fci (bool): Whether to run fci calc.

    Attributes:

    """

    def __init__(
            self,
            geometry: str,
            basis_set: str,
            mo_basis:str,
            convergence: Optional[float] = 1e-6,
            charge: Optional[int] = 0,
            max_ram_memory: Optional[int] = 4000,
            pyscf_print_level: int = 1,
            unit: Optional[str] = "angstrom",
            max_hf_cycles: int = 50,
            xc_functional: str = None,
            spin: Optional[int] = 0,
            symmetry: Optional[Union[str, bool]] = False,
            run_mp2: Optional[bool] = False,
            run_cisd: Optional[bool] = False,
            run_ccsd: Optional[bool] = False,
            run_fci: Optional[bool] = False
    ):

        if convergence > 1e-2:
            warnings.warn('note scf convergence threshold is fairly high')

        self.geometry = geometry
        self.basis_set = basis_set.lower()
        self.mo_basis = mo_basis.lower()
        self.convergence = convergence
        self.charge = charge
        self.max_ram_memory = max_ram_memory
        self.pyscf_print_level = pyscf_print_level
        self.unit = unit
        self.max_hf_cycles = max_hf_cycles
        self.symmetry = symmetry
        self.spin = spin
        self.xc_functional = xc_functional

        self.run_mp2 = run_mp2
        self.run_cisd = run_cisd
        self.run_ccsd = run_ccsd
        self.run_fci = run_fci

        ### attributes
        # TODO

    def run_driver(self):
        # method to run driver to get matrices to describe Fermionic problem
        pass

    def run_post_hf(self):
        # method to run post hartree-fock methods
        pass

    def save(self, file_path: Path):
        # save details of calculation!
        raise NotImplementedError('')