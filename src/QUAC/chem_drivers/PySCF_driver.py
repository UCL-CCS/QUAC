from typing import Optional, Union
import os
from scipy.special import comb
from pyscf import gto, scf, mp, ci, cc, fci, ao2mo, lo, symm
import warnings
import numpy as np
from quac.chem_drivers.DriverBase import BaseChemDriver


class PySCFDriver(BaseChemDriver):
    """Function run PySCF chemistry calc.

    Args:
        geometry (str): Path to .xyz file containing molecular geometry or raw xyz string.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        convergence (float): The convergence tolerance for energy calculations.
        charge (int): Charge of molecular species
        max_ram_memory (int): Amount of RAM memery in MB available for PySCF calculation
        pyscf_print_level (int): Amount of information PySCF prints
        unit (str): molecular geometry unit 'Angstrom' or 'Bohr'
        max_hf_cycles (int): max number of Hartree-Fock iterations allowed
        spin (int): 2S value
        symmetry (str, bool): Point-group symmetry of molecular system (see pyscf for details)
        hf_method (str): Type of Hartree-Fock calulcation, one of the following:
                        restricted (RHF), restricted open-shell (ROHF),
                        unrestriced (UHF) or generalised (GHF) Hartree-Fock.

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
        super().__init__(
            geometry,
            basis_set,
            mo_basis,
            convergence=convergence,
            charge=charge,
            max_ram_memory=max_ram_memory,
            pyscf_print_level=pyscf_print_level,
            unit=unit,
            max_hf_cycles=max_hf_cycles,
            spin=spin,
            symmetry=symmetry,
            xc_functional=xc_functional,
            run_mp2=run_mp2,
            run_cisd=run_cisd,
            run_ccsd=run_ccsd,
            run_fci=run_fci
        )

    def _build_mol(self) -> gto.mole:
        """Function to build PySCF molecule.

        Returns:
            full_mol (gto.mol): built PySCF molecule object
        """
        if os.path.exists(self.geometry):
            # geometry is an xyz file
            full_mol = gto.Mole(
                atom=self.geometry,
                basis=self.basis_set,
                charge=self.charge,
                unit=self.unit,
                spin=self.spin,
                symmetry=self.symmetry
            ).build()
        else:
            # geometry is raw xyz string
            full_mol = gto.Mole(
                atom=self.geometry[3:],
                basis=self.basis_set,
                charge=self.charge,
                unit=self.unit,
                spin=self.spin,
                symmetry=self.symmetry
            ).build()
        return full_mol

    def run_post_hf(self):

        assert self.SCF_obj is not None, 'need to run a SCF calculation to do post Hatree-Fock on'
        assert len(self.mo_basis)<5, f'can only run post Hartree-Fock on standard SCF object not: {self.mo_basis}'

        if self.run_mp2:
            self.pyscf_mp2 = mp.MP2(self.SCF_obj)
            self.pyscf_mp2.verbose = self.pyscf_print_level
            self.pyscf_mp2.run()

        if self.run_cisd:
            self.pyscf_cisd = ci.CISD(self.SCF_obj)
            self.pyscf_cisd.verbose = self.pyscf_print_level
            self.pyscf_cisd.run()
            if self.pyscf_cisd.converged is False:
                warnings.warn("CISD calc not converged")

        if self.run_ccsd:
            self.pyscf_ccsd = cc.CCSD(self.SCF_obj)
            self.pyscf_ccsd.verbose = self.pyscf_print_level
            # self.pyscf_ccsd.diis = False
            self.pyscf_ccsd.max_cycle = self.max_hf_cycles

            self.pyscf_ccsd.run()
            if self.pyscf_ccsd.converged is False:
                warnings.warn("CCSD calc not converged")

        if self.run_fci:
            # check how large calc will be and raise error if too big.
            n_deterimants = comb(2 * self.SCF_obj.mol.nao,
                                 self.n_electron)
            if n_deterimants > 2 ** 25:
                raise NotImplementedError(f'FCI calc too expensive. Number of determinants = {n_deterimants} ')

            self.pyscf_fci = fci.FCI(self.SCF_obj.mol,
                                     self.SCF_obj.mo_coeff)
            self.pyscf_fci.verbose = 0
            self.pyscf_fci.kernel()
            if self.pyscf_fci.converged is False:
                warnings.warn("FCI calc not converged")

    def run_driver(self):

        SCF_methods ={
            'rhf': scf.RHF,
            'rohf': scf.ROHF,
            'uhf': scf.UHF,
            'ghf': scf.GHF,

            'rks': scf.RKS,
            'roks': scf.ROKS,
            'uks': scf.UKS,
            'gks': scf.GKS
        }

        mol_full = self._build_mol()
        self.eri_ao = mol_full.intor('int2e')  # 2e- electron repulsion integrals in AO basis
        self.hcore_ao = scf.hf.get_hcore(mol_full)
        self.S_ao = mol_full.intor('int1e_ovlp')  # 1e- electron overlap

        self.SCF_obj = None
        self.C_matrix = None
        if self.mo_basis in ['rks','roks', 'uks', 'gks']:
            # DFT calculation!
            assert self.xc_functional is not None, 'DFT calculation needs exchange-correlation functional!'

            scf_function = SCF_methods[self.mo_basis]

            self.SCF_obj = scf_function(mol_full)
            self.SCF_obj.xc = self.xc_functional
            self.SCF_obj.verbose = self.pyscf_print_level
            self.SCF_obj.kernel()

            self.SCF_energy = self.SCF_obj.e_tot
            self.SCF_conv = self.SCF_obj.converged
            self.C_matrix = self.SCF_obj.mo_coeff

        elif self.mo_basis in ['rhf', 'rks', 'uhf', 'ghf']:

            scf_function = SCF_methods[self.mo_basis]

            self.SCF_obj = scf_function(mol_full)
            self.SCF_obj.verbose = self.pyscf_print_level
            self.SCF_obj.kernel()

            self.SCF_energy = self.SCF_obj.e_tot
            self.SCF_conv = self.SCF_obj.converged
            self.C_matrix = self.SCF_obj.mo_coeff

        elif self.mo_basis == 'lowdin':
            self.C_matrix = lo.lowdin(self.S_ao)
        elif self.mo_basis == 'schmidt':
            self.C_matrix = lo.schmidt(self.S_ao)
        elif self.mo_basis[:3] == 'iao':
            raise NotImplementedError()
            scf_function = SCF_methods[self.mo_basis[4:]]

            SCF_obj = scf_function(mol_full)
            SCF_obj.verbose = self.pyscf_print_level
            SCF_obj.kernel()
            self.SCF_energy = self.SCF_obj.e_tot
            self.SCF_conv = self.SCF_obj.converged

            ## localize
            C_occ = SCF_obj.mo_coeff[:, SCF_obj.mo_occ > 0]
            C_loc = lo.iao.iao(SCF_obj.mol, C_occ)
            C_temp = SCF_obj.mo_coeff.copy()
            C_temp[:, :C_loc.shape[1]] = C_loc
            self.C_matrix = lo.orth.lowdin(C_temp.T @ self.S_ao @ C_temp)
        elif self.mo_basis[:3] == 'ibo':
            raise NotImplementedError()
            scf_function = SCF_methods[self.mo_basis[4:]]

            SCF_obj = scf_function(mol_full)
            SCF_obj.verbose = self.pyscf_print_level
            SCF_obj.kernel()
            self.SCF_energy = self.SCF_obj.e_tot
            self.SCF_conv = self.SCF_obj.converged

            C_occ = lo.ibo.ibo(SCF_obj.mol, SCF_obj.mo_coeff)
            self.C_matrix = SCF_obj.mo_coeff.copy()
            self.C_matrix[:, :C_occ.shape[1]] = C_occ
        elif self.mo_basis == 'random_unitary':
            raise NotImplementedError()

        elif self.mo_basis[:19] == 'edmiston_ruedenberg':
            scf_function = SCF_methods[self.mo_basis[20:]]

            SCF_obj = scf_function(mol_full)
            SCF_obj.verbose = self.pyscf_print_level
            SCF_obj.kernel()
            self.SCF_energy = SCF_obj.e_tot
            self.SCF_conv = SCF_obj.converged

            ## localize
            self.SCF_obj = lo.EdmistonRuedenberg(SCF_obj.mol,
                                            SCF_obj.mo_coeff)
            self.SCF_obj.kernel()
            self.C_matrix = self.SCF_obj.mo_coeff
        elif self.mo_basis[:4] == 'boys':
            scf_function = SCF_methods[self.mo_basis[5:]]

            SCF_obj = scf_function(mol_full)
            SCF_obj.verbose = self.pyscf_print_level
            SCF_obj.kernel()
            self.SCF_energy = SCF_obj.e_tot
            self.SCF_conv = SCF_obj.converged

            ## localize
            self.SCF_obj = lo.boys.Boys(SCF_obj.mol,
                                            SCF_obj.mo_coeff)
            self.SCF_obj.kernel()

            self.C_matrix = self.SCF_obj.mo_coeff

        elif self.mo_basis[:11] == 'pipek_mezey':
            scf_function = SCF_methods[self.mo_basis[12:]]

            SCF_obj = scf_function(mol_full)
            SCF_obj.verbose = self.pyscf_print_level
            SCF_obj.kernel()

            self.SCF_energy = SCF_obj.e_tot
            self.SCF_conv = SCF_obj.converged

            ## localize
            self.SCF_obj = lo.PipekMezey(SCF_obj.mol,
                                        SCF_obj.mo_coeff)
            self.SCF_obj.kernel()
            self.C_matrix = self.SCF_obj.mo_coeff
        else:
            raise ValueError(f'unknown mo_basis: {self.mo_basis}')


        if isinstance(self.C_matrix, tuple):
            # open shell!
            self.C_alpha, self.C_beta = self.C_matrix
            self.C_matrix = None

            self.calculation_type = 'open_shell'

            # THIS IS THE ONLY CONDITION THAT NEEDS TO BE MET!!!!
            assert (np.allclose((self.C_alpha.conj().T @ self.S_ao @ self.C_alpha),
                                np.eye(self.S_ao.shape[0]))), 'basis is not valid!'
            assert (np.allclose((self.C_beta.conj().T @ self.S_ao @ self.C_beta),
                                np.eye(self.S_ao.shape[0]))), 'basis is not valid!'

            self.MO_energy_alpha, self.MO_energy_beta = self.SCF_obj.mo_energy
            self.hcore_ao = self.SCF_obj.get_hcore()

            self.eri_mo_aaaa_chemist = ao2mo.restore(1, ao2mo.kernel(self.SCF_obj.mol, [self.C_alpha, self.C_alpha,
                                                                                     self.C_alpha, self.C_alpha]),
                                                     self.SCF_obj.mol.nao)
            self.eri_mo_bbbb_chemist = ao2mo.restore(1, ao2mo.kernel(self.SCF_obj.mol, [self.C_beta, self.C_beta,
                                                                                     self.C_beta, self.C_beta]),
                                                     self.SCF_obj.mol.nao)
            self.eri_mo_aabb_chemist = ao2mo.restore(1, ao2mo.kernel(self.SCF_obj.mol, [self.C_alpha, self.C_alpha,
                                                                                     self.C_beta, self.C_beta]),
                                                     self.SCF_obj.mol.nao)

            # self.eri_ao_bbaa_chemist = ao2mo.restore(1, ao2mo.kernel(global_hf.mol, [self.C_beta, self.C_beta,
            #                                                         self.C_alpha, self.C_alpha]),
            #                         global_hf.mol.nao)
            # assert np.allclose(self.eri_ao_aabb_chemist,
            #                    np.einsum('abcd->cdab', self.eri_ao_bbaa_chemist))

        else:
            self.calculation_type = 'closed_shell'

            # THIS IS THE ONLY CONDITION THAT NEEDS TO BE MET!!!!
            assert (np.allclose((self.C_matrix.conj().T @ self.S_ao @ self.C_matrix),
                                np.eye(self.S_ao.shape[0]))), 'basis is not valid!'

            two_body_compressed = ao2mo.kernel(mol_full, self.C_matrix)

            # no permutation symmetry
            self.eri_spatial_mo_chemist = ao2mo.restore(1, two_body_compressed, mol_full.nao)


        self.n_alpha, self.n_beta = mol_full.nelec
        self.n_electron = self.n_alpha + self.n_beta
        self.n_ao = mol_full.nao
        self.multiplicity = mol_full.multiplicity

        ## group theory
        hf_array = np.zeros(self.n_ao)
        hf_array[::2] = np.hstack([np.ones(self.n_alpha), np.zeros(self.n_ao//2- self.n_alpha)])
        hf_array[1::2] = np.hstack([np.ones(self.n_beta), np.zeros(self.n_ao//2- self.n_beta)])
        if self.calculation_type == 'open_shell':
            orbsym_alpha = symm.label_orb_symm(self.SCF_obj.mol,
                                         self.SCF_obj.mol.irrep_name,
                                         self.SCF_obj.mol.symm_orb,
                                         self.C_alpha)

            orbsym_beta = symm.label_orb_symm(self.SCF_obj.mol,
                                         self.SCF_obj.mol.irrep_name,
                                         self.SCF_obj.mol.symm_orb,
                                         self.C_beta)

            labels_alpha = orbsym_alpha[np.where(hf_array[::2])]
            labels_beta  = orbsym_beta[np.where(hf_array[1::2])]

        else:
            orbsym = symm.label_orb_symm(self.SCF_obj.mol,
                                               self.SCF_obj.mol.irrep_name,
                                               self.SCF_obj.mol.symm_orb,
                                               self.C_matrix)

            labels_alpha = orbsym[np.where(hf_array[::2])]
            labels_beta = orbsym[np.where(hf_array[1::2])]


        self.point_group = {'groupname': self.SCF_obj.mol.groupname,
                            'topgroup': self.SCF_obj.mol.topgroup,
                            'labels_alpha': labels_alpha,
                            'labels_beta': labels_beta,
                            'labels_abab': np.dstack((labels_alpha, labels_beta)).flatten()}
        self.hf_array = hf_array
