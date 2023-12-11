# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:43:26 2023

@author: Gerhard R Wittreich, Ph.D, P.E.
"""

import numpy as np
import os

from pmutt.empirical.nasa import Nasa
from pmutt.io.thermdat import write_thermdat
from pmutt import parse_formula
from pmutt import constants as c

from pgradd.GroupAdd.Library import GroupLibrary
import pgradd.ThermoChem
lib = GroupLibrary.Load('PtSurface2023')

smiles = ['C(C)C', 'C([Pt])C', '[Pt]C', 'C([Pt])(C)C',
          'C(C[Pt])C', '[Pt]C[Pt]', 'C([Pt])([Pt])C',
          'C(C[Pt])[Pt]', 'C(O[Pt])C', 'O([Pt])C', 'C([Pt])([Pt])(C)C',
          'C(C[Pt])([Pt])C', 'C(CC)([Pt])[Pt]', 'C(CC[Pt])[Pt]',
          'C(CO[Pt])C', 'C(O)C', 'OC', 'C(CO)C', 'C([Pt])([Pt])[Pt]',
          'C([Pt])O', 'O(C[Pt])[Pt]', 'C([Pt])([Pt])([Pt])C',
          'C(C[Pt])([Pt])[Pt]', 'C([Pt])(O)C', 'CC=O',
          'C(C[Pt])O', 'C(O[Pt])C[Pt]', 'C(C[Pt])([Pt])([Pt])C',
          'C(C([Pt])[Pt])([Pt])C', 'C(C[Pt])(C[Pt])[Pt]', 'C(CO)([Pt])C',
          'C(CO[Pt])([Pt])C', 'C(CC)([Pt])([Pt])[Pt]',
          'C(CC[Pt])([Pt])[Pt]', 'C(C([Pt])O)C', 'CCC=O',
          'C(CO)C[Pt]', 'C(CO[Pt])C[Pt]', 'C([Pt])([Pt])([Pt])[Pt]',
          'C([Pt])([Pt])O',
          'C(C[Pt])([Pt])([Pt])[Pt]', 'C([Pt])([Pt])(O)C',
          'C(=O)([Pt])C', 'C(C([Pt])[Pt])([Pt])[Pt]',
          'C(CO)([Pt])[Pt]', 'C(C[Pt])([Pt])O', 'C(CO[Pt])([Pt])[Pt]',
          'O=CC[Pt]', 'C(C([Pt])[Pt])([Pt])([Pt])C',
          'C(C[Pt])(C[Pt])([Pt])[Pt]', 'C(CO)([Pt])([Pt])C',
          'C(CO[Pt])([Pt])([Pt])C', 'C(C([Pt])C)([Pt])([Pt])[Pt]',
          'C(C(C[Pt])[Pt])([Pt])[Pt]', 'C(C([Pt])O)([Pt])C',
          'O=CC([Pt])C', 'C(CO)(C[Pt])[Pt]',
          'C(CO[Pt])(C[Pt])[Pt]', 'C(CC[Pt])([Pt])([Pt])[Pt]',
          'C(CC)([Pt])([Pt])O', 'C(CC)(=O)[Pt]',
          'C(CC([Pt])[Pt])([Pt])[Pt]', 'C(CCO)([Pt])[Pt]',
          'C(CC[Pt])([Pt])O', 'C(CCO[Pt])([Pt])[Pt]',
          'O=CCC[Pt]', 'C([Pt])([Pt])([Pt])O',
          'C(C([Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(CO)([Pt])([Pt])[Pt]', 'C(C[Pt])([Pt])([Pt])O',
          'C(CO[Pt])([Pt])([Pt])[Pt]', 'C(=O)(C[Pt])[Pt]',
          'C(C([Pt])[Pt])([Pt])O', 'O=CC([Pt])[Pt]',
          'C(C([Pt])([Pt])[Pt])([Pt])([Pt])C',
          'C(C([Pt])[Pt])(C[Pt])([Pt])[Pt]',
          'C(C([Pt])O)([Pt])([Pt])C', 'O=CC([Pt])([Pt])C',
          'C(CO)(C[Pt])([Pt])[Pt]', 'C(C[Pt])(CO[Pt])([Pt])[Pt]',
          'C(C(C[Pt])[Pt])([Pt])([Pt])[Pt]', 'C(C([Pt])C)([Pt])([Pt])O',
          '[Pt]C(C)C([Pt])=O',
          'C(C(C([Pt])[Pt])[Pt])([Pt])[Pt]', 'C(C(CO)[Pt])([Pt])[Pt]',
          'C(C(C[Pt])[Pt])([Pt])O', 'C(C(CO[Pt])[Pt])([Pt])[Pt]',
          'O=CC(C[Pt])[Pt]', 'C(CC([Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(CCO)([Pt])([Pt])[Pt]', 'C(CC[Pt])([Pt])([Pt])O',
          'C(CCO[Pt])([Pt])([Pt])[Pt]', 'C(=O)(CC[Pt])[Pt]',
          'C(CC([Pt])[Pt])([Pt])O', 'O=CCC([Pt])[Pt]',
          'C(C([Pt])([Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(C([Pt])O)([Pt])([Pt])[Pt]', 'C(C([Pt])[Pt])([Pt])([Pt])O',
          'O=CC([Pt])([Pt])[Pt]',
          'C(C([Pt])[Pt])(=O)[Pt]',
          'C(C(C[Pt])([Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(C([Pt])([Pt])O)([Pt])([Pt])C',
          'C(C(O[Pt])([Pt])[Pt])([Pt])([Pt])C',
          'C(C([Pt])[Pt])(C([Pt])[Pt])([Pt])[Pt]',
          'C(C([Pt])O)(C[Pt])([Pt])[Pt]', 'C(C([Pt])[Pt])(CO)([Pt])[Pt]',
          'O=CC(C[Pt])([Pt])[Pt]',
          'C(C([Pt])[Pt])(CO[Pt])([Pt])[Pt]',
          'C(C(C([Pt])[Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(C(CO)[Pt])([Pt])([Pt])[Pt]', 'C(C(C[Pt])[Pt])([Pt])([Pt])O',
          'C(C(CO[Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(C(C[Pt])[Pt])(=O)[Pt]',
          'C(C(C([Pt])[Pt])[Pt])([Pt])O',
          'O=CC([Pt])C([Pt])[Pt]',
          'C(CC([Pt])([Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(CC([Pt])O)([Pt])([Pt])[Pt]', 'C(CC([Pt])[Pt])([Pt])([Pt])O',
          'O=CCC([Pt])([Pt])[Pt]',
          'C(=O)(CC([Pt])[Pt])[Pt]',
          'C(C([Pt])([Pt])[Pt])([Pt])([Pt])O',
          'C(C([Pt])([Pt])[Pt])(=O)[Pt]',
          'C(C(C([Pt])[Pt])([Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(C(CO)([Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(C(C[Pt])([Pt])[Pt])([Pt])([Pt])O',
          'C(C(CO[Pt])([Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(C(O[Pt])([Pt])[Pt])(C[Pt])([Pt])[Pt]',
          'C(C([Pt])O)(C([Pt])[Pt])([Pt])[Pt]',
          '[Pt]C([Pt])C([Pt])([Pt])C=O',
          'C(C(C([Pt])([Pt])[Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(C(C([Pt])O)[Pt])([Pt])([Pt])[Pt]',
          'C(C(C([Pt])[Pt])[Pt])([Pt])([Pt])O',
          'O=CC([Pt])C([Pt])([Pt])[Pt]',
          'C(C(C([Pt])[Pt])[Pt])(=O)[Pt]',
          'C(CC([Pt])([Pt])[Pt])([Pt])([Pt])O',
          'C(CC(=O)[Pt])([Pt])([Pt])[Pt]',
          'C(C(C([Pt])([Pt])[Pt])([Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(C(C([Pt])O)([Pt])[Pt])([Pt])([Pt])[Pt]',
          'C(C(C([Pt])[Pt])([Pt])[Pt])([Pt])([Pt])O',
          'O=CC([Pt])([Pt])C([Pt])([Pt])[Pt]',
          'C(C(O[Pt])([Pt])[Pt])(C([Pt])[Pt])([Pt])[Pt]',
          'C(C(C([Pt])([Pt])[Pt])[Pt])([Pt])([Pt])O',
          'C(C(C(=O)([Pt]))[Pt])([Pt])([Pt])[Pt]',
          'C(C(C([Pt])([Pt])[Pt])([Pt])[Pt])([Pt])([Pt])O',
          'C(C(C(=O)[Pt])([Pt])[Pt])([Pt])([Pt])[Pt]', 'C', 'CC']

names = ['CH3CH2CH3(S)', 'CH2CH3(S)', 'CH3(S)', 'CH3CHCH3(S)',
         'CH2CH2CH3(S)', 'CH2(S)', 'CHCH3(S)', 'CH2CH2(S)',
         'CH3CH2O(S)', 'CH3O(S)', 'CH3CCH3(S)', 'CH2CHCH3(S)', 'CHCH2CH3(S)',
         'CH2CH2CH2(S)', 'CH3CH2CH2O(S)', 'CH3CH2OH(S)', 'CH3OH(S)',
         'CH3CH2CH2OH(S)', 'CH(S)', 'CH2OH(S)', 'CH2O(S)',
         'CCH3(S)', 'CHCH2(S)', 'CH3CHOH(S)', 'CH3CHO(S)', 'CH2CH2OH(S)',
         'CH2CH2O(S)', 'CH2CCH3(S)', 'CHCHCH3(S)', 'CH2CHCH2(S)',
         'CH3CHCH2OH(S)', 'CH3CHCH2O(S)', 'CCH2CH3(S)', 'CHCH2CH2(S)',
         'CH3CH2CHOH(S)', 'CH3CH2CHO(S)', 'CH2CH2CH2OH(S)', 'CH2CH2CH2O(S)',
         'C(S)', 'CHOH(S)', 'CCH2(S)', 'CH3COH(S)', 'CH3CO(S)',
         'CHCH(S)', 'CHCH2OH(S)', 'CH2CHOH(S)', 'CHCH2O(S)', 'CH2CHO(S)',
         'CHCCH3(S)', 'CH2CCH2(S)', 'CH3CCH2OH(S)', 'CH3CCH2O(S)',
         'CCHCH3(S)', 'CHCHCH2(S)', 'CH3CHCHOH(S)', 'CH3CHCHO(S)',
         'CH2CHCH2OH(S)', 'CH2CHCH2O(S)', 'CCH2CH2(S)', 'CH3CH2COH(S)',
         'CH3CH2CO(S)', 'CHCH2CH(S)', 'CHCH2CH2OH(S)', 'CH2CH2CHOH(S)',
         'CHCH2CH2O(S)', 'CH2CH2CHO(S)', 'COH(S)', 'CCH(S)',
         'CCH2OH(S)', 'CH2COH(S)', 'CCH2O(S)', 'CH2CO(S)', 'CHCHOH(S)',
         'CHCHO(S)', 'CCCH3(S)', 'CHCCH2(S)', 'CH3CCHOH(S)', 'CH3CCHO(S)',
         'CH2CCH2OH(S)', 'CH2CCH2O(S)', 'CCHCH2(S)', 'CH3CHCOH(S)',
         'CH3CHCO(S)', 'CHCHCH(S)', 'CHCHCH2OH(S)', 'CH2CHCHOH(S)',
         'CHCHCH2O(S)', 'CH2CHCHO(S)', 'CCH2CH(S)', 'CCH2CH2OH(S)',
         'CH2CH2COH(S)', 'CCH2CH2O(S)', 'CH2CH2CO(S)', 'CHCH2CHOH(S)',
         'CHCH2CHO(S)', 'CC(S)', 'CCHOH(S)', 'CHCOH(S)', 'CCHO(S)', 'CHCO(S)',
         'CCCH2(S)', 'CH3CCOH(S)', 'CH3CCO(S)', 'CHCCH(S)', 'CH2CCHOH(S)',
         'CHCCH2OH(S)', 'CH2CCHO(S)', 'CHCCH2O(S)', 'CCHCH(S)', 'CCHCH2OH(S)',
         'CH2CHCOH(S)', 'CCHCH2O(S)', 'CH2CHCO(S)', 'CHCHCHOH(S)',
         'CHCHCHO(S)', 'CCH2C(S)', 'CCH2CHOH(S)', 'CHCH2COH(S)', 'CCH2CHO(S)',
         'CHCH2CO(S)', 'CCOH(S)', 'CCO(S)', 'CCCH(S)', 'CCCH2OH(S)',
         'CH2CCOH(S)', 'CCCH2O(S)', 'CH2CCO(S)', 'CHCCHOH(S)', 'CHCCHO(S)',
         'CCHC(S)', 'CCHCHOH(S)', 'CHCHCOH(S)', 'CCHCHO(S)', 'CHCHCO(S)',
         'CCH2COH(S)', 'CCH2CO(S)', 'CCC(S)', 'CCCHOH(S)', 'CHCCOH(S)',
         'CCCHO(S)', 'CHCCO(S)', 'CCHCOH(S)', 'CCHCO(S)', 'CCCOH(S)',
         'CCCO(S)', 'CH4(S)', 'CH3CH3(S)']

"""
Desired temperature range for NASA polynomials
"""
T = np.linspace(300, 1500, 1201)
"""
Possible mid-point temperatures
"""
T_mid = np.ndarray.tolist(np.linspace(400, 800, 33))

T_ref = c.T0('K')
species = []

for smile, name in zip(smiles, names):
    descriptors = lib.GetDescriptors(smile)
    thermochem = lib.Estimate(descriptors, 'thermochem')
    CpoR = thermochem.get_CpoR(T=T)
    HoRT_ref = thermochem.get_HoRT(T=T_ref)
    SoR_ref = thermochem.get_SoR(T=T_ref)
    phase = 'S'
    elements = parse_formula(name[:name.find('S')])
    species.append(Nasa.from_data(name=name, T=T, CpoR=CpoR,
                                  HoRT_ref=HoRT_ref, SoR_ref=SoR_ref,
                                  T_ref=T_ref, T_mid=T_mid,
                                  elements=elements, phase=phase,
                                  smiles=smile))

base_path = './'
filename = 'thermdat_demo'
filepath = os.path.join(base_path, filename)
write_thermdat(filename=filepath, nasa_species=species, write_date=True)
print(write_thermdat(nasa_species=species, write_date=True))
