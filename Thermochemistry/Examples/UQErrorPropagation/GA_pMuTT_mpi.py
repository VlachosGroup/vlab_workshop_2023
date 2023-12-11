# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:14:00 2018

@author: gerhard
"""

import numpy as np
import os
import yaml
import time
from pgradd.GroupAdd.Library import GroupLibrary
import pgradd.ThermoChem
from pmutt.empirical.nasa import Nasa
from pmutt.io.thermdat import write_thermdat
from pmutt import parse_formula
from pmutt import constants as c
from mpi4py.MPI import COMM_WORLD as M


if M.Get_rank() == 0:
    Itt = 5000   # number of files
    Cores = M.size
    Itt_Core = np.floor(Itt/Cores).astype(int) + 1
    BCastData = []
    for x in range(0, Cores):
        Itt_Data = list(np.linspace(x*Itt_Core, (x+1)*Itt_Core-1,
                                    Itt_Core).astype(int))
        Seed_Data = []
        for y in range(0, Itt_Core):
            Seed_Data.append(int(np.random.rand()*2**32-1))
        BCastData.append([Itt_Data, Seed_Data])
else:
    BCastData = None
data = M.scatter(BCastData, root=0)

lib = GroupLibrary.Load('PtSurface2023')
stram = open('C:/Users/gerhard/Documents/GitHub/PythonGroupAdditivity/pgradd/data/GRWSurface2018/uq.yaml', 'r')
YY = yaml.safe_load(stram)
stram.close()
mat = np.array(YY['UQ']['InvCovMat']['mat'])               # With covarince

groups = YY['UQ']['InvCovMat']['groups']
RMSE_H = float(YY['UQ']['RMSE']['thermochem']['H_ref'].
               split()[0])/c.R('kcal/mol/K')/c.T0('K')
RMSE_S = float(YY['UQ']['RMSE']['thermochem']['S_ref'].
               split()[0])/c.R('cal/mol/K')
Cp_data = YY['UQ']['RMSE']['thermochem']['Cp_data']
RMSE_Cp = []
RMSE_Cp_T = []
for x in Cp_data:
    RMSE_Cp_T.append(float(x[0].split()[0]))
    RMSE_Cp.append(float(x[1].split()[0])/c.R('cal/mol/K')/4.5)
WGS = ['O(S)', 'OH(S)', 'H2O(S)', 'CHO(S)', 'CO(S)',
       'COOH(S)', 'CHOO(S)', 'CO2(S)']

molecules = ['C(C)C', 'C([Pt])C', '[Pt]C', 'C([Pt])(C)C',
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

common = ['CH3CH2CH3(S)', 'CH2CH3(S)', 'CH3(S)', 'CH3CHCH3(S)',
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
size = len(molecules)
Xp = np.zeros([len(molecules), len(groups)])

cp_dist = []
h_dist = []
s_dist = []
g_dist = []
T_analysis = 725.
perturb = 1
offset = 1
per = ['Unperturbed', 'Perturbed']
CP_0 = []
H_0 = []
S_0 = []
t0 = time.time()
T = np.linspace(100, 1500, 1401)
T_mid = np.ndarray.tolist(np.linspace(300, 600, 13))
for x in range(0, len(T_mid)):
    if np.where(T == T_mid[x])[0].size == 0:
        # Insert T_mid's into Ts and save position
        Ts_index = np.where(T > T_mid[x])[0][0]
        T = np.insert(T, Ts_index, T_mid[x])
for mol in molecules:
    index = molecules.index(mol)
    descriptors = lib.GetDescriptors(mol)
    thermochem = lib.Estimate(descriptors, 'thermochem')
    for key, val in descriptors.items():
        grp = groups.index(key)
        Xp[index, grp] = val
    CP_0.append(thermochem.get_CpoR(T))
    H_0.append(thermochem.get_HoRT(c.T0('K')))
    S_0.append(thermochem.get_SoR(c.T0('K')))
Itt_List = data[0]
Seed_List = data[1]
print(RMSE_H, RMSE_S)
for counter, seed in zip(Itt_List, Seed_List):

    sigma_H = RMSE_H**2*(np.eye(len(Xp)) + Xp.dot(mat).dot(Xp.T))  # Original
    sigma_S = RMSE_S**2*(np.eye(len(Xp)) + Xp.dot(mat).dot(Xp.T))  # Original
    seeds = int(np.random.rand()*2**32-1)
    np.random.seed(seeds)
    rand_H = np.random.multivariate_normal(np.zeros(np.size(sigma_H, 0)),
                                           sigma_H)
    np.random.seed(seeds)
    rand_S = np.random.multivariate_normal(np.zeros(np.size(sigma_S, 0)),
                                           sigma_S)
    rand_Cp = []
    sigma_Cp = []
    for x in range(0, len(RMSE_Cp)):
        sigma_Cp.append(RMSE_Cp[x]**2*(np.eye(len(Xp)) +
                        Xp.dot(mat).dot(Xp.T)))
        np.random.seed(seeds)
        rand_Cp.append(np.random.multivariate_normal(np.zeros(
                np.size(sigma_Cp[x], 0)), sigma_Cp[-1]))
    rand_Cp = np.array(rand_Cp).T
    a_Cp_coef = []
    for x in range(0, len(rand_Cp)):
        a_Cp_coef.append(np.polyfit(RMSE_Cp_T, rand_Cp[x], 4, full=True)[0])

    NASA_Input = []

    for mol in molecules:
        index = molecules.index(mol)
        # print(common[index])

        CP = CP_0[index] + np.polyval(a_Cp_coef[index], T)*perturb
        H_ref = H_0[index] + rand_H[index]*perturb
        S_ref = S_0[index] + rand_S[index]*perturb
        isSurface = common[index].find('(S)')
        if isSurface != -1:
            formula = common[index][:isSurface]
            elements = parse_formula(formula)
            phase = 'S'
        else:
            formula = common[index]
            elements = parse_formula(formula)
            phase = 'G'
        NASA_Input.append(Nasa.from_data(name=common[index], T=T, CpoR=CP,
                                         HoRT_ref=H_ref, SoR_ref=S_ref,
                                         T_ref=c.T0('K'), T_mid=T_mid,
                                         elements=elements, phase=phase))

    Base_path = 'C:/Users/gerhard/Documents/Python Scripts/propane2'
    Output = 'thermdats_orig'
    Input = 'ethane'
    if os.path.isdir(os.path.join(Base_path, Output)) is False:
        os.mkdir(os.path.join(Base_path, Output))

    filename = '0000' + str(counter+offset)
    length = len(filename)
    filename = 'thermdat_TC_' + filename[length-4:length]
    filepath = os.path.join(Base_path, Output, filename)
    fid = open(os.path.join(Base_path, 'thermdat_fixed.txt'))
    supp_txt = '! ' + '-'*78 + '\n' +\
               '!  Surface Species via GA [' + per[perturb] + ']\n' +\
               '! ' + '-'*78 + '\n'
    supp_data = fid.read()
    fid.close()
    write_thermdat(nasa_species=NASA_Input, filename=filepath,
                   supp_data=supp_data, supp_txt=supp_txt,
                   write_date=True)
t1 = time.time()
# print(t1-t0)
