import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.ticker import FormatStrFormatter
from pmutt import constants as c
from pmutt.io.excel import read_excel
from pmutt.empirical.nasa import Nasa
from pmutt.empirical.references import References, Reference
from pmutt.io.thermdat import write_thermdat, read_thermdat
from pmutt.reaction import Reaction
from pmutt.statmech import StatMech
from scipy.stats import linregress
from pmutt import pmutt_list_to_dict
from itertools import combinations
import pickle


with open('./surf_NH3_S2.inp', 'r') as fid:
    num_rxns = len(fid.read().split('\n'))

Rxn_EA = np.zeros([num_rxns, 3])
Rxn_dH = np.zeros([num_rxns, 3])
Rxn_A = np.zeros([num_rxns, 3])
Species_H = np.zeros([3, 30, 1])
Species_BE = np.zeros([3, 30, 1])
Species_BE_P = np.zeros([3, 30, 1])
Species_S = np.zeros([3, 30, 1])
Species_Cp = np.zeros([3, 30, 15])
strain = np.array([-4, 0, 4])

for zz in [0, 1, 2]:
    if zz == 1:
        '''
        User inputs: 0 Strain
        '''
        # Reference information
        refs_path = './NH3_Reference.xlsx'
        # Input information
        species_path = './NH3_Dataset_new_vib_0_strain_S2.xlsx'
        # Surface information
        surfaces_path = './surfaces_S2.xlsx'
        # Output information
        thermdat_path = './thermdat_0strain_S2_dft_new'
        # Thermdat label
        label = 'Ru(0001) 0 % Strain [T & S sites]'
    elif zz == 2:
        '''
        User inputs: +4 Strain
        '''
        # Reference information
        refs_path = './NH3_Reference.xlsx'
        # Input information
        species_path = './NH3_Dataset_new_vib_+4_strain_S2.xlsx'
        # Surface information
        surfaces_path = './surfaces_S2.xlsx'
        # Output information
        thermdat_path = './thermdat_+4strain_S2_dft_new'
        # Thermdat label
        label = 'Ru(0001) +4 % Strain [T & S sites]'
    elif zz == 0:
        '''
        User inputs: -4 Strain
        '''
        # Reference information
        refs_path = './NH3_Reference.xlsx'
        # Input information
        species_path = './NH3_Dataset_new_vib_-4_strain_S2.xlsx'
        # Surface information
        surfaces_path = './surfaces_S2.xlsx'
        # Output information
        thermdat_path = './thermdat_-4strain_S2_dft_new'
        # Thermdat label
        label = 'Ru(0001) -4 % Strain [T & S sites]'
    else:
        break

    T_low = 100.
    T_high = 1500.  # K

    '''
    Processing References
    '''
    # Import from excel
    refs_input = read_excel(io=refs_path)
    refs = References(references=[Reference(**ref_input) for
                                  ref_input in refs_input])
    print(refs.offset)
    # print('Reference Input:')
    # print(refs.HoRT_element_offset)
    '''
    Processing Surfaces
    '''
    surfaces_data = read_excel(io=surfaces_path)

    '''
    Processing Input Species
    '''
    # Import from excel
    Ts = np.linspace(100, 1500, 1401)
    T_mid = np.linspace(300, 600, 13)
    species_data = read_excel(io=species_path)
    for specie_data in species_data:
        for surface_data in surfaces_data:
            if surface_data['name'] in specie_data['notes']:
                specie_data['potentialenergy'] -= \
                    surface_data['potentialenergy']
                # break
    species = [Nasa.from_model(references=refs, T_low=T_low, T_high=T_high,
               T_mid=500, **specie_data) for specie_data in species_data]
    species_noref = [Nasa.from_model(references=None, T_low=T_low,
                                     T_high=T_high, T_mid=500,
                                     **specie_data)
                     for specie_data in species_data]
    '''
    Printing Out Results
    '''
    temp = np.linspace(T_low, T_high, 15)
    title = '\n\nThermodynamic Parameters at %i %% strain' % \
        ((zz - 1)*4)
    print('\n', title)
    print('-'*(len(title)))
    print('          [kcal/mol] [<--------------------------'
          '------------------------'
          '-----[cal/mol K]-------------------------------------'
          '------------------'
          '--->]')
    print('Name         Hf298     S298   Cp0100  Cp0200  Cp0300  Cp0400  '
          'Cp0500  Cp0600  Cp0700  Cp0800  Cp0900  Cp1000  Cp1100  Cp1200  '
          'Cp1300  Cp1400  Cp1500')
    for x in range(0, len(species)):
        print(f'{species[x].name:11}', end='')
        HoRT = species[x].get_HoRT(298.15)*c.T0('K')*c.R('kcal/mol/K')
        Species_H[zz, x, 0] = species[x].get_H(298.15, units='kcal/mol')
        print(f'{HoRT:-7.2f}    ', end='')
        SoR = species[x].get_SoR(298.15)*c.R('cal/mol/K')
        Species_S[zz, x, 0] = SoR
        print(f'{SoR:-6.3f}  ', end='')
        CpoR = species[x].get_CpoR(temp)*1.987
        Species_Cp[zz, x, 0:15] = CpoR
        for cp in CpoR:
            print(f'{cp:-6.3f}  ', end='')
        print(flush=True)
        for srefs in refs:
            if species_noref[x].name.split('(')[0] == srefs.name[:-5]:
                Species_BE[zz, x, 0] = \
                    species_noref[x].get_G(298.15, units='kcal/mol') -\
                    srefs.model.get_G(T=298.15, units='kcal/mol')
                Species_BE_P[zz, x, 0] = \
                    species_noref[x].get_H(593, units='kcal/mol') -\
                    srefs.model.get_H(T=593, units='kcal/mol')
    '''
    Printing Out Results
    '''
    with open('./thermdat_NH3_gas.txt') as fp:
        supp_data = fp.read()
        supp_data = '! ' + '-'*len(label) + \
            '\n! ' + label + '\n! ' + \
            '-'*len(label) + '\n' + supp_data
    write_thermdat(nasa_species=species, supp_data=supp_data,
                   filename=thermdat_path)
    spec_nasa = read_thermdat(thermdat_path, 'dict')

    species_statmech = [StatMech(**specie_data) for
                        specie_data in species_data]
    species2 = pmutt_list_to_dict(species)
    species_statmech = pmutt_list_to_dict(species_statmech)

    SDTOT = 2.6188e-9
    RATIO_S = 0.02
    SDEN_T = SDTOT*(1-RATIO_S)
    SDEN_S = SDTOT*RATIO_S
    T = 593

    title = 'Kinetic Parameters at %5.2f with %i %% strain' % (T, (zz - 1)*4)
    print('\n', title)
    print('-'*(len(title)+2))
    print('  A(q)      A(S)      Keq      Ea [eV]    dH [eV]       Reaction')
    print('--------  --------  --------  ---------  ----------    '
          '--------------------------')
    with open('./surf_NH3_S2.inp', 'r') as fid:
        rxn = fid.read().split('\n')
        rxn_label = [(rrx.split('=')[0] + '<=>' + rrx.split('=')[2])
                     for rrx in rxn]

    for r in range(np.size(rxn)):
        ob_nasa = Reaction.from_string(reaction_str=rxn[r],
                                       species=spec_nasa)
        ob_stat = Reaction.from_string(reaction_str=rxn[r],
                                       species=species_statmech)
        A1 = ob_stat.get_A(T=T, use_q=True)
        A2 = ob_stat.get_A(T=T, use_q=False)
        A3 = ob_stat.get_q_act(T=T, include_ZPE=False) *\
            c.kb('J/K')*T/c.h('J s')
        K = ob_stat.get_Keq(T=T)
        Ea = ob_stat.get_E_act(T=T, units='eV')
        Rxn_EA[r, zz] = Ea
        delta_H = ob_stat.get_delta_H(T=T, units='eV')
        Rxn_dH[r, zz] = delta_H
        s = 0
        t = 0
        for rr in ob_nasa.reactants:
            if len(rr.name) == len(rr.name.replace('T', '')):
                s += 1
            else:
                t += 1
        if s == 2 or t == 2:
            SDEN = max(s - 1, 0)*SDEN_S + max(t - 1, 0)*SDEN_T
        elif s == 1 and t == 1:
            SDEN = (SDEN_S + SDEN_T)/2
        else:
            SDEN = 0
        # SDEN = (SDEN_S + SDEN_T)/2
        Rxn_A[r, zz] = A1/T/SDEN
        print('%4.2e  %4.2e  %4.2e   %6.3f     %7.3f       %s' %
              (A1/T/SDEN, A2/T/SDEN, K, Ea, delta_H, rxn_label[r]))

title = '\n\nBEP Parameters at %5.2f [kcal/mol]' % (T)
print('\n', title)
print('-'*(len(title)+2))
print(' Slope    Intercept   R-value     Reaction')
print('--------  ---------  ---------    --------------------------')

plt.close('all')

'''----Figure 1----'''

figure, axes = plt.subplots(squeeze=True)
figure.suptitle('BEP Relationships')
for index, (Ea, dH) in enumerate(zip(Rxn_EA, Rxn_dH)):
    graphs = len(Rxn_dH)
    g_rows = 3
    g_coll = 4
    colls =[6, 7, 8, 9]
    for index2, cc in enumerate(colls):
        if cc > graphs:
            colls[index2] = cc-g_coll
    plt.subplot(g_rows, g_coll, index+1)
    plt.plot(dH, Ea, 'o')
    reg = linregress(dH, Ea)
    print('%6.3f     %5.1f      %6.3f      %s' %
          (reg.slope,
           reg.intercept*c.convert_unit(initial='eV/molecule',
                                        final='kcal/mol'),
           reg.rvalue,
           rxn_label[index]))
    plt.plot(dH, dH*reg.slope+reg.intercept, linestyle='--', color='k',
             linewidth=1)

    txt = 'Ea = ' + str(np.round(reg.slope, 3)) +\
          r'$\Delta$H$_{Rxn}$ + ' + str(np.round(reg.intercept, 3))

    lower, upper = plt.ylim()
    plt.text(min(dH), upper-0.22*(upper-lower), txt, fontsize=10)
    if index == 9:
        plt.title('N2(S) + RU(SL) = TS4_N2(S) = N(S) + N(SL)', fontsize=10)
    else:
        plt.title(rxn_label[index], fontsize=10)
    if index in colls:
        plt.xlabel('Heat of Rxn [eV]', fontsize=10)
    if index in [0, 4, 8]:
        plt.ylabel('Ea [eV]', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.2)

'''----Figure 2----'''

figure, axes = plt.subplots(squeeze=True)
figure.suptitle('BEP Relationships')

R_N2 = [0, 4, 8, 9]
R_NH3 = [1, 5]
R_NH2 = [2, 6]
R_NH = [3, 7]
color = ['C0', 'C1', 'C2', 'C3']

plt.subplot(2, 2, 1)
sites = ['Terrace (T)', 'Step (S1)', 'Step (S1)', 'Step (S1)']
for i, x in enumerate(R_N2):
    plt.plot(Rxn_dH[x], Rxn_EA[x], 'o', color=color[i], label=rxn_label[x])
    reg = np.polyfit(Rxn_dH[x], Rxn_EA[x], 1)
    plt.plot(Rxn_dH[x], np.polyval(reg, Rxn_dH[x]), color='k',
             linewidth=1, linestyle='--')
plt.xlim([-2.5, 0.1])
plt.ylim(top=1.8)
plt.title(r'N$_2$ <=> 2N*', fontsize=20)
plt.ylabel('Ea [eV]', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=12)

plt.subplot(2, 2, 2)
sites = ['Terrace (T)', 'Step (S1)', 'Step (S1)', 'Step (S1)']
for i, x in enumerate(R_NH3):
    plt.plot(Rxn_dH[x], Rxn_EA[x], 'o', color=color[i], label=rxn_label[x])
    reg = np.polyfit(Rxn_dH[x], Rxn_EA[x], 1)
    plt.plot(Rxn_dH[x], np.polyval(reg, Rxn_dH[x]), color='k',
             linewidth=1, linestyle='--')
plt.title(r'NH$_3$* <=> NH$_2$* + H*', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=12)

plt.subplot(2, 2, 3)
sites = ['Terrace (T)', 'Step (S1)', 'Step (S1)', 'Step (S1)']
for i, x in enumerate(R_NH2):
    plt.plot(Rxn_dH[x], Rxn_EA[x], 'o', color=color[i], label=rxn_label[x])
    reg = np.polyfit(Rxn_dH[x], Rxn_EA[x], 1)
    plt.plot(Rxn_dH[x], np.polyval(reg, Rxn_dH[x]), color='k',
             linewidth=1, linestyle='--')
plt.ylim(top=1.5)
plt.title(r'NH$_2$* <=> NH* + H*', fontsize=20)
plt.xlabel('Heat of Rxn [eV]', fontsize=18)
plt.ylabel('Ea [eV]', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=12)

plt.subplot(2, 2, 4)
sites = ['Terrace (T)', 'Step (S1)', 'Step (S1)', 'Step (S1)']
for i, x in enumerate(R_NH):
    plt.plot(Rxn_dH[x], Rxn_EA[x], 'o', color=color[i], label=rxn_label[x])
    reg = np.polyfit(Rxn_dH[x], Rxn_EA[x], 1)
    plt.plot(Rxn_dH[x], np.polyval(reg, Rxn_dH[x]), color='k',
             linewidth=1, linestyle='--')
plt.title(r'NH* <=> N* + H*', fontsize=20)
plt.xlabel('Heat of Rxn [eV]', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=12)

plt.subplots_adjust(hspace=0.3)
plt.subplots_adjust(wspace=0.2)

'''----Figure 3----'''

figure, axes = plt.subplots(squeeze=True)
figure.suptitle(r'LSR: Surface Species $\Delta$Enthalpy from 0-Strain DFT')
BEP_H = []
rows = [0, 5, 10, 15, 3, 8, 13, 18]
cols = [15, 16, 17, 18, 19, 20]
for xx in range(21):
    plt.subplot(5, 5, xx+1)
    Species_H1 = Species_H*298.15*1.987e-3
    net_H = Species_H[:, xx, 0] - Species_H[1, xx, 0]
    coef1 = np.polyfit(strain, net_H, 1)
    coef2 = np.polyfit(strain, Species_H[:, xx, 0], 1)
    lr = linregress(strain, net_H)
    # print(net_H, species[xx].name)
    BEP_H.append([species[xx].name, coef1])
    strain2 = np.linspace(-4, 4, 100)
    plt.plot(strain2, np.polyval(coef1, strain2))
    plt.plot(strain, net_H, 'o')
    if xx == 12:
        plt.title('N(SL)', fontsize=10)
    else:
        plt.title(species[xx].name, fontsize=10)
    if xx in cols:
        plt.xlabel('Strain [%]', fontsize=10)
    if xx in rows:
        plt.ylabel(r'$\Delta$Enthalpy''\n[kcal/mol]', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    txt = r'R$^2$ = ' + str(np.round(lr.rvalue**2, 2)) +\
        '\nSlope = ' + str(np.round(lr.slope, 2))
    txt = r'$\Delta$H = ' + str(np.round(lr.slope, 3)) +\
          'St + ' + str(np.round(lr.intercept, 3))
    lower, upper = plt.ylim()
    if lr.slope < 0:
        x_low = -0.5
    else:
        x_low = -4
    plt.text(x_low, upper-0.2*(upper-lower), txt, fontsize=8)
plt.subplots_adjust(hspace=0.7)
plt.subplots_adjust(wspace=0.35)


'''----Figure 4----'''

figure, axes = plt.subplots(squeeze=True)
figure.suptitle('Species Enthalpy vs Strain')

S_N2 = [0, 10]
S_N = [1, 11, 12]
S_H = [2, 13]
S_NH3 = [3, 14]
S_NH2 = [4, 15]
S_NH = [5, 16]
color = ['C0', 'C1', 'C2', 'C3']
plt.subplot(2, 3, 1)
for i, x in enumerate(S_N2):
    plt.plot(strain, Species_H[:, x, 0], 'o', color=color[i],
             label=species[x].name)
    reg = np.polyfit(strain, Species_H[:, x, 0], 1)
    plt.plot(strain2, np.polyval(reg, strain2), color=color[i],
             linewidth=1, linestyle='--')
plt.title(r'N$_2$*', fontsize=20)
plt.ylabel('Enthalpy [kcal/mol]', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=16)

plt.subplot(2, 3, 2)
for i, x in enumerate(S_N):
    plt.plot(strain, Species_H[:, x, 0], 'o', color=color[i],
             label=species[x].name)
    reg = np.polyfit(strain, Species_H[:, x, 0], 1)
    plt.plot(strain2, np.polyval(reg, strain2), color=color[i],
             linewidth=1, linestyle='--')
plt.title('N*', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=16)

plt.subplot(2, 3, 3)
for i, x in enumerate(S_H):
    plt.plot(strain, Species_H[:, x, 0], 'o', color=color[i],
             label=species[x].name)
    reg = np.polyfit(strain, Species_H[:, x, 0], 1)
    plt.plot(strain2, np.polyval(reg, strain2), color=color[i],
             linewidth=1, linestyle='--')
plt.title('H*', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=16)

plt.subplot(2, 3, 4)
for i, x in enumerate(S_NH3):
    plt.plot(strain, Species_H[:, x, 0], 'o', color=color[i],
             label=species[x].name)
    reg = np.polyfit(strain, Species_H[:, x, 0], 1)
    plt.plot(strain2, np.polyval(reg, strain2), color=color[i],
             linewidth=1, linestyle='--')
plt.title(r'NH$_3$*', fontsize=20)
plt.ylabel('Enthalpy [kcal/mol]', fontsize=18)
plt.xlabel('Ru(0001) Surface Strain [%]', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=16)

plt.subplot(2, 3, 5)
for i, x in enumerate(S_NH2):
    plt.plot(strain, Species_H[:, x, 0], 'o', color=color[i],
             label=species[x].name)
    reg = np.polyfit(strain, Species_H[:, x, 0], 1)
    plt.plot(strain2, np.polyval(reg, strain2), color=color[i],
             linewidth=1, linestyle='--')
plt.title(r'NH$_2$*', fontsize=20)
plt.xlabel('Ru(0001) Surface Strain [%]', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=16)

plt.subplot(2, 3, 6)
for i, x in enumerate(S_NH):
    plt.plot(strain, Species_H[:, x, 0], 'o', color=color[i],
             label=species[x].name)
    reg = np.polyfit(strain, Species_H[:, x, 0], 1)
    plt.plot(strain2, np.polyval(reg, strain2), color=color[i],
             linewidth=1, linestyle='--')
plt.title('NH*', fontsize=20)
plt.xlabel('Ru(0001) Surface Strain [%]', fontsize=18)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=16)

plt.subplots_adjust(hspace=0.3)
plt.subplots_adjust(wspace=0.2)


'''----Figure 5----'''

figure, axes = plt.subplots(squeeze=True)
figure.suptitle('Surface Species Binding Energy vs N* Binding Energy',
                fontsize=20)

S_N2 = [0, 10]
S_N = [1, 11, 12]
S_H = [2, 13]
S_NH3 = [3, 14]
S_NH2 = [4, 15]
S_NH = [5, 16]
color = ['C0', 'C1', 'C2', 'C3']
color2 = ['C0', 'C3']
N_bind = [Species_BE[:, 1, 0], Species_BE[:, 1, 0], Species_BE[:, 1, 0],
          (Species_BE[:, 1, 0] + Species_BE[:, 1, 0])/2]
N_bind2 = [Species_BE[:, 1, 0], (Species_BE[:, 1, 0] + Species_BE[:, 1, 0])/2]

aspect = 4.2/3.0

plt.subplot(2, 3, 1)
rdata = []
for i, x in enumerate(S_N2):
    rdata.append(list(z for z in Species_BE[:, x, 0]))
    plt.plot(N_bind2[i], Species_BE[:, x, 0], 'o', color=color2[i],
             label=species[x].name)
    reg = np.polyfit(N_bind2[i], Species_BE[:, x, 0], 1)
    plt.plot(N_bind2[i], np.polyval(reg, N_bind2[i]), color=color2[i],
             linewidth=1, linestyle='--')

plt.title(r'N$_2$*', fontsize=20)
plt.ylabel('Binding Energy [kcal/mol]', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.legend(fontsize=16)

plt.subplot(2, 3, 2)
rdata = []
for i, x in enumerate(S_N):
    if i != 1:
        rdata.append(list(z for z in Species_BE[:, x, 0]))
    plt.plot(N_bind[i], Species_BE[:, x, 0], 'o', color=color[i],
             label=species[x].name)
    reg = np.polyfit(N_bind[i], Species_BE[:, x, 0], 1)
    plt.plot(N_bind[i], np.polyval(reg, N_bind[i]), color=color[i],
             linewidth=1, linestyle='--')

plt.title(r'N*', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.legend(fontsize=16)

plt.subplot(2, 3, 3)
rdata = []
for i, x in enumerate(S_H):
    if i != 1:
        rdata.append(list(z for z in Species_BE[:, x, 0]))
    plt.plot(N_bind[i], Species_BE[:, x, 0], 'o', color=color[i],
             label=species[x].name)
    reg = np.polyfit(N_bind[i], Species_BE[:, x, 0], 1)
    plt.plot(N_bind[i], np.polyval(reg, N_bind[i]), color=color[i],
             linewidth=1, linestyle='--')

plt.title('H*', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.legend(fontsize=16)

plt.subplot(2, 3, 4)
rdata = []
for i, x in enumerate(S_NH3):
    if i != 1:
        rdata.append(list(z for z in Species_BE[:, x, 0]))
    plt.plot(N_bind[i], Species_BE[:, x, 0], 'o', color=color[i],
             label=species[x].name)
    reg = np.polyfit(N_bind[i], Species_BE[:, x, 0], 1)
    plt.plot(N_bind[i], np.polyval(reg, N_bind[i]), color=color[i],
             linewidth=1, linestyle='--')

plt.title(r'NH$_3$*', fontsize=20)
plt.xlabel('N(T)* Binding Energy [kcal/mol]', fontsize=18)
plt.ylabel('Binding Energy [kcal/mol]', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.legend(fontsize=16)

plt.subplot(2, 3, 5)
rdata = []
for i, x in enumerate(S_NH2):
    if i != 1:
        rdata.append(list(z for z in Species_BE[:, x, 0]))
    plt.plot(N_bind[i], Species_BE[:, x, 0], 'o', color=color[i],
             label=species[x].name)
    reg = np.polyfit(N_bind[i], Species_BE[:, x, 0], 1)
    plt.plot(N_bind[i], np.polyval(reg, N_bind[i]), color=color[i],
             linewidth=1, linestyle='--')

plt.title(r'NH$_2$*', fontsize=20)
plt.xlabel('N(T)* Binding Energy [kcal/mol]', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.legend(fontsize=16)

plt.subplot(2, 3, 6)
rdata = []
for i, x in enumerate(S_NH):
    if i != 1:
        rdata.append(list(z for z in Species_BE[:, x, 0]))
    plt.plot(N_bind[i], Species_BE[:, x, 0], 'o', color=color[i],
             label=species[x].name)
    reg = np.polyfit(N_bind[i], Species_BE[:, x, 0], 1)
    plt.plot(N_bind[i], np.polyval(reg, N_bind[i]), color=color[i],
             linewidth=1, linestyle='--')

plt.title('NH*', fontsize=20)
plt.xlabel('N(T)* Binding Energy [kcal/mol]', fontsize=18)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.legend(fontsize=16)

plt.subplots_adjust(hspace=0.3)
plt.subplots_adjust(wspace=0.2)

''' Figure 6 '''

figure, axes = plt.subplots(squeeze=True)
figure.suptitle('Surface Species Binding Energy vs N* Binding Energy',
                fontsize=20)

S_Terrace = [1, 3, 4, 5]
S_Step = [11, 13, 14, 15]
N_bind = Species_BE[:, 1, 0]
aspect = 4.2/3.0

# plt.subplot(2, 2, 1)
rdata = []
terrace_slopes = []
step_slopes = []
for i, x in enumerate(S_Terrace):
    rdata.append(list(z for z in Species_BE[:, x, 0]))
    reg = np.polyfit(N_bind2[0], Species_BE[:, x, 0], 1)
    terrace_slopes.append(reg[0])
    plt.plot(N_bind2[0], np.polyval(reg, N_bind2[0]), color='k',
             linewidth=1, linestyle='--')
    plt.plot(N_bind2[0], Species_BE[:, x, 0], 'o', markeredgecolor='k',
             markerfacecolor='r', markersize=10)
# plt.title(r'Terrace Sites', fontsize=20)
plt.xlabel('N(T) Binding Energy [kcal/mol]', fontsize=18)
plt.ylabel('Binding Energy [kcal/mol]', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.legend(fontsize=16)

# plt.subplot(2, 2, 1)
rdata = []
for i, x in enumerate(S_Step):
    rdata.append(list(z for z in Species_BE[:, x, 0]))
    reg = np.polyfit(N_bind2[0], Species_BE[:, x, 0], 1)
    step_slopes.append(reg[0])
    plt.plot(N_bind2[0], np.polyval(reg, N_bind2[0]), color='k',
             linewidth=1, linestyle='--')
    plt.plot(N_bind2[0], Species_BE[:, x, 0], '^', markeredgecolor='k',
             markerfacecolor='c', markersize=10)

# plt.title(r'Step Sites', fontsize=20)
plt.xlabel('N(T) Binding Energy [kcal/mol]', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
# plt.legend(fontsize=16)

# plt.subplots_adjust(wspace=0.3)

'''----Figure 7----'''

figure, axes = plt.subplots(squeeze=True)
figure.suptitle('LSR: Surface Species Entropy')
BEP_S = []
rows = [0, 5, 10, 15, 3, 8, 13, 18]
cols = [18, 19, 20]
form_top = [1, 7, 8, 9, 10, 11, 12, 14, 19, 20]
for xx in range(21):
    plt.subplot(5, 5, xx+1)
    net_S = Species_S[:, xx, 0] - Species_S[1, xx, 0]
    # print(net_S, species[xx].name)
    coef = np.polyfit(strain, net_S, 2)
    lr = linregress(strain, net_S)
    BEP_S.append([species[xx].name, coef])
    strain2 = np.linspace(-4, 4, 100)
    #plt.plot(strain, strain*lr.slope + lr.intercept)
    plt.plot(strain2, np.polyval(coef, strain2), 'k:')
    plt.plot(strain, net_S, 'o')
    if xx == 12:
        plt.title('N(SL)', fontsize=10)
    else:
        plt.title(species[xx].name, fontsize=10)
    # plt.ylim([0, 20])
    if xx in cols:
        plt.xlabel('Strain [%]', fontsize=10)
    if xx in rows:
        plt.ylabel('S [cal/mol K]', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    txt = r'R$^2$ = ' + str(np.round(lr.rvalue**2, 2)) +\
        '\nSlope = ' + str(np.round(lr.slope, 2))
    txt2 = 'S = ' + str(np.round(coef[0],4)) + 'Strain^2 + ' +\
        str(np.round(coef[1],2)) + 'Strain'
    lower, upper = plt.ylim()
    if lr.slope < 0:
        x_low = 1.5
    else:
        x_low = -4
    if xx in form_top:
        vert = upper-0.2*(upper-lower)
    else:
        vert = lower+0.2*(upper-lower)
    plt.text(-3, vert, txt2, fontsize=8)
plt.subplots_adjust(hspace=0.7)
plt.subplots_adjust(wspace=0.3)


'''----Figure 8----'''

figure, axes = plt.subplots(squeeze=True)
figure.suptitle('LSR: Surface Species Cp(700 K)')
for xx in range(20):
    plt.subplot(5, 4, xx+1)
    net_Cp = Species_Cp[:, xx, 6] - Species_Cp[1, xx, 6]
    coef = np.polyfit(strain, net_Cp, 2)
    lr = linregress(strain, net_Cp)
    plt.plot(strain, strain*lr.slope + lr.intercept)
    plt.plot(strain2, np.polyval(coef, strain2), 'k:')
    plt.plot(strain, net_Cp, 'o')
    # plt.ylim([0, 20])
    plt.title(species[xx].name, fontsize=10)
    if xx >= 16:
        plt.xlabel('Strain [%]', fontsize=10)
    if xx/4 == np.floor(xx/4):
        plt.ylabel(r'$\Delta$CpoR', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    txt = r'R$^2$ = ' + str(np.round(lr.rvalue**2, 2)) +\
        '\nSlope = ' + str(np.round(lr.slope, 2))
    lower, upper = plt.ylim()
    if lr.slope < 0:
        x_low = 1.5
    else:
        x_low = -4
    plt.text(x_low, upper-0.32*(upper-lower), txt, fontsize=8)
plt.subplots_adjust(hspace=0.7)
plt.subplots_adjust(wspace=0.3)
with open('BEP_H.pkl', 'wb') as fid:
    pickle.dump(BEP_H, fid)
with open('BEP_S.pkl', 'wb') as fid:
    pickle.dump(BEP_S, fid)
LSR_A = []
for i, a in enumerate(Rxn_A):
    a = np.round(a/(10**(np.floor(np.log10(a))-2)), 0) *\
        (10**(np.floor(np.log10(a))-2))
    LSR_A.append([rxn[i], np.polyfit(np.array([-4, 0, 4]), a, 2)])
with open('LSR_A.pkl', 'wb') as fid:
    pickle.dump(LSR_A, fid)
with open('Rxn_EA.pkl', 'wb') as fid:
    pickle.dump([rxn, Rxn_EA], fid)
names = []
for x in species_noref:
    names.append(x.name)
with open('Species_BE.pkl', 'wb') as fid:
    pickle.dump([names, Species_BE], fid)

'''----Figure 9----'''
figure, axes = plt.subplots(squeeze=True)
figure.suptitle('Species Binding Energies vs Strain')
width = 0.25
plt.subplot(1, 2, 1)
plt.bar(np.arange(0, 6)-width, np.abs(Species_BE[0, 0:6, 0])/23.06,
        width, label='-4% Strain')
plt.bar(np.arange(0, 6), np.abs(Species_BE[1, 0:6, 0])/23.06,
        width, label='0% Strain')
plt.bar(np.arange(0, 6)+width, np.abs(Species_BE[2, 0:6, 0])/23.06,
        width, label='+4% Strain')
plt.xticks(np.arange(0, 6), labels=[r'N$_2$(T)', 'N(T)', 'H(T)', r'NH$_3$(T)',
                                    r'NH$_2$(T)', 'NH(T)'], fontsize=14,
           rotation=90)
plt.yticks(np.arange(0, 7),
           labels=['0', '-1', '-2', '-3', '-4', '-5', '-6'],
           fontsize=14)
plt.ylabel(r'Binding Energy $\left[eV\right]$', fontsize=16)
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(np.arange(0, 6)-width, np.abs(Species_BE[0, 10:16, 0])/23.06,
        width, label='-4% Strain')
plt.bar(np.arange(0, 6), np.abs(Species_BE[1, 10:16, 0])/23.06,
        width, label='0% Strain')
plt.bar(np.arange(0, 6)+width, np.abs(Species_BE[2, 10:16, 0])/23.06,
        width, label='+4% Strain')
plt.xticks(np.arange(0, 6), labels=[r'N$_2$(S)', 'N(S)', 'H(S)', r'NH$_3$(S)',
                                    r'NH$_2$(S)', 'NH(S)'], fontsize=14,
           rotation=90)
plt.yticks(np.arange(0, 7),
           labels=['0', '-1', '-2', '-3', '-4', '-5', '-6'],
           fontsize=14)
plt.ylabel(r'Binding Energy $\left[eV\right]$', fontsize=16)
plt.legend()

plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(bottom=0.2)

''' Figure 10 '''
if 0:
    # Species_BE_P = Species_BE_P/593/1.9872e-3
    H_max = -160
    for s in combinations(np.arange(0, 6), 2):
        sp = [r'N$_2$(T)', 'N(T)', 'H(T)', r'NH$_3$(T)',
              r'NH$_2$(T)', 'NH(T)']
        figure, axes = plt.subplots()
        coef4 = np.polyfit(Species_BE_P[:, s[0], 0],
                           Species_BE_P[:, s[1], 0], 1)
        plt.plot(np.linspace(0, H_max),
                 np.polyval(coef4, np.linspace(0, H_max)), ':')
        plt.plot(Species_BE_P[:, s[0], 0], Species_BE_P[:, s[1], 0], 'o')
        plt.plot(np.linspace(0, H_max), np.linspace(0, H_max), 'k',
                 linewidth=0.5)
        p = coef4[1]/(1-coef4[0])
        axes.add_patch(patch.Rectangle([p, p], -p, -p,
                                       facecolor='dimgrey',
                                       alpha=0.2, fill=True))
        plt.xlim([H_max, 0])
        plt.ylim([H_max, 0])
        plt.xlabel(sp[s[0]])
        plt.ylabel(sp[s[1]])
        plt.title('Enthalpy of Adsorption @ 593K [kcal/mol]')
    for s in combinations(np.arange(10, 16), 2):
        sp = [r'N$_2$(S)', 'N(S)', 'H(S)', r'NH$_3$(S)',
              r'NH$_2$(S)', 'NH(S)']
        figure, axes = plt.subplots()
        coef4 = np.polyfit(Species_BE_P[:, s[0], 0],
                           Species_BE_P[:, s[1], 0], 1)
        plt.plot(np.linspace(0, H_max),
                 np.polyval(coef4, np.linspace(0, H_max)), ':')
        plt.plot(Species_BE_P[:, s[0], 0], Species_BE_P[:, s[1], 0], 'o')
        plt.plot(np.linspace(0, H_max), np.linspace(0, H_max), 'k',
                 linewidth=0.5)
        p = coef4[1]/(1-coef4[0])
        axes.add_patch(patch.Rectangle([p, p], -p, -p,
                                       facecolor='dimgrey',
                                       alpha=0.2, fill=True))
        plt.xlim([H_max, 0])
        plt.ylim([H_max, 0])
        plt.xlabel(sp[s[0]-10])
        plt.ylabel(sp[s[1]-10])
        plt.title('Enthalpy of Adsorption @ 593K [kcal/mol]')

''' Figure 11 '''
''' Pre-exponential scaling relationships '''

figure, axes = plt.subplots(squeeze=True)
figure.suptitle('Scalaing Relationship: Pre-exponential')
rxn_label = ['N2(T) + RU(T) <=> 2N(T)',
             'NH3(T) + RU(T) <=> H(T) + NH2(T)',
             'NH2(T) + RU(T) <=> H(T) + NH(T)',
             'NH(T) + RU(T) <=> N(T) + H(T)',
             'N2(S) + RU(S) <=> 2N(S)',
             'NH3(S) + RU(S) <=> H(S) + NH2(S)',
             'NH2(S) + RU(S) <=> H(S) + NH(S)',
             'NH(S) + RU(S) <=> N(S) + H(S)',
             'N2(S) + RU(T) <=> N(S) + N(T)',
             'N2(S) + RU(T) <=> N(SL) + N(SU)']
ann_loc = 'UUUUDDDUD'
for i, j in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 9]):
    plt.subplot(3, 3, i+1)
    strain_LSR = np.linspace(-4, 4, 1000)
    ax = plt.plot(strain_LSR, np.polyval(LSR_A[j][1], strain_LSR))
    plt.plot(strain, Rxn_A[j], 'o')
    plt.title(rxn_label[j])
    if ann_loc[i] == 'U':
        loc = [0.1, 0.9]
    else:
        loc = [0.1, 0.1]

    if i == 0 or i == 3 or i == 6:
        plt.ylabel('Pre-exponential [A]')
    if i >= 6:
        plt.xlabel('Strain [%]')
    ann_text = f'{LSR_A[j][1][0]:.2e} Strain$^2$ + {LSR_A[j][1][1]:.2e} Strain + {LSR_A[j][1][2]:.2e}'
    plt.annotate(ann_text, loc, xycoords='axes fraction')
plt.subplots_adjust(wspace=0.1)
plt.subplots_adjust(bottom=0.1)
