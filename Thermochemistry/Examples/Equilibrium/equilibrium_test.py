# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:30:58 2023

@author: gerha
"""

from pmutt import equilibrium
import numpy as np
from matplotlib import pyplot as plt

network = {'CH3CH2CH3': 1, 'H2O': 0.5, 'H2': 0, 'CH2CHCH3': 0, 'CH4': 0,
           'CHCH': 0, 'CH2CH2': 0, 'CH3CH3': 0, 'CO2': 0, 'CO': 0}
filepath = './thermdat_equilibrium_unittest.txt'
equil1 = equilibrium.Equilibrium.from_thermdat(filepath, network)
disc = 100
Temp = np.linspace(400, 800, disc)
results = []
for T in Temp:
    sol = equil1.get_net_comp(T=T, P=1.01325)
    results.append(sol.moles)
final_moles = np.reshape(results, [disc, len(network)])

plt.figure(1)
plt.plot(Temp, final_moles)
plt.xlabel('Temperature [K]', fontsize=14)
plt.title('Equilibrium Moles vs Temperature [K]', fontsize=16)
plt.ylabel('Equilibrium Moles', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([min(Temp), max(Temp)])
plt.ylim([0, np.max(final_moles) + 0.5])
plt.legend(sol.species, loc='best', ncol=1)
plt.tight_layout()

plt.figure(2)
mole_total = np.sum(final_moles, 1)
mole_frac = final_moles/mole_total[:, None]
plt.plot(Temp, mole_frac)
plt.xlabel('Temperature [K]', fontsize=14)
plt.title('Equilibrium Mole Fraction vs Temperature [K]', fontsize=16)
plt.ylabel('Equilibrium Mole Fraction', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([min(Temp), max(Temp)])
plt.ylim([0, 1.0])
plt.legend(sol.species, loc='best', ncol=1)
plt.tight_layout()

plt.figure(3)
start = network['CH3CH2CH3']
end = np.array([i[0] for i in final_moles])
conv = (start-end)/start*100
plt.plot(Temp, conv)
plt.xlabel('Temperature [K]', fontsize=14)
plt.title('Equilibrium Conversion vs Temperature [K]', fontsize=16)
plt.ylabel('Equilibrium Conversion CH3CH2CH3 [%]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([min(Temp)-10, max(Temp)+10])
y_min = max(min(conv) - 5, 0.0)
y_max = min(max(conv) + 5, 101.0)
plt.ylim([y_min, y_max])
plt.tight_layout()
