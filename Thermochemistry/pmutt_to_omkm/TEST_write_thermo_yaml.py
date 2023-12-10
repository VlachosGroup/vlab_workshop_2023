# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:28:27 2023

@author: Gerhard R Wittreich, Ph.D, P.E.
"""

from pmutt.omkm.units import Units

import os
from pathlib import Path
from pmutt.io.excel import read_excel
from pmutt.empirical.references import Reference, References

from pmutt.empirical.nasa import Nasa

import numpy as np
from pmutt.empirical.shomate import Shomate

from pmutt.omkm.reaction import BEP

from pmutt import pmutt_list_to_dict
from pmutt.omkm.reaction import SurfaceReaction

from pmutt.mixture.cov import PiecewiseCovEffect

from pmutt.omkm.phase import IdealGas, InteractingInterface, StoichSolid

from pmutt.io.omkm import write_thermo_yaml

from pmutt.io.omkm import write_cti

from pmutt.io.omkm import write_yaml

units = Units(length='cm', quantity='mol', act_energy='kcal/mol',
              mass='g', energy='kcal')

# Find the location of Jupyter notebook
# Note that normally Python scripts have a __file__ variable but Jupyter
# notebook doesn't.
# Using pathlib can overcome this limiation
notebook_path = Path().resolve()
os.chdir(notebook_path)
input_path = './inputs/NH3_Input_Data.xlsx'

refs_data = read_excel(io=input_path, sheet_name='refs')
refs = [Reference(**ref_data) for ref_data in refs_data]
refs = References(references=refs)
print(refs.offset)

# Lower and higher temperatures
T_low = 298.  # K
T_high = 800.  # K

species_data = read_excel(io=input_path, sheet_name='species')
species = []
species_phases = {}
for ind_species_data in species_data:
    # Initialize NASA from statistical mechanical data
    ind_species = Nasa.from_model(T_low=T_low, T_high=T_high, references=refs,
                                  **ind_species_data)
    species.append(ind_species)

    # Group the species by phase for later use
    try:
        species_phases[ind_species.phase].append(ind_species)
    except KeyError:
        species_phases[ind_species.phase] = [ind_species]

Ar = Shomate(name='Ar', elements={'Ar': 1}, phase='gas', T_low=298.,
             T_high=6000., a=np.array([20.78600, 2.825911e-7, -1.464191e-7,
                                       1.092131e-8, -3.661371e-8, -6.19735,
                                       179.999, 0.]))

species.append(Ar)
species_phases['gas'].append(Ar)

beps_data = read_excel(io=input_path, sheet_name='beps')
beps = []
for bep_data in beps_data:
    beps.append(BEP(**bep_data))

# Combine species and BEPs to make reactions
species_with_beps = species + beps

# Convert species to dictionary for easier reaction assignment
species_with_beps_dict = pmutt_list_to_dict(species_with_beps)
reactions_data = read_excel(io=input_path, sheet_name='reactions')
reactions = []
# Store information about phases for later retrieval
reaction_phases = {}
for reaction_data in reactions_data:
    reaction = SurfaceReaction.from_string(species=species_with_beps_dict,
                                           **reaction_data)
    reactions.append(reaction)
    # Assign phase information
    reaction_species = reaction.get_species(include_TS=True)
    for ind_species in reaction_species:
        try:
            phase = species_with_beps_dict[ind_species].phase
        except AttributeError:
            pass
        # Assign if key already exists
        if phase in reaction_phases:
            if reaction not in reaction_phases[phase]:
                reaction_phases[phase].append(reaction)
        else:
            reaction_phases[phase] = [reaction]

interactions = []
interactions_data = read_excel(io=input_path,
                               sheet_name='lateral_interactions')
interaction_phases = {}
for interaction_data in interactions_data:
    interaction = PiecewiseCovEffect(**interaction_data)
    interactions.append(interaction)

    # Assign phase information
    phase = species_with_beps_dict[interaction.name_i].phase
    # Assign if key already exists
    if phase in interaction_phases:
        if interaction not in interaction_phases[phase]:
            interaction_phases[phase].append(interaction)
    else:
        interaction_phases[phase] = [interaction]

phases_data = read_excel(io=input_path, sheet_name='phases')
phases = []
for phase_data in phases_data:
    # Pre-processing relevant data
    phase_name = phase_data['name']
    phase_type = phase_data.pop('phase_type')
    phase_data['species'] = species_phases[phase_name]

    # Create the appropriate object
    if phase_type == 'IdealGas':
        phase = IdealGas(**phase_data)
    elif phase_type == 'StoichSolid':
        phase = StoichSolid(**phase_data)
    elif phase_type == 'InteractingInterface':
        phase_data['reactions'] = reaction_phases[phase_name]
        phase_data['interactions'] = interaction_phases[phase_name]
        phase = InteractingInterface(**phase_data)
    phases.append(phase)


output_path = './outputs/input.cti'
use_motz_wise = True

write_cti(reactions=reactions, species=species, phases=phases, units=units,
          lateral_interactions=interactions, filename=output_path,
          use_motz_wise=use_motz_wise)

output_path = './outputs/thermo.yaml'
use_motz_wise = True

write_thermo_yaml(reactions=reactions, species=species, phases=phases,
                  units=units, lateral_interactions=interactions,
                  filename=output_path, use_motz_wise=use_motz_wise)

Path('./outputs').mkdir(exist_ok=True)
yaml_path = './outputs/reactor.yaml'
reactor_data = read_excel(io=input_path, sheet_name='reactor')[0]
write_yaml(filename=yaml_path, phases=phases, units=units, **reactor_data)
