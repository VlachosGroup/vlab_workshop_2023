# coding: utf-8
# Author: Sashank Kasiraju.
# Adapted from petBOA examples.
# Date: January 9th, 2022.

import numpy as np
import os
import time
import shutil
import subprocess
import pandas as pd
import yaml


def edit_thermo_yaml(filename,
                     reaction_id,
                     perturb_percent=0.0,
                     newfile_name="thermo_new.yaml"):
    with open(filename, "r") as stream:
        try:
            yaml_data = yaml.safe_load(stream=stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Identify the reactions
    reactions = yaml_data['reactions']
    reaction = reactions[reaction_id - 1]
    try:
        reaction['sticking-coefficient']['A'] *= (1.0 + (perturb_percent / 100.0))
    except KeyError:
        reaction['rate-constant']['A'] *= (1.0 + (perturb_percent / 100.0))
    # print("Editing the thermo yaml file for reaction {} pre-exp by {} percent".format(reaction_id, perturb_percent))
    with open(newfile_name, "w") as stream:
        yaml.safe_dump(data=yaml_data, stream=stream, sort_keys=False, indent=True)


class OMKM:
    """ A simple omkm executable handler"""
    kind = "mkm executable"

    def __init__(self,
                 exe_path,
                 wd_path,
                 save_folders=True,
                 run_args=("reactor.yaml", "thermo.xml"),
                 **kwargs):
        self.exe_path = exe_path
        self.wd_path = wd_path
        self.save_folders = save_folders
        self.reactor_file = run_args[0]
        self.thermo_file = run_args[1]
        self.process_instance = None
        self.run_number = 0

    def run(self, exp_no):
        os.chdir(self.wd_path)
        if self.save_folders:
            if not os.path.exists("run_" + str(exp_no)):
                os.mkdir("run_" + str(exp_no))
            os.chdir("run_" + str(exp_no))
        else:
            if not os.path.exists("run"):
                os.mkdir("run")
            os.chdir("run")
        shutil.copy(os.path.join(self.wd_path, self.reactor_file), ".")
        shutil.copy(os.path.join(self.wd_path, self.thermo_file), ".")
        _tic = time.perf_counter()
        self.process_instance = subprocess.run(args=[self.exe_path, self.reactor_file, self.thermo_file],
                                               capture_output=True, text=True,
                                               )
        _toc = time.perf_counter()
        print("MKM Run {} Finished in {} seconds".format(self.run_number, _toc - _tic))
        self.run_number += 1

    def loss_function(self, reaction_id,
                      tof_to_fit,
                      perturb_percent=0.0,
                      **kwargs):

        """
        Customized loss function specific to this problem
        """
        thermo_stock_file = kwargs['thermo_file']
        os.chdir(self.wd_path)
        edit_thermo_yaml(filename=thermo_stock_file,
                         reaction_id=reaction_id,
                         perturb_percent=perturb_percent)
        tof_model = np.NAN
        qoi = kwargs['qoi']
        self.run(self.run_number)
        if not self.process_instance.returncode == 0:
            print("Model {} Failed \n {}".format(self.run_number, self.process_instance.stderr))
        else:
            if qoi == 'mass-frac':
                tof_model = pd.read_csv("gas_mass_ss.csv").iloc[-1][tof_to_fit].to_numpy()[0]
            elif qoi == 'rate':
                tof_model = pd.read_csv("gas_sdot_ss.csv").iloc[-1][tof_to_fit].to_numpy()[0]
            else:
                print('Define the QOI correctly')
        os.chdir(self.wd_path)
        return np.abs(tof_model)


def main():
    # This defines the OpenMKM wrapper used to run OpenMKM from python
    omkm_path = "/Users/skasiraj/software/openmkm/bin/omkm"
    cwd = os.getcwd()
    omkm_instance = OMKM(exe_path=omkm_path,
                         wd_path=cwd,
                         save_folders=False,
                         run_args=("reactor.yaml", "thermo_new.yaml")
                         )
    # Define the original thermodynamic specification file
    thermo_stock_file = 'thermo.yaml'

    # This defines the species for which the sensitivities to each reaction step are calculated.
    tof_to_fit = ['CH4']  # rate of this species is used in the LSA calculation.

    # Define the LSA problem.
    LSA_input = pd.read_excel('input.xlsx')

    # Identify reaction for which LSA is performed.
    reaction_ids = LSA_input.where(LSA_input['LSA-Flag'] == True).dropna()['Reaction ID'].values.astype(int)
    total_rxns = len(reaction_ids)
    print("Total number of reactions checked for sensitivity are {}".format(total_rxns))

    # Pre-exponential perturbation percentage
    perturb_percent = 5.0  # in percentage units

    # Perform LSA using finite differences for the input reaction selections.
    # Mass-Frac based
    dlnf_dlnx = np.zeros(total_rxns)
    for i, idi in enumerate(reaction_ids):
        print("LSA for reaction number {} using forward difference".format(idi))
        f_plush = omkm_instance.loss_function(reaction_id=idi,
                                              tof_to_fit=tof_to_fit,
                                              perturb_percent=perturb_percent,
                                              thermo_file=thermo_stock_file,
                                              qoi='mass-frac',
                                              )
        f_x = omkm_instance.loss_function(reaction_id=idi,
                                          tof_to_fit=tof_to_fit,
                                          perturb_percent=0.0,
                                          thermo_file=thermo_stock_file,
                                          qoi='mass-frac',
                                          )
        dolnf = np.log(np.abs(f_plush)) - np.log(np.abs(f_x))
        dolnp = np.log((1.0 + (perturb_percent / 100.0))) - np.log(1.0)
        if np.isinf(dolnf):
            dlnf_dlnx[i] = 0.0
        else:
            dlnf_dlnx[i] = (dolnf / dolnp)

        a.write("Reaction id {} f(x+h) {} f(x) {} doln(f) {} doln(p) {} LSA(i) {} \n".
                format(idi, f_plush, f_x, dolnf, dolnp, dlnf_dlnx[i]))

    # Save the LSA results to the Results.csv file using pandas.
    data = {"LSA-Mass-Frac": pd.Series(data=dlnf_dlnx,
                                       index=reaction_ids, ),
            }
    df = pd.DataFrame(data=data)
    # Rate-based LSA
    dlnf_dlnx = np.zeros(total_rxns)
    for i, idi in enumerate(reaction_ids):
        print("LSA for reaction number {} using forward difference".format(idi))
        f_plush = omkm_instance.loss_function(reaction_id=idi,
                                              tof_to_fit=tof_to_fit,
                                              perturb_percent=perturb_percent,
                                              thermo_file=thermo_stock_file,
                                              qoi='rate',
                                              )
        f_x = omkm_instance.loss_function(reaction_id=idi,
                                          tof_to_fit=tof_to_fit,
                                          perturb_percent=0.0,
                                          thermo_file=thermo_stock_file,
                                          qoi='rate',
                                          )
        dolnf = np.log(np.abs(f_plush)) - np.log(np.abs(f_x))
        dolnp = np.log((1.0 + (perturb_percent / 100.0))) - np.log(1.0)
        if np.isinf(dolnf):
            dlnf_dlnx[i] = 0.0
        else:
            dlnf_dlnx[i] = (dolnf / dolnp)

        a.write("Reaction id {} f(x+h) {} f(x) {} doln(f) {} doln(p) {} LSA(i) {} \n".
                format(idi, f_plush, f_x, dolnf, dolnp, dlnf_dlnx[i]))

    # Save the LSA results to the Results.csv file using pandas.
    df['LSA-rate'] = dlnf_dlnx
    df['Reaction names'] = [LSA_input.iloc[i - 1]['Reaction Equation'] for i in reaction_ids]
    df.index.rename(name="Reaction ID", inplace=True)
    os.chdir(cwd)
    df.to_csv('results-5percent.csv')

    # Clean up the folder
    os.remove('thermo_new.yaml')
    shutil.rmtree('run')


if __name__ == "__main__":
    a = open('debug.log', mode='w')
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"Finished running manual LSA in {toc - tic:0.4f} seconds")
    a.write(f"Finished running manual LSA in {toc - tic:0.4f} seconds")
