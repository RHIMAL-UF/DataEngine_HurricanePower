# Reginal Hurricane Transmission tower line failure calculation and visualization
import InfraS_Failure_calc
import time
import os
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
import Hrcn_SCUC
import conda

start_time = time.time()

resolution = float(input('Hurricane Model Spatial Resolution (Choose [1.0, 2.0, 4.0]): '))
if resolution not in [1.0, 2.0, 4.0]:
    print('Warning: the Resolution should be [1.0, 2.0, 4.0] km \n',
          f'------ current input {resolution} km ------', os.linesep,
          f'------ Please Run the File Again ------', os.linesep)
    sys.exit()

time_interval = float(input('Hurricane Model Temporal Resolution (Choose [0.5,1.0,1.5,2.0,2.5,3.0]) :'))
if time_interval not in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    print('Warning: the time_interval should be [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] hours \n',
          f'------ current input {time_interval} hours ------', os.linesep,
          f'------ Please Run the File Again ------', os.linesep)
    sys.exit()
if abs(time_interval - 0.5) < 1.0e-4:
    string_ts = 'Half'
elif abs(time_interval - 1.0) < 1.0e-4:
    string_ts = 'One'
elif abs(time_interval - 1.5) < 1.0e-4:
    string_ts = 'OneAndHalf'
elif abs(time_interval - 2.0) < 1.0e-4:
    string_ts = 'Two'
elif abs(time_interval - 2.5) < 1.0e-4:
    string_ts = 'TwoAndHalf'
elif abs(time_interval - 3.0) < 1.0e-4:
    string_ts = 'Three'
else:
    print('Error: the time_interval should be [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] hours \n',
          f'------current input {time_interval} hours------', os.linesep)
    sys.exit()

outputfd_input = input('Specify the output folder (the hurricane model spatial and temporla resolution ...'
                       'will be added in to the final output folder) :')

if abs(resolution - 1.0) <= 0.1:
    hurricane_model = 'WindProfile_LatLon_R1km'
    v10_speed = 'all_speedV10_R1km'
    v10_angle = 'all_angleV10_R1km'
elif abs(resolution - 2.0) <= 0.1:
    hurricane_model = 'WindProfile_LatLon_R2km'
    v10_speed = 'all_speedV10_R2km'
    v10_angle = 'all_angleV10_R2km'
elif abs(resolution - 4.0) <= 0.1:
    hurricane_model = 'WindProfile_LatLon_R4km'
    v10_speed = 'all_speedV10_R4km'
    v10_angle = 'all_angleV10_R4km'
else:
    print(f'The input resolution is: {resolution} \n,'
          f'Currently support resolution to be [1km, 2km, 4km]')
    sys.exit()

outputfd = f'{outputfd_input}~Resolution- Spatial {resolution}km-Temporal {time_interval} Hour'
outputfn = f'Tower_GIS_R{resolution}km'
RH = InfraS_Failure_calc.RegionalHurricaneTLFailureCal(outputfd, outputfn)
boundary = 'Texas_Boundary'
powernetwork_model = 'InfraS_GIS.xlsx'
TLFragility = 'TL_Fragility.xlsx'
RH.towerline_preprocessing(hurricane_model, powernetwork_model)

RH.line_failure_probability_calculation(TLFragility, v10_angle, v10_speed, time_interval)
Linefailure = f'LineFailure_P'
RH.transmission_line_failure_visu(Linefailure, time_interval, boundary)
RH.creating_video()
RH.playing_video()

print(f'-------------------------------------------------------\n'
      f'Line Failure Calculation Processing Time: {time.time() - start_time}sec')

# time. sleep(10)


print(
    '################################################################################################################')
print(
    '###############################  Optimal operation of the impacted power system...  ############################')
print(
    '################################################################################################################')

# from os import chdir, getcwd

import numpy as np
import pandas as pd
import Hrcn_SCUC
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'Library'), 'share')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import plotly.express as px

Power_system = '2K system_full.xlsx'
Load_factor = 'PS_loadfactor.txt'
TIme0 = np.array(range(0, 72))  # the hours for this network configuration

LnFlrPrblty = Linefailure + '.json'
path1 = os.getcwd()
LineFailureProbability = os.path.join(path1, outputfd, LnFlrPrblty)

path2 = os.path.join(path1, outputfd, 'Result_ps')
path3 = os.path.join(path1, outputfd, 'Images_ps')

Alfa = input('Line outage probability treshold (choose a value between 0 and 1, smaller value is more conservative): ')
alfa = float(Alfa)

PH = Hrcn_SCUC.HurricaneSCUC(path2, path3)

if 0 < alfa < 1:
    Load_Shedding, item_lsh, Summery = PH.Opt_operation(Power_system, Load_factor, LineFailureProbability, TIme0, alfa)
else:
    print('Please check the input data Alfa, the value should be between 0 and 1!')
    sys.exit()

if pd.DataFrame(Load_Shedding).empty:
    print('No load shedding!')
else:
    print(f'Creating images in the {path2} folder! It may take a few minutes...')
    PH.vis_img(item_lsh)

print('###############################  Summery  ###############################')

print('Total cost: ${:.2f}M'.format(Summery[0]['Solution Value'] / 1000000))
print('Simulation time: {:.2f} seconds'.format(Summery[0]['Total Time']))
print('Total load shedding: {:.2f} MW'.format(Summery[0]['Total Load shedding']))

print('################################  Done!  ################################')

