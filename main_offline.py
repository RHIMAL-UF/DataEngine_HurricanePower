# Reginal Hurricane Transmission tower line failure calculation and visualization
import InfraS_Failure_calc
import time
import os
import sys

start_time = time.time()

outputfd_input = input('Specify the output folder:')

time_interval = 3.0
string_ts = 'Three'

print(f'The wind field temporal resolution is set at 3 hour')

hurricane_model = 'WindProfile_LatLon_R2km'
v10_speed = 'all_speedV10_R2km'
v10_angle = 'all_angleV10_R2km'

print(f'The wind field spatial resolution is set at 2km')

outputfd = f'{outputfd_input}R2km{time_interval}H'
outputfn = f'Tower_GIS_R2km'
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

print('Line outage probability is set at 0.5; lines with outage probability equal or higher than 50% are assumed as line failures. ')

import os
import sys
import numpy as np
import pandas as pd
import datetime
import Hrcn_SCUC 
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'Library'), 'share')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import plotly.express as px

path1 = os.getcwd()
folder_name = 'Input_offline'
path2 = os.path.join(path1,outputfd, 'Result_ps')
path3 = os.path.join(path1,outputfd, 'Images_ps')
item_lsh = 'LoadShedding.xlsx'

Alfa = 0.5
Alfa = float(Alfa)  

in_ps = os.path.join(path1, item_lsh)

PH = Hrcn_SCUC.HurricaneSCUC(path2,path3)

print('It may take a few minutes...')
PH.vis_img(in_ps)
    
print('Done!')




