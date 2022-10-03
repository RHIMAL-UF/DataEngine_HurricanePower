# Reginal Hurricane Transmission tower line failure calculation and visualization
import InfraS_Failure_calc
import time
import os
import platform
import subprocess
import sys
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


# time. sleep(10) 


print('################################################################################################################')
print('###############################  Optimal operation of the impacted power system...  ############################')
print('################################################################################################################')

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'Library'), 'share')
os.environ["PROJ_LIB"] = proj_lib

Alfa = input('Line outage probability threshold (choose a value from the set:{0.1,0.5,0.9}: ')
Alfa = float(Alfa)  

path1 = os.getcwd()
folder_name = 'Input_offline'
image_folder = 'Images_ps'
path = os.path.join(path1, folder_name)
os.makedirs(path, exist_ok=True)

path2 = os.path.join(path1, image_folder)
item_lsh = 'LoadShedding'+str(int(10*Alfa))+'.xlsx'
in_ps = os.path.join(path, item_lsh)

PH = Hrcn_SCUC.HurricaneSCUC('Result_ps',image_folder)

if Alfa == 0.1 or Alfa == 0.5 or Alfa == 0.9:
    print(f'Creating images in the {path2} folder! It may take a few minutes...')
    PH.vis_img(in_ps)
    
else:
    print('Alfa value is not in set! Please choose again.')    

with open(f'Line Failure Calculation Summary-{int(resolution)}km-{string_ts}.txt', 'w') as f:
    f.write(f'-------------------------------------------------------------------\n')
    f.write(f'------------------Line Failure Calculation Summary-----------------\n')
    f.write(f'-------------------------------------------------------------------\n')
    f.write('\n')
    f.write('\n')
    f.write(f'>>>>>>>>>>>>>>>>>>>>>>>>>> Input Files >>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    f.write(f'User Specified Wind Profile Resolution: {resolution}\n')
    f.write('\n')
    f.write(f'Wind Speeed Input File:\n'
            f'{v10_speed}.mat\n')
    f.write(f'Wind Angle Input File:\n'
            f'{v10_angle}.mat\n')
    f.write(f'Power System Model:\n'
            f' {powernetwork_model}\n')
    f.write('\n')
    f.write('>>>>>>>>>>>>>>>>>>>>>>>>>>  Output Files >>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    f.write(f'All Output Folder:\n'
            f'{outputfd}\n')
    f.write(f'Pre-Processing File:\n'
            f'{outputfn}.json\n')
    f.write(f'Line Failure File:\n'
            f'{Linefailure}.json\n')
    f.write(f'Visualization Folder:\n'
            f'{string_ts}hour\n')
    f.write('######################################################################\n')
    f.write('####### Optimal operation of the impacted power system Summary #######\n')
    f.write('######################################################################\n')
    f.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>> Input Files >>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    f.write(f'Line outage probability threshold: \n'
            f'{alfa}')
    f.write('>>>>>>>>>>>>>>>>>>>>>>>>>>  Output Files >>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    f.write('Output Folder: \n'
            'Result_ps')
    f.write('Image Folder: \n'
            f'Result_ps\{path2}')
