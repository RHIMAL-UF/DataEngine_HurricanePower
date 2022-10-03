import numpy as np
import pandas as pd
from scipy.io import loadmat
import time
import json
import os
import cv2
import mat73
import matplotlib.pyplot as plt
from scipy import interpolate
from PIL import Image
from tqdm import tqdm
import subprocess
import sys

# Constants
MILE2KM = 0.621371


class RegionalHurricaneTLFailureCal:

    def __init__(self, opfolder, pre_process_fn, pvideo=True, dist=0.2):
        self.opfolder = opfolder  # output folder
        self.pre_process_fn = pre_process_fn  # pre_processing output file name
        self.pvideo = pvideo
        self.dist = dist  # km

    def towerline_preprocessing(self, hurricane_model, powernetwork_model):
        # -----For Hurricane Harvey, this function should not be called due to processing time-----
        # hurricane_model: hurricane model wind profile latitude and longtitude file name
        # powernetwork_model: the buses and station file name
        time_start = time.time()
        # windprofile = loadmat(f'{hurricane_model}.mat')['windProfilelat_lon']
        windprofile = mat73.loadmat(f'{hurricane_model}.mat')[f'{hurricane_model}']
        raw_data = pd.read_excel(powernetwork_model)
        columns_name = np.array(raw_data.columns.tolist())
        columns_need = columns_name[[2, 3, 9, 10, 24, 8, 13, 15]]
        # 'To Longitude', 'From Longitude', 'To Latitude', 'From Latitude', --> position of the line
        # 'Nom kV (Min)',  --> tower type
        # 'Distance Between Substations (km)',  ---> line length in km
        # 'From Number','To Number' ---> substation num
        LongiAndLati_N = raw_data[columns_need].drop_duplicates()
        Nlines = len(LongiAndLati_N)
        rowindex = LongiAndLati_N.index.tolist()
        # --------------------------------------------------------------------------------------------
        # ------------------------  Load the wind profile Geographic information  --------------------
        # --------------------------------------------------------------------------------------------
        Tower_info = {}
        for ii in tqdm(range(len(rowindex)), desc='Tower-Line Pre-Processing Loading...', ascii=False, ncols=150):
            i = rowindex[ii]
            lat1 = LongiAndLati_N['From Latitude'][i]
            lat2 = LongiAndLati_N['To Latitude'][i]
            lon1 = LongiAndLati_N['From Longitude'][i]
            lon2 = LongiAndLati_N['To Longitude'][i]
            linedist = self.earth2cart(lat1, lat2, lon1, lon2)
            Ntower = int(np.floor(linedist / self.dist * MILE2KM))
            if Ntower != 0:
                dlat = (lat2 - lat1) / Ntower
                dlon = (lon2 - lon1) / Ntower
                # calculate the tower coordinate
                Towerlati = np.linspace(LongiAndLati_N['From Latitude'][i] + dlat,
                                        LongiAndLati_N['To Latitude'][i] - dlat, Ntower)
                Towerlong = np.linspace(LongiAndLati_N['From Longitude'][i] + dlon,
                                        LongiAndLati_N['To Longitude'][i] - dlon, Ntower)
                tower_coord = np.transpose(np.array([Towerlati, Towerlong]))
                # local search the latest windprofile to the tower
                endi = tower_coord[0, :]
                endj = tower_coord[-1, :]
                endi_c = int(np.linalg.norm(windprofile - endi, axis=1).argmin())
                endj_c = int(np.linalg.norm(windprofile - endj, axis=1).argmin())
                cindex = int(abs(endj_c - endi_c))
                endmax = max(endi_c - cindex - 100000, endj_c - cindex - 100000, endi_c + cindex - 100000,
                             endj_c + cindex - 10000, 0)
                endmax = min(endmax, windprofile.shape[0])
                endmin = min(endi_c - cindex - 100000, endj_c - cindex - 100000, endi_c + cindex - 100000,
                             endj_c + cindex - 100000, windprofile.shape[0])
                endmin = max(endmin, 0)
                minindex = [int(np.linalg.norm(windprofile[endmin:max(endmax, endmax + 1), :] - Tower1i,
                                               axis=1).argmin() + endmin) for Tower1i in tower_coord]
                Tower_coord = np.transpose(np.array([Towerlong, Towerlati])).tolist()
            else:
                Tower_coord = []
                minindex = []
            Tower_info[i] = {'Number of Towers': Ntower,
                             'Tower coordinate': Tower_coord,
                             'Tower associated wind profile': minindex,
                             'Stations': {'From': str(LongiAndLati_N['From Number'][i]),
                                          'To': str(LongiAndLati_N['To Number'][i])},
                             'Line end coordinate': [LongiAndLati_N['To Longitude'][i],
                                                     LongiAndLati_N['From Longitude'][i],
                                                     LongiAndLati_N['To Latitude'][i],
                                                     LongiAndLati_N['From Latitude'][i]],
                             }

        time_end = time.time()
        print(f"The Pre-Processing time is: {time_end - time_start}sec\n")

        path = os.getcwd()
        folder_name = self.opfolder
        path = os.path.join(path, folder_name)
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/{self.pre_process_fn}.json", 'w') as jf:
            json.dump(Tower_info, jf, indent=4)

    def line_failure_probability_calculation(self, TLFragility, v10_angle, v10_speed, time_interval):
        # -----Calculate the line time dependent failure probability based on the time_interval-----
        start_time = time.time()
        # ----------- Read the Preprocessing file -------------
        path = os.getcwd()
        with open(f"{path}/{self.opfolder}/{self.pre_process_fn}.json", "r") as read_file:
            TLinfo = json.load(read_file)

        # ----------- import the wind fragility curve file -------------
        tower_fc = pd.read_excel(TLFragility, 'Sheet2')
        # interpolate the wind fragility curve using cubic spine C1 continuous
        x = tower_fc["WindSpeed"]
        xs = np.arange(10, 71, 0.1)
        angle_fc = ["angle00", "angle30", "angle45", "angle60", "angle90"]
        tower_fc_cubic = []
        for ang in angle_fc:
            y = tower_fc[ang]
            cs = interpolate.PchipInterpolator(x, y)
            ys = cs(xs)
            tower_fc_cubic = np.append(tower_fc_cubic, ys)
        tower_fc_cubic = np.reshape(tower_fc_cubic, (-1, 610)).T
        wfc = pd.DataFrame({'WindSpeed': xs, "angle00": tower_fc_cubic[:, 0],
                            "angle30": tower_fc_cubic[:, 1], "angle45": tower_fc_cubic[:, 2],
                            "angle60": tower_fc_cubic[:, 3], "angle90": tower_fc_cubic[:, 4]})
        # ----------- import the wind speed and angle -------------
        meanV10 = mat73.loadmat(f'{v10_speed}.mat')[f'{v10_speed}']
        angleV10 = mat73.loadmat(f'{v10_angle}.mat')[f'{v10_angle}']
        timestamp_all = np.array(open("timestamp.txt").read().split())
        if abs(time_interval - 0.5) < 1.0e-4:
            wind_profile_index = list(range(0, len(timestamp_all), 1))
            timestamp = timestamp_all[wind_profile_index]
            string_ts = 'Half'
        elif abs(time_interval - 1.0) < 1.0e-4:
            wind_profile_index = list(range(0, len(timestamp_all), 2))
            timestamp = timestamp_all[wind_profile_index]
            string_ts = 'One'
        elif abs(time_interval - 1.5) < 1.0e-4:
            wind_profile_index = list(range(0, len(timestamp_all), 3))
            timestamp = timestamp_all[wind_profile_index]
            string_ts = 'OneAndHalf'
        elif abs(time_interval - 2.0) < 1.0e-4:
            wind_profile_index = list(range(0, len(timestamp_all), 4))
            timestamp = timestamp_all[wind_profile_index]
            string_ts = 'Two'
        elif abs(time_interval - 2.5) < 1.0e-4:
            wind_profile_index = list(range(0, len(timestamp_all), 5))
            timestamp = timestamp_all[wind_profile_index]
            string_ts = 'TwoAndHalf'
        elif abs(time_interval - 3.0) < 1.0e-4:
            wind_profile_index = list(range(0, len(timestamp_all), 6))
            timestamp = timestamp_all[wind_profile_index]
            string_ts = 'Three'
        else:
            print('Error: the time_interval should be [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] hours')
        LineFP = {k: {} for k in TLinfo.keys()}
        TowerFP = {k: {} for k in TLinfo.keys()}
        # tts = meanV10.shape[0]  # total time stamps
        tts = len(wind_profile_index)
        for i in tqdm(range(tts), desc='Line_Failure_Probability_Calculation Loading...', ascii=False,
                      ncols=150):  # current time stamp
            cur_angle = angleV10[wind_profile_index[i], :]  # current angle for all lines
            cur_speed = meanV10[wind_profile_index[i], :]  # current speed for all lines

            for j in TLinfo.keys():  # for each line --> jth line
                Tfail = []
                if abs(TLinfo[j]["Number of Towers"] - 0) < 1.0e-4:
                    LineFP[j][timestamp[i]] = 0.0
                    TowerFP[j][timestamp[i]] = 0.0
                else:
                    LineCoord = TLinfo[j]['Line end coordinate']
                    if abs(LineCoord[2] - LineCoord[3]) < 1.0e-4:
                        Langle = np.sign(LineCoord[0] - LineCoord[1]) * 90
                    else:
                        Langle = np.arctan((LineCoord[0] - LineCoord[1]) / (LineCoord[2] - LineCoord[3])) * 180 / np.pi
                    Langle = abs(Langle) + 90. if Langle < 0. else Langle
                    for h in TLinfo[j]['Tower associated wind profile']:
                        Tws = cur_speed[h]
                        Twa_wp = np.abs(cur_angle[h]) + 90. if cur_angle[h] < 0. else cur_angle[h]
                        if Tws < 15.0:  # from the table, wind speed < 15 m/s, the tower is safe
                            Tfail.append(0)
                        elif Tws > 70.0:
                            Tfail.append(1.0)  # from the table, wind speed > 70, the tower failure
                        else:
                            Twa = 180. - np.abs(Twa_wp - Langle) if np.abs(Twa_wp - Langle) > 90. else np.abs(
                                Twa_wp - Langle)
                            # map the Twa to the defined angle in FC
                            angle_list = [0, 30, 45, 60, 90]
                            Twa_map = min(angle_list, key=lambda x1: abs(x1 - Twa))
                            # determine the fragility of each tower in current line
                            speed_list = list(np.arange(10, 71, 0.1))
                            Tws_map = min(speed_list, key=lambda x1: abs(x1 - Tws))
                            wfc_col = wfc.columns
                            tempval = wfc[wfc_col[int(angle_list.index(Twa_map) + 1)]][speed_list.index(Tws_map)]
                            Tfail.append(tempval)
                    frg_ti = 1.0 - np.prod(1 - np.asarray(Tfail))
                    frg_td = LineFP[j][timestamp[i - 1]] + frg_ti * (
                            1 - LineFP[j][timestamp[i - 1]]) if i > 0 else frg_ti
                    LineFP[j][timestamp[i]] = frg_td
                    TowerFP[j][timestamp[i]] = Tfail

        print(f'Line_Failure_Probability_Calculation Time: {time.time() - start_time}sec')
        path = os.getcwd()
        folder_name = self.opfolder
        path = os.path.join(path, folder_name)
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/LineFailure_P.json", 'w') as json_file:
            json.dump(LineFP, json_file, indent=4)

    def transmission_line_failure_visu(self, Linefailure, time_interval, boundary):
        # Based on the time interval visualize the line failure
        start_time = time.time()
        boudnary = loadmat(f'{boundary}.mat')['Texas_Boundary']
        plt.rc('figure', max_open_warning=0)
        path = os.getcwd()
        with open(f"{path}/{self.opfolder}/{Linefailure}.json", "r") as read_file:
            LFP = json.load(read_file)
        with open(f"{path}/{self.opfolder}/{self.pre_process_fn}.json", "r") as read_file:
            TLinfo = json.load(read_file)
        timestamp_all = np.array(open("timestamp.txt").read().split())
        if abs(time_interval - 0.5) < 1.0e-4:
            wind_profile_index = list(range(0, len(timestamp_all), 1))
            timestamp = timestamp_all[wind_profile_index]
        elif abs(time_interval - 1.0) < 1.0e-4:
            wind_profile_index = list(range(0, len(timestamp_all), 2))
            timestamp = timestamp_all[wind_profile_index]
        elif abs(time_interval - 1.5) < 1.0e-4:
            wind_profile_index = list(range(0, len(timestamp_all), 3))
            timestamp = timestamp_all[wind_profile_index]
        elif abs(time_interval - 2.0) < 1.0e-4:
            wind_profile_index = list(range(0, len(timestamp_all), 4))
            timestamp = timestamp_all[wind_profile_index]
        elif abs(time_interval - 2.5) < 1.0e-4:
            wind_profile_index = list(range(0, len(timestamp_all), 5))
            timestamp = timestamp_all[wind_profile_index]
        elif abs(time_interval - 3.0) < 1.0e-4:
            wind_profile_index = list(range(0, len(timestamp_all), 6))
            timestamp = timestamp_all[wind_profile_index]
        else:
            print('Error: the time_interval should be [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] hours')

        # deal with the original image and the mapping
        minPx = np.min(boudnary[:, 1])
        maxPx = np.max(boudnary[:, 1])
        minPy = np.min(boudnary[:, 0])
        maxPy = np.max(boudnary[:, 0])
        fimage = plt.imread('Texas_BaseMap.jpg')
        img = Image.open('Texas_BaseMap.jpg').convert("L")
        arr = np.asarray(img)
        vt, hr = fimage.shape[0:2]
        rh = float(hr / (maxPx - minPx))
        rv = float(vt / (maxPy - minPy))
        tts = len(wind_profile_index)
        for i in tqdm(range(tts), desc='Transmission_Line_Failure_Visu Loading...', ascii=False,
                      ncols=150):  # for each time
            plt.figure(i + 1)
            plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
            plt.gray()
            for j in LFP.keys():
                LineEndCoord = TLinfo[j]['Line end coordinate']
                LineFailure = LFP[j][timestamp[i]]
                if (LineFailure - 0.0) < 0.2:
                    clr = np.array([247, 202, 208]) / 255.
                    lw = 1.
                elif (LineFailure - 0.2) < 0.2:
                    clr = np.array([255, 153, 172]) / 255.
                    lw = 1.
                elif (LineFailure - 0.4) < 0.2:
                    clr = np.array([255, 112, 150]) / 255.
                    lw = 1.
                elif (LineFailure - 0.6) < 0.2:
                    clr = np.array([255, 71, 126]) / 255.
                    lw = 1.
                elif (LineFailure - 0.8) < 0.2:
                    clr = np.array([255, 10, 84]) / 255.
                    lw = 1.
                else:
                    clr = np.array([255, 0, 0]) / 255.
                    lw = 3.0
                xlinecoord = rh * (LineEndCoord[:2] - minPx)
                ylinecoord = rv * (maxPy - LineEndCoord[2:])
                plt.plot(xlinecoord, ylinecoord, color=clr.tolist(), linewidth=lw)
            plt.axis('off')
            plt.title(timestamp[i])
            path = os.getcwd()
            folder_name = f'{self.opfolder}'
            path = os.path.join(path, folder_name)
            os.makedirs(path, exist_ok=True)
            plt.savefig(f'{path}/Ldamage_{timestamp[i]}.png', bbox_inches='tight', dpi=1200)
        print(f'Transmission_Line_Failure_Visu Time: {time.time() - start_time}sec')

    def creating_video(self):
        # create video
        start_time = time.time()
        image_folder = f'{os.getcwd()}/{self.opfolder}'
        video_name = f'{os.getcwd()}/{self.opfolder}/InfraS_Damage.avi'
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort()
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, 1, (width, height))
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
        print(f'Creating_VideoTime: {time.time() - start_time}sec')

    def playing_video(self):
        # determining whether or not playing created video
        if self.pvideo:
            if sys.platform == 'win32':
                os.startfile(f'{os.getcwd()}/{self.opfolder}/InfraS_Damage.avi')
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, f'{os.getcwd()}/{self.opfolder}/InfraS_Damage.avi'])

        else:
            print('User has Called Off Playing Line Failure Visualization')

    def earth2cart(self, lat1, lat2, lon1, lon2):
        phi1 = lat1 * np.pi / 180.
        phi2 = lat2 * np.pi / 180.
        d_phi = phi2 - phi1
        d_delta = (lon2 - lon1) * np.pi / 180.
        r = 6371.0
        a = np.sin(d_phi / 2.) * np.sin(d_phi / 2.) + np.cos(phi1) * np.cos(phi2) * np.sin(d_delta / 2.) * np.sin(
            d_delta / 2.)
        c = 2. * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dist = float(r * c)
        return dist
