
import os
import numpy as np 
import pandas as pd 
import json
import sys
import networkx as nx
import psutil
import datetime
import time
from pyomo.environ import *
import cv2
import plotly.express as px
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'Library'), 'share')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

class HurricaneSCUC:   
    
    def __init__(self, out_f,out_im_f,pvideo=True):
        self.out_f = out_f  # output folder
        self.out_im_f = out_im_f  # output folder
    def clean_data(self,data0):
        xlsload = pd.ExcelFile(data0)
        
        Bus0 = pd.read_excel(xlsload, 'Bus',header=1).sort_values('Number',ignore_index=True).fillna(0)
        Bus1 = Bus0[['Number','Load MW','Gen MW','Zone Num']].to_numpy()
        Bus = np.zeros((len(Bus0),4))
        Key_org2sorted = np.zeros((len(Bus0),2))
        Bus[:,0] = np.arange(len(Bus0))+1
        Bus[:,1:] = Bus1[:,1:]
        Key_org2sorted[:,0] = Bus[:,0]
        Key_org2sorted[:,1] = Bus1[:,0]
        
        Gen0 = pd.read_excel(xlsload, 'Gen',header=1).fillna(0)
    
        Gen1 = Gen0[['Number of Bus','Max MW','Min MW','Gen MW','MWh Price 1','MWh Price 2','MWh Price 3','MWh Price 4','Fixed Cost($/hr)','Start Up Cost','Shut Down Cost','Ramp Rate','Min Up Time','Min Down Time']].to_numpy()
        #                  0             1        2        3           4              5           6               7             8                 9              10             11           12              13
        Gen = np.zeros((len(Gen1),18))
        # [gen num,bus number,pmax,pmin,v1,v2,v3,v4,cost1,cost2,cost3,cost4,startup cost,shutdown cost,no load cost,min up,min down,ramp rate]
        # [0,          1,       2,   3,  4, 5, 6, 7,  8,   9,     10,   11 ,    12,         13,           14,         15,     16,     17     ] 
        Gen[:,0] = np.arange(len(Gen1))+1 # gen number
        for i in range(0,len(Gen1)):
            Gen[i,1] = int(np.where(Gen1[i,0] == Key_org2sorted[:,1])[0]+1) #bus number
        Gen[:,2] = Gen1[:,1]#pmax
        Gen[:,3] = Gen1[:,2]#pmin
        Gen[:,4] = Gen1[:,2] #v1
        Gen[:,5] = (Gen1[:,1] - Gen1[:,2])/3 #v2     
        Gen[:,6] = (Gen1[:,1] - Gen1[:,2])/3 #v3
        Gen[:,7] = (Gen1[:,1] - Gen1[:,2])/3 #v4
        Gen[:,8] = Gen1[:,4] #cost1     
        Gen[:,9] = Gen1[:,5] #cost2
        Gen[:,10] = Gen1[:,6] #cost3  
        Gen[:,11] = Gen1[:,7] #cost4  
        Gen[:,12] = Gen1[:,9] #start up cost 
        Gen[:,13] = Gen1[:,10] #shut down cost
        Gen[:,14] = Gen1[:,8] #noload cost  
        Gen[:,15:17] = Gen1[:,12:14] #min up and down time
        Gen[:,17] = Gen1[:,11] #ramp rate
        
        Line0 = pd.read_excel(xlsload, 'Branch',header=1).fillna(0)
        Line1 = Line0[['From Number','To Number','B','X','Lim MVA A']].to_numpy()
        #                  0             1        2   3      4           
        #data from civil engineering
        
        Brnch0 = pd.read_excel(xlsload, 'Branch state',header=1).fillna(0)
        Brnch1 = Brnch0[['From Number','To Number']].to_numpy()
        
        Brnch2 = np.zeros((len(Brnch1),6))  # [Line num,from bus,to bus,B,X,thermal limit]
        Brnch2[:,0] = np.arange(len(Brnch1))+1 # Line num
        Brnch2[:,1:3] = Brnch1  # from bus,to bus
        #,B,X,thermal limit
        for i in range(0,len(Brnch1)):
            tmp_b = np.where((Brnch1[i,0]==Line1[:,0])&(Brnch1[i,1]==Line1[:,1]))[0][0]
            Brnch2[i,[3,5]] = Line1[tmp_b,[2,4]]
            Brnch2[i,4] = -1/Line1[tmp_b,3]
        
        # [Line num,from bus,to bus,B,X,thermal limit]
        # [   0,      1,       2,   3,4,    5] 
        #Line[:,0] = np.arange(len(Line1))+1 # line number
        Line = np.zeros((len(Line1),6)) 
        Line[:,[0,3,4,5]] = Brnch2[:,[0,3,4,5]]
        for i in range(0,len(Brnch2)):
            Line[i,1] = int(np.where(Brnch2[i,1] == Key_org2sorted[:,1])[0]+1) #from bus number
            Line[i,2] = int(np.where(Brnch2[i,2] == Key_org2sorted[:,1])[0]+1) #to bus number    
            
            
        return Bus, Line, Gen, Key_org2sorted

    def ReadJSN(self,Datafile):
        # just the line numbers out of the lines_failure file
        #Datafile = 'LineFailureProbability_v2_3_hour.json'
        with open(Datafile, 'r') as f:
            d1 = json.load(f)

        r1 = list(d1.keys()) #convert the dictionary keys to list 
        r2 = [] #output
        for i in range(0,len(r1)):
            iy = int(r1[i]) #intermediary
            r2.append(iy)
        
        Ot1 = np.array(r2) #output1: the line numbers as int     
            #the time stamps under study
        ts = np.array(list(d1[r1[0]].keys()))# the time stamps are the same for all of the lines: keys of the first key of original dictionary 

        #the line number as the first entry in the numpy array and the 1:18 are the probability of line outage for the timesteps under study:
        Otg_prb = np.zeros([len(r2),len(ts)+1])
        for i in range(0,len(r2)):
            Otg_prb[i,0] = Ot1[i]
            Otg_prb[i,1:] = list(d1[r1[i]].values())
            
        return Otg_prb,ts

    #data0:Line,data1:Lines_failure probability,data2:time stamps of hurricane probability data,data3:alfa
    def scenario_gen(self, data0,data1,data2='0.9'):

        Alfa = float(data2) #Probability threshold for line to be considered out
        #the first column: line number, the rest of columns the outage probability of line at various time steps
        #input data from civil E group for line failure probability
        
        ln_flr_prb,fin = self.ReadJSN(data1) #'Lines_failure.json', time steps


        time_s = np.zeros((len(fin),3),dtype=int)
        for i in range(0,len(fin)):
            datetime_object = datetime.datetime.strptime(fin[i], "'%m_%d_%H%M'")
            time_s[i,0] = datetime_object.month 
            time_s[i,1] = datetime_object.day 
            time_s[i,2] = datetime_object.hour
        if len(fin)>1:
            smltn_dur = 24*((datetime.datetime.strptime(fin[-1], "'%m_%d_%H%M'")-datetime.datetime.strptime(fin[0], "'%m_%d_%H%M'")).days+1)
        hrrcn_hour = np.zeros(len(time_s),dtype=int) #hours of hurricane probability as time series: 12, 18, 21, 0=24, 3=27, ...
        tm0 = np.unique(time_s[:,1]) # temperary variable for the days of hurricane to help with days in different months  
        for i in range(0,len(time_s)):
            for j in range(0,len(tm0)):
                if time_s[i,1] == tm0[j]:
                    hrrcn_hour[i] = 24*j+time_s[i,2]

        LnSts = np.ones([int(smltn_dur),len(data0)]) #LineStatus

        for l in range(0,len(ln_flr_prb)):
            for t in range(0,len(time_s)): 
                if ln_flr_prb[l,t+1] >= np.abs(Alfa): #if by mistake the alfa values are entered as negative values
                    k1 = int(ln_flr_prb[l,0])#line number
                    k2 = hrrcn_hour[t]
                    LnSts[k2:,k1] = 0  
                    #break                
        return LnSts

    def Island(self,data0,data1,data2,data3):#line data, line_status, Load_factor, bus data
        #data0,data1,data2,data3 = Line,LineStatus,Load_fact,Bus
        T_isl = len(data2) #time duration in which islanding is examined
        num_bus = len(data3) #number of lines of the network
        new_dict = dict.fromkeys([i for i in range(0,T_isl)]) #Graphs are saved as dictionaries, keys: hours starting from 0 
        Base_graph = nx.Graph([(data0[j,1],data0[j,2]) for j in range(0,len(data0))]) #for j in range(0,len(C))
        #for each hour create the network of nodes and edges (bus and line)
        D0 = dict.fromkeys([i for i in range(0,T_isl)]) # graph informations: number of segments formed
        A_temp = np.zeros((T_isl)) 
        
        for i in range(0,T_isl):                
            C = np.delete(data0,np.where(data1[i,:]<1)[0],axis=0)  #line outages as data1 contains only 0 and 1
            new_dict[i] = nx.Graph([(C[j,1],C[j,2]) for j in range(0,len(C))]) 
            D0[i] = Base_graph.nodes-new_dict[i].nodes  # the lone nodes #new_dict[0]          
        for i in range(0,T_isl):   
            A_temp[i] = len(list(nx.connected_components(new_dict[i])))
            
        A2 = A_temp.astype(int)
        temp = np.max(A_temp).astype(int)
        A1 = np.zeros((T_isl,temp))
        for i in range(0,T_isl):
            A3 = A2[i]
            for j in range(0,A3):
                A1[i,j] = len(list(nx.connected_components(new_dict[i]))[j])
        a_idx = np.argsort(-A1)
        A4 = np.take_along_axis(A1, a_idx, axis=1)
        
        U0,U_ind = np.unique(A4,axis=0,return_index=True)
        U_i = np.flip(U_ind) 
        U = np.flipud(U0)
          
        return new_dict,D0, U, U_i    

    def Net_info(self,Bus, Line, Gen, LineStatus, Load_fact, new_dict, U, U_i):
        dict_B = {} #dictionary of bus: segment numbers as keys
        dict_L = {} #dictionary of lines: segment numbers as key
        dict_G = {}
        dict_S = {}
        dict_g_flow = {}
        a0,a1 = LineStatus.shape
        LS_index = np.zeros((a0+1,a1)).astype(int)
        LS_index[0,:] = range(1,a1+1)
        LS_index[1:,:] = LineStatus
        HH0 = np.zeros((len(U_i),2)).astype(int)
        HH0[:,0] = U_i.astype(int)
        HH0[:-1,1] = U_i[1:].astype(int)
        HH0[-1,1] = int(len(Load_fact))
        for h in range(0,len(U_i)):
            net_graph = list(nx.connected_components(new_dict[U_i[h]]))
            net_graph.sort(key=len, reverse=True)
            for j in np.where(U[h]>0)[0]:
                Bus_r = np.array(list(net_graph[j])).astype(int)
                Bus_r = np.sort(Bus_r) #1,2,3,...
                Bus0 = Bus[Bus_r-1,:2] #Bus_r-1 : index
                
                tmp = np.where((np.isin(Line[:,2],Bus_r))&(np.isin(Line[:,1],Bus_r)))[0]
                tmp1 = np.sort(tmp)
                Line0 = Line[tmp1]
                
                
                tmp2 = np.where(np.isin(Gen[:,1],Bus_r))[0]
                tmp2 = np.sort(tmp2)
                Gen0 = Gen[tmp2]
                
                dct_gn = {}
                for b in Bus_r[1:]: #buses in Bus_r starting from index 1 
                    dct_gn[b]= np.where((Gen0[:,1]) == b)[0] #index of gens on that bus with number b
                    
                Dct_g = dict((k, v) for k, v in dct_gn.items() if len(v) > 0) #key: number of original buses, value: index of gen
                
                
                LnIsl = LS_index[:,tmp1] #line status in islanded segments
                
                S_tmp = {}
                for t in range(HH0[int(h),0],HH0[int(h),1]): 
                    S_tmp[t] = LnIsl[0,np.where(LnIsl[t+1,:] == 0)[0]]
                S = dict((k, v) for k, v in S_tmp.items() if len(v) > 0) # line number of line outages
                
                temp3 = str(h)+str(',')+str(j)
                dict_B[temp3] = Bus0        
                dict_L[temp3] = Line0
                dict_G[temp3] = Gen0
                dict_g_flow[temp3] = Dct_g  
                dict_S[temp3] = S
                
        Dct_S = dict((k, v) for k, v in dict_S.items() if len(v) > 0)              
                
        return dict_B,dict_L,dict_G,dict_g_flow,Dct_S,HH0

    def Segment_finder(self,HH2,dict_L,M):    
        seg = {} # the resultant segments
        for n, v in dict_L.items():
            seg.setdefault(n, [])
        
        #all indices: the aaa containts t,m,a:segment number by time,key: segment number
        aaa = np.zeros((1000000,4)).astype(int)
        a0 = 0
        #find segments of each (m,t)
        for t in M.keys():
            for a, b in HH2.items():  
                if t in b: #if any(b == t):
                    t_h = a 
                    aaa[a0,2] = a
            for m in M[t]:#for m,t in M.items():
                aaa[a0,0:2] = t,m            
                for key, val in dict_L.items():
                    if str(t_h)+str(',') in key:                    
                        if m+1 in (val[:,0]).astype(int): 
                            aaa[a0,3] =  list(dict_L).index(key)
                a0 += 1
        
        aaa = aaa[:a0,:]
        aac = np.delete(aaa,2,1).astype(int) # if just instances are to be used
        #unique_rows_02 = np.unique(aac[:,(0,2)], axis=0) 
        unique_rows_12 = np.unique(aac[:,(1,2)], axis=0)
        unique_seg = np.unique(aac[:,-1]).astype(int)
        
        M2 = {}       
        #seg: each dictionary key is the segment name and values are the time index and line index
        for i in unique_seg: 
            ind_l = unique_rows_12[np.where(unique_rows_12[:,1]==i)[0],0]
            M2.setdefault(list(dict_L)[i], [])    
            M3 = {}        
            for i1 in ind_l:
                M3.setdefault(i1,[])
                M3[i1]=aac[np.where((aac[:,1]==i1)&(aac[:,2]==i))[0],0]
            M2[list(dict_L)[i]] = M3
        M2 = dict((k, v) for k, v in M2.items() if len(v) > 0) #seg name:{t:m s}
        return M2


    def PTDF(self,Bus, Line, new_dict, U, U_i):
        dict_sh = {}
        for h in range(0,len(U_i)):
            net_graph = list(nx.connected_components(new_dict[U_i[h]]))
            net_graph.sort(key=len, reverse=True)
            for j in np.where(U[h]>0)[0]:
                Bus_r = np.array(list(net_graph[j])).astype(int)
                Bus_r = np.sort(Bus_r)
                Bus0 = Bus[Bus_r-1,:2]
                
                Key2 = np.zeros_like(Bus0).astype(int)
                Key2[:,0] = np.arange(len(Bus0))+1
                Key2[:,1] = Bus0[:,0] #indexing in the basis bus data (original has indexing not arranged from 0 to number of bus) basis is indexed form 0 to 2000
            
                Bus_sh = np.zeros_like(Bus0)
                Bus_sh[:,0] = np.arange(len(Bus0))+1
                Bus_sh[:,1] = Bus0[:,1]
                tmp = np.where((np.isin(Line[:,2],Bus_r))&(np.isin(Line[:,1],Bus_r)))[0]
                tmp1 = np.sort(tmp)
                Line0 = Line[tmp1]
                Line_sh = np.zeros_like(Line0)
                Line_sh[:,0] = np.arange(len(Line0))+1
                for i in range(0,len(Line0)):
                    Line_sh[i,1] = Key2[np.where(Line0[i,1]==Key2[:,1]),0]
                    Line_sh[i,2] = Key2[np.where(Line0[i,2]==Key2[:,1]),0]
                Line_sh[:,3:] = Line0[:,3:]
                   
            
                K1 = range(len(Line_sh))
                N1 = range(len(Bus_sh))
        
                Bbr = np.zeros((len(Line_sh),len(Line_sh)))#[[0 for x in range(len(K))] for y in range(len(K))]
                B = np.zeros((len(Bus_sh),len(Bus_sh)))#[[0 for x in range(len(N))] for y in range(len(N))]
                A = np.zeros((len(Line_sh),len(Bus_sh)))#[[0 for x in range(len(K))] for y in range(len(K))]
                Ared = np.zeros((len(Line_sh),len(Bus_sh)-1))#[[0 for x in range(len(N)-1)] for y in range(len(K))]
                Bred = np.zeros((len(Bus_sh)-1,len(Bus_sh)-1))#[[0 for x in range(len(N)-1)] for y in range(len(N)-1)]
                shiftfactor = np.zeros((len(Line_sh),len(Bus_sh)-1))#[[0 for x in range(len(N)-1)] for y in range(len(K))]
                
                Bbr = np.diag(Line_sh[:,4])
                Line_no = np.array(Line_sh[:,0]-1,dtype = int)
                from_b = np.array(Line_sh[:,1]-1,dtype = int)
                to_b = np.array(Line_sh[:,2]-1,dtype = int)
                A[Line_no,from_b] = 1
                A[Line_no,to_b] = -1
            
                Ared = A[:,1:]
                
                #calculate the B matrix B-inverse    
                for k in K1:
                    B[int(Line_sh[k,1]-1),int(Line_sh[k,2]-1)] += -Bbr[k,k]
                    B[int(Line_sh[k,2]-1),int(Line_sh[k,1]-1)] += -Bbr[k,k]
                for i in N1:
                    B[i,i] = -np.sum(B[i,:])
                
            
                Bred = B[1:,1:]       
              
                #claculate shift factor matrix    
                inBred = np.linalg.inv(Bred)
                shiftfactor = np.matmul(np.matmul(Bbr,Ared),inBred)
                temp2 = str(h)+str(',')+str(j)
                dict_sh[temp2] = shiftfactor
        
        return dict_sh

    def Opt_operation(self,Power_system,Load_factor,LineFailureProbability,TIme0,Alfa='0.9'):
        
        
        ct = int(10*Alfa)    
        # wd = os.getcwd()
        # os.makedirs(os.path.join(wd, 'results'), exist_ok=True)
        item = 'AllResult'+str(ct)+'.xlsx'
        item_lsh = 'LoadShedding'+str(ct)+'.xlsx'
         
        
        Bus, Line, Gen, Key = self.clean_data(Power_system) #basis bus, line and Gen data, Key is the bus numbers in the original bus data (e.g. 8230)
        
        Load_fact_0 = np.loadtxt(Load_factor) #load factor data for 24 hours
            
        LineStatus = self.scenario_gen(Line,LineFailureProbability,Alfa) 
        Load_fact = np.resize(Load_fact_0, len(LineStatus)) # extending the load factor data to the number of days in line status
        
        LineStatus = LineStatus[TIme0].astype(int) # in case of changing the duration of hours
        Load_fact = Load_fact[TIme0]
        
        #graphs, lonely nodes, segments at each new division, time index of change in segments 
        new_dict, D0, U, U_i = self.Island(Line,LineStatus,Load_fact,Bus)
        
        D0_abr = dict((k, v) for k, v in D0.items() if len(v) > 0) #abriviated form of lonely nodes and their hour(index) of occurance
        
        dict_B,dict_L,dict_G,dict_g_flow,dict_S,HH0 = self.Net_info(Bus, Line, Gen, LineStatus, Load_fact, new_dict, U, U_i)
        HH2 = {}
        for n in range(0,len(HH0)):
            HH2[n] = np.array(range(HH0[n,0],HH0[n,1]))
        
        Mlines_UC = np.array([2,10,12,15,18,32,44,62,63,70,87,94,114,126,226,239,240,273,319,344,446,547,625,732,1011,1133,1202,1250,1298,1347,1400,1477,1513,1600,1753,1800,1806,1865,2056,2059,2093,2202,2281,2312,2559,2642,2836,2880,3055])  # number of lines
        ind_M_UC = Mlines_UC-1 #index of lines
        ######################## Set the parameters ###################################
            
        tolerance = 0.05 #cplex mip gap
        Threads = int((psutil.cpu_count(logical=True)+psutil.cpu_count(logical=False))/2) #cplex core usage
        W_mem = int(psutil.virtual_memory().total/1000000) # cplex working memory
        MaxIteration = 100 #maximum number of iteration, after that the final set of monitored line will be considered as final monitored lines set.
        Penalty = 1000 #penalty factor (x) for load shedding and over generation. x times more than the most expensive generation MWhr in the network
        BigM = 5000 #Sufficiennt big number for cancellation transaction.
            
        ######################## Define Parameters ####################################
            
        start_time = datetime.datetime.now()
        N = len(Bus)
        K = len(Line)
        G = len(Gen)
        T = len(Load_fact)
        
        N1 = np.arange(N)
        K1 = np.arange(K)
        G1 = np.arange(G)
        T1 = np.arange(T)
        T2 = np.array(T1[0:-1]) #for start up
        T3 = np.array(T1[1:])
        
        LC1 = Gen[:,8]
        LC2 = Gen[:,9]
        LC3 = Gen[:,10]
        LC4 = Gen[:,11]
        
        ######################## Flag : error in the input data #######################
        
        #0: Pmax less than Pmin and is not positive for all generation
        #1: The generation units are not located in the feasible range of bus numbers
        #2: Cost index is negative
        #3: LineStatus data has problem
        #4: Load shedding price is zero.
        
        Flag = np.ones(5)
        if np.any(Gen[:,2] < Gen[:,3]) or np.any(Gen[:,3] < 0):
            Flag[0] = 0
            print('Check the generation data. Is the Pmax more than or equal to the Pmin and is Pmin positive for all generation units?')
            sys.exit()
        if np.any(Gen[:,1] > N) or np.any(Gen[:,1] < 0):
            Flag[1] = 0
            print('Check the generation data. Are all the generation units located in the feasible range of bus numbers?')
            sys.exit()
        
        
        if np.any(Gen[:,8:15] < 0):
            Flag[2] = 0
            print('Check the generation data. Negative cost!')
            sys.exit()
        ###adjusting the minimum up and down data: when the unit is turned on or off it should stay on or off for at least one hour
        Gen[:,15] = np.where(Gen[:,15] > 0, Gen[:,15], 1)  
        Gen[:,16] = np.where(Gen[:,16] > 0, Gen[:,16], 1)  
        #LineStatus: line can be either online or offline
        if np.isin(LineStatus,[0,1]).all():
            print('Reading data completed!')
            print('"  "')
        else:
            Flag[3] = 0 #line can be either online or offline
            print('Check the LineStatus data. The line status values should be 0 or 1.')
            sys.exit()
        
        ######################## organizing the input data ############################
        
        # FromBus = Line[:,1].astype(int)
        # ToBus = Line[:,2].astype(int)
        FkMax = np.zeros([T,K])
        FkMin = np.zeros([T,K])
        Load = np.zeros([T,N])
        TotalLoad = np.zeros(T)
        
        for t in np.arange(T):
            Load[t,:] = Bus[:,1]*Load_fact[t]
        TotalLoad = np.sum(Load, axis=1)
        
        L_bus_in = np.where(Bus[:,1]>0)[0] # index of buses with load attached to them
        
        Loaded_bus = {} # key: index of time, value: index of bus with load in the system with outages
        for i in range(0,len(Load_fact)):
              Loaded_bus[i] = np.setdiff1d(L_bus_in,(np.array(list(D0[i]))-1).astype(int))   
        
        # Buses that are in one of the segment at t and then are isolated at t+1
        Bus_RR = {}
        for t in range(1,len(Load_fact)):
            Bus_RR.setdefault(t,[])
            Bus_RR[t] = np.setdiff1d(list(D0[t]), list(D0[t-1])).astype(int)
        
        BusRR_abr = dict((k, v) for k, v in Bus_RR.items() if len(v) > 0) #time index as key and nodes that are in a seg in t and in D0 in t+1 as values 
              
        # L_SH = np.zeros((len(Load_fact),len(L_bus_in)))
        # OVG = np.zeros((len(Load_fact),G))   
        ######################## Initial values and outages ###########################
        
        NofOut = np.zeros(T,dtype = int)
        NofMonitor = np.zeros(MaxIteration,dtype = int) #for the first iteration no line would be monitored
        # MaxMonitor = 0
        
        endflag = 0
        COUNTER = 1
        
        P_result = np.zeros([T,G]) #dispatch result variable
        Commitment_result = np.zeros([T,G])
        FC_result = np.zeros([T,K]) #results for flow cancellation
        LSH_result = np.zeros([T,K]) #result for lost load
        OVG_result = np.zeros([T,G]) #result for over generation
        
        
        LSh_cost_bus = np.zeros(N) #load lost price for each bus(the same for all scenarios)
        Over_Gen_Cost = np.zeros(G) #Over generation cost for each generation unit(the same for all scenarios)
        
        #finding the maximum marginal generation cost    
        LSh_cost_all = Penalty*np.max(np.array([LC1,LC2,LC3,LC4])) #Load Lost penalty equals to x times maximum generation cost
        print('Shadow price is .......................',LSh_cost_all) 
            
        #LSh_cost_all = 50, if load lost penalty needed to be a constant value
        if (LSh_cost_all < 0):
            Flag[4] = 0 #Load shedding penalty can not be negative or zero
            print('Load shedding cost cannot be negative!, check the linear costs and penalty value.')
            sys.exit()
        
            
        LSh_cost_bus[:] = LSh_cost_all
        #LSh_cost_bus = LSh_cost_all #load shedding cost is considered the same value for all the buses. Any policy regarding different load lost penalties(by buses) should be applied here.
           
        Over_Gen_Cost[:] = LSh_cost_all
        #Over_Gen_Cost = LSh_cost_all #over generation cost is considered the same value for all the units and equal to load shedding cost. Any policy for different over generation penalties should be applied here.    
        
        # number of outage per hour and per scenario
        for t in T1:
            NofOut[t] = np.count_nonzero(LineStatus[t,:] == 0)
        
        # MaxOut = int(np.max(NofOut))
        
        LS_time,LS_line = np.where(LineStatus == 0)
        
        # S: compact form of LineStatus. Arrays are line index starting from zero.   
        #key: time and values: line number    
        S_FC = {}
        for i in dict_S.keys():
            S0 = dict_S[i]
            for t in S0.keys():
                if t in S_FC:
                    S_FC[t] = np.concatenate((S_FC[t],S0[t]), axis=None) 
                else: 
                    S_FC[t] = S0[t]
        
        # The monitored lines and outages are different in each scenario and hour: FkMax and Min are defined on scenarios and hours.            
        for l in K1:
            FkMax[:,l] = Line[l,5]
            FkMin[:,l] = -Line[l,5]
        # if line is out, there is no need to monitor it at that hour. The original capacity multiplied by 100 should be enoug
        FkMax[LS_time,LS_line] = FkMax[LS_time,LS_line]*100
        FkMin[LS_time,LS_line] = FkMin[LS_time,LS_line]*100  
        
        linemonitorflag = np.zeros((T,K)).astype(int) #0: not monitored, 1:monitored (for the first iteration no line would be monitored.)
        # linemonitorflag_all = np.zeros((10*T,K)).astype(int)
        
        #M is the compact form of monitored lines. Arrays are line numbers starting from zero.
        M = {} #np.zeros(K,dtype=int)
        
        Done_b4 = {} # seg: {m,t}
        for sg in dict_L.keys():
            Done_b4.setdefault(sg,[])  
            M4 = {}
            for l in (dict_L[sg][:,0]-1).astype(int):
                M4.setdefault(l,[])   
            Done_b4[sg] = M4
            
        #print("Shift factor matrix calculating...")
        shift1_time = datetime.datetime.now()
        
        Dct_shiftfactor = self.PTDF(Bus, Line, new_dict, U, U_i)
        shift2_time = datetime.datetime.now()
        print("Shift_factor calculated: ",(shift2_time-shift1_time).total_seconds())
        main_solve = datetime.datetime.now()
        print('Started solve at: ',main_solve)
        
        ######################## Iterative Optimization ###############################
            
        print("******------------------------******------------------------******")       
        print("Defining Optimization Problem...")
        
        model = ConcreteModel()
        
        ## Set Variables
        def ub_pmax(model, t,g):
            return (0, Gen[g,2])
        def ub_p1(model, t,g):
            return (0, Gen[g,4])
        def ub_p2(model, t,g):
            return (0, Gen[g,5])
        def ub_p3(model, t,g):
            return (0, Gen[g,6])
        def ub_p4(model, t,g):
            return (0, Gen[g,7])
        def ub_LSH(model, t,b):
            return (0, Load[t,b])
        def ul_FK(model, t,m):
            return (FkMin[t,m],FkMax[t,m])
        
        model.P = Var(T1,G1, bounds = ub_pmax) #Generation dispatch (Total)(6.0) different for scenarios
        model.P1 = Var(T1,G1, bounds = ub_p1) #Generation dispatch (1st segment)(6.0) different for scenarios
        model.P2 = Var(T1,G1, bounds = ub_p2) #Generation dispatch (2nd segment)(6.0) different for scenarios
        model.P3 = Var(T1,G1, bounds = ub_p3) #Generation dispatch (3rd segment)(6.0) different for scenarios
        model.P4 = Var(T1,G1, bounds = ub_p4) #Generation dispatch (4th segment)(6.0) different for scenarios
        model.Fk = Var(T1,K1, bounds = ul_FK) #model.fk_set
        model.uk = Var(T1,G1, within = Binary) #Generator status (1:on, 0:off) (6.0) same for all scenarios
        model.v = Var(T2,G1, within = Binary) #Startup variable (1:startup) (6.0) same for all scenarios
        model.w = Var(T2,G1, within = Binary) #Shutdown variable (1:shutdown) (6.0) same for all scenarios
        model.FC = Var(T1,K1, bounds = (-BigM,BigM)) # model.fc_set #Flow cancellation variable (6.0) different for scenarios
        model.LSH = Var(T1,N1, bounds = ub_LSH) #model.lsh_set #Load shedding (lost load) variable (6.0) different for scenarios
        model.OVG = Var(T1,G1, bounds = ub_pmax) #Overgeneration variable (6.0) different for scenarios
        model.flow = Block() 
        model.flow.cncl = ConstraintList() #line monitor constraint
        
        for t in D0_abr.keys():
            for n in D0_abr[t]:
                G_D0 = np.where(Gen[:,1]==n)[0]
                for g in G_D0:
                    model.P[t,g].fix(0)
                    model.uk[t,g].fix(0)
                if n-1 in L_bus_in: # n: bus number, L_bus_in: bus index
                    n1 = int(n-1)
                    model.LSH[t,n1].fix(Load[t,n1])
                    
        # Define the objective function
        def obj_rule(model):
            return sum(model.P1[t,g]*LC1[g]+model.P2[t,g]*LC2[g]+model.P3[t,g]*LC3[g]+model.P4[t,g]*LC4[g]+Over_Gen_Cost[g]*model.OVG[t,g] for t in T1 for g in G1)+sum(LSh_cost_bus[b]*model.LSH[t,b] for t in T1 for b in L_bus_in)+sum(model.v[t,g]*Gen[g,12]+ model.w[t,g]*Gen[g,13] for t in T2 for g in G1)+sum(model.uk[t,g]*Gen[g,14] for t in T1 for g in G1)                
        model.obj = Objective(rule=obj_rule)
        
        def Ptotal_rule(model,t,g):
             return model.P[t,g] == model.P1[t,g]+model.P2[t,g]+model.P3[t,g]+model.P4[t,g]
        model.Ptotal = Constraint(T1,G1,rule=Ptotal_rule)
        
        
        print('Adding network power balance constraints...')
        def node_balance_rule(model,t):
            return sum(model.P[t,g]-model.OVG[t,g] for g in G1) + sum(model.LSH[t,b] for b in L_bus_in)==TotalLoad[t] 
        model.node_balance = Constraint(T1,rule=node_balance_rule) 
        
        print('Adding commitment constraints...')
        def Ineq9_rule(model,t,g):
            return model.P[t,g]-model.uk[t,g]*Gen[g,2]<=0 
        model.Ineq9 = Constraint(T1,G1,rule=Ineq9_rule)
        
        def Ineq10_rule(model,t,g): 
            return model.uk[t,g]*Gen[g,3]-model.P[t,g]<=0 
        model.Ineq10 = Constraint(T1,G1,rule=Ineq10_rule)
        
        #start up and shut down variables
        print('Adding start-up and shut down constraints...')  
        def vw1_rule(model,t,g):
            return model.v[t-1,g]-model.w[t-1,g]-model.uk[t,g]+model.uk[t-1,g]==0    
        model.vw1 = Constraint(T3,G1,rule=vw1_rule)
        
        def vw2_rule(model,t,g):
            return model.v[t-1,g]+model.w[t-1,g] <= 1
        model.vw2 = Constraint(T3,G1,rule=vw2_rule)
        
        print('Adding ramping constraints ...')
        def ramp_rule(model,t,i5):
            if len(BusRR_abr.keys())>0:
                if t in BusRR_abr.keys():
                    if Gen[i5,1] in BusRR_abr[t]:
                        return (-Gen[i5,2],(model.P[t,i5]-model.P[t-1,i5]),Gen[i5,2])
                    else:
                        return (-Gen[i5,17],(model.P[t,i5]-model.P[t-1,i5]),Gen[i5,17])
                else:
                    return (-Gen[i5,17],(model.P[t,i5]-model.P[t-1,i5]),Gen[i5,17])
            else:
                return (-Gen[i5,17],(model.P[t,i5]-model.P[t-1,i5]),Gen[i5,17])
        if T >1:
            model.ramp = Constraint(T3,np.where(Gen[:,17] < Gen[:,2])[0],rule=ramp_rule)
        
        model.minU = ConstraintList()
        model.minD = ConstraintList()
        model.flow_can  = ConstraintList()
        
        
        #Inequality constraints
                #minimum up and down time and ramping contraints exist only when t > 1
        if T > 1:
            #adding minimum up and down time constraint
            print('Adding min up/down time constraints ...')
            for i6 in np.where(Gen[:,16]>1)[0]:
                for m in range(1, int(T+1 - Gen[i6,16])): # minimum up time constraints in the middle
                    model.minU.add(sum(model.uk[i,i6] for i in range(m,int(m+Gen[i6,16])))-Gen[i6,16]*(model.uk[m,i6]-model.uk[m-1,i6])>=0)
                for m in range(1, int(T+1 -Gen[i6,15])):
                    model.minD.add(Gen[i6,15] - sum(model.uk[i,i6] for i in range(m,int(m+Gen[i6,15])))-Gen[i6,15]*(model.uk[m-1,i6]-model.uk[m,i6])>=0) 
        
        print('Adding line flow constraints ...')
        line_con_s0 = datetime.datetime.now()
        
        for sg in dict_L.keys():
            print('...')
            Bus_seg = (dict_B[sg][:,0]).astype(int)
            shiftfactor = Dct_shiftfactor[sg]
            Line_seg = dict_L[sg]
            
            Dict_g = dict_g_flow[sg]
            h = int(str(sg).split(",",1)[0])
            for m in ind_M_UC:
                print('...')
                if m+1 in Line_seg[:,0]:
                    ind_m = int(np.where(m==Line_seg[:,0]-1)[0])
                    linemonitorflag[HH2[h],m] = 1
                    HH1 = Done_b4[sg][m]
                    for t in filter(lambda el: el not in HH1, HH2[h]):
                        print('...')
                        LF = 0
                        for b in range(1,len(Bus_seg)):
                            if Bus_seg[b]-1 in Loaded_bus[t]:
                                LF += shiftfactor[ind_m,b-1]*(model.LSH[t,Bus_seg[b]-1]-Load[t,Bus_seg[b]-1])
                            if Bus_seg[b] in Dict_g.keys():                    
                                LF += sum(shiftfactor[ind_m,b-1]*(model.P[t,i2]-model.OVG[t,i2]) for i2 in Dict_g[Bus_seg[b]])
                        if sg in dict_S.keys():
                            S = dict_S[sg]
                            if t in S.keys():
                                for o in range(0,len(S[t])):
                                    ind_mo = np.where(S[t][o]==Line_seg[:,0])[0]
                                    ind_TB = int(np.where(int(Line_seg[ind_mo,2])==Bus_seg[:])[0])
                                    ind_FB = int(np.where(int(Line_seg[ind_mo,1])==Bus_seg[:])[0])
                                    if ind_FB == 1:
                                        LF += -shiftfactor[ind_m,ind_TB-2]*model.FC[t,S[t][o]-1]
                                    elif ind_TB == 1:
                                        LF += shiftfactor[ind_m,ind_FB-2]*model.FC[t,S[t][o]-1]
                                    else:
                                        LF += (shiftfactor[ind_m,ind_FB-2]-shiftfactor[ind_m,ind_TB-2])*model.FC[t,S[t][o]-1]
                            
                        model.flow.cncl.add(LF == model.Fk[t,m])  
                        Done_b4[sg][m].append(t)
        line_con_f0 = datetime.datetime.now()
        print('Line monitoring constraints take: ',(line_con_f0-line_con_s0).total_seconds()) 
        
        print('Adding line cancelation constraints ...')
        line_con_f1 = datetime.datetime.now()
        for i in dict_S.keys(): #index of time
            Bus_seg = (dict_B[i][:,0]).astype(int)
            shiftfactor = Dct_shiftfactor[i]
            Line_seg = dict_L[i]
            S = dict_S[i]
            Dict_g = dict_g_flow[i] #key:number of bus with gen, value: index of gen
            
            for t in S.keys():
                for o in range(0,len(S[t])):
                    VT = 0
                    ind_m = int(np.where(S[t][o]==Line_seg[:,0])[0])#index of outaged line(S[t][o] = m) in dict_L
                    for b in range(1,len(Bus_seg)):
                        if Bus_seg[b]-1 in Loaded_bus[t]:
                            VT += shiftfactor[ind_m,b-1]*(model.LSH[t,Bus_seg[b]-1]-Load[t,Bus_seg[b]-1])
                        if Bus_seg[b] in Dict_g.keys():
                            VT += sum(shiftfactor[ind_m,b-1]*(model.P[t,i1]-model.OVG[t,i1]) for i1 in Dict_g[Bus_seg[b]]) 
                    VT += -model.FC[t,S[t][o]-1] #second part of eq 25
                    for oo in range(0,len(S[t])):
                        ind_mo = int(np.where(S[t][oo]==Line_seg[:,0])[0])
                        ind_TB = int(np.where(int(Line_seg[ind_mo,2])==Bus_seg)[0])
                        ind_FB = int(np.where(int(Line_seg[ind_mo,1])==Bus_seg)[0])
                        if ind_FB == 0:
                            VT += -shiftfactor[ind_m,ind_TB-1]*model.FC[t,S[t][oo]-1]
                        elif ind_TB == 0:
                            VT += shiftfactor[ind_m,ind_FB-1]*model.FC[t,S[t][oo]-1]
                        else:
                            VT += (shiftfactor[ind_m,ind_FB-1]-shiftfactor[ind_m,ind_TB-1])*model.FC[t,S[t][oo]-1]
                    model.flow_can.add(VT == 0)
                    #model.flow_cncl.pprint()
        
        line_con_f = datetime.datetime.now()
        print('Line outage constraints take: ',(line_con_f-line_con_f1).total_seconds())
        print('Constraints, Done!')
        
        
        opt = SolverFactory('cplex_persistent',executable=r'C:\Program Files\IBM\ILOG\CPLEX_Studio221\cplex\bin\x64_win64\cplex')
        opt.set_instance(model)    
        opt.options['mip_tolerances_mipgap'] = tolerance #to instruct CPLEX fot stop as soon as it has found a feasible integer solution proved to be within five percent of optimal 
        opt.options['threads'] = Threads
        opt.options['workmem'] = W_mem  
        
        Solution=opt.solve(model,tee=True,report_timing=True)
        Solution.write(num=1)
        
        print("Construction of persistent model complete.")
        def_f_p = datetime.datetime.now()
        print('construction time', (def_f_p - start_time).total_seconds())
        
        
        # res_s = datetime.datetime.now()
        for t in T1:
            for g in G1:
                P_result[t,g] = model.P[t,g].value
                OVG_result[t,g] = model.OVG[t,g].value
                Commitment_result[t,g] = model.uk[t,g].value
            for b in L_bus_in:
                LSH_result[t,b] = model.LSH[t,b].value
        for t in S_FC.keys():
            for o in S_FC[t]:
                FC_result[t,o-1] = model.FC[t,o-1].value
        
        ######################## LINE MONITOR DECISION ################################
        
        print('Calculating line flows ...')
        load = np.zeros([N-1,T])
        gen = np.zeros([N-1,T])
        endflag = 0
        
        L_bus_in_no0 = np.delete(L_bus_in,np.where(L_bus_in==0)[0])
        for t in T1:
            for b in L_bus_in_no0:
                load[b-1,t] = Load[t,b]
                gen[b-1,t] = LSH_result[t,b] #the effect of load shedding is equal to the same amount of generation at that bus
            for i4 in np.where(Gen[:,1] != 1)[0]:
                gen[int(Gen[i4,1]-2),t] += P_result[t,i4]-OVG_result[t,i4]
                        
        line_flow_load_tmp = np.zeros((K,T))
        line_flow_gen_tmp = np.zeros((K,T))
        for h in Dct_shiftfactor.keys():
            t = int(str(h).split(",",1)[0])
            L_seg = (dict_L[h][:,0]-1).astype(int) # line_flow_load_tmp of lines L_seg is to be calculated
            B_seg = (dict_B[h][:-1,0]-1).astype(int) # load and gen were created for b in range(1,N)
            shiftfactor = Dct_shiftfactor[h]
            line_flow_load_tmp[L_seg,HH0[t,0]:HH0[t,1]] += np.matmul(shiftfactor,load[B_seg,HH0[t,0]:HH0[t,1]]) 
            line_flow_gen_tmp[L_seg,HH0[t,0]:HH0[t,1]] += np.matmul(shiftfactor,gen[B_seg,HH0[t,0]:HH0[t,1]])
               
                
        
        line_flow_load = np.transpose(line_flow_load_tmp)
        line_flow_gen = np.transpose(line_flow_gen_tmp)
        line_flow = line_flow_gen-line_flow_load
        
        for i in dict_S.keys():
            Bus_seg = dict_B[i]
            shiftfactor = Dct_shiftfactor[i]
            Line_seg = dict_L[i]
            S = dict_S[i]
            
            for t in S.keys():
                for l in range(0,len(Line_seg)):    
                    for o in range(0,len(S[t])):
                        ind_m = int(np.where(S[t][o]==Line_seg[:,0])[0])#index of outaged line(S[t][o] = m) in dict_L
                        ind_TB = int(np.where(int(Line_seg[ind_m,2])==Bus_seg[:,0])[0])
                        ind_FB = int(np.where(int(Line_seg[ind_m,1])==Bus_seg[:,0])[0])
                        if ind_FB == 0:
                            line_flow[t,int(Line_seg[l,0]-1)] += -shiftfactor[l,ind_TB-1]*FC_result[t,S[t][o]-1]
                        elif ind_TB == 0:
                            line_flow[t,int(Line_seg[l,0]-1)] += shiftfactor[l,ind_FB-1]*FC_result[t,S[t][o]-1]
                        else:
                            line_flow[t,int(Line_seg[l,0]-1)] += (shiftfactor[l,ind_FB-1]-shiftfactor[l,ind_TB-1])*FC_result[t,S[t][o]-1]
            
        M = {}   
        for t in T1:
            M.setdefault(t,[])
            
        for t,l in zip(*np.where(linemonitorflag == 0)):
            if (line_flow[t,l]>FkMax[t,l]) or (line_flow[t,l]<FkMin[t,l]):
                M[t].append(l)
                linemonitorflag[t,l] = 1
                endflag = 0
        
        M = dict((k, v) for k, v in M.items() if len(v) > 0)
        
        print('Previous number of monitored instances = ',NofMonitor[0]) 
        
            
        NofMonitor[COUNTER] = len(np.where(linemonitorflag==1)[0]) 
        print('number of monitored instances = ',NofMonitor[COUNTER])  
                
        if len(M) == 0:
            endflag = 1
                
        while endflag == 0:  
            COUNTER += 1    
            print("------------------- iteration -------------------",COUNTER)          
        
            #add DC power flow constraints
            # eq 24
            print('Adding line flow constraints ...')
            line_con_s = datetime.datetime.now()
            
            M2  = {}   
            M2 = self.Segment_finder(HH2,dict_L,M)  # seg:{m:t}
            for sg in M2.keys():
                print('...')
                Bus_seg = (dict_B[sg][:,0]).astype(int)
                shiftfactor = Dct_shiftfactor[sg]
                Line_seg = dict_L[sg]
                
                Dict_g = dict_g_flow[sg]
                h = int(str(sg).split(",",1)[0])
                for m in M2[sg].keys():
                    print('...')
                    ind_m = int(np.where(m==Line_seg[:,0]-1)[0])
    
                    linemonitorflag[HH2[h],m] = 1
                    HH1 = Done_b4[sg][m]
                    for t in filter(lambda el: el not in HH1, HH2[h]):
                        print('...')
                        LF = 0
                        for b in range(1,len(Bus_seg)):
                            if Bus_seg[b]-1 in Loaded_bus[t]:
                                LF += shiftfactor[ind_m,b-1]*(model.LSH[t,Bus_seg[b]-1]-Load[t,Bus_seg[b]-1])
                            if Bus_seg[b] in Dict_g.keys():                    
                                LF += sum(shiftfactor[ind_m,b-1]*(model.P[t,i2]-model.OVG[t,i2]) for i2 in Dict_g[Bus_seg[b]])
                        if sg in dict_S.keys():
                            S = dict_S[sg]
                            if t in S.keys():
                                for o in range(0,len(S[t])):
                                    ind_mo = np.where(S[t][o]==Line_seg[:,0])[0]
                                    ind_TB = int(np.where(int(Line_seg[ind_mo,2])==Bus_seg[:])[0])
                                    ind_FB = int(np.where(int(Line_seg[ind_mo,1])==Bus_seg[:])[0])
                                    if ind_FB == 0:
                                        LF += -shiftfactor[ind_m,ind_TB-1]*model.FC[t,S[t][o]-1]
                                    elif ind_TB == 0:
                                        LF += shiftfactor[ind_m,ind_FB-1]*model.FC[t,S[t][o]-1]
                                    else:
                                        LF += (shiftfactor[ind_m,ind_FB-1]-shiftfactor[ind_m,ind_TB-1])*model.FC[t,S[t][o]-1]
                            
                        model.flow.cncl.add(LF == model.Fk[t,m])  
                        Done_b4[sg][m].append(t)
                            
                            
            line_con_f1 = datetime.datetime.now()
            print('Line monitoring constraints take: ',(line_con_f1-line_con_s).total_seconds()) 
            
            opt = SolverFactory('cplex_persistent',executable=r'C:\Program Files\IBM\ILOG\CPLEX_Studio221\cplex\bin\x64_win64\cplex')
            opt.set_instance(model)
            opt.options['mip_tolerances_mipgap'] = tolerance #to instruct CPLEX fot stop as soon as it has found a feasible integer solution proved to be within five percent of optimal 
            opt.options['threads'] = Threads
            opt.options['workmem'] = W_mem  
            
            opt.solve(model,tee=True,report_timing=True) #model,save_results=False  
            
            Solution=opt.solve(model,tee=True,report_timing=True)
            Solution.write(num=1)
            
            print(value(model.obj))
            res_s1 = datetime.datetime.now()
            for t in T1:
                for g in G1:
                    P_result[t,g] = model.P[t,g].value
                    OVG_result[t,g] = model.OVG[t,g].value
                    Commitment_result[t,g] = model.uk[t,g].value
                for b in L_bus_in:
                    LSH_result[t,b] = model.LSH[t,b].value
            for t in S_FC.keys():
                for o in S_FC[t]:
                    FC_result[t,o-1] = model.FC[t,o-1].value
        
        ######################## LINE MONITOR DECISION ################################
        
            print('Calculating line flows ...')
            load = np.zeros([N-1,T])
            gen = np.zeros([N-1,T])
            endflag = 0
        
            for t in T1:
                for b in L_bus_in_no0:
                    load[b-1,t] = Load[t,b]
                    gen[b-1,t] = LSH_result[t,b] #the effect of load shedding is equal to the same amount of generation at that bus
                for i4 in np.where(Gen[:,1] != 1)[0]:
                    gen[int(Gen[i4,1]-2),t] += P_result[t,i4]-OVG_result[t,i4]
        
                            
            line_flow_load_tmp = np.zeros((K,T))
            line_flow_gen_tmp = np.zeros((K,T))
            for h in Dct_shiftfactor.keys():
                t = int(str(h).split(",",1)[0])
                L_seg = (dict_L[h][:,0]-1).astype(int) # line_flow_load_tmp of lines L_seg is to be calculated
                B_seg = (dict_B[h][:-1,0]-1).astype(int) # load and gen were created for b in range(1,N)
                shiftfactor = Dct_shiftfactor[h]
                line_flow_load_tmp[L_seg,HH0[t,0]:HH0[t,1]] += np.matmul(shiftfactor,load[B_seg,HH0[t,0]:HH0[t,1]]) 
                line_flow_gen_tmp[L_seg,HH0[t,0]:HH0[t,1]] += np.matmul(shiftfactor,gen[B_seg,HH0[t,0]:HH0[t,1]])
                   
                    
            
            line_flow_load = np.transpose(line_flow_load_tmp)
            line_flow_gen = np.transpose(line_flow_gen_tmp)
            line_flow = line_flow_gen-line_flow_load
        
            for i in dict_S.keys():
                Bus_seg = dict_B[i]
                shiftfactor = Dct_shiftfactor[i]
                Line_seg = dict_L[i]
                S = dict_S[i]
                
                for t in S.keys():
                    for l in range(0,len(Line_seg)):    
                        for o in range(0,len(S[t])):
                            ind_m = int(np.where(S[t][o]==Line_seg[:,0])[0])#index of outaged line(S[t][o] = m) in dict_L
                            ind_TB = int(np.where(int(Line_seg[ind_m,2])==Bus_seg[:,0])[0])
                            ind_FB = int(np.where(int(Line_seg[ind_m,1])==Bus_seg[:,0])[0])
                            if ind_FB == 0:
                                line_flow[t,int(Line_seg[l,0]-1)] += -shiftfactor[l,ind_TB-1]*FC_result[t,S[t][o]-1]
                            elif ind_TB == 0:
                                line_flow[t,int(Line_seg[l,0]-1)] += shiftfactor[l,ind_FB-1]*FC_result[t,S[t][o]-1]
                            else:
                                line_flow[t,int(Line_seg[l,0]-1)] += (shiftfactor[l,ind_FB-1]-shiftfactor[l,ind_TB-1])*FC_result[t,S[t][o]-1]
                            
            M = {}   
            for t in T1:
                M.setdefault(t,[])
                
            for t,l in zip(*np.where(linemonitorflag == 0)):
                if (line_flow[t,l]>FkMax[t,l]) or (line_flow[t,l]<FkMin[t,l]):
                    M[t].append(l)
                    linemonitorflag[t,l] = 1
                    endflag = 0
        
            M = dict((k, v) for k, v in M.items() if len(v) > 0)  
            
            print('Previous number of monitored instances = ',NofMonitor[COUNTER-1])
            NofMonitor[COUNTER] = len(np.where(linemonitorflag==1)[0]) #np.where(linemonitorflag==1)[0],np.unique(np.concatenate(list(M.values())))
            print('number of monitored instances = ',NofMonitor[COUNTER])
            
            res_f1 = datetime.datetime.now()
            print('Calculating flows takes: ',(res_f1-res_s1).total_seconds())
            print('Counter {} finished at: {}'.format(COUNTER, res_f1)) 
            
            if len(M)>0:
                endflag = 0
    
            else:
                endflag = 1
        
            if COUNTER == (MaxIteration-1):
                endflag = 1
                           
            if endflag == 1:
                break
        print('All Done!')
        print('Number of Iteration: ',COUNTER)
        print('Started at: ',start_time)
        mainfinish = datetime.datetime.now()
        print('Finished at: ',mainfinish) 
        print((mainfinish-start_time).total_seconds())
        
        Cost = value(model.obj)
        
        # find what new nodes get seperated at each hour               
        LSH_b = np.where(sum(LSH_result[t,:] for t in T1) > 0)[0]
        if len(LSH_b)>0:
            LSH = np.zeros((len(LSH_b),T+2))
            for i in range(0,len(LSH_b)):
                LSH[i,0] = Key[int(LSH_b[i]),1]
                LSH[i,1] = sum(LSH_result[t,int(LSH_b[i])] for t in T1)  #the first column is the numbering used by power group(1 to number of buses)#second column is the indexing used by civil group (row position in data) and the third column is the real bus numbers
                LSH[i,2:] = LSH_result[:,int(LSH_b[i])]
        else: 
            LSH = []
                
                        
        Summery = [{'Solution Value': value(model.obj),
                    'Total Time':(mainfinish-start_time).total_seconds(),
                    'Total Load shedding':np.sum(LSH_result)}]
        np.set_printoptions(threshold=np.inf)
        df_Lsh= pd.DataFrame(LSH)
        df_cmtmnt = pd.DataFrame(Commitment_result)
        df_Mnt_l = pd.DataFrame(np.unique(np.where(linemonitorflag == 1)[1]))
        df_Mnt_ln = df_Mnt_l+1
        df_gen = pd.DataFrame(P_result)
        df_smry = pd.DataFrame(Summery)
    
        path = os.getcwd()
        folder_name = self.out_f
        path = os.path.join(path, folder_name)
        os.makedirs(path, exist_ok=True)
        item_n = os.path.join(path, item)
        writer = pd.ExcelWriter(item_n) #,item
        df_smry.to_excel(writer,sheet_name='Summery',index=False)
        df_cmtmnt.columns =[i for i in Gen[:,0].astype(int)] 
        df_cmtmnt.index=[i for i in range(1,len(Load_fact)+1)]
        df_cmtmnt.to_excel(writer,sheet_name='Commitment') 
        df_Mnt_ln.index=[i for i in range(1,len(df_Mnt_l)+1)]
        df_Mnt_ln.to_excel(writer,sheet_name='Monitor')
        df_gen.columns =[i for i in Gen[:,0].astype(int)] 
        df_gen.index=[i for i in range(1,len(Load_fact)+1)]
        df_gen.to_excel(writer,sheet_name='Power')
        #df_Lsh.columns =[i for i in range(0,T+1)] 
        df_Lsh.to_excel(writer,sheet_name='Load_shedding')
        
        writer.save()
            
        
        xlsload_t = pd.ExcelFile('Texas System Buses and Branches with Geographical information.xlsx')
            
        Bus_loc_0 = pd.read_excel(xlsload_t, 'Buses',header=1).fillna(0)
        Bus_loc_1 = Bus_loc_0[['Substation Longitude', 'Substation Latitude', 'Number']].to_numpy()
        
        Bus_loc_2 = np.zeros((len(LSH),4+len(Load_fact)))
        for i in range(0,len(LSH)):
            Bus_loc_2[i,1:3] = Bus_loc_1[np.where(LSH[i,0]==Bus_loc_1[:,2])[0],0:2]
        Bus_loc_2[:,0] = LSH[:,0] #Bus number
        Bus_loc_2[:,3] = Bus[LSH_b,-1] #Zone number
        
        Bus_loc_2[:,4:] = LSH[:,2:]
        
        Bus_loc_2[Bus_loc_2==0] = 'NaN'
    
        clmn_B1 = list(range(1,1+len(Load_fact)))
        clmn_B = ['Bus','Longt','Lat','Zone']+clmn_B1
            
        np.set_printoptions(threshold=np.inf)
        Bus_loc_2_final= pd.DataFrame(Bus_loc_2)
        item_lsh = os.path.join(path, item_lsh)
        writer = pd.ExcelWriter(item_lsh)
        Bus_loc_2_final.columns = clmn_B
        Bus_loc_2_final.index=[i for i in range(1,len(LSH)+1)]
        Bus_loc_2_final.to_excel(writer,sheet_name='Lshedding_loc')
        writer.save()    
        
        return Bus_loc_2_final, item_lsh, Summery

    def vis_img(self,data0):
        
        path = os.getcwd()
        folder_name = self.out_im_f
        path = os.path.join(path, folder_name)
        os.makedirs(path, exist_ok=True)
        # result_folder = f'{os.getcwd()}/{self.out_f}/{data0}'
        #data0 = 'LoadShedding5.xlsx'
        xlsload = pd.ExcelFile(data0)
        df0 = pd.read_excel(xlsload, 'Lshedding_loc',index_col=0).fillna(0)
        H_dur = df0.shape[1] - 4
        for i in range(0,int(H_dur)):
            item = i+1
            fig = plt.figure()
            ax = fig.subplots()
            
            item2 = str('Load Shedding in each zone at hour ')+str(i+1)
            ax.set_title(item2)
                
            # Set the dimension of the figure
            plt.rcParams["figure.figsize"]=8,8;
        
            # Make the background map
            m=Basemap(llcrnrlon=-107, llcrnrlat=25.6, urcrnrlon=-93.5, urcrnrlat=36.8)
            m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
            m.fillcontinents(color='grey', alpha=0.3)
            m.drawcountries(linewidth=0.5, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
            m.drawstates(linewidth=0.5, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
            m.drawcoastlines(linewidth=0.1, color="white")
        
            # prepare a color for each point depending on the continent.
            df0['labels_enc'] = pd.factorize(df0['Zone'])[0]
        
            # Add a point per position
            m.scatter(
                x=df0['Longt'], 
                y=df0['Lat'], 
                s=df0[item]*3,
                alpha=0.75, 
                c=df0['labels_enc'], 
                cmap="Set1")  
        
            mol = str('Hour')+str(i+1)
            fig.savefig(f'{path}/{mol}.png', bbox_inches='tight', dpi=1200)
            plt.close(fig) 
            # fig.savefig(mol+".png")     
        return None

