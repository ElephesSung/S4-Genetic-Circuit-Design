#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:00:27 2023

@author: Elephes Sung 
Research Postgraduate, DoLS, FoNS, Imperial
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


#Constant of AHL and Arab ODEs
k1 = 0.005
k2 = 0.004

#Constant of HrpR ODE
k_R = 10
K_R  = 3.4e-6
#Alpha_R = 0.03
Alpha_R = 0
n_R = 2
Sigma_R = k_R/1970


#Constant of HrpS ODE
k_S = 10
K_S = 0.513
#Alpha_S = 0.001
Alpha_S = 0
n_S = 1
Sigma_S = k_S/1.3e4

#Constant of T7 ODE
Sigma_T7 = 0.001/60
k_L = 10
n_RL = 2.4
n_SL = 1.8
K_RL = 206
K_SL = 3135

k_ini = 0.0015/60
k_f = 0.00013 / 60 * 10**6
k_b = 0.0003/60
RBS = 4000
k_PR  = (RBS/10**5)*0.018/60
CopyN = 100
k_T7 = k_ini*k_f*CopyN*k_PR/k_b


#Constant of Cl ODE
Sigma_Cl = Sigma_T7

#Constant of GFP ODE
k_C = 1000
#Alpha_C = 0.05
Alpha_C = 0
n_C = 7.6 
K_C = 111
Sigma_G = k_C/7.5e4

#Initial Concentration
init_AHL = 0.1
init_Arab = 0
init_HrpS = 0
init_HrpR = 0
init_T7 = 0
init_Cl = 0
init_GFP = k_C/Sigma_G*(Alpha_C + 1)
initial_conditions = [init_AHL, init_Arab, init_HrpR, init_HrpS, init_T7, init_Cl, init_GFP]

#ODEs
def system(concentrations, t):
    AHL, Arab, HrpR, HrpS, T7, Cl, GFP = concentrations
    
    AHL_term = AHL**n_R / (AHL**n_R + K_R**n_R) if AHL > 0 else 0
    Arab_term = Arab**n_S / (Arab**n_S + K_S**n_S) if Arab > 0 else 0
   
   
    dAHL_dt =  -k1 * AHL
    dArab_dt = -k2 * Arab
    dHrpR_dt = k_R * (Alpha_R + AHL_term) - Sigma_R * HrpR
    dHrpS_dt = k_S * (Alpha_S + Arab_term) - Sigma_S * HrpS
    dT7_dt = k_L * (HrpR**n_RL / (HrpR**n_RL + K_RL**n_RL)) * (HrpS**n_SL / (HrpS**n_SL + K_SL**n_SL)) + k_T7 * T7 - Sigma_T7 * T7
    dCl_dt = k_T7 * T7 - Sigma_Cl * Cl
    dGFP_dt = k_C * (Alpha_C + K_C**n_C / (Cl**n_C + K_C**n_C)) - Sigma_G * GFP
    
    return [dAHL_dt, dArab_dt, dHrpR_dt, dHrpS_dt, dT7_dt, dCl_dt, dGFP_dt]

end_time = 3000
t = np.linspace(0, end_time, 1000)  


concentrations_over_time = odeint(system, initial_conditions, t)


AHL, Arab, HrpR, HrpS, T7, Cl, GFP = concentrations_over_time.T

fig, ax1 = plt.subplots(figsize=(15, 10), dpi=500)

# Plot AHL, Arab, HrpR, HrpS, T7, Cl on the left y-axis
ax1.plot(t, AHL, label='AHL', linewidth=8, color='C8')
ax1.plot(t, Arab, label='Arab', linewidth=8, color='C0')
ax1.plot(t, HrpR, label='HrpR', linewidth=4, color='C1', alpha=0.4)
ax1.plot(t, HrpS, label='HrpS', linewidth=4, color='C6', alpha=0.4)
ax1.plot(t, T7, label='T7', linewidth=4, color='blue', alpha=0.4)
ax1.plot(t, Cl, label='Cl', linewidth=4, color='C7', alpha=0.4)

ax1.set_xlabel('Time (s)', fontsize=25)
ax1.set_ylim(0, 2100)
ax1.set_ylabel('Concentrations', fontsize=25)
ax1.tick_params(axis='both', labelsize=18)



# Create a secondary y-axis for GFP
ax2 = ax1.twinx()
ax2.plot(t, GFP, linewidth=8, color='g')
ax2.set_ylim(0, 90000)
ax2.set_ylabel('GFP Expression Level', fontsize=25)
ax2.tick_params(axis='y', labelsize=18)



# Set the limits for the x-axis (you can adjust these according to your needs)
ax1.set_xlim(0, end_time)
ax2.set_xlim(0, end_time)



plt.show()




