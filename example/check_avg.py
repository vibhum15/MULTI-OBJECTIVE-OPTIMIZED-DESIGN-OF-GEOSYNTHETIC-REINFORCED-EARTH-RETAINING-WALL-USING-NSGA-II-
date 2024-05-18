import numpy as np
import matplotlib.pyplot as plt

def func(x):
    length = x[0]
    Tu = x[1]
    h = x[2]
    PI = 22/7  # pi = 3.1415926
    G = 9.81  # g: acceleration due to gravity
    HE = 0.45  # Minimum embedment depth = 0.45mts.
    FI_F = 34.0 * (PI/180.0)  # Internal friction angle for fill
    GAMMA_F = 20  # Unit weight of fill
    FI_B = 30.0 * (PI/180.0)  # Internal friction angle for backfill
    GAMMA_B = 18  # Unit weight of backfill
    S_MIN = 0.2  # Minimum spacing between reinforcement
    LE_MIN = 1.0  # Minimum embedment length of reinforcement
    # Minimum recommended safety factors
    FS_OVD = 2.0  # Overturning
    FS_SLD = 1.5  # Sliding
    FS_BCD = 2.0  # Bearing capacity
    FS_ECD = 1.0  # Eccentricity
    FS_STD = 1.5  # Reinforcement strength
    FS_PLD = 2.0  # Reinforcement pullout
    TA_MIN = 30  # Allowable minimum tensile strength of geosynthetic
    TA_MAX = 60  # Allowable maximum tensile strength of geosynthetic
    # Surcharge details
    BETA1 = 0.0 * (PI/180.0)  # Backfill slope angle
    QS = 0  # Surcharge
    ALPHA_C = 0.05  # Horizontal peak seismic coefficient
    # Cost factors - for leveling pad in dollar
    CO1 = 10  # Levelling pad
    CO2 = 3  # Wall fill dollar per 1000kg
    CO4 = 60  # MCU face limit
    # Engg. & testing cost
    CO5_GX = 10  # Geotextile wall
    CO5_GD = 10  # Geogrid wall
    CO6 = 50  # Installation cost

    Kae = (np.cos(BETA1) * (np.cos(BETA1) - np.sqrt(np.cos(BETA1)**2 - np.cos(FI_B)**2))) / (np.cos(BETA1) + np.sqrt(np.cos(BETA1)**2 - np.cos(FI_B)**2))
    Kai = (np.cos(BETA1) * (np.cos(BETA1) - np.sqrt(np.cos(BETA1)**2 - np.cos(FI_F)**2))) / (np.cos(BETA1) + np.sqrt(np.cos(BETA1)**2 - np.cos(FI_F)**2))
    hd = h + HE
    theta = (PI / 4) + (0.5 * FI_F)
    l_min = LE_MIN + (hd * np.tan(theta))
    l_min = np.where(l_min < (0.7 * hd), 0.7 * hd, l_min)
    F1 = GAMMA_F * hd * length
    LA1 = length / 2.0
    M1 = F1 * LA1
    F2 = QS * length
    LA2 = length / 2.0
    M2 = F2 * LA2
    F3 = 0.5 * Kae * GAMMA_B * hd * hd
    LA3 = hd / 3.0
    M3 = F3 * LA3
    F4 = Kae * QS * hd
    LA4 = hd / 2.0
    M4 = F4 * LA4
    F5 = 0.375 * ALPHA_C * GAMMA_B * hd * hd
    LA5 = 0.6 * hd
    M5 = F5 * LA5
    F6 = 0.5 * ALPHA_C * GAMMA_F * hd * length
    LA6 = hd / 2.0
    M6 = F6 * LA6
    MR = M1 + M2
    MO = M3 + M4 + M5 + M6
    FS_over = MR / MO
    sum_V = F1 + F2
    sum_H = F3 + F4 + F5 + F6
    delta = (2.0 / 3.0) * FI_F
    SR = sum_V * np.tan(delta)
    FS_slid = SR / sum_H
    e = (length / 2) - ((MR - MO) / sum_V)  # e: eccentricity
    FS_ecc = (length / 6.0) / e
    q_max = (sum_V / length) * (1.0 + ((6.0 * e) / length))
    Nq = (np.tan((PI / 4.0) + (FI_F / 2.0))) ** 2 * np.exp(PI * np.tan(FI_F))
    Nc = (Nq - 1) / np.tan(FI_F)
    Nf = 2.0 * (Nq + 1.0) * np.tan(FI_F)
    psi = np.arctan(sum_H / sum_V) * (180.0 / PI)  # psi: angle that resultant makes with vertical
    Fci = Fqi = (1.0 - (psi / 90.0)) ** 2
    Fri = (1.0 - (psi / (FI_F * 180.0 / PI))) ** 2
    q_nu = 0.5 * GAMMA_F * Nf * (length - 2.0 * e) * Fri + QS * Nq * Fqi  # q_nu: net ultimate bearing capacity
    FS_bear = q_nu / q_max
    Td = Tu / FS_STD  # Td: safe design strength of geosynthetic
    P_lat = 0.5 * Kae * GAMMA_F * hd * hd + Kae * QS * hd
    nl = np.ceil(P_lat / Td)
    cost1 = CO1 + (CO2 * (GAMMA_F * hd * length) / 1000)
    cost2 = CO4 * nl
    cost3 = CO5_GD + CO6
    total_cost = cost1 + cost2 + cost3

    return np.array([total_cost*2, abs(h)])




# Define the range of values for x
x1_range = np.linspace(1, 10, 100)
x2_range = np.linspace(10, 100, 100)
x3_range = np.linspace(1, 10, 100)

# Create a meshgrid of the x values
X1, X2, X3 = np.meshgrid(x1_range, x2_range, x3_range)

# Evaluate the function for each combination of x values
Z1, Z2 = func([X1, X2, X3])

# Flatten the Z1 and Z2 arrays
Z1_flat = Z1.ravel()
Z2_flat = Z2.ravel()

# # Compute the mean cost for each height
unique_heights, mean_costs_indices = np.unique(Z2_flat, return_inverse=True)
# mean_costs = np.array([Z1_flat[mean_costs_indices == i].mean() for i in range(len(unique_heights))])

# Compute the root mean square cost for each height
rms_costs = np.array([np.sqrt(np.mean(np.square(Z1_flat[mean_costs_indices == i]))) for i in range(len(unique_heights))])

# Plot the root mean square cost vs height
plt.figure()
plt.scatter(unique_heights, rms_costs, marker='o', color='b', alpha=0.5)
plt.xlabel('Height of retaining wall')
plt.ylabel('Total Cost per meter (USD)')
plt.title('Height(m) vs Cost(USD)')
plt.grid(True)
plt.show()
