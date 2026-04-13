import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

def create_BL_profile(U_inf,delta,height=1.5,total_pressure = False):
    
    # Constants
    nu = 1.48e-5 # m^2/s
    kappa = 0.41
    B = 5.0

    # Compute consistent u_tau
    u_tau = compute_u_tau(U_inf, delta, nu)

    # --- Grids ---
    y = np.geomspace(1e-9, delta, 500)
    y_plus = y * u_tau / nu

    # --- Spalding velocity profile ---
    u_plus = spalding_profile(y_plus)
    u_spalding = u_plus * u_tau

    # --- Add channel height profile ---
    y_farfield = np.linspace(y[-1],height,4)
    u_farfield = U_inf*np.ones_like(y_farfield)

    u_total = np.concat((u_spalding,u_farfield[1:]))
    y_total = np.concat((y,y_farfield[1:]))

    y_total = np.insert(y_total, 0, 0)
    u_total = np.insert(u_total, 0, 0)

    y_combined = np.concatenate((y_total,y_total))
    u_combined = np.concatenate((u_total,u_total))
    velo_y=np.zeros_like(y_combined)
    velo_z=np.zeros_like(y_combined)
    x_vals = x_val*np.ones_like(y_combined)
    y_min=-W/2
    y_arr1 = y_min*np.ones_like(y_total)
    y_arr2 = -y_min*np.ones_like(y_total)
    y_arr_tot =np.concatenate((y_arr1,y_arr2))

    output_arr = pd.DataFrame({'X':x_vals,'Y':y_arr_tot,'Z':y_combined,'Velocity_X':u_combined,'Velocity_Y':velo_y,'Velocity_Z':velo_z})
    
    return output_arr
    

# --- Spalding law ---
def spalding_uplus_single(y_plus, kappa=0.41, B=5.0):
    up = y_plus if y_plus < 10 else (1/kappa)*np.log(y_plus) + B

    for _ in range(50):
        exp_term = np.exp(kappa * up)
        f = up + np.exp(-kappa * B) * (
            exp_term - 1 - kappa*up - (kappa*up)**2/2 - (kappa*up)**3/6
        ) - y_plus

        df = 1 + np.exp(-kappa * B) * (
            kappa*exp_term - kappa - kappa**2*up - (kappa**3 * up**2)/2
        )

        up_new = up - f/df
        if abs(up_new - up) < 1e-8:
            break
        up = up_new

    return up


def spalding_profile(y_plus_array):
    return np.array([spalding_uplus_single(yp) for yp in y_plus_array])


# --- Solve for friction velocity (enforce u(delta)=U_inf) ---
def compute_u_tau(U_inf, delta, nu):
    u_tau = 1.0  # initial guess

    for _ in range(100):
        y_plus_delta = delta * u_tau / nu
        u_plus_delta = spalding_uplus_single(y_plus_delta)

        u_tau_new = U_inf / u_plus_delta

        if abs(u_tau_new - u_tau) < 1e-8:
            break

        u_tau = u_tau_new

    return u_tau


def plot_BL(df,ax=None,y_lim = None):
    
    if ax is None:
        fig, ax = plt.subplots()

    mid = len(df)//2
    # Plot using axis object (not plt directly)
    ax.plot(df.iloc[:mid, 3], df.iloc[:mid, 2], label="Spalding Law")

    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Distance from wall (m)")
    ax.set_title("Turbulent Boundary Layer")
    ax.grid()


    if y_lim is not None:
        ax.set_ylim(0,y_lim)

''''''''''''''''''''''''''''''''''''''''''''
'Set Inlet Conditions for CFD Runs'
''''''''''''''''''''''''''''''''''''''''''''

Pa = 101325
rho = 1.225

x_val = -3.5000000000000027
W=4

# Speed
m_initial = 0.05
m_max = 0.3
#m_steps = 6
m_steps = 6

# BL Height Param
channel_height = 1.5 # Chord Lengths (ft-ish), should be constant for all CFD runs
h_param_min = 1e-3
h_param_max = 5e-3

plot = True
total_pressureTF = False
save = False


''''''''''''''''''''''''''''''''''''''''''''
'Calculate Profiles'
''''''''''''''''''''''''''''''''''''''''''''
m_array = np.linspace(m_initial,m_max,m_steps)
a=(1.4*287*273.15)**0.5
h_param_array = np.arange(h_param_min,h_param_max+1e-3,1e-3)

BL_array = []


# Loop Through Generation of Profiles
for mach in m_array:
    velo = mach*a
    for h_param in h_param_array:
        delta = h_param * channel_height
        df = create_BL_profile(velo, delta, total_pressure=total_pressureTF)
        BL_array.append(df)

if plot:
    fig, ax = plt.subplots()
    for BL_df in BL_array:
        plot_BL(BL_df, ax=ax,y_lim=0.01)

if total_pressureTF:
    for BL_df in BL_array:
        print()

if save:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"BL_data_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    # Loop through the DataFrame array and save each as CSV
    for i, BL_df in enumerate(BL_array):
        mach_idx = i // len(h_param_array)
        mach_value = m_array[mach_idx]
        h_idx = i % len(h_param_array)
        h_value = h_param_array[h_idx]

        csv_filename = os.path.join(folder_name, f"BL_Mach{mach_value:.2f}_h{h_value:.3f}.csv")
        BL_df.to_csv(csv_filename, index=False)
        (f"Saved {csv_filename}")


print(f'Total CFD runs: {len(BL_array)}')
if True:
    plt.show()

