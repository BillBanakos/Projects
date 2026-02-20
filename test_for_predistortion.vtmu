import numpy as np

# --- Parameters ---
filename = "ramp_signal.dat"
TimeWindow = 10e-6        # 10 us sweep
SampleRate = 160e9         # 160 GHz 
N = int(TimeWindow * SampleRate)

# --- Laser Parameters ---
I_bias = 0.020             # Start current
I_sweep = 0.4           # How much current to add for the sweep

# --- Create the Time Vector ---
t = np.linspace(0, TimeWindow, N)

# --- Generate CURRENT (The Signal) ---
# Iteration 0: Perfect Linear Ramp
current = I_bias + (I_sweep * (t / TimeWindow))

# --- Save for VPI ---
data = np.column_stack((t, current))
np.savetxt(filename, data, fmt='%.12e', delimiter='\t')

print(f"File '{filename}' created successfully.")