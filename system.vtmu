import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

# ==========================================
#           CONFIGURATION
# ==========================================
SWEEP_TIME = 10e-6        # 10 microseconds
N_POINTS_OUT = 25000         # Output resolution
SAMPLE_RATE =  160e9         # User specified: 5 GHz
TAU = 28.6e-9              # MZI Delay (28.6 ns)
TARGET_BEAT_FREQ = 1e9    # 1.1 GHz
TARGET_SLOPE = TARGET_BEAT_FREQ / TAU  # ~10^16 Hz/s

# ==========================================
#           ROBUST ALGORITHM
# ==========================================
# def calculate_predistortion_robust(time, current, freq_measured, target_freq):
#     # 1. Smooth data to remove noise
#     win_size = int(len(time) * 0.02) 
#     if win_size < 3: win_size = 3
    
#     curr_smooth = uniform_filter1d(current, size=win_size)
#     freq_smooth = uniform_filter1d(freq_measured, size=win_size)

#     # 2. Calculate di/dt (Current Slope)
#     di_dt = np.gradient(curr_smooth, time)
#     di_dt = np.maximum(di_dt, 1e-6) # Avoid division by zero

#     # 3. Tuning Efficiency (F_dist)
#     # This represents Hz per (Amp/s)
#     tuning_efficiency = np.abs(freq_smooth / di_dt)

#     # 4. Calculate NEW di/dt
#     # new_di_dt = Target_Beat_Freq / Tuning_Efficiency
#     new_di_dt = target_freq / tuning_efficiency

#     # Safety Clamp (Prevent massive spikes)
#     new_di_dt = np.clip(new_di_dt, 0.0, 1e5) 

#     # 5. Integrate to get Current
#     dt = time[1] - time[0]
#     new_current = np.cumsum(new_di_dt) * dt
    
#     # Offset to start at the same bias as the previous iteration
#     new_current = new_current - new_current[0] + current[0]
    
#     return new_current

def calculate_predistortion_robust(time, current, freq_measured, target_freq):
    # 1. Smooth data to remove noise
    win_size = int(len(time) * 0.02) 
    if win_size < 3: win_size = 3
    
    curr_smooth = uniform_filter1d(current, size=win_size)
    freq_smooth = uniform_filter1d(freq_measured, size=win_size)

    # 2. Calculate di/dt (Current Slope)
    di_dt = np.gradient(curr_smooth, time)
    di_dt = np.maximum(di_dt, 1e-6) # Avoid division by zero

    # 3. Tuning Efficiency (F_dist)
    # This represents Hz per (Amp/s)
    tuning_efficiency = np.abs(freq_smooth / di_dt)

    # 4. Calculate NEW di/dt
    # new_di_dt = Target_Beat_Freq / Tuning_Efficiency
    new_di_dt = target_freq / tuning_efficiency

    # Safety Clamp (Prevent massive spikes)
    new_di_dt = np.clip(new_di_dt, 0.0, 1e5) 

    # 5. Integrate to get Current
    dt = time[1] - time[0]
    new_current = np.cumsum(new_di_dt) * dt
    
    # Offset to start at the same bias as the previous iteration
    new_current = new_current - new_current[0] + current[0]
    
    return new_current


# ==========================================
#           MAIN EXECUTION
# ==========================================
vpi_file = "vpi_output.txt"
old_current_file = "ramp_signal.dat"  
new_current_file = "current_iter1.dat"  

try:
    print(f"--- PARAMETERS ---")
    print(f"VPI Sample Rate:  {SAMPLE_RATE/1e9:.1f} GHz")
    print(f"Target Beat Freq: {TARGET_BEAT_FREQ/1e9:.4f} GHz")
    print(f"------------------")

    # --- 1. Load Data ---
    print("Loading files...")
    
    # Load VPI Output (Column 0 = Frequency, Column 1 = Power)
    vpi_raw = np.loadtxt(vpi_file, comments=['%', '#', '"'])
    
    if vpi_raw.ndim > 1:
        freq_vpi_raw = vpi_raw[:, 0]  # <--- FIXED: Read Freq from Col 0
    else:
        freq_vpi_raw = vpi_raw        # Fallback for 1D file

    # Generate Time Axis (Implicit in VPI text exports)
    time_vpi_raw = np.linspace(0, SWEEP_TIME, len(freq_vpi_raw))

    # Load Old Current (Col 0 = Time, Col 1 = Current)
    curr_raw = np.loadtxt(old_current_file, comments=['%', '#', '"'])
    if curr_raw.ndim > 1:
        time_curr_raw = curr_raw[:, 0]
        vals_curr_raw = curr_raw[:, 1]
    else:
        # Fallback if current file has no time column
        time_curr_raw = np.linspace(0, SWEEP_TIME, len(curr_raw))
        vals_curr_raw = curr_raw

    # --- 2. Alignment ---
    print(f"Resampling to {N_POINTS_OUT} points...")
    new_time_grid = np.linspace(0, SWEEP_TIME, N_POINTS_OUT)
    
    # Interpolate VPI Data to new grid
    f_interp_vpi = interp1d(time_vpi_raw, freq_vpi_raw, fill_value="extrapolate")
    opt_freq_aligned = f_interp_vpi(new_time_grid)
    
    # Interpolate Current to new grid
    f_interp_curr = interp1d(time_curr_raw, vals_curr_raw, fill_value="extrapolate")
    current_vals_aligned = f_interp_curr(new_time_grid)

    # --- 3. Identify Data Type (Optical vs Beat) ---
    # Case A: Data is Optical Frequency (Chirping ~100 GHz) -> Needs Gradient
    # Case B: Data is Beat Frequency (Constant ~1.1 GHz) -> No Gradient
    
    data_range = np.ptp(opt_freq_aligned) # Peak-to-Peak range
    data_mean = np.mean(opt_freq_aligned)

    if data_range > 1e9: 
        # Large range implies Optical Chirp (e.g. 193.1 THz -> 193.2 THz)
        print("Detected Optical Frequency Sweep. Calculating Beat Frequency...")
        chirp_slope = np.gradient(opt_freq_aligned, new_time_grid)
        chirp_slope = uniform_filter1d(chirp_slope, size=50) # Smooth derivative
        pd_freq_measured = chirp_slope * TAU
    else:
        # Small range implies we are reading the Beat Frequency directly
        print("Detected Beat Frequency (or unscaled data). Using directly.")
        pd_freq_measured = opt_freq_aligned
        
        # Scaling Check for VPI "Normalized" output
        if np.mean(np.abs(pd_freq_measured)) < 1e6:
             print("WARNING: Data values are tiny. Multiplying by Sample Rate...")
             pd_freq_measured *= SAMPLE_RATE

    # --- 4. Diagnostics ---
    mean_meas = np.mean(pd_freq_measured)
    print(f"Measured Mean Beat Freq: {mean_meas/1e9:.4f} GHz")

    if mean_meas < 100: 
        print("CRITICAL ERROR: Measured frequency is effectively zero.")
        print("Check: 1. Is VPI Laser ON? 2. Is 'vpi_output.txt' actually frequency?")
        raise ValueError("Measured frequency too low.")

    # --- 5. Run Algorithm ---
    print("Calculating correction...")
    new_current = calculate_predistortion_robust(
        new_time_grid, current_vals_aligned, pd_freq_measured, TARGET_BEAT_FREQ
    )

    # --- 6. Save ---
    # Final check for negative currents
    if np.min(new_current) < 0:
        print("Warning: Clamping negative currents to 0")
        new_current = np.maximum(new_current, 0)

    output_data = np.column_stack((new_time_grid, new_current))
    np.savetxt(new_current_file, output_data, fmt='%.12e', delimiter='\t')
    print(f"Success! Saved '{new_current_file}'")

    # --- Plot ---
    plt.figure(figsize=(10,6))
    
    plt.subplot(2,1,1)
    plt.plot(new_time_grid*1e6, pd_freq_measured/1e9, label='Measured')
    plt.plot(new_time_grid*1e6, np.ones_like(new_time_grid)*TARGET_BEAT_FREQ/1e9, 'r--', label=f'Target ({TARGET_BEAT_FREQ/1e9:.1f} GHz)')
    plt.ylabel("Beat Freq (GHz)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(new_time_grid*1e6, current_vals_aligned*1e3, label='Old')
    plt.plot(new_time_grid*1e6, new_current*1e3, '--', label='New')
    plt.ylabel("Current (mA)")
    plt.xlabel("Time (us)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error: {e}")