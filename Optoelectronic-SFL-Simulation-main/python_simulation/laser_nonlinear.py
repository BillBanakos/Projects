import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import cumulative_trapezoid
from scipy.signal import hilbert, butter, sosfiltfilt, sosfilt 

# Global dictionaries and variables for state management
sim_data = {}
block_coords = {}
global_predistorted_ramp = None
global_target_freq = None
global_time_array = None

# Physical Constants
q_charge = 1.602e-19        
k_boltzmann = 1.38e-23      
Temp_K = 298.0              

# ==========================================
# 1. ADVANCED PHYSICAL COMPONENT MODELS 
# ==========================================
def vpi_laser_advanced(time_array, dt, current_array, I_bias, tuning_eff, thermal_mag, thermal_tau, linewidth):
    thermal_sag = -thermal_mag * (1 - np.exp(-time_array / thermal_tau))
    
    nyquist = 0.5 / dt
    res_freq = min(5e9, 0.95 * nyquist) 
    fm_filter = butter(2, res_freq / nyquist, btype='low', output='sos')
    
    dynamic_current = sosfilt(fm_filter, current_array - I_bias)
    
    # Static Quadratic Non-Linearity (Gain Compression)
    static_nonlinearity = -120.0 * tuning_eff * (dynamic_current ** 2)

    if linewidth > 0:
        phase_variance = 2 * np.pi * linewidth * dt
        phase_noise = np.cumsum(np.random.normal(0, np.sqrt(phase_variance), len(time_array)))
    else:
        phase_noise = np.zeros_like(time_array)
        
    inst_freq = (tuning_eff * dynamic_current) + thermal_sag + static_nonlinearity
    phase = 2 * np.pi * cumulative_trapezoid(inst_freq, dx=dt, initial=0) + phase_noise
    
    P_out = 0.010 
    E_field = np.sqrt(P_out) * np.exp(1j * phase)
    return E_field, inst_freq

def vpi_mzi(E_in, dt, tau):
    delay_samples = int(max(1, np.round(tau / dt)))
    E_delayed = np.zeros_like(E_in)
    if delay_samples < len(E_in):
        E_delayed[delay_samples:] = E_in[:-delay_samples]
    E_out = (E_in + E_delayed) / np.sqrt(2)
    return E_out

def vpi_photodiode_advanced(E_in, dt, pd_bandwidth, responsivity=0.8, R_load=50):
    P_opt = np.abs(E_in)**2
    I_pd_ideal = responsivity * P_opt
    
    noise_bw = 0.5 / dt 
    shot_noise_std = np.sqrt(2 * q_charge * np.mean(I_pd_ideal) * noise_bw)
    shot_noise = np.random.normal(0, shot_noise_std, len(I_pd_ideal))
    
    thermal_noise_std = np.sqrt((4 * k_boltzmann * Temp_K * noise_bw) / R_load)
    thermal_noise = np.random.normal(0, thermal_noise_std, len(I_pd_ideal))
    
    I_total = I_pd_ideal + shot_noise + thermal_noise
    V_rf = I_total * R_load * 1000 
    
    nyquist = 0.5 / dt
    safe_pd_bw = min(pd_bandwidth, 0.95 * nyquist) 
    
    pd_filter = butter(1, safe_pd_bw / nyquist, btype='low', output='sos')
    V_rf_filtered = sosfilt(pd_filter, V_rf)
    
    V_ac = V_rf_filtered - np.mean(V_rf_filtered) 
    return V_ac

def apply_electrical_delay(signal, dt, delay_ns):
    if delay_ns <= 0: return signal
    delay_samples = int(delay_ns * 1e-9 / dt)
    if delay_samples == 0: return signal
    delayed_signal = np.pad(signal, (delay_samples, 0), mode='constant')[:-delay_samples]
    return delayed_signal

def vpi_mixer(V_in, ref_freq, time_array):
    V_ref = np.cos(2 * np.pi * ref_freq * time_array - np.pi/2)
    return V_in * V_ref, V_ref


# ==========================================
# 2. SIMULATION LOGIC (FIXED STABILITY)
# ==========================================
def calculate_predistortion(time_window, sweep_bw, iterations, gain, tau, thermal_mag, thermal_tau, 
                            linewidth, ref_freq, pd_bandwidth, elec_delay):
    global sim_data, global_predistorted_ramp, global_target_freq, global_time_array
    try:
        SampleRate = 40e9 
        N = int(time_window * SampleRate)
        t = np.linspace(0, time_window, N)
        dt = time_window / N
        
        I_bias = 0.030  
        TUNING_EFFICIENCY = 10e12  
        I_sweep = sweep_bw / TUNING_EFFICIENCY
        
        current_ramp = I_bias + (I_sweep * (t / time_window))
        ideal_target_freq = TUNING_EFFICIENCY * (I_sweep * (t / time_window))
        
        freq_history, current_history, rf_history, pd_freq_history = [], [], [], []
        
        nyquist = 0.5 * SampleRate
        lpf_cutoff = min(200e3, 0.95 * nyquist)
        lpf_sos = butter(2, lpf_cutoff / nyquist, btype='low', output='sos')
        
        # --- FIX: REMOVED P-TERM FROM TRAINING PHASE ---
        # We only use the Integrator for learning the bias.
        # This prevents noise from "exploding" the ramp shape.
        
        for iteration in range(iterations):
            delayed_drive_current = apply_electrical_delay(current_ramp, dt, elec_delay)
            
            E_laser, true_inst_freq = vpi_laser_advanced(t, dt, delayed_drive_current, I_bias, 
                                                         TUNING_EFFICIENCY, thermal_mag, thermal_tau, linewidth)
            E_mzi_out = vpi_mzi(E_laser, dt, tau)
            V_pd = vpi_photodiode_advanced(E_mzi_out, dt, pd_bandwidth)
            
            analytic_signal = hilbert(V_pd)
            inst_phase = np.unwrap(np.angle(analytic_signal))
            measured_rf_freq = np.gradient(inst_phase, t) / (2 * np.pi)
            
            # Increased edge masking to 5% to ignore startup transients in RMSE calc
            ignore_idx = int(0.05 * len(measured_rf_freq))
            if ignore_idx > 0:
                measured_rf_freq[:ignore_idx] = ref_freq
                measured_rf_freq[-ignore_idx:] = ref_freq
            
            error_freq = ref_freq - measured_rf_freq
            filtered_error = sosfiltfilt(lpf_sos, error_freq)
            
            # PURE INTEGRAL LEARNING (Stable)
            current_correction = gain * cumulative_trapezoid(filtered_error, dx=dt, initial=0)
            
            freq_history.append(true_inst_freq.copy())
            current_history.append(current_ramp.copy())
            rf_history.append(V_pd.copy())
            pd_freq_history.append(measured_rf_freq.copy())
            
            if iteration < iterations - 1:
                current_ramp = current_ramp + current_correction
                
        global_predistorted_ramp = current_ramp.copy()
        global_target_freq = ideal_target_freq.copy()
        global_time_array = t.copy()
        
        V_mixer, V_ref = vpi_mixer(V_pd, ref_freq, t)
        
        sim_data = {
            'Time': t,
            'Ideal Ramp': {'sig': current_history[0]},
            'Predistortion DSP': {'sig': current_ramp},
            'Predistorted Bias': {'sig': current_ramp},
            'Adder': {'sig': current_ramp}, 
            'SCL Laser': {'freq': true_inst_freq, 'sig': delayed_drive_current}, 
            'MZI Delay': {'sig': V_pd}, 
            'Photodiode': {
                'sig': V_pd, 
                'freq': measured_rf_freq,
                'sig_before': rf_history[0],  
                'sig_after': rf_history[-1]   
            },
            'Reference': {'sig': V_ref, 'freq': np.full_like(t, ref_freq)},
            'Mixer': {'sig': V_mixer},
            'LPF & Integrator': {'sig': current_correction, 'freq': filtered_error}
        }

        # Calculate metrics using the middle 80% to avoid edge effects
        valid_slice = slice(int(0.1*N), int(0.9*N))
        
        lin_err_before = (freq_history[0][valid_slice] - ideal_target_freq[valid_slice]) / 1e9
        lin_err_after = (freq_history[-1][valid_slice] - ideal_target_freq[valid_slice]) / 1e9
        rmse_lin_before = np.sqrt(np.mean(lin_err_before**2))
        rmse_lin_after = np.sqrt(np.mean(lin_err_after**2))
        lin_improvement = (1 - rmse_lin_after / max(1e-12, rmse_lin_before)) * 100

        beat_err_before = (pd_freq_history[0][valid_slice] - ref_freq) / 1e6
        beat_err_after = (pd_freq_history[-1][valid_slice] - ref_freq) / 1e6
        rmse_beat_before = np.sqrt(np.mean(beat_err_before**2))
        rmse_beat_after = np.sqrt(np.mean(beat_err_after**2))
        beat_improvement = (1 - rmse_beat_after / max(1e-12, rmse_beat_before)) * 100

        fig2 = plt.figure(figsize=(12, 8))
        fig2.canvas.manager.set_window_title('Phase 1: Open-Loop Predistortion Learning')
        
        ax_b1 = plt.subplot(2, 1, 1)
        ax_b1.plot(t * 1e6, ideal_target_freq / 1e9, 'k--', linewidth=2, label="Perfect Target")
        ax_b1.plot(t * 1e6, freq_history[0] / 1e9, '#e74c3c', linewidth=2, label="Open Loop (Non-Linear)")
        ax_b1.plot(t * 1e6, freq_history[-1] / 1e9, '#2ecc71', linewidth=2, label=f"Predistorted (Iter {iterations-1})")
        ax_b1.set_title("Optical Frequency Sweep Comparison", fontweight='bold')
        ax_b1.set_ylabel("Frequency (GHz)")
        ax_b1.grid(True, linestyle='--')
        ax_b1.legend()
        
        stats_lin = (f"--- Linearity RMSE ---\n"
                     f"Before: {rmse_lin_before:.3f} GHz\n"
                     f"After: {rmse_lin_after:.3f} GHz\n"
                     f"Improvement: {lin_improvement:.1f}%")
        ax_b1.text(0.02, 0.92, stats_lin, transform=ax_b1.transAxes, fontsize=11, 
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

        ax_b2 = plt.subplot(2, 1, 2)
        ax_b2.axhline(0, color='k', linestyle='--', linewidth=2, label="Zero Error Target")
        ax_b2.plot(t[valid_slice] * 1e6, beat_err_before, '#e74c3c', linewidth=2, alpha=0.8, label="Error Without Predistortion")
        ax_b2.plot(t[valid_slice] * 1e6, beat_err_after, '#2ecc71', linewidth=2, alpha=0.9, label="Error With Predistortion")
        ax_b2.set_title(f"Beat Frequency Error (Deviation from {ref_freq/1e6:.2f} MHz Target)", fontweight='bold')
        ax_b2.set_xlabel("Time (µs)")
        ax_b2.set_ylabel("Error (MHz)")
        ax_b2.grid(True, linestyle='--')
        ax_b2.legend()
        
        stats_beat = (f"--- Beat Error RMSE ---\n"
                      f"Before: {rmse_beat_before:.3f} MHz\n"
                      f"After: {rmse_beat_after:.3f} MHz\n"
                      f"Improvement: {beat_improvement:.1f}%")
        ax_b2.text(0.02, 0.92, stats_beat, transform=ax_b2.transAxes, fontsize=11, 
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        plt.show()

        messagebox.showinfo("Success", "Predistortion Bias Calculated! Left-Click the 'Photodiode' to see the Thesis Spectrograms.")

    except Exception as e:
        messagebox.showerror("Simulation Error", str(e))


def run_closed_loop(time_window, tau, thermal_mag, thermal_tau, linewidth, ref_freq, pd_bandwidth, elec_delay, gain):
    global sim_data, global_predistorted_ramp, global_target_freq, global_time_array
    
    if global_predistorted_ramp is None:
        messagebox.showwarning("Warning", "You must Calculate Predistortion first!")
        return
        
    try:
        t = global_time_array
        dt = t[1] - t[0]
        SampleRate = 1.0 / dt
        I_bias = 0.030  
        TUNING_EFFICIENCY = 10e12  
        N = len(t)

        nyquist = 0.5 * SampleRate
        lpf_cutoff = min(200e3, 0.95 * nyquist)
        lpf_sos = butter(2, lpf_cutoff / nyquist, btype='low', output='sos')
        
        # 1. Base Run 
        delayed_bias = apply_electrical_delay(global_predistorted_ramp, dt, elec_delay)
        E_laser_base, true_inst_freq_base = vpi_laser_advanced(t, dt, delayed_bias, I_bias, TUNING_EFFICIENCY, thermal_mag, thermal_tau, linewidth)
        E_mzi_out = vpi_mzi(E_laser_base, dt, tau)
        V_pd_base = vpi_photodiode_advanced(E_mzi_out, dt, pd_bandwidth)
        
        analytic_signal = hilbert(V_pd_base)
        inst_phase = np.unwrap(np.angle(analytic_signal))
        measured_rf_freq_base = np.gradient(inst_phase, t) / (2 * np.pi)
        
        ignore_idx = int(0.05 * len(measured_rf_freq_base))
        if ignore_idx > 0:
            measured_rf_freq_base[:ignore_idx] = ref_freq
            measured_rf_freq_base[-ignore_idx:] = ref_freq
        
        error_freq_base = ref_freq - measured_rf_freq_base
        filtered_error = sosfiltfilt(lpf_sos, error_freq_base)
        
        # --- IMPROVED: PI CONTROL (Only for Closed Loop) ---
        # The P-term here fights instant noise without ruining the bias file.
        Kp = gain * (1.0e-4) 
        
        integral_term = cumulative_trapezoid(filtered_error, dx=dt, initial=0)
        proportional_term = filtered_error
        
        I_correction = (gain * integral_term) + (Kp * proportional_term)
        closed_loop_current = global_predistorted_ramp + I_correction
        # ---------------------------------------------------
        
        # 2. Closed Loop Run
        delayed_cl_current = apply_electrical_delay(closed_loop_current, dt, elec_delay)
        E_laser_cl, true_inst_freq_cl = vpi_laser_advanced(t, dt, delayed_cl_current, I_bias, TUNING_EFFICIENCY, thermal_mag, thermal_tau, linewidth)
        E_mzi_cl = vpi_mzi(E_laser_cl, dt, tau)
        V_pd_cl = vpi_photodiode_advanced(E_mzi_cl, dt, pd_bandwidth)
        
        analytic_signal_cl = hilbert(V_pd_cl)
        inst_phase_cl = np.unwrap(np.angle(analytic_signal_cl))
        measured_rf_freq_cl = np.gradient(inst_phase_cl, t) / (2 * np.pi)
        
        if ignore_idx > 0:
            measured_rf_freq_cl[:ignore_idx] = ref_freq
            measured_rf_freq_cl[-ignore_idx:] = ref_freq
        
        error_freq_cl = ref_freq - measured_rf_freq_cl
        
        V_mixer, V_ref = vpi_mixer(V_pd_cl, ref_freq, t)
        
        sim_data = {
            'Time': t,
            'Predistorted Bias': {'sig': global_predistorted_ramp},
            'Adder': {'sig': closed_loop_current},
            'SCL Laser': {'freq': true_inst_freq_cl, 'sig': delayed_cl_current}, 
            'MZI Delay': {'sig': V_pd_cl}, 
            'Photodiode': {
                'sig': V_pd_cl, 
                'freq': measured_rf_freq_cl,
                'sig_before': V_pd_base, 
                'sig_after': V_pd_cl      
            },
            'Reference': {'sig': V_ref, 'freq': np.full_like(t, ref_freq)},
            'Mixer': {'sig': V_mixer},
            'LPF & Integrator': {'sig': I_correction, 'freq': error_freq_base}
        }

        # Calculate metrics using middle 80% to ensure stability in reading
        valid_slice = slice(int(0.1*N), int(0.9*N))

        lin_err_base = (true_inst_freq_base[valid_slice] - global_target_freq[valid_slice]) / 1e9
        lin_err_cl = (true_inst_freq_cl[valid_slice] - global_target_freq[valid_slice]) / 1e9
        rmse_lin_base = np.sqrt(np.mean(lin_err_base**2))
        rmse_lin_cl = np.sqrt(np.mean(lin_err_cl**2))
        cl_lin_imp = (1 - rmse_lin_cl / max(1e-12, rmse_lin_base)) * 100

        beat_err_base = error_freq_base[valid_slice] / 1e6
        beat_err_cl = error_freq_cl[valid_slice] / 1e6
        rmse_beat_base = np.sqrt(np.mean(beat_err_base**2))
        rmse_beat_cl = np.sqrt(np.mean(beat_err_cl**2))
        cl_beat_imp = (1 - rmse_beat_cl / max(1e-12, rmse_beat_base)) * 100
        
        fig3 = plt.figure(figsize=(12, 8))
        fig3.canvas.manager.set_window_title('Phase 2: Closed-Loop Real-Time Locking')
        
        ax_c1 = plt.subplot(2, 1, 1)
        ax_c1.plot(t * 1e6, global_target_freq / 1e9, 'k--', linewidth=2, label="Perfect Target")
        ax_c1.plot(t * 1e6, true_inst_freq_cl / 1e9, '#8e44ad', linewidth=2, label="Final Closed-Loop Output")
        ax_c1.set_title("Active Feedback SCL Frequency Output", fontweight='bold')
        ax_c1.set_ylabel("Frequency (GHz)")
        ax_c1.grid(True, linestyle='--')
        ax_c1.legend()
        
        stats_lin_cl = (f"--- Linearity RMSE ---\n"
                        f"Open Loop: {rmse_lin_base:.3f} GHz\n"
                        f"Closed Loop: {rmse_lin_cl:.3f} GHz\n"
                        f"Improvement: {cl_lin_imp:.1f}%")
        ax_c1.text(0.02, 0.92, stats_lin_cl, transform=ax_c1.transAxes, fontsize=11, 
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

        ax_c2 = plt.subplot(2, 1, 2)
        ax_c2.axhline(0, color='k', linestyle='--', linewidth=2, label="Zero Error Target")
        ax_c2.plot(t[valid_slice] * 1e6, beat_err_base, '#e74c3c', linewidth=2, alpha=0.5, label="Residual Error (Open Loop)")
        ax_c2.plot(t[valid_slice] * 1e6, beat_err_cl, '#2980b9', linewidth=2, alpha=0.9, label="Locked Error (Closed Loop)")
        ax_c2.set_title("Real-Time Integrator Feedback Correction", fontweight='bold')
        ax_c2.set_xlabel("Time (µs)")
        ax_c2.set_ylabel("Beat Error (MHz)")
        ax_c2.grid(True, linestyle='--')
        ax_c2.legend()
        
        stats_beat_cl = (f"--- Beat Error RMSE ---\n"
                         f"Open Loop: {rmse_beat_base:.3f} MHz\n"
                         f"Closed Loop: {rmse_beat_cl:.3f} MHz\n"
                         f"Improvement: {cl_beat_imp:.1f}%")
        ax_c2.text(0.02, 0.92, stats_beat_cl, transform=ax_c2.transAxes, fontsize=11, 
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Simulation Error", str(e))

# ==========================================
# 3. INTERACTIVE EQUATION & PROBE HANDLERS
# ==========================================
def show_equations(block_name):
    equations = {
        'Ideal Ramp': r"$I_{ideal}(t) = I_{bias} + I_{sweep} \cdot \left(\frac{t}{T_{window}}\right)$",
        'Predistortion DSP': r"$f_{measured}(t) = \frac{1}{2\pi} \frac{d}{dt} \text{unwrap}(\angle \text{Hilbert}\{V_{PD}(t)\})$" + "\n\n" + r"$I_{pre}(n+1) = I_{pre}(n) + G \int (f_{ref} - f_{meas}) dt$",
        'Predistorted Bias': r"$I_{pre}(t) \quad \text{(Saved from Open-Loop Offline Training)}$",
        'Adder': r"$I_{total}(t) = I_{pre}(t) + I_{integrator}(t)$",
        'SCL Laser': r"$\nu(t) = \eta \cdot I_{total}(t-\tau_{elec}) - \Delta\nu_{th}(1 - e^{-t/\tau_{th}}) + \text{Noise}$" + "\n\n" + r"$E_{out}(t) = \sqrt{P_{out}} \cdot e^{j 2\pi \int_0^t \nu(\tau) d\tau}$",
        'MZI Delay': r"$E_{out}(t) = \frac{1}{\sqrt{2}} \left[ E_{in}(t) + E_{in}(t-\tau_{MZI}) \right]$",
        'Photodiode': r"$I_{PD}(t) = \mathcal{R} |E_{out}(t)|^2 + i_{shot}(t) + i_{thermal}(t)$" + "\n\n" + r"$V_{RF}(t) = \text{LPF}_{BW} \{ I_{PD}(t) \cdot R_{load} \}$",
        'Reference': r"$V_{ref}(t) = \cos(2\pi f_{ref} t - \pi/2)$",
        'Mixer': r"$V_{mix}(t) = V_{PD}(t) \times V_{ref}(t)$",
        'LPF & Integrator': r"$e_{filt}(t) = \text{LPF}_{200kHz} \{ f_{ref} - f_{measured}(t) \}$" + "\n\n" + r"$I_{integrator}(t) = G \int_0^t e_{filt}(\tau) d\tau$"
    }
    
    if block_name not in equations: return
    
    eq_window = tk.Toplevel()
    eq_window.title(f"Mathematical Model: {block_name}")
    eq_window.geometry("550x250")
    eq_window.configure(bg="white")
    
    fig = plt.figure(figsize=(5.5, 2.5), facecolor='white')
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.5, 0.5, equations[block_name], fontsize=14, ha='center', va='center', color='#2c3e50')
    
    canvas = FigureCanvasTkAgg(fig, master=eq_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def plot_block_data(block_name):
    if not sim_data or block_name not in sim_data: return
    data = sim_data[block_name]
    t = sim_data['Time']
    dt = t[1] - t[0]
    SampleRate = 1.0 / dt

    if block_name == 'Photodiode' and 'sig_before' in data:
        sig_before = data['sig_before']
        sig_after = data['sig_after']
        
        factor = int(SampleRate / 20e6)
        if factor < 1: factor = 1
        
        n_len = len(sig_before) - (len(sig_before) % factor)
        ds_before = np.mean(sig_before[:n_len].reshape(-1, factor), axis=1)
        ds_after = np.mean(sig_after[:n_len].reshape(-1, factor), axis=1)
        fs_new = SampleRate / factor
        
        fig_spec, axs_spec = plt.subplots(1, 2, figsize=(12, 5))
        fig_spec.canvas.manager.set_window_title("Photodiode Spectrograms (Thesis Fig 5.3 Replica)")
        
        Pxx1, freqs1, bins1, im1 = axs_spec[0].specgram(ds_before, NFFT=2048, Fs=fs_new, noverlap=1900, cmap='jet', scale='dB', vmin=-120)
        axs_spec[0].set_title("Spectrogram, Free-Running", fontweight='bold')
        axs_spec[0].set_ylabel("Photodetector Frequency (MHz)")
        axs_spec[0].set_xlabel("Time (ms)")
        axs_spec[0].set_ylim(1.0e6, 6.0e6) 
        cb1 = fig_spec.colorbar(im1, ax=axs_spec[0])
        cb1.set_label('dB')
        
        Pxx2, freqs2, bins2, im2 = axs_spec[1].specgram(ds_after, NFFT=2048, Fs=fs_new, noverlap=1900, cmap='jet', scale='dB', vmin=-120)
        axs_spec[1].set_title("Spectrogram, Predistorted", fontweight='bold')
        axs_spec[1].set_ylabel("Photodetector Frequency (MHz)")
        axs_spec[1].set_xlabel("Time (ms)")
        axs_spec[1].set_ylim(1.0e6, 6.0e6) 
        cb2 = fig_spec.colorbar(im2, ax=axs_spec[1])
        cb2.set_label('dB')
        
        for ax in axs_spec:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1e6:g}"))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*1000:g}"))
            
        plt.tight_layout()
        plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    fig.canvas.manager.set_window_title(f"Oscilloscope Probe: {block_name}")

    if 'sig' in data:
        plot_t, plot_sig = t * 1e6, data['sig']
        if block_name in ['Photodiode', 'Reference', 'Mixer', 'LPF & Integrator']:
            zoom_idx = min(len(t), int(1.5e-6 / dt)) 
            plot_t, plot_sig = plot_t[:zoom_idx], plot_sig[:zoom_idx]
        axs[0].plot(plot_t, plot_sig, color='#2980b9')
        axs[0].set_title(f"{block_name} - Amplitude vs Time")
        axs[0].set_xlabel("Time (µs)")
        axs[0].grid(True)

    if 'freq' in data:
        scale = 1e9 if block_name == 'SCL Laser' else 1e6
        unit = "GHz" if block_name == 'SCL Laser' else "MHz"
        axs[1].plot(t * 1e6, data['freq'] / scale, color='#27ae60')
        axs[1].set_title(f"{block_name} - Instantaneous Frequency")
        axs[1].set_ylabel(f"Freq ({unit})")
        axs[1].grid(True)

    if block_name == 'SCL Laser':
        axs[2].hist(data['freq'] / 1e9, bins=200, color='#8e44ad', alpha=0.7)
        axs[2].set_title("Simulated Optical Spectrum Analyzer (OSA) Trace")
        axs[2].set_xlabel("Optical Frequency Offset (GHz)")
    elif 'sig' in data and block_name in ['Photodiode', 'Reference', 'Mixer', 'LPF & Integrator']:
        sig = data['sig'] - np.mean(data['sig'])
        freqs = np.fft.rfftfreq(len(sig), d=dt)
        fft_mag = np.abs(np.fft.rfft(sig))
        fft_db = 20 * np.log10(fft_mag / np.max(fft_mag) + 1e-12)
        axs[2].plot(freqs / 1e6, fft_db, color='#c0392b')
        axs[2].set_title(f"{block_name} - RF Spectrum")
        axs[2].set_xlabel("Frequency (MHz)")
        axs[2].set_ylabel("Magnitude (dB)")
        axs[2].set_xlim(0, max(sim_data['Reference']['freq'][0] / 1e6 * 3, 50))
        axs[2].set_ylim(-80, 5)

    plt.tight_layout()
    plt.show()

def on_canvas_left_click(event):
    if not sim_data: return
    for block_name, (x1, y1, x2, y2) in block_coords.items():
        if x1 <= event.x <= x2 and y1 <= event.y <= y2:
            plot_block_data(block_name)
            break

def on_canvas_right_click(event):
    for block_name, (x1, y1, x2, y2) in block_coords.items():
        if x1 <= event.x <= x2 and y1 <= event.y <= y2:
            show_equations(block_name)
            break

# ==========================================
# 4. GUI INTERFACE
# ==========================================
def draw_block(canvas, x, y, name, color="lightblue"):
    canvas.create_rectangle(x, y, x+100, y+60, fill=color, outline="#333333", width=2)
    canvas.create_text(x+50, y+30, text=name.replace(" ", "\n"), font=("Segoe UI", 9, "bold"), justify="center")
    block_coords[name] = (x, y, x+100, y+60)

def draw_arrow(canvas, x1, y1, x2, y2):
    canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=2, fill="#555555")

def build_gui():
    root = tk.Tk()
    root.title("Thesis OPLL Simulator")
    root.geometry("900x900")
    root.configure(bg="#f8f9fa")
    
    header = tk.Frame(root, bg="#2c3e50", pady=15)
    header.pack(fill="x")
    tk.Label(header, text="Two-Stage Optoelectronic SFL Simulator", font=("Segoe UI", 16, "bold"), fg="white", bg="#2c3e50").pack()
    tk.Label(header, text="Left-Click to Probe Signals | Right-Click to View Math Equations", font=("Segoe UI", 10, "italic"), fg="#ecf0f1", bg="#2c3e50").pack()
    
    canvas_frame = tk.Frame(root, bg="#f8f9fa", pady=10)
    canvas_frame.pack()
    canvas = tk.Canvas(canvas_frame, width=850, height=300, bg="white", relief="flat", highlightthickness=1)
    canvas.pack()
    
    canvas.bind("<Button-1>", on_canvas_left_click)
    canvas.bind("<Button-3>", on_canvas_right_click)
    canvas.bind("<Button-2>", on_canvas_right_click)
    
    draw_block(canvas, 30, 40, "Ideal Ramp", "#e0e0e0")
    draw_block(canvas, 160, 40, "Predistortion DSP", "#cfd8dc")
    draw_block(canvas, 300, 40, "Predistorted Bias", "#e0e0e0")
    draw_block(canvas, 430, 40, "Adder", "#ffe0b2")
    draw_block(canvas, 560, 40, "SCL Laser", "#ffcdd2")
    draw_block(canvas, 690, 40, "MZI Delay", "#c8e6c9")
    
    draw_block(canvas, 690, 180, "Photodiode", "#b3e5fc")
    draw_block(canvas, 560, 180, "Mixer", "#d1c4e9")
    draw_block(canvas, 560, 260, "Reference", "#e0e0e0")
    draw_block(canvas, 430, 180, "LPF & Integrator", "#fff9c4")
    
    draw_arrow(canvas, 130, 70, 160, 70)
    draw_arrow(canvas, 260, 70, 300, 70)
    draw_arrow(canvas, 400, 70, 430, 70)
    draw_arrow(canvas, 530, 70, 560, 70)
    draw_arrow(canvas, 660, 70, 690, 70)
    
    draw_arrow(canvas, 740, 100, 740, 180)            
    draw_arrow(canvas, 690, 210, 660, 210)            
    draw_arrow(canvas, 560, 210, 530, 210)            
    draw_arrow(canvas, 610, 260, 610, 240) 
    
    canvas.create_line(430, 210, 400, 210, width=2, fill="#555555")
    canvas.create_line(400, 210, 400, 130, width=2, fill="#555555")
    draw_arrow(canvas, 400, 130, 450, 100)
    
    control_frame = tk.LabelFrame(root, text="Thesis Project Parameters (DFB SCL)", bg="#ffffff", font=("Segoe UI", 11, "bold"), pady=10)
    control_frame.pack(pady=10, padx=25, fill="x")
    
    time_var = tk.DoubleVar(value=1000.0)        
    bw_var = tk.DoubleVar(value=100.0)            
    iter_var = tk.IntVar(value=5)                
    gain_var = tk.DoubleVar(value=2.0e-6)         # Reverted to safer gain for stability
    tau_var = tk.DoubleVar(value=28.6)            
    thermal_mag_var = tk.DoubleVar(value=25.0)    
    thermal_tau_var = tk.DoubleVar(value=300.0)  
    ref_freq_var = tk.DoubleVar(value=2.86)       
    delay_var = tk.DoubleVar(value=0.0)           
    pd_bw_var = tk.DoubleVar(value=10.0)          
    linewidth_var = tk.DoubleVar(value=0.0)       
    
    labels = [
        ("Time Window (µs):", time_var), ("Sweep BW (GHz):", bw_var),
        ("MZI Delay τ (ns):", tau_var), ("Ref Freq (MHz):", ref_freq_var), 
        ("Predistort Iterations:", iter_var), ("Thermal Sag Mag (GHz):", thermal_mag_var),
        ("Thermal Time Const (µs):", thermal_tau_var), ("Integrator Loop Gain:", gain_var),
        ("Elec. Cable Delay (ns):", delay_var), ("PD Bandwidth (GHz):", pd_bw_var),
        ("Laser Linewidth (MHz):", linewidth_var)
    ]
    
    for i, (text, var) in enumerate(labels):
        row, col = i // 3, (i % 3) * 2
        tk.Label(control_frame, text=text, bg="#ffffff", font=("Segoe UI", 9)).grid(row=row, column=col, padx=5, pady=8, sticky="e")
        tk.Entry(control_frame, textvariable=var, width=10, font=("Segoe UI", 9)).grid(row=row, column=col+1, padx=5, sticky="w")
        
    def on_calc_pre_click():
        messagebox.showinfo("Processing", "Computing 1ms at 40 GHz...\nThis generates 40 million points per iteration and may take 10-20 seconds. Please wait after clicking OK.")
        root.update()
        calculate_predistortion(
            time_var.get() * 1e-6, bw_var.get() * 1e9, iter_var.get(), gain_var.get(),
            tau_var.get() * 1e-9, thermal_mag_var.get() * 1e9, thermal_tau_var.get() * 1e-6,
            linewidth_var.get() * 1e6, ref_freq_var.get() * 1e6, pd_bw_var.get() * 1e9,
            delay_var.get()
        )

    def on_run_loop_click():
        run_closed_loop(
            time_var.get() * 1e-6, tau_var.get() * 1e-9, thermal_mag_var.get() * 1e9, 
            thermal_tau_var.get() * 1e-6, linewidth_var.get() * 1e6, ref_freq_var.get() * 1e6, 
            pd_bw_var.get() * 1e9, delay_var.get(), gain_var.get()
        )
        
    btn_frame = tk.Frame(root, bg="#f8f9fa")
    btn_frame.pack(pady=10)
    
    tk.Button(btn_frame, text="1. CALCULATE PREDISTORTION", font=("Segoe UI", 11, "bold"), bg="#3498db", fg="white", activebackground="#2980b9", relief="flat", padx=15, pady=10, command=on_calc_pre_click).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="2. SIMULATE CLOSED LOOP", font=("Segoe UI", 11, "bold"), bg="#2ecc71", fg="white", activebackground="#27ae60", relief="flat", padx=15, pady=10, command=on_run_loop_click).pack(side=tk.LEFT, padx=10)

    root.mainloop()

if __name__ == "__main__":
    build_gui()
