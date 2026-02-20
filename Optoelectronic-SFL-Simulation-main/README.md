# Phase-Locked Optoelectronic Swept-Frequency Laser (SFL) Simulation

## Overview
This repository contains two different approaches (one using VPIphotonics and one using Python) to simulate an Optoelectronic Swept-Frequency Laser (SFL). The main objective of this project is to generate an optical signal whose instantaneous frequency follows a highly precise, prescribed linear chirp, using a simple semiconductor laser (SCL) whose frequency modulation (FM) response is inherently non linear. This becomes possible by utilizing a closed-loop phase-locked architecture and an iterative numerical predistortion algorithm to linearize the optical sweep. 

## System Architecture
The feedback loop consists of the following core components:
* **Semiconductor Laser (SCL):** The laser whose frequency needs to be linearized.
* **Mach-Zehnder Interferometer (MZI):** A delayed MZI is used to convert the optical chirp slope into an RF beat frequency.
* **Photodetector (PD):** Converts the optical intensity from the MZI into an electrical signal.
* **High-pass Filter & Amplifier:** Used to cut-off unnecessary DC components and amplify PD's output.
* **Mixer:** Acts as a phase detector, comparing the measured beat frequency with a reference frequency.
* **Integrator (Loop Filter):** Utilizes a Type-I approximation to close the loop, suppressing low-frequency noise from the laser via high loop gain.

## Predistortion Algorithm
To ensure the loop can lock, the open-loop driving current must generate a chirp that is close to the target linearity. This repository includes a Python-based numerical iterative method to calculate the required predistorted current.

The algorithm extracts a current-dependent "distortion function" and operates in four main steps:
1. **Measurement:** Apply a test current (e.g., a linear ramp) and measure the resulting beat frequency.
2. **Extraction:** Calculate the derivative of the applied current and extract the distortion function.
3. **Calculation:** Calculate the new required current slope necessary to achieve the target reference frequency.
4. **Integration:** Numerically integrate the new slope to generate the predistorted current waveform for the next iteration.

*Note: This process is typically repeated 3-4 times until the measured frequency becomes flat, effectively linearizing the sweep.*

## Repository Structure
* `/python_simulation/`: Contains the Python code for the simulation of the system
* `/vpi_simulation/`: Contains the VPIphotonics schematic files for the closed-loop system simulation along with the Python code for generation of the signals and calculation of the predistortion current.

## Setup and Usage
### Prerequisites
1. **VPIphotonics Design Suite** (for optical/electrical physical layer simulation).
2. **Python 3.8+** with the following libraries:
   * `numpy`
   * `scipy`
   * `matplotlib`

## Mathematical Background
For a complete derivation of the small-signal loop transfer function, component modeling, and the predistortion logic, please refer to the accompanying mathematical documentation in this repository.

## References
This project is based on Naresh Satyan, Arseny Vasilyev, George Rakuljic, Victor Leyva, and Amnon Yariv, "Precise control of broadband frequency chirps using optoelectronic feedback," Opt. Express 17, 15991-15999 (2009).
