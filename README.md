👉 Title
Adaptive Modulation Prototype for SDR Systems

👉 Overview
This repository contains a Python-based prototype of an adaptive modulation communication system. The system dynamically selects modulation schemes based on channel conditions and demonstrates packet-based signaling for receiver-side adaptation.

The prototype simulates a complete transmitter–receiver pipeline, including adaptive modulation selection, packet framing with modulation identifiers, transmission over a noisy channel, and runtime demodulation at the receiver.

👉 Features
- Adaptive modulation switching (BPSK, QPSK, 16-QAM) based on SNR  
- Packet-based transmission with modulation ID embedded in header  
- Dynamic demodulator selection at the receiver  
- AWGN channel simulation  
- Bit Error Rate (BER) computation  
- Visualization of:
  - Constellation diagrams  
  - BER vs SNR performance  
  - Modulation switching behavior over time  

👉 System Overview
The system follows a simplified adaptive transceiver architecture:

Bit Source → Adaptive Modulator → Packet Builder → Channel → Receiver → Adaptive Demodulator → Bit Output

A feedback mechanism based on channel conditions (e.g., SNR) is used to select the appropriate modulation scheme.

👉 Example Outputs
The prototype generates visualizations including:
- Adaptive modulation switching across packets
- BER vs SNR curves for different modulation schemes
- Constellation diagrams under clean and noisy conditions

👉 Usage
Requirements:
- Python 3.x
- NumPy
- Matplotlib

Run the main script:

python main.py

The script will simulate transmission, print modulation decisions, and display plots.

👉 Purpose
This prototype was developed as part of a GNU Radio project proposal to demonstrate adaptive modulation with packet-based signaling and runtime demodulation.

It serves as a proof-of-concept prior to full implementation using GNU Radio blocks and an Out-of-Tree (OOT) module.

👉 Future Work
- Integration with GNU Radio as custom blocks  
- Real-time SDR implementation  
- Machine learning-based modulation selection  
- Support for OFDM systems  
