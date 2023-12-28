import numpy as np
import matplotlib.pyplot as plt




import numpy as np
import matplotlib.pyplot as plt

def calculate_dft_coefficients(signal):
    """
    Calculate the DFT coefficients for a given signal.

    Parameters:
    - signal: Input signal (numpy array)

    Returns:
    - dft_coefficients: DFT coefficients (complex numpy array)
    """
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    # Constructing the matrix of complex exponentials
    e = np.exp(-2j * np.pi * k * n / N)
    
    # Calculating DFT coefficients
    dft_coefficients = np.sum(signal * e, axis=1)
    return dft_coefficients




def get_conjugate_transpose(freq,sampling_rate):
    """
    Calculate the DFT coefficients for a given signal.

    Parameters:
    - signal: Input signal (numpy array)

    Returns:
    - dft_coefficients: DFT coefficients (complex numpy array)
    """
    N = sampling_rate
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    # Constructing the matrix of complex exponentials
    e = np.exp(-2j * np.pi * k * n / N)
    return e[:,freq]





print("test")
# Example usage:
# Create a simple signal (e.g., a sine wave)
fs = 1000  # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector
frequency = 10  # Frequency of the sine wave
signal = np.sin(2 * np.pi * frequency * t)

conj_10 = get_conjugate_transpose(10,1000)

conj_20 = get_conjugate_transpose(20,1000)

plt.plot(t, signal)
plt.title('Original Signal')
plt.show()

plt.plot(np.abs(np.convolve(signal,conj_10[:400],'valid'))[::100])
plt.plot(np.abs(np.convolve(signal,conj_20[:400],'valid'))[::100])
plt.show()
