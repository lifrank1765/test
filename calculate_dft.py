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

# Example usage:
# Create a simple signal (e.g., a sine wave)
fs = 1000  # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector
frequency = 5  # Frequency of the sine wave
signal = np.sin(2 * np.pi * frequency * t)

# Calculate DFT coefficients using the custom function
dft_coefficients_custom = calculate_dft_coefficients(signal)

# Display the original signal and its DFT
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')

plt.subplot(2, 1, 2)
plt.stem(np.abs(dft_coefficients_custom))
plt.title('DFT Coefficients (Custom)')

plt.show()




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
    print(e.shape)
    
    # Calculating DFT coefficients
    dft_coefficients = np.sum(signal * e, axis=1)
    #a = signal * e
    #print(a[:,5])
    print(np.abs(dft_coefficients[:10]))
    print(np.abs(np.sum(signal * e[:,5])))
    plt.plot( e[:,10].real)
    plt.plot( e[:,10].imag)
    
    plt.show()
    #print(signal * e[:,5].T)
    #print(e[:,5])
    
    return dft_coefficients

print("test")
# Example usage:
# Create a simple signal (e.g., a sine wave)
fs = 1000  # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector
frequency = 5  # Frequency of the sine wave
signal = np.sin(2 * np.pi * frequency * t)

# Calculate DFT coefficients using the custom function
dft_coefficients_custom = calculate_dft_coefficients(signal)
#print(np.abs(dft_coefficients_custom))



plt.plot(t, signal)
plt.title('Original Signal')
plt.show()

# Display the original signal and its DFT
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')

plt.subplot(2, 1, 2)
plt.stem(np.abs(dft_coefficients_custom))
plt.title('DFT Coefficients (Custom)')

plt.show()
