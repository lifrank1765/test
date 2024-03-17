import numpy as np
from scipy.signal import find_peaks, convolve
import matplotlib.pyplot as plt
def remove_clicks(audio_data, sample_rate, threshold=0.1, window_size=0.04):
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Find peaks (clicks)
    peaks, _ = find_peaks(np.abs(audio_data), height=threshold, rel_height=0.75, width=[0, 800], distance=0.01 * sample_rate)

    # Interpolate over clicks
    cleaned_audio = np.copy(audio_data)
    for peak in peaks:
        # Smooth out the click region by averaging neighboring samples
        window_size_samples = int(sample_rate * window_size)  # Convert window size to samples
        if peak - window_size_samples // 2 > 0 and cleaned_audio.size > peak + window_size_samples // 2:
            window = np.ones(window_size_samples) / window_size_samples
            cleaned_audio[peak - window_size_samples // 2: peak + window_size_samples // 2] = \
                np.convolve(cleaned_audio[peak - window_size_samples // 2: peak + window_size_samples // 2], window, mode='same')

    print("XXXXXXXXXXXXX")
    plt.plot(cleaned_audio)
    plt.show()
    return sample_rate, cleaned_audio





import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.interpolation.UnivariateInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.FastMath;
import java.util.ArrayList;
import java.util.List;

public class ClicksRemoval {
    public static double[] removeClicks(double[] audioData, int sampleRate, double threshold) {
        // Convert to mono if stereo
        if (audioData.length > 1) {
            audioData = convertToMono(audioData);
        }

        // Find peaks (clicks)
        List<Integer> peaks = findPeaks(audioData, threshold);

        // Apply smoothing only to click regions
        double[] cleanedAudio = audioData.clone();
        for (int peak : peaks) {
            int windowSize = (int) (sampleRate * 0.02); // Tune window size
            int startIdx = Math.max(0, peak - windowSize / 2);
            int endIdx = Math.min(cleanedAudio.length, peak + windowSize / 2 + 1);
            double[] window = new double[endIdx - startIdx];
            System.arraycopy(cleanedAudio, startIdx, window, 0, endIdx - startIdx);
            double[] smoothedWindow = movingAverage(window, 3); // Tune moving average window size
            System.arraycopy(smoothedWindow, 0, cleanedAudio, startIdx, smoothedWindow.length);
        }

        return cleanedAudio;
    }

    private static double[] convertToMono(double[] stereoData) {
        double[] monoData = new double[stereoData.length / 2];
        for (int i = 0; i < stereoData.length; i += 2) {
            monoData[i / 2] = (stereoData[i] + stereoData[i + 1]) / 2;
        }
        return monoData;
    }

    private static List<Integer> findPeaks(double[] data, double threshold) {
        List<Integer> peaks = new ArrayList<>();
        for (int i = 1; i < data.length - 1; i++) {
            if (data[i] > threshold && data[i] > data[i - 1] && data[i] > data[i + 1]) {
                peaks.add(i);
            }
        }
        return peaks;
    }

    private static double[] movingAverage(double[] data, int windowSize) {
        double[] smoothedData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            int startIdx = Math.max(0, i - windowSize / 2);
            int endIdx = Math.min(data.length, i + windowSize / 2 + 1);
            smoothedData[i] = StatUtils.mean(data, startIdx, endIdx);
        }
        return smoothedData;
    }

    public static void main(String[] args) {
        // Example usage
        double[] audioData = { /* Your audio data here */ };
        int sampleRate = 44100; // Example sample rate
        double threshold = 0.1; // Example threshold
        double[] cleanedAudioData = removeClicks(audioData, sampleRate, threshold);

        // Print the cleaned audio data
        System.out.println("Cleaned Audio Data:");
        for (double value : cleanedAudioData) {
            System.out.println(value);
        }
    }
}





















def remove_clicks(audio_data, sample_rate, threshold=0.1):
    # Load the audio file

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Find peaks (clicks)
    peaks, _ = find_peaks(np.abs(audio_data), height=threshold,rel_height=0.75,width=[0,800],distance=0.01*sample_rate)

    # Interpolate over clicks
    cleaned_audio = np.copy(audio_data)
    for peak in peaks:
        # Smooth out the click region by averaging neighboring samples
        window_size = int(sample_rate * 0.04)  # 10 ms window
        window = np.ones(window_size) / window_size
        #print(window,peak,window_size,peak - window_size // 2, peak + window_size // 2)
        if peak - window_size // 2 > 0 and  cleaned_audio.size > peak + window_size // 2:
            cleaned_audio[peak - window_size // 2: peak + window_size // 2] = np.convolve(cleaned_audio[peak - window_size // 2: peak + window_size // 2], window, mode='same')
    plt.plot(cleaned_audio)
    plt.show()
    return sample_rate, cleaned_audio





import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.interpolation.UnivariateInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;

import java.util.ArrayList;
import java.util.List;

public class ClicksRemoval {
    public static double[] removeClicks(double[] audioData, int sampleRate, double threshold) {
        // Convert to mono if stereo
        if (audioData.length > 1) {
            audioData = convertToMono(audioData);
        }

        // Find peaks (clicks)
        List<Integer> peaks = findPeaks(audioData, threshold);

        // Interpolate over clicks
        double[] cleanedAudio = audioData.clone();
        for (int peak : peaks) {
            // Smooth out the click region by averaging neighboring samples
            int windowSize = (int) (sampleRate * 0.04);  // 10 ms window
            int startIdx = Math.max(0, peak - windowSize / 2);
            int endIdx = Math.min(cleanedAudio.length, peak + windowSize / 2);
            double[] window = new double[endIdx - startIdx];
            System.arraycopy(cleanedAudio, startIdx, window, 0, endIdx - startIdx);

            // Apply moving average to smooth out the click region
            double[] smoothedWindow = movingAverage(window, 5);
            System.arraycopy(smoothedWindow, 0, cleanedAudio, startIdx, smoothedWindow.length);
        }

        return cleanedAudio;
    }

    private static double[] convertToMono(double[] stereoData) {
        double[] monoData = new double[stereoData.length / 2];
        for (int i = 0; i < stereoData.length; i += 2) {
            monoData[i / 2] = (stereoData[i] + stereoData[i + 1]) / 2;
        }
        return monoData;
    }

    private static List<Integer> findPeaks(double[] data, double threshold) {
        List<Integer> peaks = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            if (Math.abs(data[i]) > threshold) {
                peaks.add(i);
            }
        }
        return peaks;
    }

    private static double[] movingAverage(double[] data, int windowSize) {
        double[] smoothedData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            int startIdx = Math.max(0, i - windowSize / 2);
            int endIdx = Math.min(data.length, i + windowSize / 2 + 1);
            smoothedData[i] = StatUtils.mean(data, startIdx, endIdx);
        }
        return smoothedData;
    }

    public static void main(String[] args) {
        // Example usage
        double[] audioData = { /* Your audio data here */ };
        int sampleRate = 44100; // Example sample rate
        double threshold = 0.1; // Example threshold
        double[] cleanedAudioData = removeClicks(audioData, sampleRate, threshold);

        // Print the cleaned audio data
        System.out.println("Cleaned Audio Data:");
        for (double value : cleanedAudioData) {
            System.out.println(value);
        }
    }
}





def remove_clapping(audio_data, threshold=0.05):
    # Compute the envelope of the audio signal
    envelope = np.abs(signal.hilbert(audio_data))
    
    # Identify time points where the envelope exceeds the threshold
    clapping_indices = np.where(envelope > threshold)[0]
    
    # Set the clapping regions to zero
    audio_data_clean = np.copy(audio_data)
    audio_data_clean[clapping_indices] = 0
    
    return audio_data_clean


import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.*;
import java.util.ArrayList;
import java.util.List;

public class ClappingRemoval {
    public static double[] removeClapping(double[] audioData, double threshold) {
        int n = audioData.length;

        // Compute the envelope of the audio signal
        double[] envelope = absHilbertEnvelope(audioData);

        // Identify time points where the envelope exceeds the threshold
        List<Integer> clappingIndices = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (envelope[i] > threshold) {
                clappingIndices.add(i);
            }
        }

        // Set the clapping regions to zero
        double[] audioDataClean = audioData.clone();
        for (int index : clappingIndices) {
            audioDataClean[index] = 0;
        }

        return audioDataClean;
    }

    private static double[] absHilbertEnvelope(double[] signal) {
        int n = signal.length;

        // Apply the Fast Fourier Transform (FFT)
        FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] transformedSignal = transformer.transform(signal, TransformType.FORWARD);

        // Apply the Hilbert transform
        for (int i = 0; i < n / 2; i++) {
            transformedSignal[i] = transformedSignal[i].multiply(2);
        }
        for (int i = n / 2 + 1; i < n; i++) {
            transformedSignal[i] = Complex.ZERO;
        }

        // Apply the Inverse Fast Fourier Transform (IFFT)
        Complex[] hilbertTransformedSignal = transformer.transform(transformedSignal, TransformType.INVERSE);

        // Calculate the absolute value of the Hilbert transformed signal
        double[] envelope = new double[n];
        for (int i = 0; i < n; i++) {
            envelope[i] = hilbertTransformedSignal[i].abs();
        }

        return envelope;
    }

    public static void main(String[] args) {
        // Example usage
        double[] audioData = { /* Your audio data here */ };
        double threshold = 0.05;
        double[] cleanedAudioData = removeClapping(audioData, threshold);

        // Print the cleaned audio data
        System.out.println("Cleaned Audio Data:");
        for (double value : cleanedAudioData) {
            System.out.println(value);
        }
    }
}
