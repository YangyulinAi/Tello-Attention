import numpy as np
from pylsl import StreamInlet, resolve_stream
from scipy.signal import butter, lfilter, welch, iirnotch, find_peaks
from scipy.integrate import simps
import matplotlib.pyplot as plt


class AttentionHandler:
    def __init__(self, channel=0, lower=5, upper=50, fs=125, notch_freq=50, quality_factor=30):
        self.fs = fs
        self.notch_freq = notch_freq
        self.quality_factor = quality_factor
        self.b_notch, self.a_notch = self.design_notch_filter()
        self.setup_plot()
        self.safe_mode = "off"  # Initialize a return value
        self.channel = channel
        self.lower = lower
        self.upper = upper

        self.alpha_threshold = 0.6
        self.beta_threshold = 20

        print("Looking for an EEG stream...")
        self.streams = resolve_stream('type', 'EEG')
        self.inlet = StreamInlet(self.streams[0])
        print("EEG stream receiving...")

        self.locked = False
    def design_notch_filter(self):
        return iirnotch(self.notch_freq / (self.fs / 2), self.quality_factor)

    def bandpass_filter(self, data, order=5):
        nyq = 0.5 * self.fs
        low = self.lower / nyq
        high = self.upper / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def setup_plot(self):
        plt.ion()
        self.fig, (self.ax_psd, self.ax_alpha, self.ax_beta) = plt.subplots(1, 3, figsize=(8, 5),
                                                                            gridspec_kw={'width_ratios': [3, 1, 1]})
        self.fig.canvas.mpl_connect('close_event', self.on_close)  # Attach the close event handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)  # Attach key press event handler

        self.line, = self.ax_psd.plot([], [], 'b-', label='Raw PSD')
        self.nms_line, = self.ax_psd.plot([], [], 'ro', label='NMS Peaks')
        self.ax_psd.set_xlabel('Frequency (Hz)')
        self.ax_psd.set_ylabel('Power Spectral Density (μV^2/Hz)')
        self.ax_psd.legend()
        self.ax_psd.set_xlim([0, 50])
        self.ax_psd.set_ylim([0, 100])

        self.ax_alpha.set_xlabel('Relative Alpha Power')
        self.ax_alpha.set_ylim([0, 1])
        self.bar_alpha = self.ax_alpha.bar(1, 0, width=0.1, color='g')

        self.ax_beta.set_xlabel('Absolute Beta Power')  # More sensitive
        self.ax_beta.set_ylim([0, 50])
        self.bar_beta = self.ax_beta.bar(1, 0, width=0.1, color='b')

    def on_close(self, event):
        print("Plot window closed!")
        self.safe_mode = "on"  # Store a message or any other value

    def on_key_press(self, event):
        if event.key == 'escape':
            self.safe_mode = "on"  # Store a message or any other value
            plt.close('all')
            print("ESC pressed, stopping data stream...")

    def update_plot(self, freqs, psd, selected_freqs, selected_psd, relative_alpha_power, relative_beta_power):
        #psd[psd < 10] = 0
        #selected_psd[selected_psd < 20] = 0

        self.line.set_data(freqs, psd)
        self.nms_line.set_data(selected_freqs, selected_psd)
        self.ax_psd.relim()
        self.ax_psd.autoscale_view()

        self.bar_alpha[0].set_height(relative_alpha_power)
        self.bar_beta[0].set_height(relative_beta_power)
        self.fig.canvas.draw()
        plt.pause(0.1)

    def update_plot_pause(self):
        self.ax_psd.cla()
        self.ax_psd.set_ylim([0, 1])
        self.ax_psd.text(0.5, 0.5, 'Task performing, EEG stream paused...', fontsize=12, ha='center', color='red',
                         transform=self.ax_psd.transAxes)
        self.fig.canvas.draw()
        plt.pause(0.1)

    def non_maximum_suppression(self, psd, window_size=5):
        peaks, _ = find_peaks(psd)
        selected_peaks = []
        for peak in peaks:
            start = max(0, peak - window_size)
            end = min(len(psd), peak + window_size)
            if psd[peak] == np.max(psd[start:end]):
                selected_peaks.append(peak)
        return selected_peaks

    def calculate_absolute_power(self, freqs, psd, band):
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        freq_res = freqs[1] - freqs[0]
        band_power = simps(psd[idx_band], dx=freq_res)
        return band_power

    def calculate_relative_power(self, freqs, psd, band):
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        freq_res = freqs[1] - freqs[0]
        band_power = simps(psd[idx_band], dx=freq_res)
        total_power = simps(psd, dx=freq_res)
        relative_power = band_power / total_power
        return relative_power

    def signal_scaling(self, type, value):
        if type == 'beta':
            if 2 * self.beta_threshold < value < 4 * self.beta_threshold:
                value = value * 0.5 + 1
            elif 4 * self.beta_threshold < value:
                value = value * 0.25 + 1
            if 50 < self.beta_threshold:
                value = self.beta_threshold + 1
        else:
            if 2 * self.alpha_threshold < value < 4 * self.alpha_threshold:
                value = value * 0.5 + 1
            elif 4 * self.alpha_threshold < value:
                value = value * 0.25 + 1
            if 1 < self.alpha_threshold:
                value = self.alpha_threshold + 1

        return value

    def threshold_based_classification(self, alpha_power, beta_power):

        if alpha_power > self.alpha_threshold:
            return 1
        elif beta_power > self.beta_threshold and alpha_power < self.alpha_threshold:
            return 2
        elif alpha_power > self.alpha_threshold and beta_power > self.beta_threshold:
            return 3
        else:
            return 4

    def clear_buffer(self):
        while True:
            temp_chunk, temp_timestamps = self.inlet.pull_chunk(max_samples=self.fs, timeout=0.01)
            #self.update_plot_pause()
            if not temp_chunk:
                break
    def process_data(self):

        is_data_coming = False
        try:
            while True:
                chunk, timestamps = self.inlet.pull_chunk(max_samples=self.fs, timeout=1.0)
                if timestamps:  # if there is data coming in
                    data = np.array(chunk)
                    data_filtered = lfilter(self.b_notch, self.a_notch, data[:, self.channel])
                    data_bandpassed = self.bandpass_filter(data_filtered)

                    freqs, psd = welch(data_bandpassed, self.fs, nperseg=self.fs)
                    selected_peaks = self.non_maximum_suppression(psd)
                    selected_freqs = freqs[selected_peaks]
                    selected_psd = psd[selected_peaks]

                    alpha_power = self.calculate_relative_power(freqs, psd, (8, 12))
                    alpha_power = self.signal_scaling("alpha", alpha_power)

                    beta_power = self.calculate_absolute_power(freqs, psd, (15, 30))
                    beta_power = self.signal_scaling("beta", beta_power)

                    #print(f"Centered Relative Alpha Power: {alpha_power}")

                    if not self.locked:

                        self.update_plot(freqs, psd, selected_freqs, selected_psd, alpha_power,
                                     beta_power)
                        command = self.threshold_based_classification(alpha_power, beta_power)
                        yield command
                    else:
                        print("Locked")
                        pass
                    is_data_coming = True
                else:
                    if is_data_coming:
                        print("Waiting for LSL Stream ...")
                        is_data_coming = False
        except KeyboardInterrupt:
            print("Program stopped manually.")
        finally:
            plt.ioff()




