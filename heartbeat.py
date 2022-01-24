import matplotlib.pyplot as plt
import numpy as np
import math
import json
from edfrd import read_header, read_data_records
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d

measures = {}


# Falls Kanal und Frequenz nicht bekannt

def sampling_frequency(file_path):
    header = read_header(file_path)
    duration = header.duration_of_a_data_record

    for i in header.signals:
        print(i.label)
        if 'ECG' in i.label:
            sampling = int(i.nr_of_samples_in_each_data_record / duration)
    print('Sampling : ' + str(sampling) + 'Hz')


# Extraktion des EKG-Signal aus einer EDF

def get_data(file_path, kanal):
    header = read_header(file_path)

    l = []
    for i, data_record in enumerate(read_data_records(file_path, header)):
        ecg = data_record[kanal]
        # Eingrenzung auf 30 Sekunden nur für Entwicklungszwecke, schnellere Testung durch kleinen Datensatz
        if i < 60:
            for j in ecg:
                l.append(j.tolist())
    return l  
    
def get_data_with_records(file_path, kanal):
    header = read_header(file_path)
    
    for i, data_record in enumerate(read_data_records(file_path, header)):
        ecg = data_record[kanal]
        # Eingrenzung auf 30 Sekunden nur für Entwicklungszwecke, schnellere Testung durch kleinen Datensatz
    return ecg  



# Definition des Butterworth Filter
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist frequency is half the sampling frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Berechnung der Moving Average mit 20%

def rolmean(dataset, hrw, fs, cutoff, order):
    filtered = butter_lowpass_filter(dataset.heart, cutoff, fs, order)
    listpos = 0

    for datapoint in dataset['heart']:
        dataset.heart[listpos] = filtered[listpos]
        listpos += 1
    mov_avg = dataset['heart'].rolling(int(hrw * fs)).mean()
    avg_hr = (np.mean(dataset.heart))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    dataset['heart_rollingmean'] = mov_avg


# Erkennung der einzelnen Herzschläge
def detect_peaks(dataset, ma_perc, fs):  # Change the function to accept a moving average percentage 'ma_perc' argument
    rolmean = [(x + ((x / 100) * ma_perc)) for x in
               dataset.heart_rollingmean]  # Raise moving average with passed ma_perc
    window = []
    peaklist = []
    listpos = 0

    for datapoint in dataset.heart:
        rollingmean = rolmean[listpos]
        if (datapoint <= rollingmean) and (len(window) <= 1):  # Here is the update in (datapoint <= rollingmean)
            listpos += 1
        elif (datapoint > rollingmean):
            window.append(datapoint)
            listpos += 1
        else:
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window)))
            peaklist.append(beatposition)
            window = []
            listpos += 1

        measures['peaklist'] = peaklist
        measures['ybeat'] = [dataset.heart[x] for x in peaklist]
        measures['rolmean'] = rolmean
        calc_RR(dataset, fs)
        measures['rrsd'] = np.std(measures['RR_list'])


def fit_peaks(dataset, fs, ma_perc_list):
    rrsd = []
    valid_ma = []

    for x in ma_perc_list:  # Detect peaks with all percentages, append results to list 'rrsd'
        detect_peaks(dataset, x, fs)
        bpm = ((len(measures['peaklist']) / (len(dataset.heart) / fs)) * 60)
        rrsd.append([measures['rrsd'], bpm, x])

    for x, y, z in rrsd:  # Test list entries and select valid measures
        if ((x > 1) and ((y > 30) and (y < 130))):
            valid_ma.append([x, z])

    measures['best'] = min(valid_ma, key=lambda t: t[0])[
        1]  # Save the ma_perc for plotting purposes later on (not needed)
    detect_peaks(dataset, min(valid_ma, key=lambda t: t[0])[1],
                 fs)  # Detect peaks with 'ma_perc' that goes with lowest rrsd


# Berechnung der RR-Intervalle
def calc_RR_simple(dataset, fs):
    RR_list = []
    peaklist = measures['peaklist']
    cnt = 0
    while (cnt < (len(peaklist) - 1)):
        RR_interval = (peaklist[cnt + 1] - peaklist[cnt])
        ms_dist = ((RR_interval / fs) * 1000.0)
        RR_list.append(ms_dist)
        cnt += 1
    measures['RR_list'] = RR_list


# Berechnng der zeitbasierten HRV-Parameter

def calc_RR(dataset, fs):
    peaklist = measures['peaklist']
    RR_list = []
    cnt = 0
    while (cnt < (len(peaklist) - 1)):
        RR_interval = (peaklist[cnt + 1] - peaklist[cnt])
        ms_dist = ((RR_interval / fs) * 1000.0)
        RR_list.append(ms_dist)
        cnt += 1
    RR_diff = []
    RR_sqdiff = []
    cnt = 0

    while (cnt < (len(RR_list) - 1)):
        RR_diff.append(abs(RR_list[cnt] - RR_list[cnt + 1]))
        RR_sqdiff.append(math.pow(RR_list[cnt] - RR_list[cnt + 1], 2))
        cnt += 1

    measures['RR_list'] = RR_list
    measures['RR_diff'] = RR_diff
    measures['RR_sqdiff'] = RR_sqdiff


def calc_ts_measures():
    RR_list = measures['RR_list']
    RR_diff = measures['RR_diff']
    RR_sqdiff = measures['RR_sqdiff']

    measures['bpm'] = 60000 / np.mean(RR_list)
    measures['ibi'] = np.mean(RR_list)
    measures['sdnn'] = np.std(RR_list)
    measures['sdsd'] = np.std(RR_diff)
    measures['rmssd'] = np.sqrt(np.mean(RR_sqdiff))
    NN20 = [x for x in RR_diff if (x > 20)]
    NN50 = [x for x in RR_diff if (x > 50)]
    measures['nn20'] = NN20
    measures['nn50'] = NN50
    measures['pnn20'] = float(len(NN20)) / float(len(RR_diff))
    measures['pnn50'] = float(len(NN50)) / float(len(RR_diff))


def calc_fd_measures(dataset, fs, output):
    peaklist = measures['peaklist']  # First retrieve the lists we need

    RR_list = measures['RR_list']
    RR_x = peaklist[1:]  # Remove the first entry, because first interval is assigned to the second beat.
    RR_y = RR_list  # Y-values are equal to interval lengths
    RR_x_new = np.linspace(RR_x[0], RR_x[-1], RR_x[
        -1])  # Create evenly spaced timeline starting at the second peak, its endpoint and length equal to position of last peak
    f = interp1d(RR_x, RR_y, kind='cubic')  # Interpolate the signal with cubic spline interpolation

    # Set variables
    n = len(dataset.heart)  # Length of the signal
    frq = np.fft.fftfreq(len(dataset.heart), d=(1 / fs))  # divide the bins into frequency categories
    frq = frq[list(range(n // 2))]  # Get single side of the frequency range

    # Do FFT
    Y = np.fft.fft(f(RR_x_new)) / n  # Calculate FFT
    Y = Y[list(range(n // 2))]  # Return one side of the FFT

    lf = np.trapz(abs(Y[(frq >= 0.04) & (
                frq <= 0.15)]))  # Slice frequency spectrum where x is between 0.04 and 0.15Hz (LF), and use NumPy's trapezoidal integration function to find the area
    # print(("LF:", lf))

    hf = np.trapz(abs(Y[(frq >= 0.16) & (frq <= 0.5)]))  # Do the same for 0.16-0.5Hz (HF)
    # print(("HF:", hf))

    measures['lf'] = lf
    measures['hf'] = hf

    # Plot
    plt.title("Frequency Spectrum of Heart Rate Variability")
    plt.xlim(0, 0.6)  # Limit X axis to frequencies of interest (0-0.6Hz for visibility, we are interested in 0.04-0.5)
    plt.ylim(0, 50)  # Limit Y axis for visibility
    plt.plot(frq, abs(Y))  # Plot it
    plt.xlabel("Frequencies in Hz")
    plt.savefig('Ergebnisse/' + output + '_frequency.png')
    plt.show()


# RR-Intervalle als Array um sie für andere Pakete nutzen zu können
def get_rr_intervals(dataset, hrw, fs):
    rolmean(dataset, hrw, fs)
    detect_peaks(dataset)
    calc_RR_simple(dataset, fs)
    calc_bpm()
    RR_list = measures['RR_list']
    return RR_list


# Berechnung der durschnittlichen Herzfrequenz
def calc_bpm():
    RR_list = measures['RR_list']
    measures['bpm'] = 60000 / np.mean(RR_list)
    print((measures['bpm']))


# plotten der Peakerkennung und der moving average
def plotter(dataset, title, range_min, range_max, output):
    peaklist = measures['peaklist']
    ybeat = measures['ybeat']
    plt.title(title)
    plt.plot(dataset.heart, alpha=0.5, color='blue', label="raw signal")
    plt.plot(dataset.heart_rollingmean, color='green', label="moving average")
    plt.scatter(peaklist, ybeat, color='red', label="average: %.1f BPM" % measures['bpm'])
    plt.legend(loc=4, framealpha=0.6)
    plt.xlim(range_min, range_max)
    plt.savefig('Ergebnisse/' + output + '_peaks.png')
    plt.show()


def plot_rrintervals(dataset, title, output):
    peaklist = measures['peaklist']
    ybeat = measures['ybeat']
    RR_list = measures['RR_list']
    plt.title(title)
    # plt.plot(dataset.heart, alpha=0.5, color='blue', label="raw signal")
    plt.plot(RR_list, color='blue', label="RR-Intervalle")
    # plt.scatter(peaklist, ybeat, color='red', label="average: %.1f BPM" % measures['bpm'])
    plt.legend(loc=4, framealpha=0.6)
    plt.xlim(0, 25)
    plt.savefig('Ergebnisse/rr_intervals.png')
    plt.show()


# plotten des gefilterten Signals

def plot_filtered(dataset, data, range_min, range_max, output, cutoff, order, fs):
    filtered = butter_lowpass_filter(dataset.heart, cutoff, fs,
                                     order)  # filter the signal with a cutoff at 2.5Hz and a 5th order Butterworth filter

    plt.subplot(211)
    plt.plot(dataset.heart, color='Blue', alpha=0.5, label='Original Signal')
    plt.legend(loc=4)
    plt.xlim(range_min, range_max)
    plt.subplot(212)
    plt.plot(filtered, color='Red', label='Filtered Signal')
    plt.ylim(min(data), max(data))
    plt.xlim(range_min, range_max)
    # limit filtered signal to have same y-axis as original (filter response starts at 0 so otherwise the plot will be scaled)
    plt.legend(loc=4)
    plt.savefig('Ergebnisse/filtered.png')
    plt.show()


def print_files(output):
    # in JSON alle Ergebnisse schreiben
    jsons = json.dumps(str(measures))
    f = open("Ergebnisse/" + output + ".json", "w")
    f.write(jsons)
    f.close()

    # Alle relevanten Ergebnisse in Textdatei schreiben
    w = open("Ergebnisse/" + output + ".txt", "w")
    for i in measures:
        if i not in ('peaklist', 'ybeat', 'rolmean', 'RR_list', 'RR_diff', 'RR_sqdiff'):
            w.write(i + " : " + str(measures[i]) + "\n")
    w.close()


# Funktion zur Durchführung aller Berechnungen
def process(dataset, ecg_signal, hrw, fs, ma_perc_list, output, range_min, range_max, cutoff, order):
    plot_filtered(dataset, ecg_signal, range_min, range_max, output, cutoff, order, fs)
    rolmean(dataset, hrw, fs, cutoff, order)
    fit_peaks(dataset, fs, ma_perc_list)
    calc_RR(dataset, fs)
    calc_bpm()
    plot_rrintervals(dataset, "RR-Intervalle", output)
    calc_ts_measures()
    calc_fd_measures(dataset, fs, output)
    print_files(output)
    plotter(dataset, "My Heartbeat Plot", range_min, range_max, output)
