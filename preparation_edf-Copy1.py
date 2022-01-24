from ctypes import ArgumentError
import pandas as pd
import csv
import matplotlib.gridspec
import matplotlib.pyplot as plt
import heartbeat as hb
import preparation_hypno as preph
import neurokit2 as nk
from preparation_hypno import System
from edfrd import read_header, read_data_records
from datetime import time
from datetime import datetime
from datetime import date
from datetime import timedelta
import edfrd
from neurokit2.hrv.hrv_utils import _hrv_get_rri
import numpy as np
import os

FILE_NAME = __import__(__name__)


TEST_EDF_SAMPLE = 'Probanden/anonymized_sample.edf'
TEST_SCHLAFPROFIL = "Schlafprofil.txt"

'''
Functions for preparating edf files and analysing
'''

#Hilfsfunktionen für das plotting

def plot_compare_signals(segments, fs):
    '''
    plot_compare_signals
    
    Parameters
    ----------
    segments: np.array
        the array with cuttet ecg-signal into segments
    fs: 
        sampling frequency of ecg
        
    Returns
    -------
    figure: fig
        a figure to compare filtered an unfiltered signal
        
    Examples
    --------
    Throw error:
    >>> plot_compare_signals()
    Traceback (most recent call last):
        ...
    TypeError: plot_compare_signals() missing 2 required positional arguments: 'segments' and 'fs'
    
    '''
    
    clean = nk.ecg_clean(segments,fs)

    fig, axs = plt.subplots(2)
    fig.suptitle('Comparison 15 seconds signal')
    axs[0].set_title("raw signal")
    axs[1].set_title("filtered signal")
    axs[1].plot(segments, alpha=0.5, color='grey',linestyle="--", label="raw")
    axs[1].plot(clean, alpha=0.5, color='red', label="cleaned")
    axs[1].legend(loc="upper left")
    axs[0].plot(segments, alpha=0.5, color='blue', label="raw")
    axs[0].legend(loc="upper left")

    #15 Sek. 256x30s
    axs[0].set_xlim(0,3840)
    axs[1].set_xlim(0,3840)
    fig.savefig('plots/compare_signals.png')
    #plt.show()
    
#Analysieren und plotten der einzelnen 300 sek Epochen

#segment = segments[0]['values']
def analyse_first_segments(fs, segments, artifacts = True):
    '''
    Analyse hrv and plot the first 300 sec epoch
    Parameters
    ----------
    fs:
        sampling fequency of ecg
    segments: 
        the array with cuttet ecg-signal into segments
    artifacts: Bool
        selection of wether artifacts should be corrected
        
    Returns
    -------
    results
        - DataFrame with HRV results
    firstsegm
        - the first segment result
    info
        - list of detected rpeaks
        
    Examples
    --------
    SKIP>>> def prep():
    SKIP...     dfh = preph.make_df_hypno_all(TEST_SCHLAFPROFIL, System.Somnomedics, start_of_hypno= None)
    SKIP...     cutpoints = preph.calculate_cutpoints(dfh, True)
    SKIP...     header = read_header(TEST_EDF_SAMPLE)
    SKIP...     ecg_channel, fs = get_ecg_channel_and_frequency(TEST_EDF_SAMPLE)
    SKIP...     cutpoints = preph.calculate_cutpoints(dfh)
    SKIP...     diff = preph.calculate_cutpoints_diff(cutpoints)
    SKIP...     diff, difference_betweens = difference_between(header,dfh,diff)
    SKIP...     ecg_signal = get_data_sections(TEST_EDF_SAMPLE,ecg_channel,diff,True)
    SKIP...     return list_of_stad_values(ecg_signal, cutpoints)
    SKIP>>> segments = segment_epochs(prep())
    SKIP>>> a = analyse_first_segments(256, segments, True)  
    SKIP>>> print(a[0:1]) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE

    '''
    firstsegm,info = my_processing(segments[0]['values'], fs, artifacts)
    #print(testsegm)
    results = nk.ecg_analyze(firstsegm, sampling_rate=fs)
    return results, firstsegm, info

def analyse_segments(idp, fs, segments,night,dia,artifacts, min=1, max=3):
    '''
    Analyse hrv and plot every 300 sec epoch
    Parameters
    ----------
    idp:
        id of patient
    fs:
        sampling fequency of ecg
    segments: 
        the array with cuttet ecg-signal into segments
    artifacts: Bool
        selection of wether artifacts should be corrected
    night: String
        night of patients PSG
    dia: String
        diagnosis of patient
        
    Returns
    -------
    results
        - DataFrame with HRV results
    process
        - 
    info
        - list of detected rpeaks
        
    Examples
    --------
    SKIP>>> def prep():
    SKIP...     dfh = preph.make_df_hypno_all(TEST_SCHLAFPROFIL, System.Somnomedics, start_of_hypno= None)
    SKIP...     cutpoints = preph.calculate_cutpoints(dfh, True)
    SKIP...     header = read_header(TEST_EDF_SAMPLE)
    SKIP...     ecg_channel, fs = get_ecg_channel_and_frequency(TEST_EDF_SAMPLE)
    SKIP...     cutpoints = preph.calculate_cutpoints(dfh)
    SKIP...     diff = preph.calculate_cutpoints_diff(cutpoints)
    SKIP...     diff, difference_betweens = difference_between(header,dfh,diff)
    SKIP...     ecg_signal = get_data_sections(TEST_EDF_SAMPLE,ecg_channel,diff,True)
    SKIP...     return list_of_stad_values(ecg_signal, cutpoints)
    SKIP>>> segments = segment_epochs(prep())
    SKIP>>> a = analyse_segments(id,256, segments,night,dia,True)  
    SKIP>>> print(a[0:1]) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE

    '''
    #global results
    results,firstsegm, rpeaks = analyse_first_segments(fs,segments, artifacts)
    ecg_plots(idp,firstsegm, rpeaks, fs, show_type='default')
    results.insert(0,'stadium',segments[0]['epoch'])
    results.insert(1,'dia',dia)
    results.insert(2,'idp',idp)

    
    for i in range(min,len(segments)):
#    for i in range(min,max):
        process,info = my_processing(segments[i]['values'], fs, artifacts)
        stad= segments[i]['epoch']
        result = nk.ecg_analyze(process, fs)
        #pd.concat([results,result], axis=0)
        #results.append([result])
        result.insert(0,'stadium',stad)
        result.insert(1,'dia',dia)
        result.insert(2,'idp',idp)
        results = results.merge(result, how='outer')
        #processneu = process[42000:47000]
        ecg_plots(idp, process,fs, show_type='default', count=i)

        #results= resulta
        #print(results)
        
    if not os.path.exists('Probanden/result_hrv'):
        os.makedirs('Probanden/result_hrv')
    
    path_dir = 'Probanden/result_hrv'
    filename = 'results_hrv.csv'
    path_file = "{}{}{}{}{}".format(path_dir, os.sep, idp, night, filename)
    
    results.to_csv(path_file, index=True)
    return results, process, info

#eigene Plotfunktion für analyse ecg von neurokit2(ecg_plots()) erweitert um eigene Funktionen

def ecg_plots(idp, ecg_signals, rpeaks=None, sampling_rate=None, show_type="default", count= 0):
    """Visualize ECG data.

    Parameters
    ----------
    ecg_signals : DataFrame
        DataFrame obtained from `ecg_process()`.
    rpeaks : dict
        The samples at which the R-peak occur. Dict returned by
        `ecg_process()`. Defaults to None.
    sampling_rate : int
        The sampling frequency of the ECG (in Hz, i.e., samples/second). Needs to be supplied if the
        data should be plotted over time in seconds. Otherwise the data is plotted over samples.
        Defaults to None. Must be specified to plot artifacts.
    show_type : str
        Visualize the ECG data with 'default' or visualize artifacts thresholds with 'artifacts' produced by
        `ecg_fixpeaks()`, or 'full' to visualize both.
    count: int
        default 0 , for counting plots 

    Returns
    -------
    fig
        Figure representing a plot of the processed ecg signals (and peak artifacts).

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=15, sampling_rate=1000, heart_rate=80)
    >>> signals, info = nk.ecg_process(ecg, sampling_rate=1000)
    >>> nk.ecg_plot(signals, sampling_rate=1000, show_type='default') #doctest: +ELLIPSIS
    <Figure ...>

    See Also
    --------
    ecg_process

    """
    # Sanity-check input.
    if not isinstance(ecg_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: ecg_plot(): The `ecg_signals` argument must be the "
            "DataFrame returned by `ecg_process()`."
        )

    # Extract R-peaks.
    peaks = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]

    # Prepare figure and set axes.
    if show_type in ["default", "full"]:
        if sampling_rate is not None:
            x_axis = np.linspace(0, ecg_signals.shape[0] / sampling_rate, ecg_signals.shape[0])
            gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[1 - 1 / np.pi, 1 / np.pi])
            fig = plt.figure(constrained_layout=False)
            ax0 = fig.add_subplot(gs[0, :-1])
            ax1 = fig.add_subplot(gs[1, :-1])
            ax2 = fig.add_subplot(gs[:, -1])
            ax0.set_xlabel("Time (seconds)")
            ax1.set_xlabel("Time (seconds)")
            ax2.set_xlabel("Time (seconds)")
        else:
            x_axis = np.arange(0, ecg_signals.shape[0])
            fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
            ax0.set_xlabel("Samples")
            ax1.set_xlabel("Samples")

        fig.suptitle("Electrocardiogram (ECG)", fontweight="bold")
        plt.subplots_adjust(hspace=0.3, wspace=0.1)

        # Plot cleaned, raw ECG, R-peaks and signal quality.
        ax0.set_title("Raw and Cleaned Signal")

        quality = nk.rescale(
            ecg_signals["ECG_Quality"], to=[np.min(ecg_signals["ECG_Clean"]), np.max(ecg_signals["ECG_Clean"])]
        )
        minimum_line = np.full(len(x_axis), quality.min())

        # Plot quality area first
        ax0.fill_between(
            x_axis, minimum_line, quality, alpha=0.12, zorder=0, interpolate=True, facecolor="#4CAF50", label="Quality"
        )

        # Plot signals
        ax0.plot(x_axis, ecg_signals["ECG_Raw"], color="#B0BEC5", label="Raw", zorder=1)
        ax0.plot(x_axis, ecg_signals["ECG_Clean"], color="#E91E63", label="Cleaned", zorder=1, linewidth=1.5)
        ax0.scatter(x_axis[peaks], ecg_signals["ECG_Clean"][peaks], color="#FFC107", label="R-peaks", zorder=2)

        # Optimize legend
        handles, labels = ax0.get_legend_handles_labels()
        order = [2, 0, 1, 3]
        ax0.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right")

        # Plot heart rate.
        ax1.set_title("Heart Rate")
        ax1.set_ylabel("Beats per minute (bpm)")

        ax1.plot(x_axis, ecg_signals["ECG_Rate"], color="#FF5722", label="Rate", linewidth=1.5)
        rate_mean = ecg_signals["ECG_Rate"].mean()
        ax1.axhline(y=rate_mean, label="Mean", linestyle="--", color="#FF9800")

        ax1.legend(loc="upper right")

        # Plot individual heart beats.
        if sampling_rate is not None:
            ax2.set_title("Individual Heart Beats")

            heartbeats = nk.ecg_segment(ecg_signals["ECG_Clean"], peaks, sampling_rate)
            heartbeats = nk.epochs_to_df(heartbeats)

            heartbeats_pivoted = heartbeats.pivot(index="Time", columns="Label", values="Signal")

            ax2.plot(heartbeats_pivoted)

            cmap = iter(
                plt.cm.YlOrRd(np.linspace(0, 1, num=int(heartbeats["Label"].nunique())))  # pylint: disable=E1101
            )  # Aesthetics of heart beats

            lines = []
            for x, color in zip(heartbeats_pivoted, cmap):
                (line,) = ax2.plot(heartbeats_pivoted[x], color=color)
                lines.append(line)

    # Plot artifacts
    if show_type in ["artifacts", "full"]:
        if sampling_rate is None:
            raise ValueError(
                "NeuroKit error: ecg_plot(): Sampling rate must be specified for artifacts" " to be plotted."
            )

        if rpeaks is None:
            _, rpeaks = ecg_peaks(ecg_signals["ECG_Clean"], sampling_rate=sampling_rate)

        fig = nk.signal_fixpeaks(rpeaks, sampling_rate=sampling_rate, iterative=True, show=False, method="Kubios")
    
    #speichern der jeweiligen Plots
    if not os.path.exists('Probanden/result_plots'):
        os.makedirs('Probanden/result_plots')
    
    path_dir = 'Probanden/result_plots'
    filename = 'resultplot'
    path_file = "{}{}{}{}{}".format(path_dir, os.sep, idp, filename, count)
    fig.savefig(path_file)
    #return fig

#Vergleich Peaks

def compare_peaks(ecg_signal):
    '''
    function for comparing peak detection corrected and uncorrected
    
    Parameters
    ----------
    ecg_signal: 
        - the ecg signal
        
    Returns
    -------
    fig:
        - shows peaks with artifact types
    artifacts: 
        - table with artifact types
        
    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=15, sampling_rate=1000, heart_rate=80)
    >>> compare_peaks(ecg)
    (<Figure size 432x288 with 1 Axes>, {'ectopic': [], 'missed': [], 'extra': [], 'longshort': []})
    '''
    
    # NeuroKit
    signal = ecg_signal
    #peaks_true = nk.signal_findpeaks(signal)["Peaks"]
    #peaks_corrected = nk.signal_fixpeaks(peaks_true, interval_min=0.5, interval_max=1.5, method="neurokit")
    # Plot and shift original peaks to the rightto see the difference.
    #fig = nk.events_plot([peaks_true + 50, peaks_corrected], signal)
    
    
    rpeaks_uncorrected = nk.ecg_findpeaks(signal, method= 'neurokit')
    artifacts, rpeaks_corrected = nk.signal_fixpeaks(rpeaks_uncorrected, iterative=True,
                                             show=True, method="Kubios")
    rate_corrected = nk.signal_rate(rpeaks_corrected, desired_length=len(signal))
    rate_uncorrected = nk.signal_rate(rpeaks_uncorrected, desired_length=len(signal))
    
    fig, ax = plt.subplots()
    ax.plot(rate_uncorrected, label="heart rate without artifact correction") 
    ax.plot(rate_corrected, label="heart rate with artifact correction") 
    ax.legend(loc="upper right")
    return fig, artifacts
#, rpeaks_corrected 
        
# Define a new process function
def my_processing(ecg_signal, fs, artifacts):
    '''
    process function for the ecg signal incl. cleaning, peak detection, ecg rate and quality
    
    Parameters
    ----------
    ecg_signal: array
        - raw ecg
    fs:
        - sampling frequency
    artifacts: Boolean
        selection of wether artifacts should be corrected
        
    Returns
    -------
    signals: DataFrame
        - different signal types
    info: 
        - rpeaks calculated
    '''    
    # Do processing
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs,  method="neurokit")
    instant_peaks, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, correct_artifacts=artifacts)
    rate = nk.ecg_rate(rpeaks, sampling_rate=fs, desired_length=len(ecg_cleaned))
    quality = nk.ecg_quality(ecg_cleaned, sampling_rate=fs)


    # Prepare output
    signals = pd.DataFrame({"ECG_Raw": ecg_signal,
                            "ECG_Clean": ecg_cleaned,
                            "ECG_Rate": rate,
                            "ECG_Quality": quality
                           })
    #blapeaks = pd.DataFrame({"ECG_R_Peaks": rpeaks["ECG_R_Peaks"]})
    signals = pd.concat([signals, instant_peaks], axis=1)
    info = rpeaks
    #print(instant_peaks)
    #print(rpeaks)

    return signals, info

#Unterteilung der einzelnen Stadienabschnitte in 300 sek Epochen
# source https://stackoverflow.com/a/1751478
def segment_epochs(list_of_stad_value, fs = 256, cutoff = 300):
    '''
    subdivion in 300 sec epochs in every stadium
    
    Parameters
    ----------
    list_of_stad_value: dict
        dictionary of stadium and ecg signal
    fs:
        - the sample frequency
    cutoff:
        - size of epochs
        
    Returns
    -------
    epochs:
        - new list with only 300 second epochs
        
    Examples
    --------
    Throw an error on missing params:
    >>> segment_epochs()
    Traceback (most recent call last):
        ...
    TypeError: segment_epochs() missing 1 required positional argument: 'list_of_stad_value'
    
    >>> def prep():
    ...     dfh = preph.make_df_hypno_all(TEST_SCHLAFPROFIL, System.Somnomedics, start_of_hypno= None)
    ...     cutpoints = preph.calculate_cutpoints(dfh, True)
    ...     header = read_header(TEST_EDF_SAMPLE)
    ...     ecg_channel, fs = get_ecg_channel_and_frequency(TEST_EDF_SAMPLE)
    ...     cutpoints = preph.calculate_cutpoints(dfh)
    ...     diff = preph.calculate_cutpoints_diff(cutpoints)
    ...     diff, difference_betweens = difference_between(header,dfh,diff)
    ...     ecg_signal = get_data_sections(TEST_EDF_SAMPLE,ecg_channel,diff,True)
    ...     return list_of_stad_values(ecg_signal, cutpoints)
    >>> result = segment_epochs(prep())
    >>> len(result)
    64
    
    Throw Error when fs is less then 0:
    >>> segment_epochs(prep(), -1)
    Traceback (most recent call last):
        ...
    ctypes.ArgumentError: fs must be greater then 0
    
    Throw Error when cutoff is less then 0:
    >>> segment_epochs(prep(), 100, -1)
    Traceback (most recent call last):
        ...
    ctypes.ArgumentError: cutoff must be greater then 0
    '''
    if fs <= 0:
        raise ArgumentError("fs must be greater then 0") 

    if cutoff <= 0:
        raise ArgumentError("cutoff must be greater then 0") 
    
    
    epochs = []
    epochlength = cutoff * fs
    
    for i, value in enumerate(list_of_stad_value):
        valuelength = calc_segment_len(value['values'], fs, cutoff)
        
        if valuelength < 1: 
            continue
        
        chunks = list(chunk(value['values'], epochlength))
        chunks = list(filter(lambda x: len(x) == epochlength, chunks))
        
        
        #print(len(list(chunks)))
        
        for j in chunks: 
            epochs.append({'epoch': value['stadium'], 'values': j})
        
    return epochs


def chunk(l, n):
    '''
    helper function for cutting the epochs
    source https://stackoverflow.com/a/1751478
    
    Parameters
    ----------
    l : list
        Array of elements
    n: int
        size of elements
        
    Returns
    -------
    list: 
        TODO
        
    Examples
    --------
    It fails, if no arguments are passed in:
    >>> chunk()
    Traceback (most recent call last):
        ...
    TypeError: chunk() missing 2 required positional arguments: 'l' and 'n'
    
    
    It chunks a list in equal parts:
    >>> list(chunk(list(range(1,10)), 3))
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    evnen when the sizes not as equal as they should be:
    >>> list(chunk(list(range(0,10)), 4))
    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    '''
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def calc_segment_len(values, fs, cutoff):
    '''
    calculates length of given epochs
    
    Parameters
    ----------
    values: array
        input values
    fs: int
        sampling frequency
    cutoff: int 
        goal size 
        
    Returns
    -------
    int
        calculated value from dividing seconds with cutoff
        
    Examples
    --------
    It fails, if no arguments are passed in:
    >>> calc_segment_len()
    Traceback (most recent call last):
        ...
    TypeError: calc_segment_len() missing 3 required positional arguments: 'values', 'fs', and 'cutoff'
     
    >>> calc_segment_len(list(range(0,1000)), 8, 30)
    4
    
    It fails, if 0 are passed in:
    >>> calc_segment_len(list(range(0,1024)), 0, 0)
    Traceback (most recent call last):
        ...
    ZeroDivisionError: division by zero
    '''
    
    # seconds in epoch 
    seconds = len(values) / fs  
    return int(seconds / cutoff)

def get_ecg_channel(file_path):
    '''Determinate the EDF Channel
    
    Parameters
    ----------
    file_path: string
        Path to EDF File
        
    Returns
    -------
    int
        The channel number
        
    Examples
    --------
    It requires a filename:
    >>> get_ecg_channel()
    Traceback (most recent call last):
        ...
    TypeError: get_ecg_channel() missing 1 required positional argument: 'file_path'
    
    
    The given filename should be a string:
    >>> get_ecg_channel(123)
    Traceback (most recent call last):
        ...
    ValueError: file_path should be a string
    
    Print the EKG / ECG channel number:
    >>> get_ecg_channel(TEST_EDF_SAMPLE)
    6
    '''
    if not isinstance(file_path, str):
       raise ValueError('file_path should be a string')
    
    header = read_header(file_path)
    channel = 0

    for i in header.signals:
        #print(i.label)
        #print(channel)
        if 'ECG' in i.label or 'EKG' in i.label:
            ecg_channel= channel
        channel = channel + 1 
    #print('Channel : ' + str(ecg_channel))

    return ecg_channel;

def get_ecg_channel_and_frequency(file_path):
    '''Determinate the EDF Channel and sampling frequency
    
    Parameters
    ----------
    file_path: string
        Path to EDF File
        
    Returns
    -------
    ecg_channel: int
        The channel number
    fs: int
        the sampling frequency of ecg
    Examples
    --------
    It requires a filename:
    >>> get_ecg_channel_and_frequency()
    Traceback (most recent call last):
        ...
    TypeError: get_ecg_channel_and_frequency() missing 1 required positional argument: 'file_path'
    
    
    The given filename should be a string:
    >>> get_ecg_channel_and_frequency(123)
    Traceback (most recent call last):
        ...
    ValueError: file_path should be a string
    
    Print the EKG / ECG channel number and frequency:
    >>> get_ecg_channel_and_frequency(TEST_EDF_SAMPLE)
    (6, 256)
    '''
    
    header = read_header(file_path)
    channel = 0

    for i in header.signals:
        #print(i.label)
        #print(channel)
        if 'ECG' in i.label or 'EKG' in i.label:
            ecg_channel= channel
            fs = i.nr_of_samples_in_each_data_record
        channel = channel + 1 
    #print('Channel : ' + str(ecg_channel))
    #print('Frequency : ' + str(fs))

    return ecg_channel, fs;


def get_absolute_time_diff(time1, time2, maxDeltaTime=timedelta(hours=25)):
    '''calculate diff between two dimestamps
    Parameters
    ----------
    time1: datetime
        - time of hypnogram
    time2: datetime
        - time of edf
    maxDeltaTime: timedelta default 5hours
        - for calculating difference after midnight
    
    Returns
    -------
    diffTime:
        difftime in seconds, with optional negative value 
    absTime:
        timedifference in seconds
        
    Examples
    --------
    Throw error if no attributes are passed in:
    >>> get_absolute_time_diff()
    Traceback (most recent call last):
        ...
    TypeError: get_absolute_time_diff() missing 2 required positional arguments: 'time1' and 'time2'
    
    Calulate the timedifference:
    >>> time1 = datetime.strptime("23:24:59", "%H:%M:%S")
    >>> time2 = datetime.strptime("23:24:30", "%H:%M:%S")
    >>> get_absolute_time_diff(time1, time2)
    (29.0, 29.0)
    
    Even when the time entiers are flipped:
    >>> time1 = datetime.strptime("23:24:59", "%H:%M:%S")
    >>> time2 = datetime.strptime("23:24:30", "%H:%M:%S")
    >>> get_absolute_time_diff(time2, time1)
    (-29.0, 29.0)
    '''
    time1delta = timedelta(hours=time1.hour, minutes=time1.minute, seconds=time1.second)
    time2delta = timedelta(hours=time2.hour, minutes=time2.minute, seconds=time2.second)
    addDay = timedelta(days=1)

    # debug
    #print("time1_edf=", time1delta)
    #print("time2_hypno=", time2delta)

    absTime = abs(time1delta - time2delta)
    diffTime = time1delta - time2delta
    
    #new = time2delta - timedelta(seconds=1199)
    print(absTime)
    if (absTime > maxDeltaTime):
        return diffTime.total_seconds(),(time2delta - time1delta + addDay)
        #return diffTime.total_seconds(),(time2delta - time1delta)

    return diffTime.total_seconds(), absTime.total_seconds()

# eigentlich datetime genannt
def get_absolute_timedate_diff(timedate1, timedate2):
    '''TODO
    eigentlich datetime genannt
    
    Parameters
    ----------
        TODO
        
    Returns
    -------
        TODO
        
    Examples
    --------
        TODO
    '''
        
    dateDiff = timedate1 - timedate2

    # debug
    #print("time1=", timedate1)
    #print("time2=", timedate2)
    
    if (dateDiff.days < 0):
        return timedate2 - timedate1

    return dateDiff;

#Funktion zur Ermittlung der Differenz zwischen Hypno und EDF

def difference_between(header, dfh, diff):
    '''function for adjustment of the hypnogram to edf
    
    Parameters
    ----------
    header:
        starttime of hypnogram
    dfh: 
        DataFrame with timestamps and stadium
    diff:
        array with stadium epoch length
        
    Returns
    -------
    diff:
        new array with stadium epoch length with new starttime 
    differnce_between:
        difference as seconds
    
    Examples
    --------
    >>> dfh = preph.make_df_hypno_all(TEST_SCHLAFPROFIL, System.SOMNOMEDICS, start_of_hypno= None)
    >>> header = read_header(TEST_EDF_SAMPLE)
    >>> cutpoints = preph.calculate_cutpoints(dfh)
    >>> diff = preph.calculate_cutpoints_diff(cutpoints)
    >>> diff, diff_between = difference_between(header,dfh,diff)
    >>> diff_between
    29.0
    >>> len(diff) 
    62
    '''
    startedf = header.starttime_of_recording
    starthypno = dfh.iloc[0][0]
    #print("starthypno", starthypno)

    # DTF
    startedfdto = datetime.strptime(startedf, "%H.%M.%S")

    # hypno
    starthypnodto = datetime.strptime(starthypno, "%H:%M:%S")

    diffTime, difference_between = get_absolute_time_diff(startedfdto, starthypnodto)
    #print("difference_between", difference_between)
    
    diff = new_list(diff, diffTime, difference_between)
    
    return diff, difference_between

#cut ECG in parts nach Schnittmuster diff in Liste, flip ob ekg falsch herum
def get_data_sections(file_path, kanal, diff, flip=False):
    '''cut ECG in parts from cutpoints
    
    Parameters
    ----------
    file_path: string
        -edf file
    kanal: number
        -ecg channel number of edf
    diff: list
        -- frame for cutpoints
    flip: boolean
        - if its necessary to switch ecg after  wrong placement of electrodes
        
    Returns
    -------
    list
         - with cutted sections by sleep stage
    Examples
    --------
    Thorw errors:
    >>> get_data_sections()
    Traceback (most recent call last):
       ...
    TypeError: get_data_sections() missing 3 required positional arguments: 'file_path', 'kanal', and 'diff'
    
    Thorw error on invalid channal:
    >>> diff = [901, 210, 360, 1260, 30, 330, 150, 60, 600, 270, 780, 1830, 150, 270, 60, 1350, 540, 120, 180, 30, 150, 450, 540, 480, 420, 480, 660, 120, 450, 960, 420, 630, 60, 330, 1410, 120, 240, 510, 1020, 300, 90, 180, 90, 330, 1170, 210, 240, 90, 180, 90, 660, 210, 300, 210, 780, 30, 60, 120, 1770, 120, 240, 30]
    
    TODO: Some Description:
    >>> get_data_sections(TEST_EDF_SAMPLE, -1, diff)
    Traceback (most recent call last):  
       ...
    ctypes.ArgumentError: channal must be greater or equal to 1
    
    Thorw error on invald file path:
    >>> diff = [901, 210, 360, 1260, 30, 330, 150, 60, 600, 270, 780, 1830, 150, 270, 60, 1350, 540, 120, 180, 30, 150, 450, 540, 480, 420, 480, 660, 120, 450, 960, 420, 630, 60, 330, 1410, 120, 240, 510, 1020, 300, 90, 180, 90, 330, 1170, 210, 240, 90, 180, 90, 660, 210, 300, 210, 780, 30, 60, 120, 1770, 120, 240, 30]
    >>> get_data_sections(123, 1, diff)
    Traceback (most recent call last):  
       ...
    ctypes.ArgumentError: file_path is not valid
    
    it does something usefull ... TODO:
    >>> diff = [901, 210, 360, 1260, 30, 330, 150, 60, 600, 270, 780, 1830, 150, 270, 60, 1350, 540, 120, 180, 30, 150, 450, 540, 480, 420, 480, 660, 120, 450, 960, 420, 630, 60, 330, 1410, 120, 240, 510, 1020, 300, 90, 180, 90, 330, 1170, 210, 240, 90, 180, 90, 660, 210, 300, 210, 780, 30, 60, 120, 1770, 120, 240, 30]
    >>> res = get_data_sections(TEST_EDF_SAMPLE, 6, diff)
    >>> len(res)
    63
    '''
    if not isinstance(file_path, str):
        raise ArgumentError("file_path is not valid") 

    if kanal < 1:
        raise ArgumentError("channal must be greater or equal to 1") 

    
    header = read_header(file_path)

    zaehler = 1

    l = []
    m = []
    for i, data_record in enumerate(read_data_records(file_path, header)):
        ecg = data_record[kanal]

        for j in ecg:
            # ein ecg hat 256
            if flip:
                j = j * -1
            l.append(j.tolist())
            # print(len(l))
        if i >= diff_count(diff, zaehler) - 1:
            if zaehler <= len(diff):
                m.append(l)
                zaehler = zaehler + 1
                l = []

    m.append(l)
    return m

def list_of_stad_values(ecg_signal, cutpoints):
    ''' Creates a list of sleeping stages with its corresponding values
   
    Parameters
    ----------
    ecg_signal:
        a ecg singnal steam
    cutpoint:
        a list where the sleeping stage change
                    Time Stadium
            0   23:24:30    Wach
            1   23:40:00      N1
            2   23:43:30      N2
            3   23:49:30      N3
                ....
        
    Returns
    -------
    list_of_stad_value
        A list with with sleepning stages and its values
        [
            { 'stadium': 'Wach', 'values': [] },
            { 'stadium': 'N!', 'values': [] },
            { 'stadium': 'N2', 'values': [] },
        ]
        
    Examples
    --------
    >>> dfh = preph.make_df_hypno_all(TEST_SCHLAFPROFIL, System.SOMNOMEDICS, start_of_hypno= None)
    >>> cutpoints = preph.calculate_cutpoints(dfh, True)
    >>> header = read_header(TEST_EDF_SAMPLE)
    >>> ecg_channel, fs = get_ecg_channel_and_frequency(TEST_EDF_SAMPLE)
    >>> cutpoints = preph.calculate_cutpoints(dfh)
    >>> diff = preph.calculate_cutpoints_diff(cutpoints)
    >>> diff, diff_between = difference_between(header,dfh,diff)
    >>> ecg_signal = get_data_sections(TEST_EDF_SAMPLE,ecg_channel,diff,True)
    >>> res = list_of_stad_values(ecg_signal, cutpoints)
    >>> res[1]['stadium']
    ' N1'
    >>> len(res[1]['values'])
    53760

    Throws an expeption 
    >>> list_of_stad_values(ecg_signal, [])
    Traceback (most recent call last):
        ...
    ctypes.ArgumentError: cutpoints must me a type of pandas.corde.frame.DataFrame
    '''
    list_of_stad_value = []

    for i,value in enumerate(ecg_signal):
        list_of_stad_value.append({'stadium':cutpoints['Stadium'][i], 'values': value})
    
    
    return list_of_stad_value


def diff_count(diff, zaehler):
    '''Summiert die Werte in diff bis zur position des Zaehlers
     
    Parameters
    ----------
    diff: list
        -- a list of diff values
    zaehler: Int
        --  the list index to sum to 
        
    Returns
    -------
    sum: int
        the sum up value to a given point
        
    Examples
    --------
    >>> diff = [901, 210, 360, 1260, 30, 330, 150, 60, 600, 270]
    >>> diff_count(diff, 1)
    901
    >>> diff_count(diff, 5)
    2761
    >>> diff_count(diff, 10)
    4171

    >>> diff_count(diff, '1')
    Traceback (most recent call last):
        ...
    ctypes.ArgumentError: zaehler is not an integer

    >>> diff_count(diff, -4)
    Traceback (most recent call last):
        ...
    ValueError: zaehler must greater the 0
    '''
    sum = 0

    if zaehler <= len(diff):
        for i in range(zaehler):
            sum = sum + diff[i]

    return sum

def new_list(diff, diffTime, difference_between):
    '''
        it adds or subtract to the first element of the diff list
        to match up the Hypno and EDF signal
   
    Parameters
    ----------
    diff:
        THe list of values
    diffTime: int | float
        An indicator to determin if the secodns shoud be add or subtract from the
        first element of the diff's list
    difference_between: int | float
        The diffecnce of seconds between the Hypno and the EDF starting points 
        
    Returns
    -------
    diff: list
        a list with the first item altered accordingly 
        
    Examples
    --------
    >>> diff = [930, 210, 360, 1260, 30, 330, 150]
    >>> new_list(diff, -29.0 , 29.0 )
    [959, 210, 360, 1260, 30, 330, 150]

    >>> new_list(diff, "negativ", 39)
    Traceback (most recent call last):
        ...
    ctypes.ArgumentError: diffTime indicator should be a intager of float

    >>> new_list(diff, -29.0 , "29.0")
    Traceback (most recent call last):
        ...
    ctypes.ArgumentError: difference_between should be a intager of float

    '''
    
    print(difference_between)
    print(diff[0])
    print(type(difference_between))

    if diffTime >= 0:
        diff[0] = int(diff[0]-difference_between)
    else: 
        diff[0] = int(diff[0]+difference_between)


    
    return diff

#neues Dataframe ohne Epochen mit zu vielen Artefakten
def clean_df_from_bad_signal(hrv_results, idp, night, selected):
    '''creates a new datafreom with cleaned artifacts 
   
    Parameters
    ----------
    hrv_results:
         DataFrame with HRV results
    idp: string
        The id of the patient
    night: string
        the recorded night
    selected: list
        drop selected values
    
    Returns
    -------
    hrv_results:
        a clean df 
            
    '''
    #todo prüfen selected array
    #for i in selected: 
    #hrv_results.drop(hrv_results.index[i], inplace=True)
    hrv_results.drop(selected, inplace=True)
        
    if not os.path.exists('Probanden/result_hrv'):
        os.makedirs('Probanden/result_hrv')
    
    path_dir = 'Probanden/result_hrv'
    filename = 'results_hrv_cleaned.csv'
    path_file = "{}{}{}{}{}".format(path_dir, os.sep, idp, night, filename)
    #path_file = "{}{}{}{}".format(path_dir, os.sep, idp, filename)
    
    hrv_results.to_csv(path_file, index=True)
        
    return hrv_results

#Function for plotting Raw Signal, checking signal quality
def plot_raw(ecg_signal, limit):
    '''Function for plotting Raw Signal, checking signal quality
   
    Parameters
    ----------
    ecg_signal:
        the ecg singla to print
    limit: int 
        limit the values to print in plot
        
    Returns
    -------
        void
    '''
    plt.rcParams['figure.figsize'] = [15, 9] 
    plt.plot(ecg_signal, alpha=0.5, color='blue', label="raw signal")
    plt.xlim(0,limit)
    plt.show()
    
