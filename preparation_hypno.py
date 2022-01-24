import pandas as pd
import csv
from datetime import time
from datetime import datetime
from datetime import date
from datetime import timedelta
from enum import Enum


FILE_NAME = __import__(__name__)

'''
Functions for data preparation of hypnogram files
'''

testfile_somno = "Schlafprofil.txt"
testfile_nihon = "Nihon_hypno_anon.html"
testfile_alice = "STADIUManonym_alice.csv"
start_of_hypno_nihon = '23:09:09'




class System(Enum):
    Nihon = 'Nihon'
    Alice = 'Alice'
    Somnomedics = 'Somnomedics'
    
def fileopener(filename):
    '''
    fileopener reads a given somnomedics hypnogram file txt and returns it as a pandas DataFrame
    
    Parameters
    ----------
    filename : string
        The name of the File to open
        
    Returns
    -------
    pandas.core.frame.DataFrame
        A Dataframe of the readed file
    
    
    Examples
    --------
    It requires a filename:
    >>> fileopener()
    Traceback (most recent call last):
        ...
    TypeError: fileopener() missing 1 required positional argument: 'filename'
    
    
    The given filename should be a string:
    >>> fileopener(123)
    Traceback (most recent call last):
        ...
    ValueError: filename should be a string
    
    
    It returns a pandas.core.frame.DataFrame class:
    >>> type(fileopener(testfile_somno))
    <class 'pandas.core.frame.DataFrame'>
    
    It returns all readed lines:
    >>> fileopener(testfile_somno).size
    1820
    '''
    if not isinstance(filename, str):
        raise ValueError('filename should be a string')
    
    return pd.read_csv(filename, skiprows=5, delimiter=";", sep=" ", names=["Time", "Stadium"], header=0)


def fileopener_alice(filename):
    '''
    fileopener_alice reads a given Alice5 hypnogram csv file and returns it as a pandas DataFrame and header as String
    
    Parameters
    ----------
    filename : string
        The name of the File to open
        
    Returns
    -------
    pandas.core.frame.DataFrame
        A Dataframe of the readed file
    header    
        and header as String
    
    Examples
    --------
    It requires a filename:
    >>> fileopener_alice()
    Traceback (most recent call last):
        ...
    TypeError: fileopener_alice() missing 1 required positional argument: 'filename'
    
    
    The given filename should be a string:
    >>> fileopener_alice(123)
    Traceback (most recent call last):
        ...
    ValueError: filename should be a string
    
    
    It returns a pandas.core.frame.DataFrame class:
    >>> header, dfh = fileopener_alice(testfile_alice)
    >>> type(dfh)
    <class 'pandas.core.frame.DataFrame'>
    
    returns the starttime of hypnogram from header:
    >>> header
    '22:37:30'
    '''
    if not isinstance(filename, str):
        raise ValueError('filename should be a string')
        
    header = []
    with open(filename, 'r', newline='') as csvheader:
        csv_header = csv.reader(csvheader)
        for index, row in enumerate(csv_header):
            if index == 0:
                #get starttime of hypno
                header = row[4]
    return header, pd.read_csv(filename, skiprows=1, delimiter=",", sep=" ", names=["Stadium"], header=None)


def make_df_hypno_all(file, system, start_of_hypno= None):
    '''
    creates a Dataframe from different Hypnogram types with 'Stadium' and belonging timestamps
    
    Parameters
    ----------
    file : string
        The name of the File to open
    system: System
        The system the hypnogram was recorded by
    start_of_hypno: String or None
        if the hypnogram does not have a timestamp, enter it manually
        
    Returns
    -------
    pandas.core.frame.DataFrame
        A Dataframe of the created hypnogram
    
    Examples
    --------
    The given file must be a string:
    >>> make_df_hypno_all(123, System.Nihon, start_of_hypno_nihon)
    Traceback (most recent call last):
        ...
    ValueError: filename must be a string
    
    The it raises an error, if a system is not supported:
    >>> make_df_hypno_all(testfile_nihon, "unknown" , start_of_hypno_nihon)
    Traceback (most recent call last):
        ...
    AttributeError: The given System is not supported
    
    The it raises an error, if start_of_hypno is not a string or none:
    >>> make_df_hypno_all(testfile_nihon, System.Nihon , 123)
    Traceback (most recent call last):
        ...
    ValueError: start_of_hypno must be a string or none
    
    It returns a dataframe for the given system
    >>> dfh = make_df_hypno_all(testfile_alice, System.Alice)
    >>> dfh.iloc[:3] #doctest: +NORMALIZE_WHITESPACE
           Time Stadium
    0  22:37:30    Wach
    1  22:37:31    Wach
    2  22:37:32    Wach

    '''
    
    if not isinstance(file, str):
        raise ValueError('filename must be a string')
    
    if not ( isinstance(start_of_hypno, str) or start_of_hypno is None):
        raise ValueError('start_of_hypno must be a string or none')
    
    if system == System.Nihon :
        states, header = read_sleeping_state_nihon_kohden(file)
        dfh = pd.DataFrame(states,columns =['Stadium'])
        dfh = dfh.replace(['W', 'R', 'N1', 'N2', 'N3'],['Wach', 'Rem', 'N1', 'N2', 'N3'])

        indexNames = dfh[ dfh['Stadium'] == 'L' ].index
        dfh.drop(indexNames , inplace=True)
        dfh.reset_index(inplace=True, drop=True)
        seconds = 30
        dfh = make_time_stamps(start_of_hypno, dfh, seconds)
    
    elif system == System.Alice :
        start_of_hypno, dfh = fileopener_alice(file)
        dfh = dfh.replace([11, 12, 13, 14, 15],['Wach', 'Rem', 'N1', 'N2', 'N3'])
        seconds = 1
        dfh = make_time_stamps(start_of_hypno, dfh, seconds)

        
    elif system == System.Somnomedics :
        dfh = fileopener(file)
        dfh["Time"]= dfh["Time"].str[:8]
        
    else:
        raise AttributeError("The given System is not supported")
            
    #dfh = make_time_stamps(start_of_hypno, dfh, seconds)
    
    return dfh


def read_sleeping_state_nihon_kohden(file, startValue = 3, sleeping_state_row = 20):
    """
    read out sleep_stages from html file
    
    Parameters
    ----------
    file:   html file
        the epoch report as html file 
    startValue: int
        table number in html
        
    sleeping_state_row: int
        the colums wich include the states in table 
    
        
    Returns
    -------
    states
        returned list of sleep stages incl. stadium 'L' for 'Lights on'
    
    header:
        first entry of the Table cuttet from rest
    
    Examples
    --------
    It requires a dataframe:
    SKIP>>> result = read_sleeping_state_nihon_kohden(testfile_nihon)
    SKIP>>> len(result[0])
    1197
    
    """
    table = pd.read_html(file)
    states = [];

    for i in range(startValue, len(table)):
        header, *rest = table[i][sleeping_state_row].to_list()
        states.extend(rest)

    return states, header


def get_timedate_plus(timedate1, seconds):
    """
    calculates the next time entry for hypnogram, if only starttime ist known
    
    Parameters
    ----------
    timedate1: timedate.timedate
        
    seconds: int
        the time, sleep stages events are recorded in hypnogram
        
    Returns
    -------
    dateDiff
        the datetime object x seconds later than the datetime before
    
    Examples
    --------
    It requires a dataframe:
    >>> start = datetime.strptime(start_of_hypno_nihon, "%H:%M:%S")
    >>> print(get_timedate_plus(start, 30))
    1900-01-01 23:09:39
    >>> type(get_timedate_plus(start, 30))
    <class 'datetime.datetime'>
    """

    dateDiff = timedate1 + timedelta(seconds=seconds)

    return dateDiff;

def make_time_stamps(start_of_hypno, dfh, seconds):
    """
    generates the column "Time" of dfh with only starttime
    
    Parameters
    ----------
    start_of_hypno: datetime or String
        the time where hypnogram starts
        
    dfh: pandas.core.Dataframe
        the given list of stages from raw file
        
    seconds: int
        the time where stages are recorded every x seconds
    
        
    Returns
    -------
    dfh: pandas.core.Dataframe
        a new DataFrame with mapped time stamps and stages
    
    Examples
    --------
    It requires a dataframe:
    >>> header, dfh = fileopener_alice(testfile_alice)
    >>> dfhnew = make_time_stamps(header, dfh, 1)
    >>> len(dfh)==len(dfhnew)
    True
    
    """
    
    startedhypno = datetime.strptime(start_of_hypno, "%H:%M:%S")
    new = get_timedate_plus(startedhypno, seconds)

    times = []

    for i in range(len(dfh)):
        if i>0 :
            newdate = get_timedate_plus(times[i-1],seconds)
            #newdate = newdate.strftime("%H:%M:%S")
            times.append(newdate)
            #times.append(strptime(newdate, "%H:%M:%S"))
        else:
            times.append(startedhypno)


    dfh.insert(0,'Time',times)
    dfh['Time'] = dfh['Time'].dt.strftime("%H:%M:%S")
    
    return dfh


def calculate_cutpoints(dfh, strip= False):
    """
    It calculates the cutoff values by its sleep state change, positions where sleep stages are changing
    
    Parameters
    ----------
    dfh: pandas.core.frame.DataFrame
        
    strip: Boolean
        if the date has to be cutted of
        
    Returns
    -------
    cutpoints: pandas.core.frame.DataFrame
        new DataFrame with timestamp and stadium for each entry with new index
    
    Examples
    --------
    It requires a dataframe:
    >>> calculate_cutpoints()
    Traceback (most recent call last):
        ...
    TypeError: calculate_cutpoints() missing 1 required positional argument: 'dfh'
    
    
    It calulates the cutoff values:
    >>> f = fileopener(testfile_somno)
    >>> calculate_cutpoints(f,True).iloc[[0,1,2,3,4]] #doctest: +NORMALIZE_WHITESPACE
               Time Stadium
    0  23:24:30    Wach
    1  23:40:00      N1
    2  23:43:30      N2
    3  23:49:30      N3
    4  00:10:30      N1
    
    It returns the correct count of values:
    >>> f = fileopener(testfile_somno)
    >>> len(calculate_cutpoints(f, True).values)
    63
    """
    # cutoff the miliseconds
    if strip == True:
        dfh["Time"] = dfh["Time"].str[:8]

    change_list = [dfh.loc[0]]

    # create list of cutpoints where stadium changes
    for i in range(len(dfh)):
        if dfh.Stadium.iloc[i] != dfh.Stadium.iloc[i - 1] and i != 0:
            change_list.append(dfh.loc[i])

    # create new dataframe from List
    cutpoints = pd.DataFrame(change_list, columns=['Time', 'Stadium'])

    # pd.to_datetime(cutpoints['Time'], format='%H:%M:%S').dt.time

    return cutpoints.reset_index(drop=True)


def calculate_cutpoints_diff(cutpoints):
    """
    it calculates the ellapse time between two sleeping states 
    
    Parameters
    ----------
    cutpoints: DataFrame
        the DataFrame with entries of changing
    Returns
    -------
    diff :list
        a list with determined time spans in seconds
    
    Examples
    --------
    It requires a dataframe:
    >>> calculate_cutpoints_diff()
    Traceback (most recent call last):
        ...
    TypeError: calculate_cutpoints_diff() missing 1 required positional argument: 'cutpoints'
   
    It returns an array of the ellaped time:
    >>> cp = calculate_cutpoints(fileopener(testfile_somno), True)
    >>> test = calculate_cutpoints_diff(cp)
    >>> [test[i] for i in range(0,10)]
    [930, 210, 360, 1260, 30, 330, 150, 60, 600, 270]
    
    It counts the array size of the return points :
    >>> cp = calculate_cutpoints(fileopener(testfile_somno), True)
    >>> len(calculate_cutpoints_diff(cp))
    62
    """
    diff = []
    for i in range(len(cutpoints)):
        # print(cutpoints.Time.iloc[i])
        # print(i)
        if i < len(cutpoints) - 1:
            subs = pd.Timedelta(cutpoints.Time.iloc[i + 1]).seconds - pd.Timedelta(cutpoints.Time.iloc[i]).seconds
            if subs > 0:
                diff.append(subs)
            else:
                pos = subs + 86400
                diff.append(pos)

    return diff



