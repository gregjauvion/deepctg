
import numpy as np
import pandas as pd
import os


LINK = 'https://www.physionet.org/static/published-projects/ctu-uhb-ctgdb/ctu-chb-intrapartum-cardiotocography-database-1.0.0.zip'
PATH = 'data/ctu-chb-intrapartum-cardiotocography-database-1.0.0'


def download():

    import urllib.request
    import zipfile

    # Download and extract zip
    urllib.request.urlretrieve(LINK, 'database.zip')
    with zipfile.ZipFile('database.zip', 'r') as z:
        z.extractall('data')

    os.remove('database.zip')


def get_record_names():

    names = []
    for p in os.listdir(PATH):
        if p[-4:]=='.dat':
            names.append(p[:-4])

    return names


def read(record_name):
    """
    Read a record and the pH
    """

    import wfdb

    record = wfdb.rdrecord(f'{PATH}/{record_name}')
    header = wfdb.rdheader(f'{PATH}/{record_name}')

    comments = header.comments
    ph = float([c for c in comments if 'pH' in c][0].replace('pH', '').strip())

    fhr, uc = record.p_signal[:,0], record.p_signal[:,1]

    return fhr, uc, ph


# Maximum time before delivery
MAX_NB_VALUES = 4 * (60 * 90)
# Size of FHR time-series used
NB_VALUES = 4 * (60 * 15)
# Maximum number of consecutive NaN
NB_NAN = 4 * 60


def cum_na(v):
    """
    Returns the cumsum of an array, with reset at every nan value
    Example:
    v = np.array([1,np.nan,2,3,4,np.nan,np.nan,5,4,3,np.nan])
    gives 
    """

    v = pd.Series(v)
    cumsum = v.cumsum().fillna(method='pad')
    reset = -cumsum[v.isnull()].diff().fillna(cumsum)
    result = v.where(v.notnull(), reset).cumsum().values
    result[np.isnan(result)] = np.inf

    return result


def process_fhr(fhr):
    """
    Process fetal heart rate
    """

    fhr = fhr[-MAX_NB_VALUES:]
    fhr[fhr==0] = np.nan

    # Interpolate fhr to replace nan
    fhr_interp = pd.DataFrame(fhr).interpolate(method='linear').values.reshape(-1)
    #fhr_interp = pd.DataFrame(fhr).interpolate(method='spline', order=3).values.reshape(-1)

    # Get distance to closest non-NaN
    is_not_nan = np.where(np.isnan(fhr), 1, np.nan)
    dist_left = cum_na(is_not_nan)
    dist_right = cum_na(is_not_nan[::-1])[::-1]
    fhr_filter = (dist_left < NB_NAN) | (dist_right < NB_NAN)
    fhr_filtered = np.where(fhr_filter, fhr_interp, np.nan)

    # Returns longest FHR series without nan
    fhr_cumna = cum_na(np.where(np.isnan(fhr_filtered), np.nan, 1))
    indices = np.where(fhr_cumna>NB_VALUES)[0]
    if len(indices)==0 or indices[-1]<NB_VALUES:
        return None
    else:
        return fhr_interp[indices[-1]-NB_VALUES:indices[-1]]



if __name__=='__main__':

    import matplotlib.pyplot as plt

    # Download database
    download()
    record_names = get_record_names()

    # Read and plot a given record
    record = record_names[0]
    fhr, uc, ph = read(record)

    fig = plt.figure(figsize=(12, 8))
    g = fig.add_subplot(211) ; plt.plot(signal[:,0]) ; plt.title('FHR')
    g = fig.add_subplot(212) ; plt.plot(signal[:,1]) ; plt.title('UC')
    plt.show()

    # Process FHR signals
    data = [(record, process_fhr(read(record)[0])) for record in record_names]
    print(len([i for i in data if i is not None]))

