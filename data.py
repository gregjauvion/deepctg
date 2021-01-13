
import numpy as np
import pandas as pd
import os


###
# Data download/read
###


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


def get_train_test_records(records, p=0.8):

    np.random.seed(123456)
    records_train = set(np.random.choice(records, int(p * len(records)), replace=False))
    records_test = set([r for r in records if not r in records_train])

    return records_train, records_test


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


###
# Data processing
###

# Maximum number of consecutive NaN when interpolating (DeepFHR - 15seconds)
NB_NAN_INTERPOLATION = 4 * 20


def process_fhr(fhr, nb_values_total, nb_values_part):
    """
    Processing is performed the following way:
    - We keep only the last {nb_values_total} points of the signal
    - Holes lasting less than {NB_NAN_INTERPOLATION} are interpolated
    - Split the signal into different parts (without nan values) with at least {nb_values_part} points

    Returns:
    - whole signal interpolated
    - list of start/end indices of the different parts of the signal
    """

    # Keep last points and replace 0 with nan
    fhr = fhr[-nb_values_total:]
    fhr[fhr==0] = np.nan

    # Interpolated signal
    fhr_interp = pd.DataFrame(fhr).interpolate(method='linear', limit=NB_NAN_INTERPOLATION).values.reshape(-1)
    #fhr_interp = pd.DataFrame(fhr).interpolate(method='spline', order=3, limit=NB_NAN_INTERPOLATION).values.reshape(-1)

    # Build start/end indices of different parts of the signal with more than {nb_values_part} points
    parts = []
    na_indices = np.concatenate(([-1], np.argwhere(np.isnan(fhr_interp)).reshape(-1), [len(fhr_interp)]))
    na_indices_diff = na_indices[1:] - na_indices[:-1]
    for start in np.argwhere(na_indices_diff-1>=nb_values_part).reshape(-1):
        parts.append((na_indices[start]+1, na_indices[start+1]))

    return fhr_interp, parts


def build_dataset(nb_values_total, nb_values_part):
    """
    Returns pd DataFrame with one row per part of signal with:
    - record name
    - fhr part of signal
    - pH
    """

    # Concatenate all sections with corresponding pH
    dataset = []
    record_names = get_record_names()
    for record in record_names:
        fhr, uc, ph = read(record)
        fhr_, parts = process_fhr(fhr, nb_values_total, nb_values_part)
        for s, e in parts:
            dataset.append((record, fhr_[s:e], ph))

    return pd.DataFrame(dataset, columns=['record', 'fhr', 'ph'])


PH_LIMIT = 7.15

def sample(dataset, nb_samples, nb_values_part):
    """
    Samples nb_samples in train/test, 0/1 groups
    Every part is sampled according to its size
    """

    records = get_record_names()
    r_train, r_test = get_train_test_records(records, p=0.8)

    d_train, d_test = dataset[dataset.record.isin(r_train)], dataset[dataset.record.isin(r_test)]
    d_train_0, d_train_1 = d_train[d_train.ph>=PH_LIMIT], d_train[d_train.ph<PH_LIMIT]
    d_test_0, d_test_1 = d_test[d_test.ph >= PH_LIMIT], d_test[d_test.ph < PH_LIMIT]

    # Sample nb_samples in each group
    d_train_test = []
    for d in [d_train_0, d_train_1, d_test_0, d_test_1]:
        rows = d.sample(nb_samples, replace=True, weights=d.fhr.apply(len).values)
        indices = [np.random.randint(0, len(f) - nb_values_part) for f in rows.fhr]
        x = np.expand_dims(np.stack([f[i:i+nb_values_part] for i, f in zip(indices, rows.fhr)]), 2)
        y = np.where(rows.ph>=PH_LIMIT, 0, 1)
        d_train_test.append((x, y))

    return d_train_test


def build_evaluation_dataset(nb_values_total, nb_values_part):
    """
    The evaluation is performed on the last part of the signal
    """

    evaluation = []
    records = get_record_names()
    r_train, r_test = get_train_test_records(records, p=0.8)

    for record in records:
        fhr, uc, ph = read(record)
        fhr_, parts = process_fhr(fhr, nb_values_total, nb_values_part)
        if len(parts) > 0:
            s, e = parts[-1]
            evaluation.append((record, fhr_[e - nb_values_part:e], ph))

    df_evaluation = pd.DataFrame(evaluation, columns=['record', 'fhr', 'ph'])

    return df_evaluation[df_evaluation.record.isin(r_train)], df_evaluation[df_evaluation.record.isin(r_test)]


if __name__=='__main__':

    import matplotlib.pyplot as plt

    # Download database
    #download()
    records = get_record_names()
    r_train, r_test = get_train_test_records(records, p=0.8)

    # Build dataset
    nb_values_total = 4 * 60 * 60
    nb_values_part = 4 * 60 * 15
    dataset = build_dataset(nb_values_total, nb_values_part)

    # Read and plot a given record
    for i in range(10):
        record = record_names[i]
        fhr, uc, ph = read(record)
        fhr_, sections = process_fhr(fhr)
        print(len(sections))

        plt.plot(range(len(fhr)), fhr, label='Raw', color='black', linewidth=0.5)
        plt.title(f'pH = {ph}') ; plt.ylim([0, 220]) ; plt.grid()
        for start, end in sections:
            plt.plot(range(start, end), [fhr_[i]+30 for i in range(start, end)], color='red', linewidth=0.5)

        plt.savefig(f'figures/{record}') ; plt.close()

    # Build dataset
    data = [(record, *process_fhr(read(record)[0])) for record in record_names]
    print(len([i for i in data if len(i[2])>0]))


# Short-time Fourier transform

# In order to convert our 1-D
#signals to 2-D using logarithmic spectrogram, we use the signal.Spectrogram module from the Scipy toolbox [26] in python.
#As for the spectrogram’s hyper-parameters, we follow Zihlmann
#et al. [17], i.e., Tukey window of length 64 and hop length of
#32 (i.e., 50% window overlap), and shape parameter of 0.25.

