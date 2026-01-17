import os
from collections import deque
from io import StringIO

import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler


# ======================================================================================================================================================================= #
# Func for data Processing
# ======================================================================================================================================================================= #


# ========================================= Convert NILM dataset to Classif ========================================= #
def nilmdataset_to_clfdataset(data, st_date=None, return_df=True):
    """
    Input:
        data: Convention of 4D array obtained with every NILM Databuilder available in data_processing

    Output:
        X: 2D array of input time series
        y: 2D array as (len(X), 1) -> sum of energy consumed in each window
    """
    list_label = []
    for i in range(len(data)):
        list_label.append(1) if data[i, 1, 1, :].any() > 0 else list_label.append(0)

    if st_date is not None and return_df:
        return pd.concat([pd.DataFrame(data=data[:, 0, 0, :], index=st_date.index).reset_index(), pd.DataFrame(data=list_label, columns=["label"])], axis=1).set_index('index')
    else:
        return data[:, 0, 0, :], np.array(list_label)


# ========================================= Random Under sampler ========================================= #
def undersampler(X, y=None, sampling_strategy='auto', seed=0, nb_label=1):
    np.random.seed(seed)
    
    if isinstance(X, pd.core.frame.DataFrame):
        col = X.columns
        y = X.values[:, -nb_label].astype(int)
        X = X.values[:, :-nb_label]
        X_, y_ = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X, y)
        Mat = np.concatenate((X_, np.reshape(y_, (y_.shape[0],  1))), axis=1)
        Mat = pd.DataFrame(data=Mat, columns=col)
        Mat = Mat.sample(frac=1, random_state=seed)
        
        return Mat
    else:
        assert y is not None, f"For np.array, please provide an y vector."
        X_, y_ = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X, y)
        Mat = np.concatenate((X_, np.reshape(y_, (y_.shape[0],  1))), axis=1)
        np.random.shuffle(Mat)
        Mat = Mat.astype(np.float32)
        
        return Mat[:, :-1], Mat[:, -1]

    
def split_train_valid_test(data, test_size=0.2, valid_size=0, nb_label_col=1, seed=0):
    
    if isinstance(data, pd.core.frame.DataFrame):
        X = data.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
        y = data.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)
    elif isinstance(data, np.ndarray):
        X = data[:,:-nb_label_col]
        y = data[:,-nb_label_col:]
    else:
        raise Exception('Please provide pandas Dataframe or numpy array object.')

    if valid_size != 0:
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=valid_size, random_state=seed)
                
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        
        return X_train, y_train, X_test, y_test
        
        
def split_train_valid_test_pdl(df_data, test_size=0.2, valid_size=0, nb_label_col=1, seed=0, return_df=False):
    """
    Split DataFrame based on index ID (ID PDL for example)
    
    - Input : df_data -> DataFrame
              test_size -> Percentage data for test
              valid_size -> Percentage data for valid
              nb_label_col -> Number of columns of label
              seed -> Set seed
              return_df -> Return DataFrame instances, or Numpy Instances
    - Output:
            np.arrays or DataFrame Instances
    """

    np.random.seed(seed)
    list_pdl = np.array(df_data.index.unique())
    np.random.shuffle(list_pdl)
    pdl_train_valid = list_pdl[:int(len(list_pdl) * (1-test_size))]
    pdl_test = list_pdl[int(len(list_pdl) * (1-test_size)):]
    np.random.shuffle(pdl_train_valid)
    pdl_train = pdl_train_valid[:int(len(pdl_train_valid) * (1-valid_size))]
    
    df_train = df_data.loc[pdl_train, :].copy()
    df_test = df_data.loc[pdl_test, :].copy()
    

    df_train = df_train.sample(frac=1, random_state=seed)
    df_test = df_test.sample(frac=1, random_state=seed)
    
    if valid_size != 0:
        pdl_valid = pdl_train_valid[int(len(pdl_train_valid) * (1-valid_size)):]
        df_valid = df_data.loc[pdl_valid, :].copy()
        df_valid = df_valid.sample(frac=1, random_state=seed)
            
    if return_df:
        if valid_size != 0:
            return df_train, df_valid, df_test
        else:
            return df_train, df_test
    else:
        X_train = df_train.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
        y_train = df_train.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)
        X_test  = df_test.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
        y_test  = df_test.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)

        if valid_size != 0:
            X_valid = df_valid.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
            y_valid = df_valid.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)

            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            return X_train, y_train, X_test, y_test


def split_train_test_pdl_nilmdataset(data, st_date, seed=0,
                                     nb_house_test=None,  perc_house_test=None,  
                                     nb_house_valid=None, perc_house_valid=None):
    
    assert nb_house_test is not None or perc_house_test is not None
    assert len(data)==len(st_date)
    assert isinstance(st_date, pd.DataFrame)
    
    np.random.seed(seed)
    
    if nb_house_valid is not None or perc_house_valid is not None:
        assert (nb_house_test is not None and nb_house_valid is not None) or (perc_house_test is not None and perc_house_valid is not None)
    
    if len(data.shape) > 2:
        tmp_shape = data.shape
        data = data.reshape(data.shape[0], -1)
        
    data = pd.concat([st_date.reset_index(), pd.DataFrame(data)], axis=1).set_index('index')
    list_pdl = np.array(data.index.unique())
    np.random.shuffle(list_pdl)
    
    if nb_house_test is None:
        nb_house_test = max(1, int(len(list_pdl) * perc_house_test))
        if perc_house_valid is not None and nb_house_valid is None:
            nb_house_valid = max(1, int(len(list_pdl) * perc_house_valid))
        
    if nb_house_valid is not None:
        assert len(list_pdl) > nb_house_test + nb_house_valid
    else:
        assert len(list_pdl) > nb_house_test
    
    pdl_test = list_pdl[:nb_house_test]
    
    if nb_house_valid is not None:
        pdl_valid = list_pdl[nb_house_test:nb_house_test+nb_house_valid]
        pdl_train = list_pdl[nb_house_test+nb_house_valid:]
    else:
        pdl_train  = list_pdl[nb_house_test:]
    
    df_train = data.loc[pdl_train, :].copy()
    df_test  = data.loc[pdl_test, :].copy()
    
    st_date_train = df_train.iloc[:, :1]
    data_train    = df_train.iloc[:, 1:].values.reshape((len(df_train), tmp_shape[1], tmp_shape[2], tmp_shape[3]))
    st_date_test  = df_test.iloc[:, :1]
    data_test     = df_test.iloc[:, 1:].values.reshape((len(df_test), tmp_shape[1], tmp_shape[2], tmp_shape[3]))
    
    if nb_house_valid is not None:
        df_valid      = data.loc[pdl_valid, :].copy()
        st_date_valid = df_valid.iloc[:, :1]
        data_valid    = df_valid.iloc[:, 1:].values.reshape((len(df_valid), tmp_shape[1], tmp_shape[2], tmp_shape[3]))
        
        return data_train, st_date_train, data_valid, st_date_valid, data_test, st_date_test
    else:
        return data_train, st_date_train, data_test, st_date_test


def split_train_test_nilmdataset(data, st_date, perc_house_test=0.2, seed=0):
    np.random.seed(seed)

    data_len = np.arange(len(data))
    np.random.shuffle(data_len)

    split_index = int(len(data_len) * (1-perc_house_test))
    train_idx, test_idx = data_len[:split_index], data_len[split_index:]

    data_train, st_date_train = data[train_idx], st_date.iloc[train_idx]
    data_test,  st_date_test  = data[test_idx],  st_date.iloc[test_idx]

    return data_train, st_date_train, data_test, st_date_test
        

def reshape_data(X, y, win, win_max=1024):
    factor = int(win_max // win)
    
    X = X.reshape(X.shape[0], X.shape[1], factor, win).transpose(0, 2, 1, 3).reshape(-1, X.shape[1], win)
    tmp = np.empty((len(y), factor), dtype=y.dtype)
    
    for k in range(factor):
        tmp[:, k] = y.ravel()
        
    tmp = tmp.ravel()
    tmp = tmp.reshape(len(tmp), 1)
        
    return X, tmp

# ===================== REFIT DataBuilder =====================#
class REFIT_DataBuilder(object):
    def __init__(self, 
                 data_path,
                 mask_app,
                 sampling_rate,
                 window_size,
                 window_stride=None,
                 soft_label=False,
                 use_status_from_kelly_paper=False,
                 flag_week=False, flag_day=False):
        
        # =============== Class variables =============== #
        self.data_path = data_path
        self.mask_app = mask_app
        self.sampling_rate = sampling_rate 
        self.soft_label = soft_label

        if isinstance(self.mask_app, str):
            self.mask_app = [self.mask_app]

        if isinstance(window_size, str):
            if window_size=='week':
                self.flag_week = True
                self.flag_day  = False
                if (self.sampling_rate=='1min') or (self.sampling_rate=='1T'):
                    self.window_size = 10080
                elif (self.sampling_rate=='10min') or (self.sampling_rate=='10T'):
                    self.window_size = 1008
                else:
                    raise ValueError(f"Only sampling rate 1min and 10min supported for window size='week', got: {sampling_rate}")
            elif window_size=='day':
                self.flag_week = False
                self.flag_day  = True
                if self.sampling_rate=='30s':
                    self.window_size = 2880
                elif (self.sampling_rate=='1min') or (self.sampling_rate=='1T'):
                    self.window_size = 1440
                elif (self.sampling_rate=='10min') or (self.sampling_rate=='10T'):
                    self.window_size = 144
                else:
                    raise ValueError(f"Only sampling rate 30s, 1min and 10min supported for window size='day', got: {sampling_rate}")
            else:
                raise ValueError(f'Only window size = "day" or "week" for window period related (i.e. str type), got: {window_size}')
        else:
            self.flag_week = flag_week
            self.flag_day  = flag_day
            self.window_size = window_size
        
        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size

        # ======= Add aggregate to appliance(s) list ======= #
        self._check_appliance_names()
        self.mask_app = ['Aggregate'] + self.mask_app

        # ======= Dataset Parameters ======= #
        self.cutoff = 10000

        self.use_status_from_kelly_paper = use_status_from_kelly_paper

        # All parameters are in Watts and adapted from Kelly et all. NeuralNILM paper
        if self.use_status_from_kelly_paper:
            # Threshold parameters are in Watts and time parameter in 10sec
            self.appliance_param = {'Kettle': {'min_threshold': 1000, 'max_threshold': 6000, 'min_on_duration': 1, 'min_off_duration': 0, 'min_activation_time': 0},
                                    'WashingMachine': {'min_threshold': 20, 'max_threshold': 3500, 'min_on_duration': 6, 'min_off_duration': 16, 'min_activation_time': 12},
                                    'Dishwasher': {'min_threshold': 50, 'max_threshold': 3000, 'min_on_duration': 2, 'min_off_duration': 180, 'min_activation_time': 12},
                                    'Microwave': {'min_threshold': 200, 'max_threshold': 6000, 'min_on_duration': 1, 'min_off_duration': 3, 'min_activation_time': 0},
                                    'TumbleDryer': {'min_threshold': 20, 'max_threshold': 3500, 'min_on_duration': 2, 'min_off_duration': 16, 'min_activation_time': 12},
                                }
        else:
            # Threshold parameters are in Watts
            self.appliance_param = {'Kettle': {'min_threshold': 500, 'max_threshold': 6000},
                                    'WashingMachine': {'min_threshold': 300, 'max_threshold': 4000},
                                    'Dishwasher': {'min_threshold': 300, 'max_threshold': 4000},
                                    'Microwave': {'min_threshold': 200, 'max_threshold': 6000},
                                    'TumbleDryer': {'min_threshold': 20, 'max_threshold': 3500}
                                }


    def get_house_data(self, house_indicies):

        assert len(house_indicies)==1, f'get_house_data() implemented to get data from 1 house only at a time.'

        return self._get_dataframe(house_indicies[0])

    def get_classif_dataset(self, house_indicies):
        """
        Process data to build classif dataset
        
        Return : 
            -Time Series: np.ndarray of size [N_ts, Win_Size]
            -Associated label for each subsequences: np.ndarray of size [N_ts, 1] in 0 or 1 for each TS
            -st_date: pandas.Dataframe as :
                - index: id of each house
                - column 'start_date': Starting timestamp of each TS
        """

        nilm_dataset, st_date = self.get_nilm_dataset(house_indicies)
        y = np.zeros(len(nilm_dataset))

        for idx in range(len(nilm_dataset)):    
            if (nilm_dataset[idx, 1, 1, :] > 0).any():
                y[idx] = 1

        return nilm_dataset[:, 0, 0, :], y, st_date        
        
    def get_nilm_dataset(self, house_path, padding=None):
        """
        Process data to build NILM usecase
        
        Return : 
            - np.ndarray of size [N_ts, M_appliances, 2, Win_Size] as :
        
                -1st dimension : nb ts obtained after slicing the total load curve of chosen Houses
                -2nd dimension : nb chosen appliances
                                -> indice 0 for aggregate load curve
                                -> Other appliance in same order as given "appliance_names" list
                -3rd dimension : access to load curve (values of consumption in Wh) or states of activation 
                                of the appliance (0 or 1 for each time step)
                                -> indice 0 : access to load curve
                                -> indice 1 : access to states of activation (0 or 1 for each time step) or Probability (i.e. value in [0, 1]) if soft_label
                -4th dimension : values

            - pandas.Dataframe as :
                index: id of each house
                column 'start_date': Starting timestamp of each TS
        """
        
        output_data = np.array([])
        st_date = pd.DataFrame()
        
        tmp_list_st_date = []

        check_error, data = self._get_dataframe(house_path)
        if check_error == -3:
            return check_error, None, None

	# Quante righe aggiungere
        missing = self.window_size - len(data)

        if missing > 0:
            last_ts = data.index[-1]
            last_row = data.iloc[-1]
            for i in range(1, missing + 1):
                new_ts = last_ts + i * pd.Timedelta(seconds=10)
                data.loc[new_ts] = last_row
        stems, st_date_stems = self._get_stems(data)

        if padding is not None:
            pad_with = [(0, 0), (padding, padding)]
            stems = np.pad(stems, pad_with)
            st_date_stems = list(pd.date_range(end=st_date_stems[0], periods=padding+1, freq=self.sampling_rate))[:-1] + st_date_stems
            st_date_stems = st_date_stems + list(pd.date_range(start=st_date_stems[-1], periods=padding+1, freq=self.sampling_rate))[1:]

            assert len(stems[0])==len(st_date_stems)

        if self.window_size==self.window_stride:
            n_wins = len(data) // self.window_stride
        else:
            n_wins = 1 + ((len(data) - self.window_size) // self.window_stride)

        X = np.empty((n_wins, len(self.mask_app), 2, self.window_size))

        cpt = 0
        for i in range(n_wins):
            tmp = stems[:, i*self.window_stride:i*self.window_stride+self.window_size]

            if not self._check_anynan(tmp): # Check if nan -> skip the subsequences if it's the case
                tmp_list_st_date.append(st_date_stems[i*self.window_stride + int(self.window_size / 2)])

                X[cpt, 0, 0, :] = tmp[0, :]
                X[cpt, 0, 1, :] = (tmp[0, :] > 0).astype(dtype=int)

                key = 1
                for j in range(1, len(self.mask_app)):
                    X[cpt, j, 0, :] = tmp[key, :]
                    X[cpt, j, 1, :] = tmp[key+1, :]
                    key += 2

                cpt += 1 # Add one subsequence

        tmp_st_date = pd.DataFrame(data=tmp_list_st_date, columns=['start_date'])
        output_data = np.concatenate((output_data, X[:cpt, :, :, :]), axis=0) if output_data.size else X[:cpt, :, :, :]
        st_date = pd.concat([st_date, tmp_st_date], axis=0) if st_date.size else tmp_st_date
                
        return 1, output_data, st_date
    
    def _get_stems(self, dataframe):
        """
        Extract load curve for each chosen appliances.
        
        Return : np.ndarray instance
        """
        stems = np.empty((1 + (len(self.mask_app)-1)*2, dataframe.shape[0]))
        stems[0, :] = dataframe['Aggregate'].values

        key = 1
        for appliance in self.mask_app[1:]:
            stems[key, :] = dataframe[appliance].values
            stems[key+1, :] = dataframe[appliance+'_status'].values
            key+=2

        return stems, list(dataframe.index)
    
    def _compute_status(self, initial_status, min_on, min_off, min_activation_time):
        tmp_status = np.zeros_like(initial_status)
        status_diff = np.diff(initial_status)
        events_idx  = status_diff.nonzero()

        events_idx  = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx     = events_idx.reshape((-1, 2))
        on_events      = events_idx[:, 0].copy()
        off_events     = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events    = on_events[off_duration > min_off]
            off_events   = off_events[np.roll(off_duration, -1) > min_off]

            on_duration  = off_events - on_events
            on_events    = on_events[on_duration  >= min_on]
            off_events   = off_events[on_duration >= min_on]
            assert len(on_events) == len(off_events)

        # Filter activations based on minimum continuous points after applying min_on and min_off
        activation_durations = off_events - on_events
        valid_activations = activation_durations >= min_activation_time
        on_events = on_events[valid_activations]
        off_events = off_events[valid_activations]

        for on, off in zip(on_events, off_events):
            tmp_status[on:off] = 1

        return tmp_status
    
    def _get_dataframe(self, house_path):
        """
        Load houses data and return one dataframe with aggregate and appliance resampled at chosen time step.
        
        Return : pd.core.frame.DataFrame instance
        """
        print(f"Carico casa {house_path}")
        file = house_path
        self._check_if_file_exist("data/HOUSES_Labels")
        labels_houses = pd.read_csv("data/HOUSES_Labels").set_index('House_id')
        with open(file, "r") as f:
            last_30_lines = deque(f, 300)  # header + 301 righe
        csv_snippet = "".join(last_30_lines)
        house_data = pd.read_csv(StringIO(csv_snippet))
        house_data.columns = list(labels_houses.loc[22].values)
        house_data = house_data.set_index('Time').sort_index()
        house_data.index = pd.to_datetime(house_data.index)
        time_diff = (house_data.index[-1] - house_data.index[0]).total_seconds()
        if time_diff > 3600:
            return -3, None

        idx_to_drop = house_data[house_data['Issues']==1].index
        house_data = house_data.drop(index=idx_to_drop, axis=0)
        #house_data = house_data.resample(rule='10s').mean().ffill(limit=9) # Resample to minimum of 10sec and ffill for 1min30
        # Resample a 10s
        house_data = house_data.resample(rule='10s').mean()
        
        # Interpola i NaN in mezzo usando i valori vicini (media lineare nel tempo)
        house_data = house_data.interpolate(method='time')
        
        # Come sicurezza extra, se rimane ancora qualche NaN ai bordi:
        house_data = house_data.ffill().bfill()
        
        house_data[house_data < 5] = 0 # Remove small value
        house_data = house_data.clip(lower=0, upper=self.cutoff) # Apply general cutoff
        house_data = house_data.sort_index()

        if self.flag_week:
            tmp_min = house_data[(house_data.index.weekday == 0) & (house_data.index.hour == 0) & (house_data.index.minute == 0) & (house_data.index.second == 0)]
            house_data = house_data[house_data.index >= tmp_min.index[0]]
        elif self.flag_day:
            tmp_min = house_data[(house_data.index.hour == 0) & (house_data.index.minute == 0) & (house_data.index.second == 0)]
            house_data = house_data[house_data.index >= tmp_min.index[0]]
        
        for appliance in self.mask_app[1:]:
            # Check if appliance is in this house
            if appliance in house_data:

                # Replace nan values by -1 during appliance activation status filtering
                house_data[appliance] = house_data[appliance].replace(np.nan, -1)

                # Creating status
                initial_status = ((house_data[appliance] >= self.appliance_param[appliance]['min_threshold']) & (house_data[appliance] <= self.appliance_param[appliance]['max_threshold'])).astype(int).values
                
                if self.use_status_from_kelly_paper:
                    house_data[appliance+'_status'] = self._compute_status(initial_status, 
                                                                           self.appliance_param[appliance]['min_on_duration'], 
                                                                           self.appliance_param[appliance]['min_off_duration'],
                                                                           self.appliance_param[appliance]['min_activation_time'])
                else:
                    house_data[appliance+'_status'] = initial_status

                # Finally replacing nan values put to -1 by nan
                house_data[appliance] = house_data[appliance].replace(-1, np.nan)

        if self.sampling_rate!='10s':
            house_data = house_data.resample(self.sampling_rate).mean()

        tmp_list = ['Aggregate']
        for appliance in self.mask_app[1:]:
            tmp_list.append(appliance)
            tmp_list.append(appliance+'_status')
            if appliance in house_data:
                if not self.soft_label:
                    house_data[appliance+'_status'] = (house_data[appliance+'_status'] > 0).astype(int)
                else:
                    continue
            else:
                house_data[appliance] = 0
                house_data[appliance+'_status'] = 0

        #aggiungo io le prossime due righe per eliminare i nan
        house_data = house_data.interpolate(method='time')
        house_data = house_data.ffill().bfill()

        house_data = house_data[tmp_list]

        return 1, house_data
    
    def _check_appliance_names(self):
        """
        Check appliances names for UKDALE case.
        """
        for appliance in self.mask_app:
            assert appliance in ['WashingMachine', 'Dishwasher', 'Kettle', 'Microwave', "TumbleDryer"], f"Selected applicance unknow for REFIT Dataset, got: {appliance}"
        return
            
    def _check_if_file_exist(self, file):
        """
        Check if file exist at provided path.
        """
        if os.path.isfile(file):
            pass
        else:
            raise FileNotFoundError
        return
    
    def _check_anynan(self, a):
        """
        Fast check of NaN in a numpy array.
        """
        return np.isnan(np.sum(a))
