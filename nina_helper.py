"""Utility functions to help with working with NinaPro database."""

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from itertools import combinations, chain
from sklearn import preprocessing
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation





def DA_Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise
'''Scaling'''
def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

'''Magnitude Warping'''
## This example using cubic splice is not the best approach to generate random curves.
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    tt = np.arange(0, X.shape[0], (X.shape[0]-1)/(knot+1))
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    curves = []
    for i in range(X.shape[1]):
        cs = CubicSpline(tt, yy[:,i])
        curves.append(cs(x_range))
    return np.array(curves).transpose()
def DA_MagWarp(X, sigma=0.2):
    return X * GenerateRandomCurves(X, sigma)


'''Time Warping'''
def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum
def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
    return X_new
'''Rotation'''
def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))


#db1_path='../../ninaproDB1/'+'DB1_s'+str(subject)+'/DB1_s'+str(subject)+  '/S' + str(subject) + '_A1_E1.mat'
def import_db1(folder_path, subject,rest_length_cap=5):
    """Function for extracting data from raw NinaiPro files for ninaproDB1.

    Args:
        folder_path (string): Path to folder containing raw mat files
        subject (int): 1-27 which subject's data to import
        rest_length_cap (int, optional): The number of seconds of rest data to keep before/after a movement

    Returns:
        Dictionary: Raw EMG data, corresponding repetition and movement labels, indices of where repetitions are
            demarked and the number of repetitions with capped off rest data


    """
    fs = 100


    data = sio.loadmat(
        folder_path + 'DB1_s' + str(subject) + '/DB1_s' + str(subject) + '/S' + str(subject) + '_A1_E2.mat')
    emg = np.squeeze(np.array(data['emg']))
    rep = np.squeeze(np.array(data['rerepetition']))
    move = np.squeeze(np.array(data['restimulus']))

    # data = sio.loadmat(folder_path + 'DB1_s' + str(subject) + '/DB1_s' + str(subject) + '/S' + str(subject) + '_A1_E1.mat')
    # emg = np.squeeze(np.array(data['emg']))
    # rep = np.squeeze(np.array(data['rerepetition']))
    # move = np.squeeze(np.array(data['restimulus']))
    #
    # data = sio.loadmat(folder_path + 'DB1_s' + str(subject) + '/DB1_s' + str(subject) + '/S' + str(subject) +'_A1_E2.mat')
    # emg = np.vstack((emg, np.array(data['emg'])))
    # rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
    # move_tmp = np.squeeze(np.array(data['restimulus']))  # Fix for numbering
    # move_tmp[move_tmp != 0] += max(move)
    # move = np.append(move, move_tmp)

    # data = sio.loadmat(folder_path + 'DB1_s' + str(subject) + '/DB1_s' + str(subject) + '/S' + str(subject) + '_A1_E3.mat')
    # emg = np.vstack((emg, np.array(data['emg'])))
    # rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
    # move_tmp = np.squeeze(np.array(data['restimulus']))  # Fix for numbering
    # move_tmp[move_tmp != 0] += max(move)
    # move = np.append(move, move_tmp)

    move = move.astype('int8')  # To minimise overhead

    # Label repetitions using new block style: rest-move-rest regions
    move_regions = np.where(np.diff(move))[0]
    rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
    nb_reps = int(round(move_regions.shape[0] / 2))
    last_end_idx = int(round(move_regions[0] / 2))
    nb_unique_reps = np.unique(rep).shape[0] - 1  # To account for 0 regions
    nb_capped = 0
    cur_rep = 1

    rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
    for i in range(nb_reps - 1):
        rep_regions[2 * i] = last_end_idx
        midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                  move_regions[2 * (i + 1)]) / 2)) + 1

        trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
        if trailing_rest_samps <= rest_length_cap * fs:
            rep[last_end_idx:midpoint_idx] = cur_rep
            last_end_idx = midpoint_idx
            rep_regions[2 * i + 1] = midpoint_idx - 1

        else:
            rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                           int(round(rest_length_cap * fs)))
            rep[last_end_idx:rep_end_idx] = cur_rep
            last_end_idx = ((move_regions[2 * (i + 1)] -
                             int(round(rest_length_cap * fs))))
            rep_regions[2 * i + 1] = rep_end_idx - 1
            nb_capped += 2

        cur_rep += 1
        if cur_rep > nb_unique_reps:
            cur_rep = 1
    # emg=DA_Jitter(emg)
    end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
    rep[last_end_idx:end_idx] = cur_rep
    rep_regions[-2] = last_end_idx
    rep_regions[-1] = end_idx - 1
    return {'emg': emg,
            'rep': rep,
            'move': move,
            'rep_regions': rep_regions,
            'nb_capped': nb_capped
            }
def import_db5(folder_path, subject, rest_length_cap=5):
    """Function for extracting data from raw NinaiPro files for ninaproDB1.

    Args:
        folder_path (string): Path to folder containing raw mat files
        subject (int): 1-27 which subject's data to import
        rest_length_cap (int, optional): The number of seconds of rest data to keep before/after a movement

    Returns:
        Dictionary: Raw EMG data, corresponding repetition and movement labels, indices of where repetitions are
            demarked and the number of repetitions with capped off rest data


    """
    fs = 100

    data = sio.loadmat(
        folder_path + 's' + str(subject) + '/s' + str(subject)+ '/S' + str(subject) + '_E2_A1.mat')
    emg = np.squeeze(np.array(data['emg']))
    rep = np.squeeze(np.array(data['rerepetition']))
    move = np.squeeze(np.array(data['restimulus']))

    # data = sio.loadmat(folder_path+'DB1_s'+str(subject)+'/DB1_s'+str(subject)+  '/S' + str(subject) + '_A1_E1.mat')
    # emg = np.squeeze(np.array(data['emg']))
    # rep = np.squeeze(np.array(data['rerepetition']))
    # move = np.squeeze(np.array(data['restimulus']))
    #
    # data = sio.loadmat(folder_path+'DB1_s'+str(subject)+'/DB1_s'+str(subject)+  '/S' + str(subject) + '_A1_E2.mat')
    # emg = np.vstack((emg, np.array(data['emg'])))
    # rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
    # move_tmp = np.squeeze(np.array(data['restimulus']))  # Fix for numbering
    # move_tmp[move_tmp != 0] += max(move)
    # move = np.append(move, move_tmp)
    #
    # data = sio.loadmat(folder_path+'DB1_s'+str(subject)+'/DB1_s'+str(subject)+  '/S' + str(subject) + '_A1_E3.mat')
    # emg = np.vstack((emg, np.array(data['emg'])))
    # rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
    # move_tmp = np.squeeze(np.array(data['restimulus']))  # Fix for numbering
    # move_tmp[move_tmp != 0] += max(move)
    # move = np.append(move, move_tmp)

    move = move.astype('int8')  # To minimise overhead

    # Label repetitions using new block style: rest-move-rest regions
    move_regions = np.where(np.diff(move))[0]
    rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
    nb_reps = int(round(move_regions.shape[0] / 2))
    last_end_idx = int(round(move_regions[0] / 2))
    nb_unique_reps = np.unique(rep).shape[0] - 1  # To account for 0 regions
    nb_capped = 0
    cur_rep = 1

    rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
    for i in range(nb_reps - 1):
        rep_regions[2 * i] = last_end_idx
        midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                  move_regions[2 * (i + 1)]) / 2)) + 1

        trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
        if trailing_rest_samps <= rest_length_cap * fs:
            rep[last_end_idx:midpoint_idx] = cur_rep
            last_end_idx = midpoint_idx
            rep_regions[2 * i + 1] = midpoint_idx - 1

        else:
            rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                           int(round(rest_length_cap * fs)))
            rep[last_end_idx:rep_end_idx] = cur_rep
            last_end_idx = ((move_regions[2 * (i + 1)] -
                             int(round(rest_length_cap * fs))))
            rep_regions[2 * i + 1] = rep_end_idx - 1
            nb_capped += 2

        cur_rep += 1
        if cur_rep > nb_unique_reps:
            cur_rep = 1

    end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
    rep[last_end_idx:end_idx] = cur_rep
    rep_regions[-2] = last_end_idx
    rep_regions[-1] = end_idx - 1
    print(emg.shape)
    return {'emg': emg,
            'rep': rep,
            'move': move,
            'rep_regions': rep_regions,
            'nb_capped': nb_capped
            }
def gen_split_balanced(rep_ids, nb_test, base=None):
    """Create a balanced split for training and testing based on repetitions (all reps equally tested + trained on) .

    Args:
        rep_ids (array): Repetition identifiers to split
        nb_test (int): The number of repetitions to be used for testing in each each split
        base (array, optional): A specific test set to use (must be of length nb_test)

    Returns:
        Arrays: Training repetitions and corresponding test repetitions as 2D arrays [[set 1], [set 2] ..]
    """
    nb_reps = rep_ids.shape[0]
    nb_splits = nb_reps

    # Generate all possible combinations
    all_combos = combinations(rep_ids, nb_test)
    all_combos = np.fromiter(chain.from_iterable(all_combos), int)
    all_combos = all_combos.reshape(-1, nb_test)
    all_combos = np.delete(all_combos, np.where(np.all(all_combos == base, axis=1))[0][0], axis=0)
    all_combos_copy = all_combos

    test_reps = np.zeros((nb_splits, nb_test), dtype=int)
    if base is not None:
        test_reps[0, :] = base

    train_reps = np.zeros((nb_splits, nb_reps - nb_test,), dtype=int)

    cur_split = 1
    reset_counter = 0
    while cur_split < (nb_splits):
        if reset_counter >= 10 or all_combos.shape[0] == 0:
            all_combos = all_combos_copy
            test_reps = np.zeros((nb_splits, nb_test), dtype=int)
            if base is not None:
                test_reps[0, :] = base

            cur_split = 1
            reset_counter = 0

        randomIndex = np.random.randint(0, all_combos.shape[0])
        test_reps[cur_split, :] = all_combos[randomIndex, :]
        all_combos = np.delete(all_combos, randomIndex, axis=0)

        _, counts = np.unique(test_reps[:cur_split + 1], return_counts=True)

        if max(counts) > nb_test:
            test_reps[cur_split, :] = np.zeros((1, nb_test), dtype=int)
            reset_counter += 1
            continue
        else:
            cur_split += 1
            reset_counter = 0

    for i in range(nb_splits):
        train_reps[i, :] = np.setdiff1d(rep_ids, test_reps[i, :])

    return train_reps, test_reps


def gen_split_rand(rep_ids, nb_test, nb_splits, base=None):
    """Randomly generate nb_splits out of nb_reps training-test splits.

    Args:
        rep_ids (array): Repetition identifiers to split
        nb_test (int): The number of repetitions to be used for testing in each each split
        nb_splits (int): The number of splits to produce
        base (array, optional): A specific test set to use (must be of length nb_test)

    Returns:
        Arrays: Training repetitions and corresponding test repetitions as 2D arrays [[set 1], [set 2] ..]
    """
    nb_reps = rep_ids.shape[0]

    all_combos = combinations(rep_ids, nb_test)
    all_combos = np.fromiter(chain.from_iterable(all_combos), int)
    all_combos = all_combos.reshape(-1, nb_test)
    all_combos = np.delete(all_combos, np.where(np.all(all_combos == base, axis=1))[0][0], axis=0)

    test_reps = np.zeros((nb_splits, nb_test), dtype=int)
    if base is not None:
        test_reps[0, :] = base

    train_reps = np.zeros((nb_splits, nb_reps - nb_test,), dtype=int)

    for i in range(1, nb_splits):
        rand_idx = np.random.randint(all_combos.shape[0])
        test_reps[i, :] = all_combos[rand_idx, :]
        train_reps[i, :] = np.setdiff1d(rep_ids, test_reps[i, :])
        all_combos = np.delete(all_combos, rand_idx, axis=0)

    return train_reps, test_reps

def Jitter(emg, reps, train_reps):
    train_targets = get_idxs(reps, train_reps)
    # print(emg.shape)
    # print(train_targets.shape)
    '''(142976, 10)
(114010,)'''
    return DA_Jitter(emg[train_targets, :])
def rotaion(emg, reps, train_reps):
    train_targets = get_idxs(reps, train_reps)
    print(emg.shape)
    print(train_targets.shape)
    '''(142976, 10)
(114010,)'''
    return DA_Rotation(emg[train_targets, :])
def Scale(emg, reps, train_reps):
    train_targets = get_idxs(reps, train_reps)
    print(emg.shape)
    print(train_targets.shape)
    '''(142976, 10)
(114010,)'''
    return DA_Scaling(emg[train_targets, :])
def magnitude_warping(emg, reps, train_reps):
    train_targets = get_idxs(reps, train_reps)
    print(emg.shape)
    print(train_targets.shape)
    '''(142976, 10)
(114010,)'''
    return DA_MagWarp(emg[train_targets, :])
def time_wraping(emg, reps, train_reps):
    train_targets = get_idxs(reps, train_reps)
    print(emg.shape)
    print(train_targets.shape)
    '''(142976, 10)
(114010,)'''
    return DA_TimeWarp(emg[train_targets, :])
def normalise_emg(emg, reps, train_reps):
    """Preprocess train+test data to mean 0, std 1 based on training data only.

    Args:
        emg (array): Raw EMG data
        reps (array): Corresponding repetition information for each EMG observation
        train_reps (array): Which repetitions are in the training set

    Returns:
        array: Rescaled EMG data
    """
    train_targets = get_idxs(reps, train_reps)
    print(emg.shape)
    print(train_targets.shape)
    '''(142976, 10)
(114010,)'''
    scaler = StandardScaler(with_mean=True,
                            with_std=True,
                            copy=False).fit(emg[train_targets, :])

    return scaler.transform(emg)
def normalise_emg_all(emg, reps, train_reps):
    """Preprocess train+test data to mean 0, std 1 based on training data only.

    Args:
        emg (array): Raw EMG data
        reps (array): Corresponding repetition information for each EMG observation
        train_reps (array): Which repetitions are in the training set

    Returns:
        array: Rescaled EMG data
    """
    train_targets = get_idxs(reps, train_reps)
    print(train_targets.shape)
    scaler = StandardScaler(with_mean=True,
                            with_std=True,
                            copy=False).fit(emg[train_targets, :])

    return scaler.transform(emg)

def get_windows(subject,which_reps, window_len, window_inc, emg, movements, repetitons, which_moves=None, dtype=np.float32):
    """Get set of windows based on repetition and movement criteria and associated label + repetition data.

    Args:
        which_reps (array): Which repetitions to return
        window_len (int): Desired window length
        window_inc (int): Desired window increment
        emg (array): EMG data (should be normalise beforehand)
        movements (array): Movement labels
        repetitons (array): Repetition labels
        which_moves (array, optional): Which movements to return - if None use all
        dtype (TYPE, optional): What precision to use for EMG data

    Returns:
        X_data (array): Windowed EMG data
        Y_data (array): Movement label for each window
        R_data (array): Repetition label for each window
    """
    nb_obs = emg.shape[0]
    nb_channels = emg.shape[1]

    # All possible window end locations given an increment size
    possible_targets = np.array(range(window_len - 1, nb_obs, window_inc))

    targets = get_idxs(repetitons[possible_targets], which_reps)

    # Re-adjust back to original range (for indexinging into rep/move)
    targets = (window_len - 1) + targets * window_inc

    # Keep only selected movement(s)
    if which_moves is not None:
        move_targets = get_idxs(movements[targets], which_moves)
        targets = targets[move_targets]

    X_data = np.zeros([targets.shape[0], window_len, nb_channels, 1],
                      dtype=dtype)
    Y_data = np.zeros([targets.shape[0], ], dtype=np.int8)
    R_data = np.zeros([targets.shape[0], ], dtype=np.int8)
    for i, win_end in enumerate(targets):
        win_start = win_end - (window_len - 1)
        X_data[i, :, :, 0] = emg[win_start:win_end + 1, :]  # Include end
        Y_data[i] = movements[win_end]
        R_data[i] = repetitons[win_end]
        # Create a mask where Y_data is not zero
    # #去除休息手势
    # mask = Y_data != 0
    #
    # # Apply the mask to X_data, Y_data, and R_data
    # X_data = X_data[mask]
    # Y_data = Y_data[mask]
    # R_data = R_data[mask]

    #只使用休息手势
    mask = Y_data == 0

    # Apply the mask to X_data, Y_data, and R_data
    X_data = X_data[mask]
    Y_data = Y_data[mask]
    R_data = R_data[mask]
    Y_data[:] = subject
    # # Instantiate the LabelEncoder
    # le = preprocessing.LabelEncoder()

    # Fit and transform Y_data to encode labels
    # Y_data = le.fit_transform(Y_data)
    return X_data, Y_data, R_data

def jitter(x, snr_db=25):
    if isinstance(snr_db, list):
        snr_db_low = snr_db[0]
        snr_db_up = snr_db[1]
    else:
        snr_db_low = snr_db
        snr_db_up = 45
    snr_db = np.random.randint(snr_db_low, snr_db_up, (1,))[0]
    snr = 10 ** (snr_db/10)
    Xp = np.sum(x**2, axis=0, keepdims=True) / x.shape[0]
    Np = Xp / snr
    n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)
    xn = x + n
    return xn
def to_categorical(y, nb_classes=None):
    """Convert a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to nb_classes).
        nb_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.

    Taken from:
    https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py
    v2.0.2 of Keras to remove unnecessary Keras dependency
    """
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def get_idxs(in_array, to_find):
    """Utility function for finding the positions of observations of one array in another an array.

    Args:
        in_array (array): Array in which to locate elements of to_find
        to_find (array): Array of elements to locate in in_array

    Returns:
        TYPE: Indices of all elements of to_find in in_array
    """
    targets = ([np.where(in_array == x) for x in to_find])
    return np.squeeze(np.concatenate(targets, axis=1))


