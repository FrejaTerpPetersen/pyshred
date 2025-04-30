import argparse
import torch
import numpy as np
import models
from processdata import TimeSeriesDataset
import optuna
import pickle

from sklearn.preprocessing import MinMaxScaler
import os
from help_funcs import load_data, train_val_test_split, pick_sensor_locations

parser = argparse.ArgumentParser(description='In sample reconstructing with SHRED')

parser.add_argument('--dataset', type=str, default='oresund', help='Dataset for reconstruction/forecasting. Choose between "cylinder" or "oresund" or "oresund_forcing"')

parser.add_argument('--item', type=int, default=0, help='Which item to use in the dfsu file. 0 for surface elevation, 1,2 for velocity components')

parser.add_argument('--num_sensors', type=int, default=10, help='Number of sensors to use')

parser.add_argument('--placement', type=str, default='file', help='Placement of sensors (random, QR, or file)')

parser.add_argument('--sensor_location_file', type=str, default='Data/sensor_locations.npy', help='.npy file with sensor locations')

parser.add_argument('--epochs', type=int, default=1000, help='Maximum number of epochs')

parser.add_argument('--val_length', type=int, default=20, help='Length of validation set (Training set of 0.85*N, test set remainder)')

parser.add_argument('--lags', type=int, default=52, help='Length of sensor trajectories used')

parser.add_argument('--dest', type=str, default='', help='Destination folder')

parser.add_argument('--suffix', type=str, default='', help='Suffix for the output files')

# python ./reconstructions.py --dataset 'cylinder' --num_sensors 10 --placement 'qr' --dest 'cylinder' --val_length 5 --lags 10 --suffix '_sensor10lag10'
# python ./reconstructions.py --dataset 'oresund_forcing' --num_sensors 6 --dest 'oresund_forcing' --val_length 20 --lags 52 --suffix '_sensor6lag52'
# python ./reconstructions.py --dataset 'oresund_forcing' --num_sensors 6 --dest 'oresund_forcing' --val_length 500 --lags 52 --suffix '_sensor6lag52_1y'

# U and V components
# python ./reconstructions.py --dataset 'oresund_forcing' --item 1 --num_sensors 6 --dest 'oresund_forcing_U' --val_length 500 --lags 52 --suffix '_sensor6lag52_1y'
# python ./reconstructions.py --dataset 'oresund_forcing' --item 2 --num_sensors 6 --dest 'oresund_forcing_V' --val_length 500 --lags 52 --suffix '_sensor6lag52_1y'


args = parser.parse_args()
lags = args.lags
num_sensors = args.num_sensors


load_X, load_y = load_data(args)

n = load_X.shape[0]
m = load_X.shape[1]

train_indices, valid_indices, test_indices = train_val_test_split(n,lags,args.val_length)

print("Train set size:", len(train_indices))
print("Val set size:", len(valid_indices))
print("Test set size:", len(test_indices),'\n')

# Pick sensor locations according to args.placement
sensor_locations, U_r, Sigma = pick_sensor_locations(args, num_sensors)

### Fit min max scaler to training data, and then scale all data
sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])

transformed_X = sc.transform(load_X)

# Save scaler for later
if not os.path.exists('ReconstructingResults/' + args.dest):
    os.makedirs('ReconstructingResults/' + args.dest)
scalerfile = 'ReconstructingResults/' + args.dest + '/scaler' + args.suffix + '.sav'
pickle.dump(sc, open(scalerfile, 'wb'))

if args.dataset.lower() == 'oresund_forcing':
    # Transform the forcing data
    sc_y = MinMaxScaler()
    sc_y = sc_y.fit(load_y[train_indices])
    transformed_y = sc_y.transform(load_y)

    # Save scaler for later
    scalerfile = 'ReconstructingResults/' + args.dest + '/scaler_y' + args.suffix + '.sav'
    pickle.dump(sc_y, open(scalerfile, 'wb'))

### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))

for i in range(len(all_data_in)):
    if args.dataset.lower() == 'oresund_forcing':
        # Use the forcing data as sensor input
        all_data_in[i] = transformed_y[i:i+lags, sensor_locations]
    else:
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using cuda: ",torch.cuda.is_available(),'\n')

train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

train_dataset_sdn = TimeSeriesDataset(train_data_in[:,-1,:], train_data_out)
valid_dataset_sdn = TimeSeriesDataset(valid_data_in[:,-1,:], valid_data_out)
test_dataset_sdn = TimeSeriesDataset(test_data_in[:,-1,:], test_data_out)

# Set up objective function for hyperparameter optimization
def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 1,512)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 5)
    l1 = trial.suggest_int('l1', hidden_size, 512)
    l2 = trial.suggest_int('l2', l1, 512)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)


    ### Initialize and train SHRED network for reconstruction
    shred = models.SHRED(num_sensors, m, hidden_size=hidden_size, hidden_layers=hidden_layers, l1=l1, l2=l2, dropout=dropout).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=batch_size, num_epochs=args.epochs, lr=lr, verbose=True, patience=5)
    return validation_errors[-1]

# Run optuna optimization
n_trials = 100
study = optuna.create_study(direction="minimize")
study.optimize(lambda trial: objective(trial), n_trials=n_trials)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
best_trial = study.best_trial
print("  Value: ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))