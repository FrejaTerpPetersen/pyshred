import argparse
import torch
import numpy as np
import models
from processdata import TimeSeriesDataset
from processdata import qr_place

from sklearn.preprocessing import MinMaxScaler
import os
import mikeio

parser = argparse.ArgumentParser(description='In sample reconstructing with SHRED')

parser.add_argument('--dataset', type=str, default='SST', help='Dataset for reconstruction/forecasting. Choose between "cylinder" or "oresund" or "oresund_forcing"')

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


args = parser.parse_args()
lags = args.lags
num_sensors = args.num_sensors

# load_X = load_data(args.dataset)
if args.dataset.lower() == 'oresund':
    ds = mikeio.read("Data/Area_5m.dfsu",time=slice("2022-01-01", "2022-12-31"), items=[0])
    load_X = ds[0].to_numpy()
if args.dataset.lower() == 'oresund_forcing':
    ds = mikeio.read("Data/Area_5m.dfsu",time=slice("2022-01-01", "2022-12-31"), items=[0])
    load_X = ds[0].to_numpy()
    dsn = mikeio.read("Data/oresund/BCn.dfs1")
    dss = mikeio.read("Data/oresund/BCs.dfs1")
    # Concatenate boundary data
    load_y = np.concatenate((dsn.to_numpy().squeeze(), dss.to_numpy().squeeze()[:,1:-1]), axis=1)

    args.placement = "distributed"
    args.suffix = '_forcing' + args.suffix

elif args.dataset.lower() == 'cylinder':
    load_X = np.loadtxt("Data/cylinder/VORTALL.csv", delimiter=',').T

n = load_X.shape[0]
m = load_X.shape[1]

# FRTP:
# n_test = 174
# n_valid = 174
# n_train = 1000
# train_indices = np.arange(0, n_train)
# valid_indices = np.arange(n_train,n_train+n_valid)
# test_indices = np.arange(n_train+n_valid,n_train+n_valid+n_test)
train_indices = np.arange(0, int(n*0.85))
valid_indices = np.arange(int(n*0.85), int(n*0.85) + args.val_length)
test_indices = np.arange(int(n*0.85) + args.val_length, n - lags)

print("Train set size:", len(train_indices))
print("Val set size:", len(valid_indices))
print("Test set size:", len(test_indices),'\n')

### Set sensors randomly or according to QR
if args.placement == 'QR':
    sensor_locations, U_r, Sigma = qr_place(load_X[train_indices].T, num_sensors)
elif args.placement == 'file':
    sensor_locations = np.load(args.sensor_location_file)
    if len(sensor_locations) != num_sensors:
        print("num_sensors changed to ", len(sensor_locations), "to match sensor location file")
        num_sensors = len(sensor_locations)
    _, U_r, Sigma = qr_place(load_X[train_indices].T, num_sensors)
elif args.placement == 'semirandom':
    locn = np.random.choice(np.arange(0,13), size=int(np.floor(num_sensors/2)), replace=False)
    locs = np.random.choice(np.arange(13,(13+27)), size=int(np.ceil(num_sensors/2)), replace=False)
    sensor_locations = np.concatenate((locn, locs), axis=0)
    print(f"Sampled {len(locn)} sensor locations on North boundary and {len(locs)} sensor locations on South boundary")
    _, U_r, Sigma = qr_place(load_X[train_indices].T, num_sensors)
elif args.placement == 'distributed':
    locn = np.round(np.linspace(0,13,int(np.floor(num_sensors/2)) + 2)[1:-1])
    locs = np.round(np.linspace(13,(13+27),int(np.ceil(num_sensors/2)) + 2)[1:-1])
    sensor_locations = np.concatenate((locn, locs), axis=0)
    print(f"Sampled {len(locn)} sensor locations on North boundary and {len(locs)} sensor locations on South boundary")
    _, U_r, Sigma = qr_place(load_X[train_indices].T, num_sensors)
else:
    _, U_r, Sigma = qr_place(load_X[train_indices].T, num_sensors)
    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

### Fit min max scaler to training data, and then scale all data
sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)

if args.dataset.lower() == 'oresund_forcing':
    # Transform the forcing data
    sc_y = MinMaxScaler()
    sc_y = sc_y.fit(load_y[train_indices])
    transformed_y = sc_y.transform(load_y)

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


### Train SHRED network for reconstruction
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.0).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=args.epochs, lr=1e-3, verbose=True, patience=5)

### Train SDN network for reconstruction
# sdn = models.SDN(num_sensors, m, l1=350, l2=400, dropout=0.0).to(device)
# validation_errors_sdn = models.fit(sdn, train_dataset_sdn, valid_dataset_sdn, batch_size=64, num_epochs=args.epochs, lr=1e-3, verbose=True, patience=5)

### Generate reconstructions from SHRED and SDN
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
# test_recons_sdn = sc.inverse_transform(sdn(test_dataset_sdn.X).detach().cpu().numpy())

test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())

### Generate reconstructions from QR/POD
qrpod_sensors = load_X[test_indices][:, sensor_locations]
C = np.zeros((num_sensors, m))
for i in range(num_sensors):
    C[i, sensor_locations[i]] = 1

qrpod_recons = (U_r @ np.linalg.inv(C @ U_r) @ qrpod_sensors.T).T

### Plot and save error
if not os.path.exists('ReconstructingResults/' + args.dest):
    os.makedirs('ReconstructingResults/' + args.dest)
np.save('ReconstructingResults/' + args.dest + '/reconstructions'+args.suffix+'.npy', test_recons)
# np.save('ReconstructingResults/' + args.dest + '/sdnreconstructions.npy', test_recons_sdn)
np.save('ReconstructingResults/' + args.dest + '/qrpodreconstructions'+args.suffix+'.npy', qrpod_recons)
np.save('ReconstructingResults/' + args.dest + '/truth.npy', test_ground_truth)
np.save('ReconstructingResults/' + args.dest + '/sensor_locations'+args.suffix+'.npy', sensor_locations)
np.save('ReconstructingResults/' + args.dest + '/singularvals'+args.suffix+'.npy', Sigma)

# Save model weights
if not os.path.exists('models/' + args.dest):
    os.makedirs('models/' + args.dest)
shred.save_weights('models/' + args.dest + '/shred_reconstruction'+args.suffix+'.pt')
