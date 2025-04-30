import mikeio
import numpy as np
from processdata import qr_place


def load_data(args):
    # load_X = load_data(args.dataset)
    if args.dataset.lower() == 'oresund':
        ds = mikeio.read("Data/Area_5m.dfsu", items=[args.item])
        load_X = ds[0].to_numpy()
        load_y = None

    elif args.dataset.lower() == 'oresund_forcing':
        # ds = mikeio.read("Data/Area_5m.dfsu",time=slice("2022-01-01", "2022-12-31"), items=[0])
        ds = mikeio.read("Data/Area_1y.dfsu", items=[args.item])
        load_X = ds[0].to_numpy()
        dsn = mikeio.read("Data/oresund/BCn_1y.dfs1", items=[args.item])
        dss = mikeio.read("Data/oresund/BCs_1y.dfs1", items=[args.item])
        # Concatenate boundary data
        load_y = np.concatenate((dsn.to_numpy().squeeze(), dss.to_numpy().squeeze()[:,1:-1]), axis=1)

        args.placement = "distributed"
        args.suffix = '_forcing' + args.suffix

    elif args.dataset.lower() == 'cylinder':
        load_X = np.loadtxt("Data/cylinder/VORTALL.csv", delimiter=',').T
        load_y = None

    return load_X, load_y


def train_val_test_split(n,lags,val_length):
    train_indices = np.arange(0, int(n*0.85))
    valid_indices = np.arange(int(n*0.85), int(n*0.85) + val_length)
    test_indices = np.arange(int(n*0.85) + val_length, n - lags)
    return train_indices, valid_indices, test_indices


def pick_sensor_locations(args, num_sensors):

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
        locn = np.round(np.linspace(0,13,int(np.floor(num_sensors/2)) + 2)[1:-1]).astype(int)
        locs = np.round(np.linspace(13,(13+27),int(np.ceil(num_sensors/2)) + 2)[1:-1]).astype(int)
        sensor_locations = np.concatenate((locn, locs), axis=0)
        print(f"Picked {len(locn)} sensor locations on North boundary and {len(locs)} sensor locations on South boundary")
        _, U_r, Sigma = qr_place(load_X[train_indices].T, num_sensors)
    else:
        _, U_r, Sigma = qr_place(load_X[train_indices].T, num_sensors)
        sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

    return sensor_locations, U_r, Sigma