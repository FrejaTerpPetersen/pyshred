import mikeio
import numpy as np
import pandas as pd
from processdata import qr_place

def load_obs(num_sensors,times,ds=None,fldr = "Data/oresund/obs/"):
    stations = ["Drogden","Klagshamn","Barseback","Dragor","Flinten7","Helsingborg","Hornbaek",
            "Kobenhavn","MalmoHamn","Skanor","Vedbaek"]
    files = [f + "_wl.csv" for f in stations]
    locs = pd.read_csv(fldr + "stations.csv", delimiter=',', header=0,index_col=0)
    obs = []
    coords = []
    for i,f in enumerate(files[:num_sensors]):
        try:
            data = pd.read_csv(fldr + f, delimiter=',', header=0,index_col=0)
            data.index = pd.to_datetime(data.index)
            # use dsn.time to subset data

            data = data.loc[times[0]:times[-1]]
            # for the indices that are in dsn.time but not in data, set to NaN
            data = data.reindex(times)

    
            # Fill nan values with the last valid observation
            obs.append(data.ffill().values)

            # # Find the closest point in the ds geometry to the station location
            if ds is not None:
                closest_point = ds.sel(x=locs.iloc[i,0],y=locs.iloc[i,1]).geometry
                # Get the index of that point in the ds geometry
                coords.append(np.array([closest_point.x,closest_point.y]).reshape(1,-1))
        except:
            print(f"File {f} not found")

    return np.concatenate(obs,axis=1), coords

def load_data(args,suff = '_1y'):
    # load_X = load_data(args.dataset)
    if args.dataset.lower() == 'oresund':
        ds = mikeio.read("Data/Area_5m.dfsu", items=[args.item])
        load_X = ds[0].to_numpy()
        load_y = None

    elif args.dataset.lower() == 'oresund_forcing':
        # ds = mikeio.read("Data/Area_5m.dfsu",time=slice("2022-01-01", "2022-12-31"), items=[0])
        ds = mikeio.read(f"Data/Area{suff}.dfsu", items=[args.item])
        load_X = ds[0].to_numpy()
        dsn = mikeio.read(f"Data/oresund/BCn{suff}.dfs1", items=[args.item])
        dss = mikeio.read(f"Data/oresund/BCs{suff}.dfs1", items=[args.item])
        # Concatenate boundary data
        load_y = np.concatenate((dsn.to_numpy().squeeze(), dss.to_numpy().squeeze()[:,1:-1]), axis=1)
        
        

        if args.num_sensors>0:
            
            fldr = "Data/oresund/obs/"
            obs, coords = load_obs(args.num_sensors,times=dsn.time,ds=ds,fldr=fldr)


            # 
            # stations = ["Drogden","Klagshamn","Barseback","Dragor","Flinten7","Helsingborg","Hornbaek",
            #         "Kobenhavn","MalmoHamn","Skanor","Vedbaek"]
            # files = [f + "_wl.csv" for f in stations]
            # locs = pd.read_csv(fldr + "stations.csv", delimiter=',', header=0,index_col=0)
            # obs = []
            # coords = []
            # for i,f in enumerate(files[:args.num_sensors]):
            #     try:
            #         data = pd.read_csv(fldr + f, delimiter=',', header=0,index_col=0)
            #         data.index = pd.to_datetime(data.index)
            #         # use dsn.time to subset data

            #         data = data.loc[dsn.time[0]:dsn.time[-1]]
            #         # for the indices that are in dsn.time but not in data, set to NaN
            #         data = data.reindex(dsn.time)

         
            #         # Fill nan values with the last valid observation
            #         obs.append(data.ffill().values)

            #         # # Find the closest point in the ds geometry to the station location
            #         closest_point = ds.sel(x=locs.iloc[i,0],y=locs.iloc[i,1]).geometry
            #         # Get the index of that point in the ds geometry
            #         coords.append(np.array([closest_point.x,closest_point.y]).reshape(1,-1))


            #     except:
            #         print(f"File {f} not found")


            # Save indices of the sensors according to load_y
            sensor_locations = np.arange(0,args.num_sensors) + load_y.shape[1]
            np.save(fldr + "sensor_locations_load_y.npy", sensor_locations)
            args.sensor_location_file = fldr + "sensor_locations_load_y.npy"
            args.placement = 'file'


            load_y = np.concatenate((load_y,obs), axis=1)

            coordinates = np.concatenate(coords,axis=0)
            np.save(fldr + 'coordinates'+args.suffix+'.npy', coordinates)
            # args.sensor_location_file = fldr + "coordinates.npy"
            

            args.suffix = '_forcing_obs' + args.suffix
        else:
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


def pick_sensor_locations(args, num_sensors,X):

    m = X.shape[1]

    ### Set sensors randomly or according to QR
    if args.placement == 'QR':
        sensor_locations, U_r, Sigma = qr_place(X.T, num_sensors)
    elif args.placement == 'file':
        sensor_locations = np.load(args.sensor_location_file)
        if len(sensor_locations) != num_sensors:
            print("num_sensors changed to ", len(sensor_locations), "to match sensor location file")
            num_sensors = len(sensor_locations)
        if args.num_forcings == 0:
            _, U_r, Sigma = qr_place(X.T, num_sensors)
        else:
            U_r = None; Sigma = None
    elif args.placement == 'semirandom':
        locn = np.random.choice(np.arange(0,13), size=int(np.floor(num_sensors/2)), replace=False)
        locs = np.random.choice(np.arange(13,(13+27)), size=int(np.ceil(num_sensors/2)), replace=False)
        sensor_locations = np.concatenate((locn, locs), axis=0)
        print(f"Sampled {len(locn)} sensor locations on North boundary and {len(locs)} sensor locations on South boundary")
        _, U_r, Sigma = qr_place(X.T, num_sensors)
    else:
        _, U_r, Sigma = qr_place(X.T, num_sensors)
        sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

    return sensor_locations, U_r, Sigma


def pick_forcing_locations(args, num_forcings,num_sensors,X):

    locn = np.round(np.linspace(0,13,int(np.floor(num_forcings/2)) + 2)[1:-1]).astype(int)
    locs = np.round(np.linspace(13,(13+27),int(np.ceil(num_forcings/2)) + 2)[1:-1]).astype(int)
    forcing_locations = np.concatenate((locn, locs), axis=0)
    print(f"Picked {len(locn)} sensor locations on North boundary and {len(locs)} sensor locations on South boundary")
    _, U_r, Sigma = qr_place(X.T, num_forcings + num_sensors)

    return forcing_locations, U_r, Sigma