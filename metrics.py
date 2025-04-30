""" Metrics module
Author: frtp, 2024

"""

import numpy as np
import pandas as pd
import warnings
import mikeio
from typing import Optional, Union
import sklearn.metrics as skm

def nanaverage(A,axis=0,weights=None):
    if weights is None:
        weights = np.ones_like(A)
    sums = np.nansum(A*weights,axis=axis)
    not_na = ~pd.isna(A) if isinstance(A,np.ndarray) else not pd.isna(A)
    num_vals = ((not_na)*weights).sum(axis=axis)
    if len(A.shape)>1:
        all_nan = pd.isna(A).all(axis=axis)
        # To avoid division with zero, we sort out the points where all values (across the axis) are nan
        sums[~all_nan] = sums[~all_nan]/num_vals[~all_nan]
        averages = sums.copy()
        averages[all_nan] = np.nan # Set values to nan where all values are nan
    else:
        averages = sums/num_vals
    return averages

def rmse(y_true: mikeio.DataArray, y_pred: mikeio.DataArray, return_dim: Optional[int]=None, weights: Optional[np.ndarray]=None, suppress_weight_warnings: bool=False) -> Union[float,mikeio.DataArray]:
    """Weighted RMSE metric for measuring the difference between two data arrays.

    Parameters
    ----------
    y_true : mikeio.DataArray
        The true data array.
    y_pred : mikeio.DataArray
        The predicted data array.
    return_dim : Optional[int], optional
        The dimension to return the RMSE. Default is None. If None, the RMSE is calculated across all dimensions and returned as a float, by default None
    weights : Optional[np.ndarray], optional
        he weights for the RMSE calculation. Weights should be given by the element areas.
        If None is given, unit weights will be used.
        
    Returns
    -------
    Union[float,mikeio.DataArray]
        if return_dim == 0: 
            (mikeio DataArray) The RMSE value for each time step (across space)
        elif return_dim == 1: 
            (mikeio DataArray) The RMSE value for each spatial point (across time)
        else:
            float: The RMSE value.

    """
    
    if return_dim == 0:
        if weights is None:
            if not suppress_weight_warnings: warnings.warn("No weights provided. RMSE is calculated without a spatial weight.")
            return np.mean((y_true - y_pred)**2, axis = 1)**0.5
        else:
            y_out = y_pred.isel(element=0).copy()
            y_out.values = nanaverage((y_true.values - y_pred.values)**2,axis=1,weights=weights)**0.5
            return y_out
        
    elif return_dim == 1:
        errs = nanaverage(((y_true - y_pred).values)**2,axis=0,weights=None)**0.5
        err_da = y_true.isel(time=0).copy()
        if isinstance(errs,np.ndarray):
            err_da.values = np.array(errs,dtype='float64')
        else:
            err_da.values = errs
        return err_da
    
    elif return_dim is None:
        if weights is None:
            if not suppress_weight_warnings: warnings.warn("No weights provided. RMSE is calculated without a spatial weight.")
            return  np.sqrt(nanaverage(nanaverage((y_true.values - y_pred.values)**2,axis=0),weights=None))
        else:
            return np.sqrt(nanaverage(nanaverage((y_true.values - y_pred.values)**2,axis=0),weights=weights))
    else:
        raise ValueError("return_dim must be 0, 1 or None.")


def mae(y_true: mikeio.DataArray, y_pred: mikeio.DataArray, return_dim: Optional[int]=None, weights: Optional[np.ndarray]=None,suppress_weight_warnings: bool=False) -> Union[float,mikeio.DataArray]:
    """Weighted mean absolute error metric for measuring the difference between two data arrays.
    
    Parameters:
    y_true (mikeio DataArray): The true data array.
    y_pred (mikeio DataArray): The predicted data array.
    weights (np.array): The weights for the MAE calculation. Weights should be given by the element areas.
                        Default is None. If default, the DataArrays only have a time dimension, and no spatial components.
    return_dims (int): The dimension to return the MAE. Default is None. If None, the MAE is calculated across all dimensions and returned as a float. 

    Returns:
    if return_dim == 0: 
        (mikeio DataArray) The MAE value for each time step (across space)
    elif return_dim == 1: 
        (mikeio DataArray) The MAE value for each spatial point (across time)
    else:
        float: The MAE value.
    """
    if (y_true.values == np.nan).any():
        warnings.warn("NaN values are present in the true data array. They are not handled, so output might be nan")
    if (y_pred.values == np.nan).any():
        warnings.warn("NaN values are present in the pred data array. They are not handled, so output might be nan")
    
    if return_dim == 0:
        if weights is None:
            if not suppress_weight_warnings: warnings.warn("No weights provided. MAE is calculated without a spatial weight.")
            return np.mean(abs(y_true - y_pred), axis = 1)
        else:
            y_out = y_pred.isel(element=0).copy()
            y_out.values = nanaverage(abs(y_true.values - y_pred.values),axis=1,weights=weights)
            return y_out
        
    elif return_dim == 1:
        return np.mean(abs(y_true - y_pred), axis = 0)
    
    elif return_dim is None:
        if weights is None:
            if not suppress_weight_warnings: warnings.warn("No weights provided. MAE is calculated without a spatial weight.")
            return nanaverage(nanaverage(abs(y_true.values - y_pred.values)))
        else:
            return nanaverage(nanaverage(abs(y_true.values - y_pred.values),axis=0),weights=weights)
    else:
        raise ValueError("return_dim must be 0, 1 or None.")
    
def mape(y_true: mikeio.DataArray, y_pred: mikeio.DataArray, return_dim: Optional[int]=None) -> Union[float,mikeio.DataArray]:
    """Mean absolute percentage error.
    
    Parameters:
    y_true (mikeio DataArray): The true data array.
    y_pred (mikeio DataArray): The predicted data array.
    weights (np.array): The weights for the MAE calculation. Weights should be given by the element areas.
                        Default is None. If default, the DataArrays only have a time dimension, and no spatial components.
    return_dims (int): The dimension to return the MAE. Default is None. If None, the MAE is calculated across all dimensions and returned as a float. 

    Returns:
    if return_dim == 0: 
        (mikeio DataArray) The MAE value for each time step (across space)
    elif return_dim == 1: 
        (mikeio DataArray) The MAE value for each spatial point (across time)
    else:
        float: The MAE value.
    """
    if (y_true.values == np.nan).any():
        warnings.warn("NaN values are present in the true data array. They are not handled, so output might be nan")
    if (y_pred.values == np.nan).any():
        warnings.warn("NaN values are present in the pred data array. They are not handled, so output might be nan")
    
    if return_dim == 0:
        return np.mean(abs((y_true - y_pred)/y_true), axis = 1) * 100
   
        
    elif return_dim == 1:
        return np.mean(abs((y_true - y_pred)/y_true), axis = 0) * 100
    
    elif return_dim is None:
        return np.mean(np.mean(abs((y_true.values - y_pred.values )/y_true.values )))* 100
    else:
        raise ValueError("return_dim must be 0, 1 or None.")

def maxAbsError(y_true: mikeio.DataArray, y_pred: mikeio.DataArray, return_dim: Optional[int]=None) -> Union[float,mikeio.DataArray]:
    """Maximum absolute error metric for measuring the difference between two data arrays.
    
    Parameters:
    y_true (mikeio DataArray): The true data array.
    y_pred (mikeio DataArray): The predicted data array.
    return_dims (int): The dimension to return the maxAbsError. Default is None. If None, the maxAbsError is calculated across all dimensions and returned as a float. 

    Returns:
    if return_dim == 0: 
        (mikeio DataArray) The maxAbsError value for each time step (across space)
    elif return_dim == 1: 
        (mikeio DataArray) The maxAbsError value for each spatial point (across time)
    else:
        float: The maxAbsError value.
    """
    
    if return_dim == 0:
        return np.max(abs(y_true - y_pred), axis = 1)
        
    elif return_dim == 1:
        return np.max(abs(y_true - y_pred), axis = 0)
    
    elif return_dim is None:
        return np.nanmax(np.nanmax(abs(y_true.values - y_pred.values),axis=0))
    else:
        raise ValueError("return_dim must be 0, 1 or None.")
    


def top_vals_unweighted_rmse(da_true: mikeio.DataArray,da_pred: mikeio.DataArray,q: float=0.98) -> float:
    """Finds the observations where the absolute value of the true data array is in the top (1-q)*100 %. 
    Calculates the RMSE of the predicted data array at these observations.

    Parameters
    ----------
    da_true : mikeio.DataArray
        The true values
    da_pred : mikeio.DataArray
        The predicted values
    q : float, optional
        The quantile defining the top (1-q)*100 % observations, by default 0.98

    Returns
    -------
    float
        The RMSE of the predicted data array at the top (1-q)*100 % observations.
    """

    top_q = np.quantile(abs(da_true.values[~pd.isna(da_true.values)]),q=q)
    top_obs = np.where(abs(da_true.values)>top_q)
    top_pred  = da_pred.values[top_obs]
    top_true  = da_true.values[top_obs]

    top_rmse=np.sqrt(nanaverage(nanaverage((top_pred-top_true)**2)))
    return top_rmse

def table_error_metrics(config,das_true,das_pred,weights,q=0.98,suppress_weight_warnings: bool=False):
    
    state_names = config["state_names"]
    err_units = config["err_units"]
    err_fac = config["err_factor"]
    idxs = [sn + " [" + eu + "]" for sn,eu in zip(state_names,err_units)]
    df = pd.DataFrame(columns=["RMSE", "State top 2% RMSE","MeanAbsError", "MaxAbsError", "R2"],index=idxs)
    for i in range(len(state_names)):
        da_pred = das_pred[i]
        da_true = das_true[i].sel(time=da_pred.time)

        df.loc[idxs[i],"RMSE"] =  rmse(y_pred=da_pred,y_true=da_true,weights=weights,suppress_weight_warnings=suppress_weight_warnings)*err_fac
        df.loc[idxs[i],"MeanAbsError"] =  mae(y_pred=da_pred,y_true=da_true,weights=weights,suppress_weight_warnings=suppress_weight_warnings)*err_fac
        # df.loc[idxs[i],"MAPE"] =  mape(y_pred=da_pred,y_true=da_true)
        # df.loc[idxs[i],"pMAE"] =  100*mae(y_pred=da_pred,y_true=da_true,weights=weights,suppress_weight_warnings=suppress_weight_warnings) / np.mean(np.mean(abs(da_true.values)))
        df.loc[idxs[i],"MaxAbsError"] =  maxAbsError(y_pred=da_pred,y_true=da_true)*err_fac
        df.loc[idxs[i],"State top 2% RMSE"] =  top_vals_unweighted_rmse(da_pred=da_pred,da_true=da_true,q=q)*err_fac
        non_nan_mask = ~pd.isna(da_pred.values)
        df.loc[idxs[i],"R2"] =  skm.r2_score(y_true=da_true.values[non_nan_mask],y_pred=da_pred.values[non_nan_mask])
    return df

def table_result_metrics(config,das_pred,q=0.98):
    state_names = config["state_names"]
    err_units = config["err_units"]
    err_fac = config["err_factor"]
    idxs = [sn + " [" + eu + "]" for sn,eu in zip(state_names,err_units)]
    df = pd.DataFrame(columns=["Mean abs value", "State top 2% abs value","MaxAbsValue"],index=idxs)
    for i in range(len(state_names)):
        da_pred = das_pred[i]

        df.loc[idxs[i],"Mean abs value"] =  nanaverage(nanaverage(abs(da_pred.values)))*err_fac
        df.loc[idxs[i],"MaxAbsValue"] =  np.nanmax(np.abs(da_pred.values))*err_fac
        df.loc[idxs[i],"State top 2% abs value"] =  np.quantile(abs(da_pred.values),q=q)*err_fac

    return df