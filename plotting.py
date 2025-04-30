""" Plotting module
Author: frtp, 2024

"""


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import numpy as np
# import fmrom.rom as rom
import metrics as ms
# import fmrom.common as cmn
import mikeio
import os
import modelskill
import pandas as pd
# import contextily as cx
from mikeio.spatial import crs
# import statsmodels.graphics.tsaplots as tsaplt
import ast
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


FONT_SIZE = 20
mpl.style.use('default')

## Plots:
#        - reconstruction error dfs file
#        - whole timeseries in one point with training/test periods marked. (Klagshamn)
#        - contourf map plot with reconstruction error in one specific time step
#        - contourf map plot with RMSE in one specific time step
#        - ACF and pACF of state PCs.
#        - Plot of principal components. 


############################## Preprocess data ################################################

def plot_percent_nan_over_time(ds,item):
    plt.rcParams['font.size'] = FONT_SIZE
    nans = np.sum(np.isnan(ds[item].values),axis=1)
    frac_nan_time = nans/ds[item].values.shape[1]
    plt.plot(frac_nan_time*100)
    plt.title("Percent NaNs over time")
    plt.ylabel("NaNs (%)")
    plt.xlabel("Time")

def plot_points_with_nan(ds,item):
    plt.rcParams['font.size'] = FONT_SIZE
    nans_time = np.sum(np.isnan(ds[item].values),axis=1)
    idx_max_nan = nans_time.argmax() # Timestep with the most NaNs
    time_with_maxnan = ds[item].isel(time=idx_max_nan)
    nan_points = np.where(np.isnan(time_with_maxnan.values))[0]

    fig,axs = plt.subplots(1,1)
    time_with_maxnan.plot(ax=axs)
    for i in range(len(nan_points)):
        spatial_point_with_nan = time_with_maxnan.isel(element=nan_points[i]).geometry
        axs.plot(spatial_point_with_nan.x,spatial_point_with_nan.y,'ro')
    axs.set_title("Time step with the most NaNs: \n"+str(time_with_maxnan.time.item()))




#################################################################################################
#################################################################################################
############################## EDA: Explorative data analysis ###################################
#################################################################################################
#################################################################################################

def make_animation(das,da_idx,fig_path,cmap_surfEle='RdBu_r'):
    plt.rcParams['font.size'] = FONT_SIZE

    da = das[da_idx]
    minval = np.min(da.values)
    maxval = np.max(da.values)
    maxabs = max(abs(minval),abs(maxval))

    fig, ax = plt.subplots()
    

    ts = da.time[::2]
    def animate(t):
        ax.clear()
        ax.axis('off')
        da[ts[t]].plot(ax=ax,vmin=-maxabs,vmax=maxabs,cmap=cmap_surfEle,add_colorbar=False)
        ax.set_title((da[ts[t]].time.strftime('%Y-%m-%d %H:%M:%S')[0])[:10])

    animation = FuncAnimation(fig, animate, frames=24*30, interval=2)
    animation  # Assign the animation to a variable

    animation.save(fig_path+f"{da.name}_animation.gif", writer="pillow")

def make_EDA_plots(das,fig_path,config=None,show_fig=True,save_fig=False):
    plt.rcParams['font.size'] = FONT_SIZE-10
    pos = config["plot_point_position"]
    pos_name = config["plot_point_name"]
    case_str = config["case_str"]
    all_slice = config["all_times"]
    split_time = config["split_time"]
    save_animation = config["save_animation"]
    err_fac = config["err_factor"]
    err_units = config["err_units"]

    # Plot of domain
    plt.figure();
    das[0][20].plot();
    # ax = plt.gca()
    # ax.axis('off')
    if save_fig: plt.savefig(fig_path+"EDA_domain.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
    if not show_fig: plt.close()
    # Plot of mesh
    plt.figure();
    das[0].plot.mesh(title="");
    if save_fig: plt.savefig(fig_path+"EDA_mesh.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
    if not show_fig: plt.close();
    # Animation of item 0
    if save_animation: 
        make_animation(das,das_idx=0,fig_path=fig_path)
        if not show_fig: plt.close();
    # Time plot of single point
    
    plt.figure();
    da_point = das[0].sel(x=pos[0],y=pos[1],time=all_slice)*err_fac
    ax = da_point.plot(title=f"Point {pos_name}",figsize=(14,6));
    ax.set_ylabel(f"{das[0].name} [{err_units[0]}]");
    ax.set_xlabel("Time");
    ax.axvspan(split_time, das[0].time[-1], facecolor='0.2', alpha=0.2);
    ax.margins(x=0);

    if save_fig: plt.savefig(fig_path+f"EDA_single_point_{case_str}.png", bbox_inches='tight', pad_inches=0.02, dpi=300);
    if show_fig: plt.show();
    else: plt.close();

    if config["use_mda"] or config["separate_train_times"]:
        train_times_all = all_slice
        train_times_mda = config["train_times"]

        fig,ax = plt.subplots(1,1,figsize=(18,6));
        da_point = das[0].sel(x=pos[0],y=pos[1],time=train_times_all)
        da_point_mda = das[0].sel(x=pos[0],y=pos[1],time=train_times_mda)

        da_point.plot(title="Training period",ax=ax);
        ax.plot(da_point_mda.time,da_point_mda.values,'ro',markersize=2,label = "Selected by MDA");
        ax.set_ylabel(f"{das[0].name} [m]");
        ax.set_xlabel("Time");
        ax.margins(x=0);
        plt.legend(loc="upper right");
        if save_fig and config["use_mda"]: plt.savefig(fig_path+f"MDA_selected_points_{case_str}.png", bbox_inches='tight', pad_inches=0.02, dpi=300);
        elif save_fig and config["separate_train_times"]: plt.savefig(fig_path+f"training_points_{case_str}.png", bbox_inches='tight', pad_inches=0.02, dpi=300);
        if show_fig: plt.show();
        else: plt.close();


# def make_geometry_plot(da,ax)

def visualize_geometry_on_map(das,fig_path,config,show_fig=True,save_fig=False):
    plt.rcParams['font.size'] = FONT_SIZE
    case_str = config["case_str"]
    plot_str = config["plot_str"]
    cb_location = config["cb_location"]
    fig_name = f"geometry_on_map_{case_str}.png"
    file = fig_path+fig_name
    if os.path.exists(file): 
        print(f"Figure '{fig_name}' has previously been generated. The figure will not be updated.")
        return

    da = das[0]
    xmin = da.geometry.node_coordinates[:,0].min()
    xmax = da.geometry.node_coordinates[:,0].max()
    ymin = da.geometry.node_coordinates[:,1].min()
    ymax = da.geometry.node_coordinates[:,1].max()

    ax = da.geometry.plot.contourf(figsize=(8,8), cmap="Blues_r", add_colorbar=False)
    ax.set_xlim([xmin-(xmax-xmin), xmax+(xmax-xmin)])
    ax.set_ylim([ymin-(ymax-ymin), ymax+(ymax-ymin)])
    cx.add_basemap(ax, crs=crs.CRS(da.geometry.projection).to_pyproj(),
                source=cx.providers.CartoDB.Positron,
                zoom=9)
    ax.set_title(plot_str)
    ax.set_aspect(1.0/np.cos(0.5*(ymin+ymax)*np.pi/180))

    if save_fig: plt.savefig(fig_path+f"geometry_on_map_{case_str}.png", bbox_inches='tight', pad_inches=0.02, dpi=300);
    if show_fig: plt.show();
    else: plt.close();

    min_depth = min(da.geometry.element_coordinates[:,2])
    levels = np.linspace(np.ceil(min_depth/10)*10,0,11)

    fig,ax = plt.subplots(1,1,figsize=(8,8))
    pcf = da.geometry.plot.contourf(ax=ax, levels = levels,cmap="Blues_r", add_colorbar=False)
    ax.set_xlim([xmin-0.1, xmax+0.1])
    ax.set_ylim([ymin-0.05, ymax+0.05])
    ax.set_title(plot_str)
    cx.add_basemap(ax, crs=crs.CRS(da.geometry.projection).to_pyproj(),
                source=cx.providers.CartoDB.Positron,
                zoom=9)
    ax.set_aspect(1.0/np.cos(0.5*(ymin+ymax)*np.pi/180))
    axins = inset_axes(ax,
                width="5%",  
                height="30%",
                loc=cb_location,
                # borderpad=4.5
                bbox_to_anchor=(-0.2, 0,0.98,1),
                bbox_transform=ax.transAxes
                )
    fig.colorbar(ax.collections[0], cax=axins)
    if config["obs_positions"] is not None:
        for o in config["obs_positions"]:
            ax.plot(o[0],o[1],'ko',markersize=15)
    # fig.colorbar(pcf.collections[0], pad=0.01, shrink=0.5)
    if save_fig: plt.savefig(fig_path+f"geometry_on_map_zoom_{case_str}.png", bbox_inches='tight', pad_inches=0.02, dpi=300);
    if show_fig: plt.show();
    else: plt.close();

def visualize_first_N_components_spatial(da,N=10,fig_path="",save_fig=False,show_fig=True,cmap_surfEle='RdBu_r'):
    plt.rcParams['font.size'] = FONT_SIZE
    fig_name = f"{da.name}_PC_visualizations_PC{N}.png"
    file = fig_path+fig_name
    if os.path.exists(file): 
        print(f"Figure '{fig_name}' has previously been generated. The figure will not be updated.")
        return
    stateT = rom.StateTransformer(N).fit(da)
    da_PC = da[0].copy() # Copy data array to have correct geometry
    da_PC_list = []
    for i in range(N):
        da_PC.values[~np.isnan(da_PC.values)] = stateT.model[1].components_[i,:]
        da_PC.name = "PC"+str(i+1)
        da_PC_list.append(da_PC.copy())

    ds_PC = mikeio.Dataset(da_PC_list)

    vmin = np.min(ds_PC.min().to_numpy())
    vmax = np.max(ds_PC.max().to_numpy())
    # fig, axs = plt.subplots(2, n_comp_S)
    fig, axs = plt.subplots(2, int(N/2),figsize=(14,7));

    for i in range(N):
        row = i // (N // 2)
        col = i % (N // 2)
        # vmin = ds_PC[i].values.min()
        # vmax = ds_PC[i].values.max()
        # largest_abs = np.max((abs(vmin),abs(vmax)))
        # ds_PC[i].plot(ax=axs[row, col], add_colorbar=False, vmin = -largest_abs, vmax = largest_abs, cmap=cmap_surfEle);
        ds_PC[i].plot(ax=axs[row, col], add_colorbar=False, cmap=cmap_surfEle);
        axs[row, col].axis('off');
        axs[row, col].set_title("PC" + str(i+1));

    plt.subplots_adjust(left=0.01,
                        bottom=0.1, 
                        right=0.99, 
                        top=0.9, 
                        wspace=-0.8, 
                        hspace=0.15);
    if save_fig: plt.savefig(file, bbox_inches='tight', pad_inches=0.02, dpi=300);
    if show_fig: plt.show();
    else: plt.close();

def visualize_first_N_components_time(da,N=10,fig_path="",save_fig=False,show_fig=True):
    plt.rcParams['font.size'] = FONT_SIZE
    fig_name = f"{da.name}_PC_projections_time_PC{N}.png"
    fig_name_ACF = f"{da.name}_PC_projections_time_ACF_PC{N}.png"
    fig_name_dACF = f"{da.name}_PC_projections_time_dACF_PC{N}.png"
    fig_name_ddACF = f"{da.name}_PC_projections_time_ddACF_PC{N}.png"

    if os.path.exists(fig_path+fig_name_ddACF): 
        print(f"Figure '{fig_path+fig_name_ddACF}' has previously been generated. The figure will not be updated.")
    da_t = rom.StateTransformer(N).fit_transform(da)


    fig, axs = plt.subplots(2, int(N/2),figsize=(14,7));
    for i in range(N):
        row = i // (N // 2)
        col = i % (N // 2)
        # ds_PC[i].plot(ax=axs[row, col], add_colorbar=False, vmin=vmin, vmax=vmax, cmap=colormap)
        da_t.iloc[:,i].plot(ax=axs[row, col])
        axs[row, col].set_title("PC" + str(i+1));

    if save_fig: plt.savefig(fig_path+fig_name, bbox_inches='tight', pad_inches=0.02, dpi=300);
    if show_fig: plt.show();
    else: plt.close();

    fig, axs = plt.subplots(2, int(N/2),figsize=(14,7));
    for i in range(N):
        row = i // (N // 2)
        col = i % (N // 2)
        # ds_PC[i].plot(ax=axs[row, col], add_colorbar=False, vmin=vmin, vmax=vmax, cmap=colormap)
        tsaplt.plot_acf(da_t.iloc[:,i], ax=axs[row, col],lags=10)
        # da_t.iloc[:,i].plot(ax=axs[row, col]);
        axs[row, col].set_title("acf PC" + str(i+1));

    if save_fig: plt.savefig(fig_path+fig_name_ACF, bbox_inches='tight', pad_inches=0.02, dpi=300);
    if show_fig: plt.show();
    else: plt.close();

def visualize_train_test_comps(da_train,da_test,name,fig_path="",save_fig=False,show_fig=True):
    plt.rcParams['font.size'] = FONT_SIZE

    if not os.path.isdir(fig_path+"/scatters"):
        os.mkdir(fig_path+"/scatters")

    da_train_t = rom.StateTransformer(2).fit_transform(da_train)
    da_test_t = rom.StateTransformer(2).fit_transform(da_test)

    fig, axs = plt.subplots(1,1,figsize=(10,7));
    plt.scatter(da_train_t.iloc[:,0],da_train_t.iloc[:,1],label="Training data",s=5);
    plt.scatter(da_test_t.iloc[:,0],da_test_t.iloc[:,1],label="Test data",s=5);
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(name)
    if save_fig: plt.savefig(fig_path+"/scatters/"+"PC1vsPC2_scatter_"+name+".png", bbox_inches='tight', pad_inches=0.02, dpi=300);
    if show_fig: plt.show();
    else: plt.close();


def pair_plot(da1,da2,name,fig_path="",save_fig=False,show_fig=True):
    plt.rcParams['font.size'] = FONT_SIZE

    if not os.path.isdir(fig_path+"/scatters"):
        os.mkdir(fig_path+"/scatters")

    da1_t = rom.StateTransformer(10).fit_transform(da1).add_prefix('S')
    da2_t = rom.StateTransformer(2).fit_transform(da2).add_prefix('F')

    fig, axs = plt.subplots(1,1,figsize=(10,7));
    plt.scatter(da_train_t.iloc[:,0],da_train_t.iloc[:,1],label="Training data",s=5);
    plt.scatter(da_test_t.iloc[:,0],da_test_t.iloc[:,1],label="Test data",s=5);
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(name)
    if save_fig: plt.savefig(fig_path+"/scatters/"+"PC1vsPC2_scatter_"+name+".png", bbox_inches='tight', pad_inches=0.02, dpi=300);
    if show_fig: plt.show();
    else: plt.close();


def plot_explained_variance_and_recon_error(das,config=None,fig_path="",save_fig=False,show_fig=True):
    plt.rcParams['font.size'] = FONT_SIZE-7
    for i,da in enumerate(das):
        # da = das[0]
        if config is None:
            err_fac = 1.0
            err_unit = ""
        else:
            err_fac = config["err_factor"]
            err_unit = config["err_units"][i]
        
        fig_name = f"{da.name}_PCA_reconstruction_error.png"
        file = fig_path+fig_name
        if os.path.exists(file): 
            print(f"Figure '{fig_name}' has previously been generated. The figure will not be updated.")
            continue

        recon_errors = []
        lwr = 1
        upr = 104 if da.values.shape[1]>104 else da.values.shape[1]+1
        stp = 10 if da.values.shape[1]>40 else 1
        for nC in range(lwr,upr,stp):
            stateT = rom.StateTransformer(nC).fit(da)
            recon_err = da - stateT.to_physical(stateT.transform(da))
            recon_errors.append(np.sqrt( np.average(np.average((recon_err.values**2),axis=0))))

        plt.figure();
        plt.plot(np.arange(lwr,upr,stp,dtype=int),np.array(recon_errors)*err_fac,'-o');
        plt.grid();
        plt.xlabel("Number of components");
        plt.ylabel(f"RMSE [{err_unit}]");
        plt.yscale("log");
        plt.title("Reconstruction error (on training data)");
        if save_fig: plt.savefig(fig_path+f"{da.name}_PCA_reconstruction_error.png",dpi=300, bbox_inches='tight');
        if show_fig: plt.show();
        else: plt.close();
        
        plt.figure();
        stateT.plot_explained_variance(yscale='linear')
        if save_fig: plt.savefig(fig_path+f"{da.name}_PCA_explained_variance_ratio.png",dpi=300, bbox_inches='tight');
        if show_fig: plt.show();
        else: plt.close();

def plot_recon_errors_all(comps,das,name,config=None,save_fig=False,show_fig=True,fig_path="",cmap_err='Reds'):
    plt.rcParams['font.size'] = FONT_SIZE-5
    # Set default values if config dict is not given. This allows for flexibility, 
    # if the function is used in a case where a config file does not exist.
    if config is None:
        err_fac = 1.0
        err_units = [""]*len(das)
    else:
        err_fac = config["err_factor"]
        err_units = config["err_units"]


    # Find reconstruction errors for each state
    recon_errs = []
    rmse_recons = []
    for i,n_comp in enumerate(comps):
        try: # Set element weights
            element_weight = das[i].geometry.get_element_area()
            skip_weights=False
        except AttributeError:
            skip_weights=True

        
        # Asses reconstruction error
        if 'forc' in name:
            forcingT = rom.ForcingTransformer(n_comp).fit(das[i])
            dfs = forcingT.transform(das[i])
            da_recon = forcingT.inverse_transform(dfs).to_numpy().reshape(das[i].shape)
        else:
            stateT = rom.StateTransformer(n_comp).fit(das[i])
            dfs = stateT.transform(das[i])
            da_recon = stateT.to_physical(dfs)
        # da_err = abs(das_train[i] - da_recon)
        rec_err = ms.rmse(das[i],da_recon,return_dim=1)*err_fac
        rec_err.name=das[i].name.split(" <")[0]
        recon_errs.append(rec_err)
        if skip_weights: rmse_recons.append(ms.nanaverage(ms.nanaverage(rec_err.values.squeeze())))
        else: rmse_recons.append(ms.nanaverage(ms.nanaverage(rec_err.values.squeeze(),weights=element_weight)))
    # ds_recon_errs = mikeio.Dataset(recon_errs)
    print("Reconstruction errors: ",rmse_recons)

    plt.figure();
    n_cols=(min(3,len(recon_errs)))
    n_rows = int(np.ceil( len(recon_errs) / n_cols))
    # Set the height of the figure based on the number of rows
    fig_height = n_rows * 6
    # Create the subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, fig_height))
    # Now axs is a 2D array. Flatten it for easy iteration.
    axs = axs.flatten()
    levels = np.linspace(0,np.ceil(np.max(rmse_recons)),11)

    for i in range(len(recon_errs)):
        try:
            # For contour plots
            recon_errs[i].plot(ax=axs[i],levels=levels,cmap=cmap_err,add_colorbar=False);
            # axs[i].set_ylabel('RMSE');
            cbar = fig.colorbar(axs[i].collections[0], ax=axs[i], location='right')
            cbar.ax.set_ylabel(f'RMSE [{err_units[i]}]')
        except AttributeError:
            # Sometimes it is as line plot if we're plotting a dfs1
            recon_errs[i].plot(ax=axs[i]);
            axs[i].set_ylabel('RMSE');

        axs[i].set_title(f"{recon_errs[i].name}, {comps[i]} comps\nRMSE="+str(np.round(rmse_recons[i],2))+err_units[i]);
        
    plt.tight_layout()
    if save_fig: plt.savefig(fig_path+f"recon_errs_{name}_rulebased.png", dpi=300);
    if show_fig: plt.show();
    else: plt.close();


#################################################################################################
#################################################################################################
##################################### Model selection ###########################################
#################################################################################################
#################################################################################################


def plot_decision_on_lags(collect_data,best_idx,case_str,file_name,legend_col="Parameters",model_name="PCLR",text="Chosen"):
    plt.rcParams['font.size'] = FONT_SIZE
    cols = ["surface elevation", "u velocity", "v velocity", "mean"]
    # cols = ["mean"]
    for c in cols:
        col = "Train RMSE "+c
        col_val = "Validation RMSE "+c
        plot_data = collect_data[collect_data[col]<100].reset_index(drop=True)
        fig,ax = plt.subplots()
        for i in range(len(plot_data["N parameters"])):
            p = ax.plot(plot_data.loc[i,"N parameters"],plot_data.loc[i,col],'o',label=f"{i}: {plot_data.loc[i,legend_col]}")
            ax.plot(plot_data.loc[i,"N parameters"],plot_data.loc[i,col_val],'o',alpha=0.5, color=p[0].get_color(), label='_nolegend_')
        ax.set_xlabel("Number of parameters")
        ax.set_ylabel("RMSE "+c)
        ax.legend(bbox_to_anchor=(1.9, 1))
        ax.set_title(f"{case_str} - {model_name}")
        ax.text(collect_data.loc[best_idx,"N parameters"],collect_data.loc[best_idx,col],text)
        plt.savefig(file_name+".png", bbox_inches='tight')
        plt.show()






#################################################################################################
#################################################################################################
######################### Fit selected model and make plots #####################################
#################################################################################################
#################################################################################################

def plot_loss_curve(config,fmrom,fig_path,fig_name_prefix="",save_fig=True,show_fig=False):
    if isinstance(fmrom.model,list) and len(fmrom.model)==fmrom.n_states:
        for i in range(fmrom.n_states):
            if "mlp" in str(fmrom.model[i]).lower():
                plt.figure()
                plt.plot(fmrom.model[i].loss_curve_)
                plt.title(f"Loss curve for state: {config['state_names'][i]}")
                if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_loss_curve_{config['state_names'][i]}.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
                plt.show() if show_fig else plt.close()
    else:
        if "mlp" in str(fmrom.model).lower() and not "multioutput" in str(fmrom.model).lower():
            plt.figure()
            plt.plot(fmrom.model.loss_curve_)
            plt.title("Loss curve")
            if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_loss_curve.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
            plt.show() if show_fig else plt.close()

def plot_residuals_MLP(config,fmrom,fig_path,fig_name_prefix="",save_fig=True,show_fig=False):
    if isinstance(fmrom.model,list):
        for i in range(fmrom.n_states):
            fig, axs = plt.subplots(1, 3,figsize=(18,5))
            for j in range(3):
                axs[j].scatter(fmrom.residuals[i].index,fmrom.residuals[i].iloc[:,j],s=2)
                axs[j].set_title(f"Residuals for state: {config['state_names'][i]}, PC: {j+1}")
            if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_residuals_{config['state_names'][i]}.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
            plt.show() if show_fig else plt.close()

            
            
            fig, axs = plt.subplots(1, 3,figsize=(18,5))
            for j in range(3):
                maxidx = fmrom.residuals[i].iloc[:,j].idxmax()
                maxidxx = np.where(fmrom.residuals[i].index == maxidx)[0]
                largest_residuals_window = fmrom.residuals[i].iloc[np.arange(max(0,maxidxx-50),min(maxidxx+50,fmrom.residuals[i].shape[0])),j]
                axs[j].scatter(largest_residuals_window.index,largest_residuals_window)
                axs[j].set_title(f"Residuals for state: {config['state_names'][i]}, PC: {j+1}")
            if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_residuals_largest_{config['state_names'][i]}.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
            plt.show() if show_fig else plt.close()


def plot_residuals(ds_pred,ds_true, fig_path, config=None, fig_name_prefix="",save_fig = True, show_fig = False,var_names=None):

    pos = config["plot_point_position"]
    pos_name = config["plot_point_name"]
    if var_names is None:
        var_names = ds_pred.names
    
    fig, axs = plt.subplots(1, len(ds_pred),figsize=(3*len(ds_pred),5))
    if len(ds_pred)==1:
        axs=[axs]
    for j in range(len(ds_pred)):
        res = (ds_true[j].sel(x=pos[0],y=pos[1],time=ds_pred.time).values - ds_pred[j].sel(x=pos[0],y=pos[1]).values).flatten()
        axs[j].scatter(np.arange(len(res)),res,s=2)
        axs[j].set_title(f"Residuals for state: {var_names[j]}")
        axs[j].set_ylabel("Residual")
        axs[j].set_xlabel("Time index")
    if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_residuals_vs_time.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.show() if show_fig else plt.close()

    fig, axs = plt.subplots(1, len(ds_pred),figsize=(3*len(ds_pred),5))
    if len(ds_pred)==1:
        axs=[axs]
    for j in range(len(ds_pred)):
        res = (ds_true[j].sel(x=pos[0],y=pos[1],time=ds_pred.time).values - ds_pred[j].sel(x=pos[0],y=pos[1]).values).flatten()
        axs[j].scatter(ds_pred[j].sel(x=pos[0],y=pos[1]).values,res,s=2)
        axs[j].set_title(f"{var_names[j]}")
        axs[j].set_ylabel("Residual")
        axs[j].set_xlabel("Value")
    if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_residuals_vs_val.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.show() if show_fig else plt.close()

        
        
        # fig, axs = plt.subplots(1, 3,figsize=(18,5))
        # for j in range(3):
        #     maxidx = fmrom.residuals[i].iloc[:,j].idxmax()
        #     maxidxx = np.where(fmrom.residuals[i].index == maxidx)[0]
        #     largest_residuals_window = fmrom.residuals[i].iloc[np.arange(max(0,maxidxx-50),min(maxidxx+50,fmrom.residuals[i].shape[0])),j]
        #     axs[j].scatter(largest_residuals_window.index,largest_residuals_window)
        #     axs[j].set_title(f"Residuals for state: {config['state_names'][i]}, PC: {j+1}")
        # if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_residuals_largest_{config['state_names'][i]}.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
        # plt.show() if show_fig else plt.close()

def plot_RMSE_spatial(ds_pred,ds_true, fig_path, config=None, fig_name_prefix="",cmap_err="Reds",save_fig = True, show_fig = False,var_names=None):

    plt.rcParams['font.size'] = FONT_SIZE
    if config is None:
        err_fac = 1.0
        err_units = [""]*len(ds_pred.names)
    else:
        err_fac = config["err_factor"]
        err_units = config["err_units"]
        individual_error_bars = config["individual_error_bars"]
    element_weight = ds_pred.geometry.get_element_area()
    # element_weight=None
    if var_names is None:
        var_names = ds_pred.names

    err_space_list = []
    err_list = []
    for i,var in enumerate(var_names):
        da_pred = ds_pred[i].copy()
        da_true = ds_true[i].sel(time=da_pred.time) # True data

        da_pred_cm = da_pred*err_fac 
        da_true_cm = da_true*err_fac
        # RMSE across time (out: spatial rmse)
        err_space_list.append(ms.rmse(da_true_cm,da_pred_cm,return_dim=1))
        err_list.append(ms.rmse(da_true_cm,da_pred_cm,return_dim=None,weights=element_weight))

    for i,var in enumerate(var_names):
        err_space = err_space_list[i]
        err = err_list[i]
        if np.max(err_list)<=0.5:
            levels = None
        elif individual_error_bars:
            levels = np.linspace(0,4.0,11)
        else:
            # levels = np.linspace(0,np.max(err_list)+np.mean(err_list),11)
            levels = np.linspace(0,np.ceil(np.max(err_list)),11)

        fig, axs = plt.subplots(1, 1,figsize=(6,5))
        if err_fac != 1.0:
            err_space.values[np.isnan(err_space.values)]=0 # Set nan values to 0
            err_space.plot.contourf(ax=axs,cmap=cmap_err,levels=levels,add_colorbar=False)
            plt.colorbar(axs.collections[0], ax=axs,location="right",label=f'RMSE [{err_units[i]}]')
        else:
            err_space.plot.contourf(ax=axs,cmap=cmap_err,levels=levels)
        # cbar.ax.set_ylabel(f'RMSE [{err_units[i]}]')
        axs.axis('off')
        axs.set_facecolor('gray')
        axs.set_title(f"{var}\nRMSE: {np.round(err,3)} {err_units[i]}")
        # if var == var_names[-1]:
        if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_rmse_map_{var}.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
        plt.show() if show_fig else plt.close()

def rmse_on_map(da_pred,da_true,fig_path,config,show_fig=True,save_fig=False):
    plt.rcParams['font.size'] = FONT_SIZE
    case_str = config["case_str"]
    plot_str = config["plot_str"]
    cb_location = config["cb_location"]
    err_fac = config["err_factor"]
    err_units = config["err_units"]
    fig_name = f"geometry_on_map_{case_str}.png"
    file = fig_path+fig_name
    if os.path.exists(file): 
        print(f"Figure '{fig_name}' has previously been generated. The figure will not be updated.")
        return

    da_true = da_true.sel(time=da_pred.time) # True data

    element_weight = da_pred.geometry.get_element_area()

    da_pred_cm = da_pred*err_fac 
    da_true_cm = da_true*err_fac
    # RMSE across time (out: spatial rmse)
    err_space = ms.rmse(da_true_cm,da_pred_cm,return_dim=1)

    xmin = da_pred.geometry.node_coordinates[:,0].min()
    xmax = da_pred.geometry.node_coordinates[:,0].max()
    ymin = da_pred.geometry.node_coordinates[:,1].min()
    ymax = da_pred.geometry.node_coordinates[:,1].max()

    if config["individual_error_bars"]:
            levels = np.linspace(0,4.0,11)
    else:
        levels = np.linspace(0,np.ceil(err_space.values.max()),11)

    fig,ax = plt.subplots(1,1,figsize=(8,8))
    pcf = err_space.plot.contourf(ax=ax, levels = levels,cmap="Reds", add_colorbar=False)
    ax.set_xlim([xmin-0.1, xmax+0.1])
    ax.set_ylim([ymin-0.05, ymax+0.05])
    ax.set_title(plot_str)
    cx.add_basemap(ax, crs=crs.CRS(err_space.geometry.projection).to_pyproj(),
                source=cx.providers.CartoDB.Positron,
                zoom=9)
    ax.set_aspect(1.0/np.cos(0.5*(ymin+ymax)*np.pi/180))
    # axins = inset_axes(ax,
    #             width="5%",  
    #             height="30%",
    #             loc=cb_location,
    #             # borderpad=4.5
    #             bbox_to_anchor=(-0.2, 0,0.98,1),
    #             bbox_transform=ax.transAxes
    #             )
    plt.colorbar(ax.collections[0], ax=ax,location="right",label=f'RMSE [{err_units[0]}]')
    # fig.colorbar(pcf.collections[0], pad=0.01, shrink=0.5)
    if save_fig: plt.savefig(fig_path+f"rmse_map_zoom_{case_str}.png", bbox_inches='tight', pad_inches=0.02, dpi=300);
    if show_fig: plt.show();
    else: plt.close();

def plot_abs_err_quantiles(ds_pred,ds_true,fig_path,fig_name_prefix="",config=None,save_fig = True, show_fig = False,var_names=None,q=[0.01,0.5,0.99]):
    plt.rcParams['font.size'] = FONT_SIZE-1
    if config is None:
        err_fac = 1.0
        err_units = [""]*len(ds_pred.names)
    else:
        err_fac = config["err_factor"]
        err_units = config["err_units"]
    
    if var_names is None:
        var_names = ds_pred.names
    fig, axs = plt.subplots(1, len(var_names),figsize=(len(var_names)*6,5))

    for v in range(len(var_names)):
        if len(var_names)>1: 
            cur_ax = axs[v]
        else:
            cur_ax = axs
        da = ds_true[v].sel(time=ds_pred[v].time) # True data
        err = abs(da-ds_pred[v])
        (err_fac*err.quantile(q=q,axis=1)).plot(ax=cur_ax,ls="-")
        cur_ax.set_title(f"{var_names[v]}")
        cur_ax.set_ylabel(f"Absolute error  [{err_units[v]}]")
        cur_ax.grid()
        if v == 0:
            cur_ax.legend(["Quantile 0.01", "Quantile 0.5", "Quantile 0.99"])
        else:
            cur_ax.get_legend().set_visible(False)
    plt.tight_layout()

    if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_error_quantiles_time.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.show() if show_fig else plt.close()

def plot_err_quantiles(ds_pred,ds_true,fig_path,fig_name_prefix="",config=None,save_fig = True, show_fig = False,var_names=None):
    plt.rcParams['font.size'] = FONT_SIZE-1
    if config is None:
        err_fac = 1.0
        err_units = [""]*len(ds_pred.names)
    else:
        err_fac = config["err_factor"]
        err_units = config["err_units"]
    
    if var_names is None:
        var_names = ds_pred.names

    fig, axs = plt.subplots(1, len(var_names),figsize=(len(var_names)*6,5))
    for v in range(len(var_names)):
        if len(var_names)>1: 
            cur_ax = axs[v]
        else:
            cur_ax = axs
        da = ds_true[v].sel(time=ds_pred[v].time) # True data
        err = err_fac*(da-ds_pred[v])
        err.quantile(q=[0.99],axis=1).plot(ax=cur_ax,ls="-",color="tab:green")
        err.quantile(q=[0.01],axis=1).plot(ax=cur_ax,ls="-",color="tab:green",legend=False)
        err.quantile(q=[0.9],axis=1).plot(ax=cur_ax,ls="-",color="tab:blue")
        err.quantile(q=[0.1],axis=1).plot(ax=cur_ax,ls="-",color="tab:blue",legend=False)
        err.quantile(q=[0.5],axis=1).plot(ax=cur_ax,ls="-",color="tab:orange",label="Median")
        cur_ax.set_title(f"{var_names[v]}")
        cur_ax.set_ylabel(f"Error  [{err_units[v]}]")
        cur_ax.grid()
        if v == 0:
            cur_ax.legend(handles = [cur_ax.get_legend_handles_labels()[0][0],cur_ax.get_legend_handles_labels()[0][2],cur_ax.get_legend_handles_labels()[0][4]],
                            labels =  ["1, 99%","10, 90%","Median"])
        else:
            cur_ax.get_legend().set_visible(False)
    plt.tight_layout()
    if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_error_res_quantiles_time.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.show() if show_fig else plt.close()


    fig, axs = plt.subplots(1, len(var_names),figsize=(len(var_names)*6,5))
    for v in range(len(var_names)):
        if len(var_names)>1: 
            cur_ax = axs[v]
        else:
            cur_ax = axs
        da = ds_true[v].sel(time=ds_pred[v].time) # True data
        err = err_fac*(da-ds_pred[v])
        p10 = err.quantile(q=[0.25,0.75],axis=1)
        p1 = err.quantile(q=[0.01,0.99],axis=1)
        cur_ax.fill_between(p1.time,p1[0].values,p1[1].values,color="k",alpha=0.2)
        cur_ax.fill_between(p10.time,p10[0].values,p10[1].values,color="k",alpha=0.5)
        cur_ax.plot(err.time,err.quantile(q=[0.5],axis=1)[0].values,color="k")
        # err.quantile(q=[0.5],axis=1).plot(ax=cur_ax,ls="-",color="k")
        cur_ax.set_title(f"{var_names[v]}")
        cur_ax.set_ylabel(f"Error  [{err_units[v]}]")
        cur_ax.grid()
        # cur_ax.get_legend().set_visible(False)
    plt.tight_layout()
    if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_error_res_quantiles_fill_time.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.show() if show_fig else plt.close()


def plot_value_in_point(ds_pred,ds_true,mod_name,fig_path,true_mod_name="MIKE 21",fig_name_prefix="",config=None,save_fig=True, show_fig=False,vars_in_same_plot=False):
    plt.rcParams['font.size'] = FONT_SIZE-5
    pos = config["plot_point_position"]
    pos_name = config["plot_point_name"]
    units = config["state_units"]
    var_names = ds_pred.names
    n_tsteps = len(ds_pred.time)
    pwin = 300 # plot window
    if not vars_in_same_plot:
        fig, axs = plt.subplots(1, len(var_names),figsize=(len(var_names)*5,5))
        for v in range(len(var_names)):
            if len(var_names)>1: 
                cur_ax = axs[v]
            else:
                cur_ax = axs
            da = ds_true[var_names[v]].sel(time=ds_pred[v].time) # True data
            # Select only a window around the maximum error
            if v==0:
                t_max = np.argmax(abs(da.sel(x=pos[0],y=pos[1]).values-ds_pred[v].sel(x=pos[0],y=pos[1]).values))
                if t_max + pwin > n_tsteps:
                    t_max = n_tsteps-pwin-1
                elif t_max - pwin < 0:
                    t_max = pwin+1
            da.sel(x=pos[0],y=pos[1]).isel(time = np.arange(t_max-pwin,t_max+pwin)).plot(ax=cur_ax,ls="-",label = f"{true_mod_name}")
            ds_pred[v].sel(x=pos[0],y=pos[1]).isel(time = np.arange(t_max-pwin,t_max+pwin)).plot(ax=cur_ax,ls="--",alpha=0.5,label=mod_name)
            cur_ax.set_ylabel(f"{var_names[v]} [{units[v]}]")
            cur_ax.set_title(f"{var_names[v]} at point {pos_name}")
            cur_ax.grid()
    else:
        fig, axs = plt.subplots(1, 1,figsize=(6,5))
        for v,var_name in enumerate(ds_true.names):
            da = ds_true[v].sel(time=ds_pred[0].time) # True data
            da.sel(x=pos[0],y=pos[1]).plot(ax=axs,ls="-",label = f"{true_mod_name} {var_name}")
        for v in range(len(var_names)):
            ds_pred[v].sel(x=pos[0],y=pos[1]).plot(ax=axs,ls="--",alpha=0.5,label=mod_name+" "+var_names[v])
        axs.set_ylabel(f"[{units[0]}]")
        axs.set_title(f"At point {pos_name}")
        axs.grid()
    plt.tight_layout()
    axs[0].legend(loc="best")
    if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_value_point_time.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.show() if show_fig else plt.close()


def plot_scatter(ds_pred,ds_true,mod_name,fig_path,fig_name_prefix="",config=None,save_fig=True, show_fig=False,var_names=None):
    plt.rcParams['font.size'] = FONT_SIZE+4
    pos = config["plot_point_position"]
    pos_name = config["plot_point_name"]
    if var_names is None:
        var_names = ds_pred.names

    # fig, axs = plt.subplots(1, 3,figsize=(15,5))
    for v in range(len(var_names)):
        plt.figure()
        # Predicted data
        # True data
        da = ds_true[v].sel(time=ds_pred[v].time) # True data
        da_point = da.sel(x=pos[0],y=pos[1])
        da_point.name="MIKE 21"
        mike = modelskill.PointObservation(da_point*100, x=pos[0],y=pos[1],name="MIKE 21",quantity=modelskill.Quantity(name="Surface Elevation", unit="centimeter"))
        fmrom = modelskill.PointModelResult(ds_pred[v].sel(x=pos[0],y=pos[1])*100,x=pos[0],y=pos[1], name=mod_name,quantity=modelskill.Quantity(name="Surface Elevation", unit="centimeter"))
        # da_point.values[np.isnan(da_point.values)]=1e-35

        # Create model skill comparer
        cc = modelskill.match(mike, fmrom)
        cmp = cc.remove_bias()
        # cc2 = modelskill.ComparerCollection(cmp)
        # cc2[0]
        # q = modelskill.Quantity(...)
        # 

        # cc2.plot.scatter(skill_table=["rmse", "mae", "cc", "si"]);
        cmp.plot.scatter();
        plt.xlabel("MIKE 21")
        plt.ylabel(mod_name)
        plt.title(var_names[v]+" at point "+pos_name)
        if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_{var_names[v]}_scatter.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
        plt.show() if show_fig else plt.close()

def plot_point_with_maxAbsError(ds_pred,ds_true,fig_path,mod_name,fig_name_prefix="",save_fig=True, show_fig=False):
    plt.rcParams['font.size'] = FONT_SIZE
    idx = np.where(abs(ds_pred[0]-ds_true[0].sel(time=ds_pred[0].time)).values == ms.maxAbsError(y_pred=ds_pred[0],y_true=ds_true[0].sel(time=ds_pred[0].time)))
    max_time = ds_true[0].sel(time=ds_pred[0].time).isel(idx[0]).time
    max_point = ds_true[0].sel(time=ds_pred[0].time).isel(idx[0])[idx[1]].geometry
    max_val = ds_true[0].sel(time=max_time,x=max_point.x,y=max_point.y)
    freq = ds_pred[0].time[1]-ds_pred[0].time[0]
    if len(ds_pred[0].time)>300:
        plot_times = pd.date_range(max(max_time[0]-150*freq,ds_pred[0].time[0]),min(max_time[0]+150*freq,ds_pred[0].time[-1]),freq=freq)
    else:
        plot_times = ds_pred[0].time


    fig,axs = plt.subplots(1,2,figsize=(16,5))
    ds_true[0].sel(time=max_time).plot(ax=axs[0])
    axs[0].plot(max_point.x,max_point.y,'ro');


    ds_true[0].sel(time=plot_times,x=max_point.x,y=max_point.y).plot(ax=axs[1],ls="-",label = "MIKE 21")
    ds_pred[0].sel(time=plot_times,x=max_point.x,y=max_point.y).plot(ax=axs[1],ls="--",alpha=0.5,label=mod_name)
    axs[1].plot(max_time,max_val,"ro")
    axs[1].legend()
    axs[1].tick_params(axis='x',labelrotation=45)
    axs[1].set_title(f"{ds_pred[0].name}")
    if save_fig: plt.savefig(fig_path+f"{fig_name_prefix}_point_with_maxAbsError.png", bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.show() if show_fig else plt.close()
    


def plot_model_skill(config,das_pred,das_true,fig_path="", save_figs = True, show_figs = False):
    plt.rcParams['font.size'] = FONT_SIZE-5

    make_plots = False if not save_figs and not show_figs else True
    
    da_true = das_true[0].sel(time=das_pred[0].time)
    obs = []
    for i in range(len(config["obs_directories"])):
        xp, yp = config["obs_positions"][i][0], config["obs_positions"] [i][1]
        o = modelskill.PointObservation(config["obs_directories"][i], x=xp, y=yp, item=config["item_names"][i], name=config["obs_names"][i],
                                        quantity=modelskill.Quantity(name="Surface Elevation", unit="meter"))
        obs.append(o)
    
    mr1 = modelskill.DfsuModelResult(da_true, name="MIKE 21",quantity=modelskill.Quantity(name="Surface Elevation", unit="meter"))
    mr2 = modelskill.DfsuModelResult(das_pred[0], name="FMROM",quantity=modelskill.Quantity(name="Surface Elevation", unit="meter"))
    mrs = [mr1,mr2]

    if make_plots: 
        modelskill.plotting.spatial_overview(obs, mr1);
        plt.savefig(fig_path+"model_skill_spatial_overview.png")
        if show_figs: plt.show()
        else: plt.close()

    # Initialize comparer collection
    cc = modelskill.match(obs, mrs)
    cmps = []
    for cmp in cc:
        cmps.append(cmp.remove_bias())
    cc2 = modelskill.ComparerCollection(cmps)

    # Print skill table
    print(cc2.skill())
    
    if make_plots: 
        for cmp in cc2:
            cmp.plot.qq();
            plt.savefig(fig_path+"model_skill_qqplot_"+cmp.name+".png")
            if show_figs: plt.show()
            else: plt.close()

            # plt.figure(figsize = (7,5))
            cmp.sel(model="MIKE 21").plot(figsize = (6,5),skill_table=["rmse", "mae", "cc", "si","r2"])
            # plt.tight_layout()
            plt.savefig(fig_path+"model_skill_scatter_"+cmp.name+"_MIKE.png",bbox_inches='tight')
            if show_figs: plt.show()
            else: plt.close()
            cmp.sel(model="FMROM").plot(figsize = (6,5),skill_table=["rmse", "mae", "cc", "si","r2"])
            # plt.tight_layout()
            plt.savefig(fig_path+"model_skill_scatter_"+cmp.name+"_FMROM.png",bbox_inches='tight')
            if show_figs: plt.show()
            else: plt.close()



def plot_data_sensitivity_results(config_dict):
    plt.rcParams['font.size'] = FONT_SIZE-5
    df_data_sensitivity = pd.read_csv(f"{config_dict['fldr']}/data/mda/MDA_data_sensitivity_{len(config_dict['state_names'])}states.csv",index_col=0)
    
    units = ["cm","cm/s","cm/s"]
    fig,axs = plt.subplots(1,3,figsize=(13,5))
    for i in range(len(config_dict["state_names"])):
        axs[i].plot(df_data_sensitivity["Fraction selected"],[ast.literal_eval(df_data_sensitivity["RMSEs [cm]"][s])[i] for s in range(len(df_data_sensitivity))],marker='o')
        axs[i].set_title(f"{config_dict['state_names'][i]}")
        axs[i].set_xlabel("Fraction selected")
        axs[i].set_ylabel(f"RMSE {units[i]}")
    plt.tight_layout()
    plt.savefig(f"{config_dict['fig_path']}MDA_data_sensitivity.png", bbox_inches='tight', pad_inches=0.02, dpi=300)

def plot_MDA_selected_points(config_dict,train_times_all=None,train_times_mda=None):
    plt.rcParams['font.size'] = FONT_SIZE-2
    das,das_bc = cmn.load_data(config_dict,config_dict["all_times"])

    if train_times_all is None:
        train_times_all = config_dict["forecast_times"]
    if train_times_mda is None:
        train_times_mda = config_dict["train_times"]

    pos = config_dict["plot_point_position"]
    fig,ax = plt.subplots(1,1,figsize=(18,6));
    da_point = das[0].sel(x=pos[0],y=pos[1],time=train_times_all)
    da_point_mda = das[0].sel(x=pos[0],y=pos[1],time=train_times_mda)

    da_point.plot(title="MDA in training period",ax=ax);
    ax.plot(da_point_mda.time,da_point_mda.values,'ro',markersize=2,label = "Selected by MDA");
    ax.set_ylabel(f"{das[0].name} [m]");
    ax.set_xlabel("Time");
    ax.margins(x=0);
    # ax.axis("off")
    plt.legend(loc="upper right");



































def plot_map_pred_vs_truth(ds_in,cmaps,save_fig=False,fig_path=None,fig_name=None,plot_type="regular"):
    vmin = np.min([ds_in['Ground truth'].values, ds_in["Predicted"].values])
    vmax = np.max([ds_in['Ground truth'].values, ds_in["Predicted"].values])

    # Make sure that 0 has color white
    min_val = -max(abs(vmin),abs(vmax))
    max_val = max(abs(vmin),abs(vmax))

    min_vals = [min_val,min_val,0]
    max_vals = [max_val,max_val,abs(ds_in['Error'].values).max()]

    fig, axs = plt.subplots(1, 3,figsize=(12,5))
    for i in range(len(ds_in.names)):
        if plot_type == "contourf":
            levels = np.linspace(0,max_vals[i],11)
            ds_in[i].plot(ax=axs[i],levels=levels,cmap=cmaps[i],add_colorbar=False)
        elif plot_type == "regular":
            ds_in[i].plot(ax=axs[i],vmin=min_vals[i],vmax=max_vals[i],cmap=cmaps[i],add_colorbar=False)
        axs[i].axis('off')
        axs[i].set_title(ds_in[i].name)
    cbar = fig.colorbar(axs[0].collections[0], ax=axs[0], location='left')
    cbar.ax.set_ylabel('Surface elevation [cm]')
    cbar = fig.colorbar(axs[2].collections[0], ax=axs[2], location='right')
    cbar.ax.set_ylabel('Absolute error [cm]')

    plt.subplots_adjust(left=0.01,
                            bottom=0.1, 
                            right=0.99, 
                            top=0.85, 
                            wspace=-0.9, 
                            hspace=0.15)
    
    if save_fig:
        if fig_path is None or fig_name is None:
            raise ValueError("fig_path and fig_name must be provided")
        else:
            plt.savefig(fig_path+fig_name+".png", bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close()


def make_error_plot_time(dif,element_weight,split_time,cols):

    max_idx = int(np.argmax(dif.values.max(axis=0)))
    # Max across space, min across time! Find time step with smallest maximum absolute error
    min_idx = int(np.argmin(np.max(dif.values,axis=0)))

    # vals = [np.min(np.max(dif.values,axis=0)),np.max(dif.values.max(axis=0))]
    idxs = [min_idx,max_idx]

    txt = ["smallest","largest"]
    cols = cols

    for i in range(len(idxs)):
        da_pt_cm = dif.isel(idxs[i],axis=1) * 100
        ax = da_pt_cm.plot(title="Error at point with "+txt[i]+" MaxAE",c = cols[i])
        ax.set_ylabel("Error [cm]")
        ax.axvspan(split_time, dif.time[-1], facecolor='0.2', alpha=0.2)

    ## Errors over time
    
    dif = dif[1:] # Skip first time step

    maxAE_time = dif.max(axis=1)*100
    maxAE_time.plot(c=cols[3])
    plt.title("MaxAE")
    plt.ylabel("MaxAE [cm]")
    plt.axvspan(split_time, dif.time[-1], facecolor='0.2', alpha=0.2)

    MAE_time = dif.average(axis=1,weights=element_weight)*100
    MAE_time.plot(c=cols[2])
    plt.title("MAE")
    plt.ylabel("MAE [cm]")
    plt.axvspan(split_time, dif.time[-1], facecolor='0.2', alpha=0.2)

def plot_geometry_error(dif):
    # This function plots the error on the geometry.
    # Plots time step with the largest and smallest MaxAE across time. 

    dif = abs(dif)

    max_idx = int(np.argmax(dif.values.max(axis=1)))
    # Max across space, min across time! Find time step with smallest maximum absolute error
    min_idx = int(np.argmin(np.max(dif.values,axis=1)))

    min_val = min(dif[max_idx].values.min(),dif[min_idx].values.min())
    max_val = max(dif[max_idx].values.max(),dif[min_idx].values.max())

    dif[max_idx].plot(vmin = min_val,vmax=max_val)
    dif[min_idx].plot(vmin = min_val,vmax=max_val)