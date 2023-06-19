import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math

from numpy import sqrt, pi, exp, linspace

from lmfit import Model
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from natsort import natsorted


###################################################### MATH FUNCTIONS ##################################################################

def gaussian(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/np.sqrt(wid)*np.exp(-(x-cen)**2/(2.*wid**2)))
    #return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2))

def line(x, slope, intercept):
    "line"
    return slope * x + intercept

def gauss1(x,a,x0,sigma,offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

def gauss1_new(x,a,x0,sigma,c,offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + c*x + offset

def gauss3(x,amp,cen,wid):
    return amp*np.exp(-(x-cen)**2/(2*wid**2))


###################################################### UTILITY FUNCTIONS ##################################################################

def csv_to_df_list(csv_files):
    # create empty list
    dfs_list = []
    name_dfs_list = []
    
    
    #append datasets into the list
    for i in natsorted(csv_files):
        df = pd.read_csv(i,header=None)
        dfs_list.append(df)
        name_dfs_list.append(csv_files)
    
    return dfs_list, name_dfs_list


def interpolate(xval, df, xcol): # NOT USED YET, but may be important to interpolate outlier values for the best parameters
    '''compute xval as the linear interpolation of xval where df is a dataframe and
 df.x are the x coordinates, and df.y are the y coordinates. df.x is expected to be sorted'''
    return np.interp([xval], df[xcol])


###################################################### PLOTTING ##########################################################################

def plotFit(dataframe, parameters_df, model, filename,fitting_method = "lmfit", profile_type = "NMY-2"):
    ''' Plots the fit for every timepoint on the same plot for only one example, being that a single embryo or data pooled from multiple embryos.
    inputs:
        - dataframe: should be the original dataframe, the same used in the fitProfile2 function, for single or multiple embryos
        - parameters_df: should be the dataframe with the parameters extracted after fitting: Amplitude, mean, sigma - and offset or slope and offset if fitting a
        gaussian with constant offset or gaussian with linear regression, respectively
        - model: the method that defines the fitting equation: gaussian with constant offset or gaussian with linear regression for curve_fit or the one defined by a composite model for lmfit
        - filename: the name of the file with the final best fit plot
        - fitting_method: can be curve_fit or lmfit - check python documentation on these methods if you want to know more'''
    
    x_eval = np.linspace(0,len(dataframe.columns), 1000) #generate a continuous variable from the dataframe dimensions
    if "curve_fit" in fitting_method:
        plt.figure(figsize=(10,6))
        for i, row in enumerate(parameters_df.itertuples()):
            parameters = np.array(list(row)[1:])
            y_eval = gauss1_new(x_eval,*parameters) ## calculate against a continuous variable so it looks like a "real" gaussian and not a discretized form
            #y_smooth_continuous = a*exp(-(xx-b)**2/(2*c**2)) + d*xx + e
            plt.plot(x_eval,y_eval)
            plt.title("Best fits for line profiles of " + profile_type + "::GFP for each timepoint",fontsize=15)
            plt.xlabel("Membrane length (s/L) (%)",fontsize=15)
            plt.ylabel(profile_type + "::GFP \n fluorescence intensity (AU)",fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.savefig('./Results/Plots/' + fitting_method + '_' + filename + '.png')
    
    elif "lmfit" in fitting_method:
        plt.figure(figsize=(10,6))
        for i, row in enumerate(parameters_df.itertuples()):
            parameters = np.array(list(row)[1:])
            pars  = model.make_params( amp = parameters[2] , 
                                    cen = parameters[3], 
                                    wid = parameters[4], 
                                    slope=parameters[0], 
                                    intercept=parameters[1])
            y_eval = model.eval(pars,x=x_eval) ## calculate against a continuous variable so it looks like a "real" gaussian and not a discretized form
            plt.plot(x_eval,y_eval)
            plt.title("Best fits for line profiles of " + profile_type + "::GFP for each timepoint",fontsize=15)
            plt.xlabel("Membrane length (s/L) (%)",fontsize=15)
            plt.ylabel(profile_type + "::GFP \n fluorescence intensity (AU)",fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.savefig('./Results/Plots/' + fitting_method + '_' + filename + '.png')
    else:
        raise Exception("Invalid fitting method!")
    


def plotParameters(dataframes_list, profile_type,filename,condition="one condition",plot="pooled profiles"): #name_dataframes
    ''' Input: - dataframes_list - list of dataframes containing the best parameters for each embryo or pooled from multiple embryos
                 for a single condition
               - parameters_to_plot - a string with the name of the parameter to plot or a list of strings containing the parameters to plot
               - profile_type - can be "NMY-2", "RhoA(AHPH)" or "MLC-4AA" 
               - filename - name with which to save the plot. DO NOT DISCLOSE THE NAME OF THE PARAMETER, that is included by default below
               - condition - CAN BE "one condition" or "several conditions" depending on whether you want to compare controls with RNAis on the same plot
               - plot - can be "individual profiles" or "pooled profiles", data pooled from multiple embryos already
        Output: - plot files according to the options selected '''
    for dataframe in dataframes_list:
        timestep=5
        time_new = np.arange(0, (len(dataframe)-1)*timestep, timestep)
        if ("all conditions" in condition and "pooled profiles" in plot) or ("one condition" in condition and "pooled profiles" in plot):
            for name,values in dataframe.iteritems():
                y=np.array(values[1:],dtype=float)
                if values[0] == "wid":
                    y = abs(y)
                plt.figure(figsize=(10,6))
                plt.title(values[0] + " of " + profile_type + "::GFP) profiles' fittings over time",fontsize=15)
                plt.plot(time_new,y)
                plt.xlabel("Time (seconds)",fontsize=15)
                plt.ylabel(values[0],fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.savefig('./Results/Plots/' + values[0] + '_' + filename + '.png')
                #plt.savefig('./plots/best_parameters/' + filename + '.png')
        elif "all conditions" in condition and "individual profiles" in plot:
            for name,values in dataframe.iteritems():
                y=np.array(values[1:],dtype=float)
                if values[0] == "wid":
                    y = abs(y) # HAVEN'T CONSIDERED THE 2*SIGMA HERE, IT IS JUST SIGMA, SO HALF OF THE GAUSSIAN, SO DON'T BE SURPRISED IT IS SO SMALL
                plt.figure(figsize=(10,6))
                plt.title(values[0] + " of " + profile_type + "::GFP) profiles' fittings over time",fontsize=15)
                plt.plot(time_new,y)
                plt.xlabel("Time (seconds)",fontsize=15)
                plt.ylabel(values[0],fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.savefig('./Results/Plots/' + values[0] + '_' + filename + '.png')
        elif "one condition" in condition and "individual profiles" in plot:
            for name,values in dataframe.iteritems():
                y=np.array(values[1:],dtype=float)
                if values[0] == "wid":
                    y = abs(y)
                plt.figure(figsize=(10,6))
                plt.title(values[0]+ " of " + profile_type + "::GFP) profiles' fittings over time",fontsize=15)
                plt.plot(time_new,y)
                plt.xlabel("Time (seconds)",fontsize=15)
                plt.ylabel(values[0], fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.savefig('./Results/Plots/' + values[0] + '_' + filename + '.png')

        else:
            raise Exception("Invalid option!")


###################################################### FITTING FUNCTIONS ###################################################################

def fitProfile(dataframe,function):
    ''' Fits the data using curve_fit '''
    x= np.arange(len(dataframe.columns))
    A, B, C, D, E = [], [], [], [], []
    for i, row in enumerate(dataframe.itertuples()):
        y = np.array(list(row)[1:])
        #n=len(x)
        n=sum(y)
        amp=1
        #amp = max(y)-min(y)
        #mean = sum(x)/len(x)
        mean = sum(x*(y-min(y)))/n
        #mean = sum(x)/x[len(df.columns)]
        sigma_1 = sum((y-min(y))*(x-mean)**2)/n
        #sigma_1=sum((x-mean)**2)/len(x)
        slope = (y[len(dataframe.columns)-1]-y[0])/(x[len(dataframe.columns)-1]-x[0])  
        #slope = 0
        offset = 1
        init_vals = [ amp, mean, sigma_1, slope, offset]
        #init_vals = [ amp, mean, sigma_1, slope , offset] #[1, mean, sigma_1, slope , 0]
        # calculate fit. The curve_fit method uses the least squares to estimate the error by default, but other methods can be employed
        popt, cov = curve_fit(function, x, y, init_vals, maxfev=1000000) # ## p0=init_vals = [ 1, mean, sigma_1, 0 , 0 ] --> leave out the first estimation of the parameters

        # use fit params with original x data to calculate fit y data points
        a,b,c,d,e = popt[0], popt[1], popt[2], popt[3], popt[4]
        #print(best_parameters)
        A.append(a)
        B.append(b)
        C.append(c)
        D.append(d)
        E.append(e)
    return A, B, C, D, E


def fitProfile2(dataframe,model,fitting = "MULTIPLE EMBRYOS", plot=False):
    ''' fitRows2 used lmfit to fit a composite model
    inputs:
    - dataframe: if trying to fit data from a single embryo, the dataframe must have n_rows = n_timepoints and n_cols = n_positions
                 otherwise, if trying to fit data pooled from multiple embryos, the dataframe must be in the following format:
                 n_rows = n_positions and n_colums = n_timepoints; of note, each cell in the dataframe is a 1D array with 
                 length = number of examples/embryos.
    - model: can be a simple or a composite model previously established using mathematical functions
    - fitting: defines whether we want to fit data from single or pooled from multiple embryos
    - plot: boolean variable that defines whether to plot the data + fitting or not. Default = False. If you set it to True,
            it will show a plot for every timepoint, but may be worth checking the first runs to access the quality of the fitting visually'''
    
    best_parameters = []

    '''pars  = model.make_params( amp = 1, 
                                          cen = mean, 
                                          wid = init_parameters[2], 
                              slope=init_parameters[3], 
                              intercept=init_parameters[4])'''
    
    if "MULTIPLE EMBRYOS" in fitting:
        for col in dataframe.columns:
            result_x=np.array([])
            result_y=np.array([])
            for x, y_array in dataframe[col].iteritems():
                xx=[x] * len(y_array)
                result_x= np.concatenate([result_x, xx])
                result_y = np.concatenate([result_y,y_array])
            #n=sum(y)
            amp=1 # or amp = max(result_y)-min(result_y)
            mean = sum(result_x)/len(result_x)
            #mean = sum(x*(y-min(y)))/n
            #sigma = sum((y-min(y))*(x-mean)**2)/n'''
            pars  = model.make_params( amp = 1, 
                                    cen = mean, 
                                    wid = 1, 
                                    slope=0, 
                                    intercept=1)
            result = model.fit(result_y, pars, x=result_x)
            #print(result.fit_report())
            best_parameters.append(result.best_values)

            if plot == True:
                plt.plot(result_x, result_y,         'bo')
                plt.plot(result_x, result.init_fit, 'k--')
                plt.plot(result_x, result.best_fit, 'r-')
                uncertainty = result.eval_uncertainty(sigma=3)
                plt.fill_between(result_x, result.best_fit-uncertainty, result.best_fit+uncertainty, color="#ABABAB", label='3-$\sigma$ uncertainty band')
                plt.show()

    elif "SINGLE EMBRYO" in fitting:
        for i, row in enumerate(dataframe.itertuples()):
            y = np.array(list(row)[1:])
            pars  = model.make_params( amp = 1, 
                                    cen = mean, 
                                    wid = 1, 
                                    slope=0, 
                                    intercept=1)
            result = model.fit(result_y, pars, x=x)
            best_parameters.append(result.best_values)

            if plot:
                plt.plot(x, y,         'bo')
                plt.plot(x, result.init_fit, 'k--')
                plt.plot(x, result.best_fit, 'r-')
                uncertainty = result.eval_uncertainty(sigma=3)
                plt.fill_between(x, result.best_fit-uncertainty, result.best_fit+uncertainty, color="#ABABAB", label='3-$\sigma$ uncertainty band')
                plt.show()
    
    else: 
        raise Exception("This is not a valid fitting option!")
    
    return best_parameters

def fitParameters(best_parameters_df,model,plot=True): #NOT YET IMPLEMENTED
    return 0



