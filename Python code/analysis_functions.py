import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data


'''
Returns the difference in rankings from year to year, calculating the difference in ranking change
and the difference in name count. Also has options load
'''
def calc_yearly_diff(rc,rid,load_saved=[True,True],save_file=[False,False],filename=['MaleRankCountDiffIndex.csv','MaleRankCountDiffCount.csv']):
    # loading data
    isloaded = [False,False]
    if len(load_saved) == 2 and len(filename) == 2:
        if load_saved[0]:
            rc_diffi = data.import_csv(file=filename[0])
            if rc_diffi.size > 0:
                isloaded[0] = True
        if load_saved[1]:
            rc_diffc = data.import_csv(file=filename[1])
            if rc_diffc.size > 0:
                isloaded[1] = True
        if isloaded[0] and isloaded[1]:
            return rc_diffi, rc_diffc
    if not isloaded[0]: rc_diffi = pd.DataFrame(index=rc.index,columns=rc.columns[1:],dtype='int')
    if not isloaded[1]: rc_diffc = pd.DataFrame(index=rc.index,columns=rc.columns[1:],dtype='int')
    prev_yr = rc.columns[0]
    # Calculating first index of NaN values to improve looping efficiency when searching previous year data
    lens = rc.count(axis=0)
    # Starting calculations
    for cur_yr in rc.columns[1:]:
        for i in rc.index[:lens[cur_yr]]:
            prev_count = 0
            cur_id = rid[cur_yr][i]
            res = rid[prev_yr][rid[prev_yr]==cur_id].index
            if res.size:
                prev_count = rc[prev_yr][res[0]]
            else:
                res = [lens[cur_yr]]
            rc_diffi[cur_yr][i] = rc[cur_yr][i] - prev_count
            rc_diffc[cur_yr][i] = i - res[0]
        print("Year: "+str(cur_yr))
        prev_yr = cur_yr

    #Saving data
    if len(save_file) == 2:
        if save_file[0]:
            data.export_data(df=rc_diffi,filename=filename[0],path='data')
        if save_file[1]:
            data.export_data(df=rc_diffi,filename=filename[1],path='data')
    return rc_diffi,rc_diffc


def plot_top_perc_name_stats(rcm,rcf,lims=[0.001,0.005,0.01,0.1],figs = [1]):
    plt.figure(figs[0])
    L = len(lims)
    N = int(np.ceil(np.sqrt(L)))
    while L%N !=0:
        N+=1
    M = int(L/N)
    nrcm = df_norm(rcm,use_lim=False,as_pdf=True)
    nrcf = df_norm(rcf,use_lim=False,as_pdf=True)
    count_m = nrcm.count()
    count_f = nrcf.count()
    nrcm_lims = [[]]*len(lims)
    nrcf_lims = [[]]*len(lims)
    for i in range(len(lims)):
        nrcm_lims[i] = pd.Series(index=nrcm.columns)
        for column in nrcm.columns:
            nrcm_lims[i][column] = nrcm[column][:int(lims[i]*count_m[column])].sum()
        nrcf_lims[i] = pd.Series(index=nrcf.columns)
        for column in nrcf.columns:
            nrcf_lims[i][column] = nrcf[column][:int(lims[i]*count_f[column])].sum()
    for i in range(len(nrcm_lims)):
        plt.subplot(N,M,i+1)
        nrcm_lims[i].plot(c='b')
        nrcf_lims[i].plot(c='r')
        plt.title('Probability of top '+str(lims[i]*100)+'% Names per Year')
        plt.ylabel('Probability of top '+str(lims[i]*100)+'% Names')
        plt.xlabel('Year')
        plt.legend(['Male','Female'])
    plt.show()


# Normalizing DataFrame column data by column maxes and using limit
def df_norm(df,use_lim=True,lim=5000,as_pdf=False):
    if not use_lim or lim < 0 or lim > df.shape[0]:
        lim = df.shape[0]
    if as_pdf:
        dists = df[0:lim]/df[0:lim].sum()
    else:
        dists = df[0:lim]/df[0:lim].max()
    return dists

# Normalizing Series Data
def s_norm(s,as_pdf=False):
    if as_pdf:
        return s/s.sum()
    else:
        return s/s.max()
