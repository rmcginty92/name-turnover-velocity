import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data
from analysis_functions import *

years = np.zeros(1)

def main(verbose=False):
    global years
    ''' Importing data and setting variables '''
    name_id, rank_count, rank_id = data.import_data(save=False,try_from_csv=True)
    m='M';f='F'
    clean()
    # Splitting all gender data into individual dataframes and dictionaries
    nidm = name_id[m]; nidf = name_id[f]
    rcm = rank_count[m]; rcf = rank_count[f]
    ridm = rank_id[m]; ridf = rank_id[f]
    years = np.arange(start=1880,stop=2013+1)
    rcm.columns = years;rcf.columns = years;ridm.columns = years;ridf.columns = years
    #normalizing count of birth names for mean/variance analysis
    n_rcm = df_norm(rcm)
    n_rcf = df_norm(rcf)
    ''' Starting basic analysis of rankings and count data '''
    print("\nStarting Analysis...")
    cur_fig = 1
    if 0: # calculating basic stuff
        print("Plotting total births per year and number of Unique names used per year...")
        calc_the_basics(rcm,rcf)
        cur_fig+=1
    if 0:
        print("Plotting probability of top proportions of names used per year...")
        plot_top_perc_name_stats(rcm,rcf,figs=[cur_fig,cur_fig+1])
        cur_fig+=2

    ''' Analysis of yearly differences in rankings '''
    if 1:
        rcm_d = calc_yearly_diff(rcm,ridm,load_saved=[True,False],save_file=[True,True],
                                 filename=['MaleRankCountDiffIndex.csv','MaleRankCountDiffCount.csv'])
        rcf_d = calc_yearly_diff(rcf,ridf,load_saved=[True,False],save_file=[True,True],
                                 filename=['FemaleRankCountDiffIndex.csv','FemaleRankCountDiffCount.csv'])



'''
Calculate mean, variance, total births per year, and number of names used per year
type(rcm): DataFrame
type(rcf): DataFrame
'''
def calc_the_basics(rcm,rcf):
    plt.figure(1)
    plt.subplot(2,1,1)
    rcm.sum().plot(color='b')
    rcf.sum().plot(color='r')
    plt.title('Total Number of Births per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Births')
    plt.legend(['Male','Female'])
    plt.subplot(2,1,2)
    rcm.count().plot(color='b')
    rcf.count().plot(color='r')
    plt.title('Number of Unique Names Used per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Unique Names')
    plt.legend(['Male','Female'])
    year_cm = np.array([[(years[i]-years.min())/(years.max()-years.min()) for i in range(len(years))],
                       [0]*len(years),
                       [(years.max()-years[i])/(years.max()-years.min()) for i in range(len(years))]])
    plt.show()

'''
Plotting the combined likelihood of the highest N used names, where N is calculated from a list of percentages
specified in lims

type(rcm): DataFrame
type(rcf): DataFrame
type(lims): list
type(figs): list
'''
def plot_topN_name_stats(rcm,rcf,lims=[0.005,0.01,0.05,0.1,0.15,0.25],figs = [1,2]):
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
    nrcm_lims = [df_norm(rcm,lim=int(count_m)) for val in lims]
    nrcf_lims = [df_norm(rcf,lim=val) for val in lims]
    for i in range(len(nrcm_lims)):
        plt.subplot(N,M,i+1)
        (nrcm_lims[i]/nrcm.sum(axis=0)).sum().plot(c='b')
        (nrcf_lims[i]/nrcf.sum(axis=0)).sum().plot(c='r')
        plt.title('Probability of First N '+str(lims[i])+' Names per Year')
        plt.ylabel('Probability of First '+str(lims[i])+' Names')
        plt.xlabel('Year')
        plt.legend(['Male','Female'])
    plt.show()
    plt.figure(figs[1])
    for i in range(len(nrcm_lims)):
        plt.subplot(N,M,i+1)
        (nrcm_lims[i]).std().plot(c='b')
        (nrcf_lims[i]).std().plot(c='r')
        plt.title('Variance of First N '+str(lims[i])+' Names per Year')
        plt.ylabel('Variance of First '+str(lims[i])+' Names')
        plt.xlabel('Year')
        plt.legend(['Male','Female'])
    plt.show()


def clean():
    plt.close('all')


if __name__ == "__main__":
    main()



