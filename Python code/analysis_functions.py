from modules import *
import data




##########################################################
####### Plottinbg and Data Visualization Functions #######
##########################################################



'''
Calculate mean, variance, total births per year, and number of names used per year
type(rcm): DataFrame - Rank count information for male names
type(rcf): DataFrame - Rank count information for female names
'''
def plot_the_basics(rcm,rcf,fignum = 1,verbose=False):
    global years
    plt.figure(fignum)
    plt.subplot(2,2,1)
    rcm.sum().plot(color='b')
    rcf.sum().plot(color='r')
    plt.title('Total Number of Births per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Births')
    plt.legend(['Male','Female'],loc=0)
    plt.subplot(2,2,3)
    rcm.count().plot(color='b')
    rcf.count().plot(color='r')
    plt.title('Number of Unique Names Used per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Unique Names')
    plt.legend(['Male','Female'],loc=0)
    plt.subplot(1,2,2)
    rcm.mean().plot(color='b')
    rcf.mean().plot(color='r')
    plt.title('Average Use of Name per Year')
    plt.xlabel('Year')
    plt.ylabel('Average Number of Births per Name')
    plt.legend(['Male','Female'],loc=0)
    plt.show()


'''
Plotting the combined likelihood of the highest N used names, where N is calculated from a list of percentages
specified in lims
type(rcm): DataFrame
type(rcf): DataFrame
type(lims): list
type(figs): list
'''
def plot_top_perc_name_stats(rcm,rcf,lims=[0.001,0.005,0.01,0.05],fignum = 1):
    plt.figure(fignum)
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
    ax = plt.subplot(1,2,1)
    for i in range(len(nrcm_lims)):
        nrcm_lims[i].plot(ax=ax)
    plt.legend(['Top '+str(lims[i]*100)+'% ' for i in range(len(nrcm_lims))],loc=0)
    plt.title('Probability of Top Name Usage per Year for Males')
    plt.ylabel('Probability of Usage')
    plt.xlabel('Year')
    ax = plt.subplot(1,2,2)
    for i in range(len(nrcf_lims)):
        nrcf_lims[i].plot(ax=ax)
    plt.legend(['Top '+str(lims[i]*100)+'% ' for i in range(len(nrcf_lims))],loc=0)
    plt.title('Probability of Top Name Usage per Year for Females')
    plt.ylabel('Probability of Usage')
    plt.xlabel('Year')
    plt.show()


def plot_zipf_comparison(rc,fignum=1):
    years = np.arange(start=rc.columns[0],stop=rc.columns[-1])
    year_cm = np.array([[(years[i]-years.min())/(years.max()-years.min()) for i in range(len(years))],
                        [0]*len(years),
                        [(years.max()-years[i])/(years.max()-years.min()) for i in range(len(years))]])
    for i,col in enumerate(rc.columns):
        plt.plot(np.log10(rc.index))

def zipf(n,b=1.07):
    return 1/n**b


def plot_rank_count_values(rc,fignum=1,use_intervals=True,interval=5,lim = 50,ismale=True):
    global mymap
    interval=int(min(max(interval,1),rc.columns.size))
    intervals=[]
    cur=rc.columns[0]

    if not use_intervals:
        interval = 1
    while cur+interval < rc.columns[-1]:
        intervals.append([cur,cur+interval])
        cur+=interval
    intervals.append([cur,rc.columns[-1]+1])
    plt.figure(fignum)
    year_cm = np.array([[(intervals[i][0]-min(intervals)[0])/(max(intervals)[0]-min(intervals)[0]) for i in range(len(intervals))],
                        [0]*len(intervals),
                        [(max(intervals)[0]-intervals[i][0])/(max(intervals)[0]-min(intervals)[0]) for i in range(len(intervals))]])
    #mymap = mcolors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])
    CS3 = plt.contourf([[0,0],[0,0]],rc.columns.values, cmap=mymap);plt.clf()
    nrc = df_norm(rc,use_lim=True,lim=lim,as_pdf=True)
    x=np.linspace(0,1,lim)
    for i in range(len(intervals)):
        rng=intervals[i]
        y = nrc[list(range(rng[0],rng[1]))][:].mean(axis=1)
        y.plot(xdata=x*y.count()/rc[list(range(rng[0],rng[1]))].count().mean(),color=year_cm[:,i],legend=False,markersize=5,markeredgecolor=year_cm[:,i])
        #y.plot(color=year_cm[:,i],legend=False)
        plt.plot(y.index[0],y.values[0],linestyle='',marker='o',markersize=5,markeredgecolor=year_cm[:,i],color=year_cm[:,i])
    max_val = 0
    ax = plt.gca()
    for line in ax.lines:
        max_val = max(max_val,np.nanmax(line.get_data()[1].data))

    #plt.axis([-1,lim,0,max_val*1.1])
    #plt.axis([0,nrc.sum(axis=0).max()*1.1,0,max_val*1.1])
    plt.axis([0,lim/rc.count(axis=0).min()*1.1,0,max_val*1.1])
    plt.colorbar(CS3)
    if ismale: gender=' for Males'
    else: gender=' for Females'
    plt.title('Probability Distribution of Top '+str(lim)+' Names vs Ratio of Top '+str(lim)+' Names to All Available Names' + gender)
    plt.ylabel('Probability of Given Name\'s Use')
    plt.xlabel('Ratio of Rank to Total Number of Names Used')
    plt.show()


def plot_usage_vs_prop(rc,fignum=1,top=50,ismale=True):
    plt.figure(fignum)
    nrc = df_norm(rc,use_lim=False,as_pdf=True,full_pdf=True)
    cl = plt.scatter(top/nrc.count(),nrc.loc[:top].sum(),s=50,c=nrc.columns.values,cmap=mymap)
    plt.grid(1)
    plt.colorbar(cl)
    if ismale: gender=' for Males'
    else: gender=' for Females'
    plt.title('% of Births with top '+str(top)+' Names vs Ratio of Top '+str(top)+' Names to All Available Names'+gender)
    plt.ylabel('Percentage of Births with one the Top '+str(top)+' Names')
    plt.xlabel('Proportion of Top '+str(top)+' Names to All Available names')
    plt.show()


def plot_top_N_diff_values(rcmdi,rcfdi,cur_fig=1,lims=[5,10,25,50],use_filt=False,window_len=5):
    nrcmdi = abs(rcmdi)
    nrcfdi = abs(rcfdi)
    plt.figure(cur_fig)
    ax1 = plt.subplot(1,2,1)
    mclrs=[[0.1,0,0.3],[0.1,0,0.4],[0.1,0,0.5],[0.1,0,0.6]]
    fclrs=[[0.25,0.05,0],[0.4,0.05,0],[0.55,0.05,0],[0.7,0.05,0]]
    ls=['-.','--','-','-.']
    Legend_decription = []
    for val in lims:
        Legend_decription.append('N = '+str(val))
        if use_filt: Legend_decription.append('N = '+str(val)+', '+ str(window_len)+' Year Average')
    prev = np.zeros(nrcmdi.columns.size)
    for i in range(len(lims)):
        x = nrcmdi[:lims[i]].sum()
        plt.plot(nrcmdi.columns, x,lw=2,c=mclrs[i])
        ax1.fill_between(nrcmdi.columns,prev,x,facecolor=mclrs[i],alpha=0.4)
        #prev = x
        if use_filt: # Use simple FIR filter to smooth results
            res = signal.filtfilt(b=np.ones(window_len)/window_len,a=[1],x=x)
            plt.plot(nrcmdi.columns,res,ls=ls[i],lw=5,c=[0.05,0,0.95])

    plt.legend(Legend_decription,loc=0)
    plt.grid(1)
    plt.title('Summation of top N Names\' Yearly Difference in Rank for Males')
    plt.xlabel('Years')
    plt.ylabel('Top N Names\' Yearly Rank Difference')
    ax3 = plt.subplot(1,2,2)
    prev = np.zeros(nrcfdi.columns.size)
    for i in range(len(lims)):
        x = nrcfdi[:lims[i]].sum()
        plt.plot(nrcfdi.columns,nrcfdi[:lims[i]].sum(),lw=2,c=fclrs[i])
        ax3.fill_between(nrcfdi.columns,prev,x,facecolor=fclrs[i],alpha=0.4)
        #prev = x
        if use_filt: # Use simple FIR filter to smooth results
            res = signal.filtfilt(b=np.ones(window_len)/window_len,a=[1],x=x)
            plt.plot(nrcfdi.columns, res,ls=ls[i],lw=5,c=[0.95,0,0.05])

    plt.legend(Legend_decription,loc=0)
    plt.grid(1)
    plt.title('Summation of top N Names\' Yearly Difference in Rank for Females')
    plt.xlabel('Years')
    plt.ylabel('Top N Names\' Yearly Rank Difference')
    # Changing Axes
    ymax = max(nrcmdi[:lims[-1]].sum().iloc[-1],nrcfdi[:lims[-1]].sum().iloc[-1])
    plt.subplot(1,2,1)
    plt.axis([nrcmdi.columns[0],nrcmdi.columns[-1],0,ymax*1.25])
    plt.subplot(1,2,2)
    plt.axis([nrcfdi.columns[0],nrcfdi.columns[-1],0,ymax*1.25])
    plt.show()


def plot_top_name_trajectories(rid_df,name_usage_df,N=10,cur_fig=1):
    topN_st = rid_df[name_usage_df.columns[0]][0:N]
    topN_end = rid_df[name_usage_df.columns[-1]][0:N]

    stonly = list(set(topN_st)-set(topN_end))
    endonly = list(set(topN_end)-set(topN_st))
    common = list(set(topN_end)-set(stonly)-set(endonly))
    plt.figure(cur_fig)
    plt.subplot(2,3,1)
    for id in stonly:
        plt.plot(name_usage_df.columns,name_usage_df[:].loc[int(id)]/name_usage_df.count(),c='b',marker='o',ms=5,lw=2)
    plt.title('Top '+str(N)+' '+str(rid_df.columns[0])+' Name Rank of Trajectories')
    plt.ylabel('Percentage Rank Position')
    plt.xlabel('Years')
    plt.axis([rid_df.columns[0],rid_df.columns[-1], 0, 0.85])
    plt.grid(1)
    plt.subplot(2,3,2)
    for id in endonly:
        plt.plot(name_usage_df.columns,name_usage_df[:].loc[int(id)]/name_usage_df.count(),c='r',marker='>',ms=5)
    plt.title('Top '+str(N)+' '+str(rid_df.columns[-1])+' Name Rank of Trajectories')
    plt.ylabel('Percentage Rank Position')
    plt.xlabel('Years')
    plt.axis([rid_df.columns[0],rid_df.columns[-1], 0, 0.85])
    plt.grid(1)
    plt.subplot(2,3,3)
    for id in common:
        plt.plot(name_usage_df.columns,name_usage_df[:].loc[int(id)]/name_usage_df.count(),c=[0.65,0,0.65],lw=5,marker='o',ms=5)
    plt.title('Top '+str(N)+' Names\'s Rank Trajectories (In Both '+str(rid_df.columns[0])+' and '+str(rid_df.columns[-1])+') ')
    plt.ylabel('Percentage Rank Position')
    plt.xlabel('Years')
    plt.grid(1)

    plt.axis([rid_df.columns[0],rid_df.columns[-1], 0, 0.85])
    plt.subplot(2,1,2)
    for id in endonly:
        plt.plot(name_usage_df.columns,name_usage_df[:].loc[int(id)]/name_usage_df.count(),c='r',marker='>',ms=5)
    for id in stonly:
        plt.plot(name_usage_df.columns,name_usage_df[:].loc[int(id)]/name_usage_df.count(),c='b',marker='o',ms=5,lw=2)
    for id in common:
        plt.plot(name_usage_df.columns,name_usage_df[:].loc[int(id)]/name_usage_df.count(),c=[0.65,0,0.65],marker='',lw=5)
    plt.title('Top '+str(N)+' Name Rank of Trajectories')
    plt.ylabel('Percentage Rank Position')
    plt.xlabel('Years')
    plt.grid(1)
    plt.show()


def plot_counts_historic_rank(ridm,ridf,N=1000,cur_fig=1):
    countsm = np.zeros(N)
    for i in range(N):
        countsm[i] = len(set(ridm.loc[i]))
    plt.plot(list(range(1,N+1)),countsm,marker='o',ms=5,lw=2,c='b')
    countsf = np.zeros(N)
    plt.hold(1)
    for i in range(N):
        countsf[i] = len(set(ridf.loc[i]))
    plt.plot(list(range(1,N+1)),countsf,marker='o',ms=5,lw=2,c='r')
    plt.grid(1)
    plt.title('Count of Different Names at Specified Rank Values')
    plt.legend(['Males','Females'],loc=0)
    plt.ylabel('Number of Names')
    plt.xlabel('Rank')
    plt.show()


def plot_mf_name_histograms(id2attrm,id2attrf,cur_fig=1):
    xm = np.array(list(map(lambda x: x[0],id2attrm.values()))) # Letter
    ym = np.array(list(map(lambda x: x[1],id2attrm.values()))) # Length
    zm = np.array(list(map(lambda x: x[2],id2attrm.values()))) # Length of Name
    xf = np.array(list(map(lambda x: x[0],id2attrf.values()))) # Letter
    yf = np.array(list(map(lambda x: x[1],id2attrf.values()))) # Length
    zf = np.array(list(map(lambda x: x[2],id2attrf.values()))) # Length of Name

    plt.figure(cur_fig)
    ax1 = plt.subplot(1,3,1)
    a = ax1.hist(xm,bins=len(set(xm)),lw=3,fc=(0.05,0,0.85,0.4),normed=1,align='left')
    ax1.hist(xf,bins=len(set(xf)),lw=3,fc=(0.85,0,0.05,0.4),normed=1,align='left')
    plt.xticks(a[1])
    labels = [chr(int(val)) for val in np.arange(65,91)]
    ax1.set_xticklabels(labels)
    x1,x2,y1,y2 = plt.axis()
    plt.axis([ord('A')-.5,ord('Z')-.5,y1,y2])
    plt.legend(['Male','Female'])
    plt.title('Histogram of First Letters')
    plt.xlabel('Letters (A-Z)')
    plt.ylabel('Probability')

    ax2 = plt.subplot(1,3,2)
    ax2.hist(ym,bins=len(set(ym)),lw=3,fc=(0.05,0,0.85,0.4),normed=1)
    ax2.hist(yf,bins=len(set(yf)),lw=3,fc=(0.85,0,0.05,0.4),normed=1)
    plt.legend(['Male','Female'])
    plt.title('Histogram of Syllable Count')
    plt.xlabel('Number of Syllables')
    plt.ylabel('Probability')

    ax3 = plt.subplot(1,3,3)
    ax3.hist(zm,bins=len(set(zm)),lw=3,fc=(0.05,0,0.85,0.4),normed=1)
    ax3.hist(zf,bins=len(set(zf)),lw=3,fc=(0.85,0,0.05,0.4),normed=1)
    plt.legend(['Male','Female'])
    plt.title('Histogram of Name Length')
    plt.xlabel('Length of Name')
    plt.ylabel('Probability')
    plt.show()


def plot_heatmap_len_vs_letter(id2attr,cur_fig=1,gender='Males'):
    x = np.array(list(map(lambda x: x[0],id2attr.values()))) # Letter
    z = np.array(list(map(lambda x: x[2],id2attr.values()))) # Length of Name
    plt.figure(cur_fig)
    #Heatmap
    ax1 = plt.subplot2grid((8,8),(0,1),colspan=7,rowspan=7)
    xhist,xedges= np.histogram(x, bins=len(set(x)),normed=1)
    plt.hist2d(x, z, bins=(len(set(x)),len(set(z))),cmap=cm.YlOrRd_r,normed=1)
    labels = [chr(int(val)) for val in np.arange(65,91)]
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    x1,x2,y1,y2 = plt.axis()
    plt.axis([ord('A'),ord('Z'),y1,y2])
    plt.title('Heatmap of Most Used Name Length to First Letter for ' + gender)
    # histograms
    ax2 = plt.subplot2grid((8,8),(0,0),colspan=1,rowspan=7)
    plt.hist(z,bins=len(set(z)),orientation='horizontal',facecolor='r',align='left')
    plt.ylabel('Number of Syllables')
    ax2.set_xticklabels([])
    ax3 = plt.subplot2grid((8,8),(7,1),colspan=7,rowspan=1)
    a = plt.hist(x,bins=len(set(x)),facecolor='r',align='left')
    plt.xticks(a[1])
    plt.xlabel('Letter (A-Z)')
    ax3.set_xticklabels(labels)
    ax3.set_yticklabels([])
    x1,x2,y1,y2 = plt.axis()
    plt.axis([ord('A')-.5,ord('Z')-.5,y1,y2])
    plt.show()


def plot_heatmap_len_vs_syllables(id2attr,cur_fig=1,gender='Males'):
    x = np.array(list(map(lambda x: x[1],id2attr.values()))) # syllable
    y = np.array(list(map(lambda x: x[2],id2attr.values()))) # Length
    fig = plt.figure(cur_fig)
    ax1 = plt.subplot2grid((8,8),(0,1),colspan=7,rowspan=7)
    plt.hist2d(x, y, bins=(len(set(x)),len(set(y))),cmap=cm.YlOrRd_r,normed=1)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.title('Heatmap of Histogram Name Length and Number of Syllables for '+ gender)
    # histograms
    ax2 = plt.subplot2grid((8,8),(0,0),colspan=1,rowspan=7)
    ax2.set_xticklabels([])
    plt.hist(y,bins=len(set(y)),orientation='horizontal',facecolor='r')
    plt.xlabel('Number of Letters in Name')
    ax3 = plt.subplot2grid((8,8),(7,1),colspan=7,rowspan=1)
    plt.hist(x,bins=len(set(x)),facecolor='r')
    plt.xlabel('Number of Syllables')
    ax3.set_yticklabels([])
    plt.show()


def plot_mf_name_letter_counts(id2nm,id2nf,cur_fig=1):
    freqs = pd.DataFrame(data.import_letters(),columns=['General'])
    freqs = freqs.sort_index()
    mfreqd = letter_freq(id2nm.values())
    mfreqs = pd.Series(data=list(mfreqd.values()),index=list(mfreqd.keys())).sort_index().to_frame(name='Males')
    ffreqd = letter_freq(id2nf.values())
    ffreqs = pd.Series(data=list(ffreqd.values()),index=list(ffreqd.keys())).sort_index().to_frame(name='Females')
    freqs.index = mfreqs.index
    freqs = pd.concat([freqs,mfreqs,ffreqs],join='outer',axis=1)
    nfreqs = df_norm(freqs,use_lim=False,as_pdf=True)
    ax_mf = plt.subplot2grid((3,8),(0,0),colspan=3,rowspan=1)
    plt.tight_layout()
    nfreqs[['Males','Females']].plot.bar(ax=ax_mf,color=[(0.05,0,0.85,0.4),(0.85,0,0.05,0.4)],width=0.8,align='center')
    ax_mg = plt.subplot2grid((3,8),(1,0),colspan=3,rowspan=1)
    nfreqs[['General','Males']].plot.bar(ax=ax_mg,color=[(0,0.75,0.05,0.4),(0.05,0,0.85,0.4)],width=0.8,align='center')
    ax_fg = plt.subplot2grid((3,8),(2,0),colspan=3,rowspan=1)
    nfreqs[['General','Females']].plot.bar(ax=ax_fg,color=[(0,0.75,0.05,0.4),(0.85,0,0.05,0.4)],width=0.8,align='center')
    ax_mfg = plt.subplot2grid((3,8),(0,3),colspan=5,rowspan=3)
    nfreqs.plot.bar(ax=ax_mfg,color=[(0,0.75,0.05,0.4),(0.05,0,0.85,0.4),(0.85,0,0.05,0.4)],width=0.9)
    plt.show()
    plt.figure(cur_fig+1)
    ax = plt.subplot(111)
    nfreqs.loc[list('aeiouy')].plot.bar(ax=ax,color=[(0,0.75,0.05,0.4),(0.05,0,0.85,0.4),(0.85,0,0.05,0.4)],width=0.75)
    plt.title('Vowel Use')
    plt.tight_layout()
    plt.show()


def plot_diff_hist_len_syll_data(olddf,newdf,cur_fig=1):
    plt.figure(cur_fig)

    ax2 = plt.subplot(1,2,1)
    olddf['Syllables'].plot.hist(ax=ax2,bins=len(set(olddf['Syllables'].unique())),lw=3,fc=(0.05,0,0.85,0.4),normed=1)
    newdf['Syllables'].plot.hist(ax=ax2,bins=len(set(newdf['Syllables'].unique())),lw=3,fc=(0.0,0.75,0.05,0.35),normed=1)
    plt.legend(['Top Names in 1880','Top Names in 2013'])
    plt.title('Histogram of Syllable Count')
    plt.xlabel('Number of Syllables')
    plt.ylabel('Probability')

    ax3 = plt.subplot(1,2,2)
    olddf['Length'].plot.hist(bins=len(set(olddf['Length'].unique())),lw=3,fc=(0.05,0,0.85,0.4),normed=1)
    newdf['Length'].plot.hist(bins=len(set(newdf['Length'].unique())),lw=3,fc=(0.0,0.75,0.05,0.35),normed=1)
    plt.legend(['Top Names in 1880','Top Names in 2013'])
    plt.title('Histogram of Name Length')
    plt.xlabel('Length of Name')
    plt.ylabel('Probability')
    plt.show()



#############################################################
####### Functions for Calculating New data structures #######
#############################################################

'''
Returns the difference in rankings from year to year, calculating the difference in ranking change
and the difference in name count. Also has options to load csv files representing already computed values.
'''
def calc_yearly_diff(rc,rid,load_saved=[True,True],save_file=[False,False],
                     filename=['MaleRankCountDiffIndex.csv','MaleRankCountDiffCount.csv'],verbose=False):
    # loading data
    isloaded = [False,False]
    if len(load_saved) == 2 and len(filename) == 2:
        if verbose: print("Trying to import from "+filename[0]+" and "+filename[1])
        if load_saved[0]:
            rc_diffi = data.import_csv(file=filename[0])
            if rc_diffi.__len__() > 0:
                isloaded[0] = True
        if load_saved[1]:
            rc_diffc = data.import_csv(file=filename[1])
            if rc_diffc.__len__() > 0:
                isloaded[1] = True
        if isloaded[0] and isloaded[1]:
            if verbose: print("Import successful")
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
            rc_diffi[cur_yr][i] = i - res[0]
            rc_diffc[cur_yr][i] = rc[cur_yr][i] - prev_count
        if verbose: print("Year: "+str(cur_yr))
        prev_yr = cur_yr

    #Saving data
    if len(save_file) == 2:
        if save_file[0]:
            data.export_data(df=rc_diffi,filename=filename[0],path='data')
        if save_file[1]:
            data.export_data(df=rc_diffi,filename=filename[1],path='data')
    return rc_diffi,rc_diffc



#Returns a list of each name's Ranking trajectory.
def calc_name_usage(rid,rc,nid,load_saved=[True,True],save_file=[False,False],
                    filename=['MaleNameUsageStatsIndex.csv','MaleNameUsageStatsCount.csv'],verbose=False):
    [nu_statsi,nu_statsc],isloaded = import_dfs(filenames=filename,load_saved=load_saved,verbose=verbose)
    if isloaded[0] and nu_statsi.index[0:5].tolist() != list(nid.keys())[0:5]: nu_statsi.index = list(nid.keys())
    else: nu_statsi = pd.DataFrame(index=list(nid.keys()),columns=rc.columns)
    if isloaded[1] and nu_statsc.index[0:5].tolist() != list(nid.keys())[0:5]:nu_statsc.index = list(nid.keys())
    else: nu_statsc = pd.DataFrame(index=list(nid.keys()),columns=rc.columns)
    if all(isloaded):
        if any(save_file): save_dfs(dfs=[nu_statsi,nu_statsc],filenames=filename,save_file=save_file,verbose=verbose)
        return nu_statsi, nu_statsc
    if verbose: print("Recalculating..")
    tally = 0
    if verbose: print("Starting computations")
    for id in nid.keys():
        for yr in rid.columns:
            ind,count = np.nan,np.nan
            res = rid[yr][rid[yr]==id]
            if res.size:
                ind = res.index[0]
                count = rc[yr][ind]
            if not isloaded[0]:
                nu_statsi[yr][id] = ind
            if not isloaded[1]:
                nu_statsc[yr][id] = count
        tally+=1
        perc = tally/len(nid.keys())
        if verbose and int(100*perc)%1 == 0 and perc > 0.02: data.show_status(perc)
    if verbose: print("\nFinished.")
    if any(save_file): save_dfs(dfs=[nu_statsi,nu_statsc],filenames=filename,save_file=save_file,verbose=verbose)
    return nu_statsi,nu_statsc

# Extracts name length, syllables, first letter
def calc_name_features(id2n):
    id2attr = {}
    for key in id2n.keys():
        name = id2n[key]
        ascii_val = ord(name[0])
        num_sylls = syllables(name)
        length = len(name)
        id2attr[key] = [ascii_val,num_sylls,length]
    return id2attr

# Creates counter of all characters used in list of words passed
def letter_freq(wordlist):
    freq_dict = Counter()
    for name in wordlist:
        for c in name:
            if ord(c)>=65 and ord(c)<91:
                c = chr(ord(c)+32)
            freq_dict[c]+=1
    return freq_dict



###############################
####### Basic Functions #######
###############################


# Imports filenames provided
def import_dfs(filenames,load_saved,verbose):
    dfs = [[]]*len(filenames)
    isloaded = [False]*len(filenames)
    if len(load_saved) == len(filenames) and len(load_saved)==len(dfs):
        if verbose:
            print("Trying to import from these files:")
            for file in filenames:
                print(file)
        for i in range(len(dfs)):
            if load_saved[i]:
                dfs[i] = data.import_csv(file=filenames[i])
                if dfs[i].__len__() > 0:
                    isloaded[i] = True
        if all(isloaded):
            if verbose: print("Import successful")
    return dfs,isloaded


# Saves passed DataFrames to Files with specified filenames
def save_dfs(dfs,filenames,save_file,verbose):
    if len(save_file) == len(dfs) and len(filenames) == len(dfs):
        if verbose: print("\nSaving.")
        for i in range(len(dfs)):
            if save_file[i]:
                data.export_data(df=dfs[i],filename=filenames[i],path='data')


# Normalizing DataFrame column data by column maxes and using limit
def df_norm(df,use_lim=True,lim=5000,as_pdf=False,full_pdf=True):
    if not use_lim or lim < 0 or lim > df.shape[0]:
        lim = df.shape[0]
    if full_pdf: lim2=df.index.size
    else: lim2 = lim
    if as_pdf:
        dists = df[0:lim]/df[0:lim2].sum()
    else:
        dists = df[0:lim]/df[0:lim].max()
    return dists


# Normalizing Series Data
def s_norm(s,as_pdf=False):
    if as_pdf:
        return s/s.sum()
    else:
        return s/s.max()


# Calculates number of syllables in the string variable word
def syllables(word):
    count = 0
    vowels = 'aeiouy'
    word = word.lower().strip(".:;?!")
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count+=1
    if count == 0:
        count +=1
    return count