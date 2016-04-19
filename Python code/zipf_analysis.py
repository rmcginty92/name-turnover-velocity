from modules import *
import analysis_functions as af
import scipy.stats as ss
import scipy.optimize as opt
import data



def plot_zipf_comparison(rc):
    years = np.arange(start=rc.columns[0],stop=rc.columns[-1]+1)
    year_cm = np.array([[(years[i]-years.min())/(years.max()-years.min()) for i in range(len(years))],
                        [0]*len(years),
                        [(years.max()-years[i])/(years.max()-years.min()) for i in range(len(years))]])
    for i,col in enumerate(rc.columns):
        plt.plot((rc.index+1),(rc[col].values),c=year_cm[:,i])


def zipf(n,b=1.07):
    return 1/n**b


name_mapping, rank_count, rank_id = data.import_data(save=False,try_from_csv=True)
m='M';f='F'
# Splitting all gender data into individual dataframes and dictionaries
id2nm = name_mapping['id2str'][m]
id2nf = name_mapping['id2str'][f]
n2idm = name_mapping['str2id'][m]
n2idf = name_mapping['str2id'][f]
rcm = rank_count[m]
rcf = rank_count[f]
ridm = rank_id[m]
ridf = rank_id[f]
nrcm = af.df_norm(rcm,use_lim=False,as_pdf=True)
nrcf = af.df_norm(rcf,use_lim=False,as_pdf=True)

ax = plt.subplot(1,2,1)
plot_zipf_comparison(nrcm)
plt.title('log-log comparison of Male Name Frequency and Rank Data',fontsize=16)
plt.xlabel(r'${log}_{10} {(Rank)}$',fontsize=15)
plt.ylabel(r'${log}_{10} {(Frequency)}$',fontsize=15)
years = list(range(nrcm.columns[0],nrcm.columns[100])) #specifying years for training
ydata = nrcm[years][10:5000].values.flatten(order='F')
xdata = np.tile(nrcm[years].index[10:5000]+1,nrcm[years].columns.size)
xdata = xdata[np.logical_not(np.isnan(ydata))]# removing nans
ydata = ydata[np.logical_not(np.isnan(ydata))]# removing nans
popt,pcov = opt.curve_fit(ss.zipf.logpmf,xdata,np.log10(ydata),p0=[1.5],bounds=(1.07,5))
#popt,pcov = opt.curve_fit(ss.zipf.pmf,xdata,ydata,p0=[1.5],bounds=(1.07,5))
x_zipf = np.linspace(nrcm.index.min()+1,nrcm.index.max()+1,nrcm.index.max()-nrcm.index.min()+1)
y_zipf = ss.zipf.pmf(x_zipf,a=popt[0])
plt.plot(x_zipf,y_zipf,color=(0,0.85,0,0.8),markeredgecolor=(0,0.85,0,0.8),lw=4,ms=5,marker='o')
plt.grid(1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.axis('tight')
plt.legend((ax.lines[0],ax.lines[-2],ax.lines[-1]),
           (str(nrcm.columns[0])+' Rank PMF',
            str(nrcm.columns[-1])+' Rank PMF',
            r'$\mathit{Zipfs Law, } \alpha = $' + str(popt[0])))
ax = plt.subplot(1,2,2)
plot_zipf_comparison(nrcf)
plt.title('log-log comparison of Female Name Frequency and Rank Data',fontsize=16)
plt.xlabel(r'${log}_{10} {(Rank)}$',fontsize=15)
plt.ylabel(r'${log}_{10} {(Frequency)}$',fontsize=15)
ydata = nrcf[years][10:5000].values.flatten(order='F')
xdata = np.tile(nrcf[years].index[10:5000]+1,nrcf[years].columns.size)
xdata = xdata[np.logical_not(np.isnan(ydata))]# removing nans
ydata = ydata[np.logical_not(np.isnan(ydata))]# removing nans
popt,pcov = opt.curve_fit(ss.zipf.logpmf,xdata,np.log10(ydata),p0=[1.5],bounds=(1.07,5))
#popt,pcov = opt.curve_fit(ss.zipf.pmf,xdata,ydata,p0=[1.5],bounds=(1.07,5))
x_zipf = np.linspace(nrcf.index.min()+1,nrcf.index.max()+1,nrcf.index.max()-nrcf.index.min()+1)
y_zipf = ss.zipf.pmf(x_zipf,a=popt[0])
plt.plot(x_zipf,y_zipf,color=(0,0.85,0,0.8),markeredgecolor=(0,0.85,0,0.8),lw=4,ms=5,marker='o')
plt.grid(1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.axis('tight')
plt.legend((ax.lines[0],ax.lines[-2],ax.lines[-1]),
           (str(nrcf.columns[0])+' Rank PMF',
            str(nrcf.columns[-1])+' Rank PMF',
            r'$\mathit{Zipfs Law, } \alpha = $' + str(popt[0])))
plt.show()




plt.figure(2)
nrcm =rcm# af.df_norm(rcm,use_lim=False,as_pdf=True)
nrcf =rcf# af.df_norm(rcf,use_lim=False,as_pdf=True)
ax = plt.subplot(1,2,1)
plot_zipf_comparison(nrcm)
plt.title('log-log comparison of Male Name Frequency and Rank Data',fontsize=16)
plt.xlabel(r'${log}_{10} {(Rank)}$',fontsize=15)
plt.ylabel(r'${log}_{10} {(Frequency)}$',fontsize=15)
plt.grid(1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.axis('tight')
plt.legend((ax.lines[0],ax.lines[-1]),
           (str(nrcm.columns[0])+' Rank PMF',
            str(nrcm.columns[-1])+' Rank PMF'))
ax = plt.subplot(1,2,2)
plot_zipf_comparison(nrcf)
plt.title('log-log comparison of Female Name Frequency and Rank Data',fontsize=16)
plt.xlabel(r'${log}_{10} {(Rank)}$',fontsize=15)
plt.ylabel(r'${log}_{10} {(Frequency)}$',fontsize=15)
plt.grid(1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.axis('tight')
plt.legend((ax.lines[0],ax.lines[-1]),
           (str(nrcf.columns[0])+' Rank PMF',
            str(nrcf.columns[-1])+' Rank PMF',))
plt.show()