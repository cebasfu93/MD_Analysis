import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
Z = 20
RS = 666
mpl.rcParams['axes.linewidth'] = 1.8  #set the value globally
pd.set_option('display.max_rows', 20)

#GENERIC FUNCITONS
def center_bins(bins):
    return 0.5*(bins[1:] + bins[:-1])

#IMPORT AND CLEANING FUNCTIONS
def clean_input_data(data, target):
    colnames = ["aref", "atarget", "rref", "rtarget", "tmin", "tmax", "dt", "dist", "alive"]
    dtypes = {"aref"   : int,
             "atarget" : int,
             "rref"    : int,
             "rtarget" : int,
             "tmin"    : float,
             "tmax"    : float,
             "dt"      : float,
             "dist"    : float,
             "alive"   : bool}#,
    df = pd.DataFrame(data, columns=colnames)
    df = df.astype(dtypes)
    df["tmin"] /= 1000 #ps to nm
    df["tmax"] /= 1000 #ps to nm
    df["dt"] /= 1000 #ps to nm
    ndx_cols = ["rref", "rtarget", "aref", "atarget"]
    df.set_index(ndx_cols, inplace=True)
    df.sort_values(ndx_cols, inplace=True)
    return df

def import_events(fname, ignore_ns=0.0):
    ignore_ps = ignore_ns*1000
    print("Importing {}".format(fname))
    f = open(fname, "r")
    fl = f.readlines()
    f.close()
    data = defaultdict(list)
    for line in tqdm(fl):
        if "TARGET GROUP" in line:
            key = line.split()[-1]
        if "#" not in line and "@" not in line:
            ls = line.split()
            if float(ls[6])>ignore_ps:
                if "***" not in line:
                    data[key].append(ls+[False])
                else:
                    data[key].append(ls[:-1] + [True])
    for key in data.keys():
        data[key] = clean_input_data(data[key], key[:3])
    return data

#FUNCTIONS FOR COMPRESSING CHEMICAL SIMILARITY, EQUIVALENT POSITIONS AND RESIDUES
def touch(i, j):
    if i.tmin <= j.tmin and j.tmin <= i.tmax:
        return True
    elif i.tmin <= j.tmax and j.tmax <= i.tmax:
        return True
    elif i.tmin >= j.tmin and i.tmax <= j.tmax:
        return True
    else:
        return False

def clean_subdf(x):
    tmin = x.tmin.min()
    tmax = x.tmax.max()
    weighted = ((x.dt*x.dist)/x.dt.sum()).sum()
    x = x.assign(tmin=tmin, tmax=tmax, dist=weighted, dt=tmax-tmin)
    return x

def clean_group(df, DT=0.02):
    if len(df) > 1:
        contacts = np.array([[touch(i,j) for m, j in df.iterrows()] for n, i in df.iterrows()])
        pivots = [i for i in range(1, len(contacts)) if np.sum(contacts[:i,i])==0] + [len(contacts)]
        for pi, pf in zip(pivots[:-1], pivots[1:]):
            df.iloc[pi:pf] = clean_subdf(df.iloc[pi:pf])
    return df

def compress_chemical_positions(data, propname=None):
    """If propname is none, then the identical chemical positions are compressed.
    This should be done compressing any of the properties"""
    groupcols = list(data.index.names)
    data = data.reset_index()
    data.sort_values("tmin", inplace=True)
    try:
        groupcols.remove(propname)
        data.drop(propname, axis=1, inplace=True)
    except:
        print("Removing chemical redundancy in binding events")
    tqdm.pandas()
    data = data.groupby(groupcols).progress_apply(clean_group)
    data.drop_duplicates(inplace=True)
    data.set_index(groupcols, inplace=True)
    data.sort_values(groupcols, inplace=True)
    return data

#PLOTTING
def plot_btime(key, btimes, thresh_ns=25, ignore_ns=0.5, xlim=(0,25), ylim=(1,2.5), starson=False):
    fig, axs = plt.subplots(figsize=(12,3), ncols=2, nrows=1, subplot_kw={'xlim':xlim}, gridspec_kw={'wspace':0.25})
    print(key)
    data = btimes[key]
    data_masked = data[data["dt"]>ignore_ns]
    stars = data_masked.loc[data_masked["alive"]]
    axs[0].set_ylim(ylim)
    axs[0].tick_params(labelsize=Z)
    axs[0].set_title(key, fontsize=Z)
    axs[0].scatter(data_masked["dt"],data_masked["dist"], s=2, alpha=0.5, c=data_masked["dist"])
    if starson:
        axs[0].scatter(stars["dt"], stars["dist"], s=10, marker='x', c='k')
    axs[0].set_xlabel('Residence time (ns)', fontsize=Z)
    counts, bins = np.histogram(data_masked["dt"], bins=np.linspace(*xlim,50))
    axs[-1].fill_between(bins[1:], counts, alpha=0.2)
    axs[-1].plot(bins[1:], counts, lw=2, label=key)
    axs[-1].set_ylabel("Events", fontsize=Z)
    axs[-1].tick_params(labelsize=Z)
    axs[-1].set_ylim(0,20)
    axs[-1].legend(fontsize=Z-4)
    axs[-1].set_xlabel('Residence time (ns)', fontsize=Z)
    axs[0].set_ylabel('Mean distance\nto Au COM (nm)', fontsize=Z)
    plt.show()
    plt.close()

def prop_boxplot(propname, btimes, colors, ignore_ns=0.5, ylim=(0,1), ylabel='Residence time (ns)', nticks=4, starson=False, normdistr=False):
    toplot = ['DOP', 'DON', 'SER', 'SEN', 'PHC', 'PHN', 'PHE', 'PHA']
    labels = ['Dop(+)', 'Dop', 'Ser(+)', 'Ser', 'Phe(+)', 'Phe', 'Phe(+/-)', 'Phe(-)']
    bp = dict(linestyle='-', lw=2, color='k', facecolor='r')
    fp = dict(marker='o', ms=8, ls='none', mec='k', mew=1, alpha=0.1)
    mp = dict(ls='-', lw=1.5,color='k')
    cp = dict(ls='-', lw=2, color='k')
    wp = dict(ls='-', lw=2, color='k')
    nmin = []
    for key in toplot:
        data = btimes[key]
        data_masked = data.loc[data["dt"]>ignore_ns]
        nmin.append(len(data_masked))
    nmin = min(nmin)
    if normdistr:
        labels = [lab+'\n[{}]'.format(nmin) for lab, key in zip(labels, toplot)]
    else:
        labels = [lab+'\n[{}]'.format(len(btimes[key].loc[btimes[key]["dt"]>ignore_ns])) for lab, key in zip(labels, toplot)]
    fig, ax = plt.subplots(figsize=(12,4), ncols=1, nrows=1, sharex=True)
    ax.axvspan(-0.5,1.5, color=(0.9,0.95,1.0), zorder=1)
    ax.axvspan(1.5, 3.5, color=(1.0,0.95,0.9), zorder=1)
    ax.axvspan(3.5, 7.5, color=(0.98,1.0,0.98), zorder=1)
    ax.tick_params(labelsize=Z-4)
    ax.set_ylabel(ylabel, fontsize=Z)
    for i, (key, lab) in enumerate(zip(toplot, labels)):
        data = btimes[key]
        data_masked = data.loc[data["dt"]>ignore_ns].sample(nmin, random_state=RS)
        stars = data_masked.loc[data_masked["alive"]]
        bpl = ax.boxplot(data_masked.loc[:,propname], positions=[i], whis=[0,90], widths=0.6, patch_artist=True, boxprops=bp, flierprops=fp, medianprops=mp, capprops=cp, whiskerprops=wp)
        bpl['boxes'][0].set_facecolor(colors[key])
        bpl['fliers'][0].set_markerfacecolor(colors[key])
        if starson:
            ax.scatter([i]*len(stars), stars.loc[:,propname], marker='x', c='k', s=30, zorder=10)
    ax.set_xticks(np.linspace(0,len(toplot)-1, len(toplot), dtype='int'))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.linspace(*ylim, nticks))
    ax.set_ylim(ylim)
    plt.show()
    plt.close()

def plot_cluster(toplot, btimes, colors, ignore_ns=0.5, xlim=(0,20), ylim=(1,2.5), starson=False):
    fig, ax = plt.subplots(figsize=(7,3), nrows=1, ncols=1)
    ax.tick_params(labelsize=Z)
    ax.set_xlabel('Residence time (ns)', fontsize=Z)
    ax.set_ylabel('Mean distance\nto Au COM (nm)', fontsize=Z)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for key in toplot:
        data = btimes[key]
        data_masked = data.loc[data["dt"]>ignore_ns]
        stars = data_masked.loc[data_masked["alive"]]
        ax.errorbar(data_masked["dt"], data_masked["dist"], alpha=0.3, label=key, fmt='o', mew=0.0, mec='k', color=colors[key], ms=3)
        if starson:
            ax.scatter(stars["dt"], stars["dist"], s=20, c='k', marker='x', zorder=40)
    ax.legend(fontsize=Z, loc='center', bbox_to_anchor=(1.2,0.5), markerscale=4)
    plt.show()
    plt.close()

def plot_cumevents(toplot, btimes, colors, ignore_ns=0.5, xlim=(0,5), nbins=80, ylim=None, normed=False):
    ns = []
    fig, ax = plt.subplots(figsize=(7,2.5), nrows=1, ncols=1)
    ax.tick_params(labelsize=Z)
    ax.set_xlabel("Residence time (ns)", fontsize=Z)
    ax.set_ylabel("Cumulated\nevents", fontsize=Z)
    for key in toplot:
        data = btimes[key]
        data_masked = data.loc[data["dt"]>ignore_ns]
        counts, bins = np.histogram(data_masked["dt"], bins=nbins, density=False, range=(min(xlim), max(xlim)))
        bins = center_bins(bins)
        if xlim[1] < xlim[0]:
            bins, counts = np.flip(bins), np.flip(counts)
        cumcount = np.cumsum(counts)
        if normed:
            cumcount = cumcount/np.max(cumcount)
        ax.errorbar(bins, cumcount, c=colors[key], label=key, lw=2.5)
        ns.append(len(data_masked))
    ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
        ax.set_yticks(np.linspace(*ylim, 3))
    else:
        ax.set_ylim(0, max(ns)*1.05)
    ax.legend(fontsize=Z, loc='center', bbox_to_anchor=(1.2,0.5))
    plt.show()
    plt.close()

def plot_positions(keys, btimes, colors, ignore_ns=0.5, req_sample_size=100, normdistr=False):
    n_samples_list, size_list, speeds_list, times_list, naref_list, natarget_list = [], [], [], [], [], []
    for key in keys:
        data = btimes[key]
        data_masked = data.loc[data['dt']>ignore_ns]
        data_grouped = data_masked.groupby(['aref', 'atarget'])
        sizes = data_grouped['dt'].size()
        times = data_grouped['dt'].sum()
        naref = len(data_masked.index.unique(level='aref'))
        natarget = len(data_masked.index.unique(level='atarget'))
        n_samples_list.append(sizes.loc[sizes>req_sample_size].min())
        size_list.append(sizes.max())
        times_list.append(times.max())
        naref_list.append(naref)
        natarget_list.append(natarget)
    n_samples_min = min(n_samples_list)
    sizes_max = max(size_list)
    times_max = max(times_list)
    naref_max = max(naref_list)
    natarget_max = max(natarget_list)
    
    for key in keys:
        data = btimes[key]
        data_masked = data.loc[data['dt']>ignore_ns]
        data_grouped = data_masked.groupby(['aref', 'atarget'])
        if normdistr:
            speeds = data_grouped['dt'].sample(min(n_samples_min, data_grouped['dt'].size().min()), random_state=RS).groupby(['aref', 'atarget']).mean()
        else:
            speeds = data_grouped['dt'].mean()
        speeds_list.append(speeds.loc[data_grouped.size()>req_sample_size].max())
    speeds_max = max(speeds_list)
    fig, all_axs = plt.subplots(figsize=(max(3*0.7*naref_max, 12), 1.3*len(keys)*0.7*natarget_max), nrows=len(keys), ncols=3,
                                subplot_kw={'xlim':(-0.5,naref_max-0.5), 'ylim':(-0.5,natarget_max-0.5)},
                                gridspec_kw={'hspace':0.4})
    all_axs[0,0].set_title('Total number of events', fontsize=Z, pad=20)
    all_axs[0,1].set_title('Total binding time', fontsize=Z, pad=20)
    for key, axs, naref, natarget in zip(keys, all_axs, naref_list, natarget_list):
        axs[0].set_ylabel("{} position".format(key), fontsize=Z)
        for ax in axs:
            ax.tick_params(labelsize=Z)
            ax.set_xlabel("Thiol position", fontsize=Z)
        data = btimes[key]
        data_masked = data.loc[data['dt']>ignore_ns]
        data_grouped = data_masked.groupby(['aref', 'atarget'])
        sizes = data_grouped['dt'].size()
        times = data_grouped['dt'].sum()
        if normdistr:
            speeds = data_grouped['dt'].sample(min(n_samples_min, sizes.min()), random_state=RS).groupby(['aref', 'atarget']).mean()
            all_axs[0,2].set_title("Average binding time\n[{}]".format(n_samples_min), fontsize=Z, pad=20)
        else:
            speeds = data_grouped['dt'].mean()
            all_axs[0,2].set_title("Average binding time", fontsize=Z, pad=20)
        for (n, szs), (m, tms), (o, sps) in zip(sizes.iteritems(), times.iteritems(), speeds.iteritems()):
            axs[0].errorbar(*n, fmt='o', ms=50*szs/sizes_max, c=colors[key], mec='k', mew=2)
            axs[1].errorbar(*n, fmt='o', ms=50*tms/times_max, c=colors[key], mec='k', mew=2)
            if szs > req_sample_size:
                axs[2].errorbar(*n, fmt='o', ms=50*sps/speeds_max, c=colors[key], mec='k', mew=2)
            else:
                axs[2].errorbar(*n, fmt='x', ms=8, c='gray', mec='k', mew=2)
    plt.show()
    plt.close()

def plot_barpositions(keys, btimes, colors, ignore_ns=0.5, req_sample_size=100, ylim1=(0,12000), ylim2=(0, 10000)):
    naref_list, natarget_list = [], []
    for key in keys:
        data = btimes[key]
        data_masked = data.loc[data['dt']>ignore_ns]
        data_grouped = data_masked.groupby(['aref', 'atarget'])
        naref = len(data_masked.index.unique(level='aref'))
        natarget = len(data_masked.index.unique(level='atarget'))
        naref_list.append(naref)
        natarget_list.append(natarget)
    naref_max = max(naref_list)
    natarget_max = max(natarget_list)
    fig, axs = plt.subplots(figsize=(max(2*naref_max, 12), 9), nrows=2, ncols=2, gridspec_kw={'hspace':0.4, 'wspace':0.4})
    for ax in axs[:,0]:
        ax.set_ylabel("Total number\nof events", fontsize=Z)
    for ax in axs[:,1]:
        ax.set_ylabel("Total binding\ntime (ns)", fontsize=Z)
    for ax in axs[0,:]:
        ax.set_xlabel("Thiol position", fontsize=Z)
        ax.set_xlim(-0.5, naref_max-0.5)
    for ax in axs[-1,:]:
        ax.set_xlabel("Analyte position", fontsize=Z)
        ax.set_xlim(-0.5, natarget_max-0.5)
    for key in keys:
        data = btimes[key]
        data_masked = data.loc[data['dt']>ignore_ns]
        for col, ax in zip(['aref', 'atarget'], axs):
            grouped = data_masked.groupby(col)['dt']
            sizes = grouped.size()
            times = grouped.sum()
            ax[0].errorbar(sizes.index, sizes, c=colors[key], fmt='o-', ms=10, mec='k', mew=1.5, lw=1.5)
            ax[1].errorbar(times.index, times, c=colors[key], fmt='o-', ms=10, mec='k', mew=1.5, lw=1.5)
    for ax in axs.flatten():
        ax.tick_params(labelsize=Z)
        ax.set_ylim((0,ax.get_ylim()[1]))
    plt.show()
    plt.close()
 
        