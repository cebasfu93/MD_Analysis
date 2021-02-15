import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import Normalize
from boxplot_2d import boxplot_2d
from scipy.optimize import curve_fit

Z = 17
RS = 666
mpl.rcParams['axes.linewidth'] = 1.8  # set the value globally
mpl.rcParams['xtick.major.width'] = 1.6  # set the value globally
mpl.rcParams['ytick.major.width'] = 1.6  # set the value globally
mpl.rcParams['xtick.major.size'] = 4  # set the value globally
mpl.rcParams['ytick.major.size'] = 4  # set the value globally
pd.set_option('display.max_rows', 20)


# GENERIC FUNCITONS
def exponential(x, A, B, C):
    """
    Standard exponential function (used for fittings)
    """
    return A * np.exp(-x * B) + C


def exponential_solvex(y, A, B, C):
    """
    Standard exponential function solving for x
    """
    return 1 / B * (np.log(A / (y - C)))


def center_bins(bins):
    """
    Converts n bin edges into n-1 centered bins.
    """
    return 0.5 * (bins[1:] + bins[:-1])


# IMPORT AND CLEANING FUNCTIONS
def clean_input_data(data, target, resSTD=False):
    """
    Converts raw data imported by import_events() (STD or Water STD) and makes it a
    pandas dataframe indexed by chemical positions of the reference and target, and the residue id
    of the reference and target.

    Target is the key of the data dictionary to process.

    The resulting dataframe reports the starting and finishing times of each binding event, the duration,
    the distance to the anchor and wether the event was turned on when the simulation finished.

    Setting resSTD to True, imports data according to .sfu files that consider Hs as distinguishable.
    That is, less columns are imported (because they are absent in the imported file).
    """
    colnames = ["rtarget", "tmin", "tmax", "dt", "dist", "alive"]
    dtypes = {"rtarget": int,
              "tmin": float,
              "tmax": float,
              "dt": float,
              "dist": float,
              "alive": bool}
    if not resSTD:
        colnames = ["aref", "atarget", "rref"] + colnames
        dtypes["aref"] = int
        dtypes["atarget"] = int
        dtypes["rref"] = int

    df = pd.DataFrame(data, columns=colnames)
    df = df.astype(dtypes)
    df["tmin"] /= 1000  # ps to nm
    df["tmax"] /= 1000  # ps to nm
    df["dt"] /= 1000  # ps to nm
    if not resSTD:
        ndx_cols = ["rref", "rtarget", "aref", "atarget"]
        df.set_index(ndx_cols, inplace=True)
        df.sort_values(ndx_cols, inplace=True)
    return df


def import_wshell(fname):
    """
    Imports text file with water exchange rate between shells. The columns should be organized as followed:
    i) Distance of the water from the gold core center of mass
    ii) Water exchange rate
    """
    f = open(fname, "r")
    fl = f.readlines()
    f.close()

    data = []
    for line in tqdm(fl):
        if "#" not in line:
            data.append(line.split())
    data = np.array(data, dtype='float')
    clean = {'dist': data[:, 0],
             'exc': data[:, 1]
             }
    return clean


def import_pistacking(fname):
    """
    Imports text file with pi-stacking data. The columns should be organized as followed:
    i) Distance between centroids of the aromatic rings stacked (nm)
    ii) Tilt angle, that is, the angle between the aromatic rings' planes (deg)
    iii) Phase (i.e. offset) angle between the aromatic rings. This is the angle between
    the plane of one ring and the vector connecting the centroids of the rings
    """
    f = open(fname, "r")
    fl = f.readlines()
    f.close()

    data = []
    for line in tqdm(fl):
        if "#" not in line:
            data.append(line.split())
    data = np.array(data, dtype='float')
    clean = {'dist': data[:, 0],
             'tilt': data[:, 1],
             'phase': data[:, 2]}
    return clean


def import_events(fname, ignore_ns=0.0, resSTD=False):
    """
    Reads STD or Water STD text file into a dictionary with one element for each target analyte in the file.
    It call clean_input_data() to convert each value of the dictionary into a pandas dataframe.
    Setting resSTD to True, imports data according to .sfu files that consider Hs as distinguishable.
    """
    tndx = 6
    if resSTD:
        tndx = 3
    ignore_ps = ignore_ns * 1000
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
            if float(ls[tndx]) > ignore_ps:
                if "***" not in line:
                    data[key].append(ls + [False])
                else:
                    data[key].append(ls[:-1] + [True])
    for key in data.keys():
        data[key] = clean_input_data(data[key], key[:3], resSTD=resSTD)
    return data


# FUNCTIONS FOR COMPRESSING CHEMICAL SIMILARITY, EQUIVALENT POSITIONS AND RESIDUES
def touch(i, j):
    """
    Return True if the binding event i overlaps at any point with j.
    An overlap at a single point is still considered an overlap.
    """
    if i.tmin <= j.tmin and j.tmin <= i.tmax:
        return True
    elif i.tmin <= j.tmax and j.tmax <= i.tmax:
        return True
    elif i.tmin >= j.tmin and i.tmax <= j.tmax:
        return True
    else:
        return False


def clean_subdf(x):
    """
    Takes a dataframe x and assigns the minimum starting time to tmin,
    the maximum finishing time to tmax, and
    a linearly weighted distance according to the overlap of each binding event to dist.

    The resulting dataframe is redundant (all the rows are the same event).
    This function is to be followed by drop_duplicates().
    """
    tmin = x.tmin.min()
    tmax = x.tmax.max()
    weighted = ((x.dt * x.dist) / x.dt.sum()).sum()
    x = x.assign(tmin=tmin, tmax=tmax, dist=weighted, dt=tmax - tmin)
    return x


def clean_group(df):
    """
    Consecutive or overlapping binding events of the dataframe df are merged with clean_subdf() and touch().

    The resulting dataframe has redundant entries.
    That is, the overlapped events are combined into copies of the same, elongated event.
    """
    if len(df) > 1:
        contacts = np.array([[touch(i, j) for m, j in df.iterrows()] for n, i in df.iterrows()])
        pivots = [i for i in range(1, len(contacts)) if np.sum(contacts[:i, i]) == 0] + [len(contacts)]
        for pi, pf in zip(pivots[:-1], pivots[1:]):
            df.iloc[pi:pf] = clean_subdf(df.iloc[pi:pf])
    return df


def compress_chemical_positions(data, propname=None):
    """
    Takes the data dataframe of STD or Water STD events imported by import_events().
    It merges consecutive or overlapping binding events that share a property propname.

    If propname is none, then the identical chemical positions are compressed.

    The resulting dataframe does not have propname (it collapses upon grouping)
    """
    groupcols = list(data.index.names)
    data = data.reset_index()
    data.sort_values("tmin", inplace=True)
    try:
        groupcols.remove(propname)
        data.drop(propname, axis=1, inplace=True)
    except ValueError:
        print("Removing chemical redundancy in binding events")
    tqdm.pandas()
    data = data.groupby(groupcols).progress_apply(clean_group)
    data.drop_duplicates(inplace=True)
    data.set_index(groupcols, inplace=True)
    data.sort_values(groupcols, inplace=True)
    return data


# PLOTTING
def plot_wshell(wshell, frac_c=0.95, savefig=False, prefix='test'):
    """
    Plots the water exchange rate between shells and it fits an exponential function for values greater than 0
    It shows straight lines when the exchange rate is frac_c times the long-rage value of the fit
    That is, C in the A * np.exp(-x * B) + C fit
    """
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    ax.tick_params(labelsize=Z)
    ax.set_xlim(0, 4.4)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel('Distance from gold C.O.M. (nm)', fontsize=Z)
    ax.set_ylabel('Water exchange', fontsize=Z)
    dist = wshell['dist'] / 10  # A to nm

    fit_mask = wshell['exc'] > 0
    fit_x = dist[fit_mask]
    fit_y = wshell['exc'][fit_mask]
    popt, _ = curve_fit(exponential, fit_x, fit_y, p0=(-4, 1, 0.6))
    print("{:.2f} e^-{:.2f}x + {:.2f}".format(*popt))

    ax.plot(dist, exponential(dist, *popt), lw=2, c='k', ls='--', label='Exponential fit')
    ax.plot(dist, wshell['exc'], lw=2.5, c=(1, 0.1, 0.3), alpha=0.6, label='MD simulations')

    mark_y = popt[2] * frac_c
    mark_x = exponential_solvex(mark_y, *popt)
    print("{:.1f}% --> ({:.2f}, {:.2f})".format(frac_c * 100, mark_x, mark_y))
    ax.axvline(mark_x, c='k', lw=0.7)
    ax.axhline(mark_y, c='k', lw=0.7)
    ax.legend(fontsize=Z, handletextpad=0.4)
    if savefig:
        plt.savefig(prefix + "_expfit.png", format='png',
                    bbox_inches='tight', dpi=300, transparent=True)
        plt.savefig(prefix + "_expfit.svg", format='svg',
                    bbox_inches='tight')
    plt.show()
    plt.close()


def plot_pi_angles_dist(pistack, d_max=0.5, sk=1, savefig=False, prefix='test'):
    """
    Scatters the tilt and phase angles, coloring each point according to the distance between the centroids

    d_max is the threshold distance between centroids for the contact to be considered as a pi-pi interaction
    sk is the stride for extracting the data (for when there are too many points to visualize)
    """
    keys = pistack.keys()
    subplots_kw = {'xlim': (0, 90), 'ylim': (0, 90)}
    fig, axs = plt.subplots(figsize=(15, 4.5), ncols=3, nrows=1, subplot_kw=subplots_kw,
                            gridspec_kw={'hspace': 0, 'wspace': 0.15})
    for ax in axs:
        ax.set_xlabel("Tilt (deg)", fontsize=Z)
        ax.tick_params(labelsize=Z)
        ax.set_xticks(np.linspace(0, 90, 4))
        ax.set_yticks(np.linspace(0, 90, 4))
    axs[0].set_ylabel("Offset (deg)", fontsize=Z)
    for k, key in enumerate(keys):
        mask = pistack[key]['dist'] <= d_max
        data = {key_pi: val[mask] for key_pi, val in pistack[key].items()}
        axs[k].set_title(key, fontsize=Z)
        cax = axs[k].scatter(data['tilt'][::sk], data['phase'][::sk], s=1, alpha=0.3,
                             c=data['dist'][::sk], cmap=cm.terrain, vmin=0.3, vmax=d_max)
    a = plt.axes([0.92, 0.125, 0.03, 0.75])
    cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0.3, vmax=d_max), cmap=cm.terrain),
                        cax=a, ax=a, ticks=np.linspace(0.3, d_max, 3))
    cbar.ax.tick_params(labelsize=Z)
    cbar.ax.set_ylabel("Centroid distance (nm)", fontsize=Z)
    if savefig:
        plt.savefig(prefix + "_angdist.png", format='png',
                    bbox_inches='tight', dpi=300, transparent=True)
    plt.show()
    plt.close()


def plot_pi_angles_hist(pistack, d_max=0.5, sk=1, savefig=False, prefix='test'):
    """
    Makes 2D histogram for the tilt and phase angles
    hmax is the maximum number of events per bin (only matters for the coloring)
    The histogram uses 90 bins on each direction. This is important to easily visualize with imshow

    d_max is the threshold distance between centroids for the contact to be considered as a pi-pi interaction
    sk is the stride for extracting the data (for when there are too many points to visualize)
    """
    hmax = 200
    keys = pistack.keys()
    cmap = cm.ocean_r
    subplots_kw = {'xlim': (0, 90), 'ylim': (0, 90)}
    fig, axs = plt.subplots(figsize=(15, 4.5), ncols=3, nrows=1,
                            subplot_kw=subplots_kw, gridspec_kw={'hspace': 0.0, 'wspace': 0.15})
    for ax in axs:
        ax.set_xlabel("Tilt (deg)", fontsize=Z)
        ax.tick_params(labelsize=Z)
        ax.set_xticks(np.linspace(0, 90, 4))
        ax.set_yticks(np.linspace(0, 90, 4))
    axs[0].set_ylabel("Offset (deg)", fontsize=Z)
    for k, key in enumerate(keys):
        mask = pistack[key]['dist'] <= d_max
        data = {key_pi: val[mask] for key_pi, val in pistack[key].items()}
        axs[k].set_title(key, fontsize=Z)
        h, x, y = np.histogram2d(data['tilt'][::sk], data['phase'][::sk],
                                 bins=90, range=[[0, 90], [0, 90]])

        # Converts number of events to density of events (deg^-2). 1**2 because each bin has dimensions 1 deg X 1 deg
        h = h / 1**2
        print(np.max(h))
        axs[k].imshow(h.T, interpolation="gaussian", cmap=cmap, vmin=0, vmax=hmax)
    a = plt.axes([0.92, 0.125, 0.03, 0.75])
    cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=hmax),
                                          cmap=cmap), cax=a, ax=a, ticks=np.linspace(0, hmax, 5))
    cbar.ax.tick_params(labelsize=Z)
    cbar.ax.set_ylabel("Number of events", fontsize=Z)
    if savefig:
        plt.savefig(prefix + "_anghist.png", format='png',
                    bbox_inches='tight', dpi=300, transparent=True)
    plt.show()
    plt.close()


def plot_box2d_angles(pistack, colors={}, d_max=0.5, sk=1, savefig=False, prefix='test'):
    """
    Makes 2D boxplot for the tilt and phase angles
    The boxplot is done with a function from StackOverFlow

    colors is a dictionary with the same keys as pistack with the color for each boxplot
    d_max is the threshold distance between centroids for the contact to be considered as a pi-pi interaction
    sk is the stride for extracting the data (for when there are too many points to visualize)
    """
    keys = pistack.keys()
    subplots_kw = {'xlim': (0, 90), 'ylim': (0, 90)}
    fig, axs = plt.subplots(figsize=(15, 4.5), ncols=3, nrows=1,
                            subplot_kw=subplots_kw, gridspec_kw={'hspace': 0.0, 'wspace': 0.15})
    for ax in axs:
        ax.set_xlabel("Tilt (deg)", fontsize=Z)
        ax.tick_params(labelsize=Z)
        ax.set_xticks(np.linspace(0, 90, 4))
        ax.set_yticks(np.linspace(0, 90, 4))
    axs[0].set_ylabel("Offset (deg)", fontsize=Z)
    for k, key in enumerate(keys):
        mask = pistack[key]['dist'] <= d_max
        data = {key_pi: val[mask] for key_pi, val in pistack[key].items()}
        axs[k].set_title(key, fontsize=Z)
        boxplot_2d(data['tilt'][::sk], data['phase'][::sk], c=colors.get(key, 'b'), ax=axs[k])
    if savefig:
        plt.savefig(prefix + "_box2d.png", format='png',
                    bbox_inches='tight', dpi=300, transparent=True)
    plt.show()
    plt.close()


def plot_btime(key, btimes, thresh_ns=25, ignore_ns=0.5, xlim=(0, 25), ylim=(1, 2.5), starson=False):
    """
    Scatters dist vs. dt for the dataframe btimes[key] ignoring the events shorter than ignore_ns.
    It also plots a histogram of dt truncating the histogram between ignore_ns and thresh_ns.

    xlim and ylim set the limits of the plots' axes.

    If starson is True, it draws black crosses on the events that were alive when the simulation finished.
    """
    fig, axs = plt.subplots(figsize=(12, 3), ncols=2, nrows=1, subplot_kw={
                            'xlim': xlim}, gridspec_kw={'wspace': 0.25})
    print(key)
    data = btimes[key]
    data_masked = data[data["dt"] > ignore_ns]
    stars = data_masked.loc[data_masked["alive"]]
    axs[0].set_ylim(ylim)
    axs[0].tick_params(labelsize=Z)
    axs[0].set_title(key, fontsize=Z)
    axs[0].scatter(data_masked["dt"], data_masked["dist"], s=2, alpha=0.5, c=data_masked["dist"])
    if starson:
        axs[0].scatter(stars["dt"], stars["dist"], s=10, marker='x', c='k')
    axs[0].set_xlabel('Residence time (ns)', fontsize=Z)
    counts, bins = np.histogram(data_masked["dt"], bins=np.linspace(*xlim, 50))
    axs[-1].fill_between(bins[1:], counts, alpha=0.2)
    axs[-1].plot(bins[1:], counts, lw=2, label=key)
    axs[-1].set_ylabel("Events", fontsize=Z)
    axs[-1].tick_params(labelsize=Z)
    axs[-1].set_ylim(0, 20)
    axs[-1].legend(fontsize=Z - 4)
    axs[-1].set_xlabel('Residence time (ns)', fontsize=Z)
    axs[0].set_ylabel('Mean distance\nto Au COM (nm)', fontsize=Z)
    plt.show()
    plt.close()


def prop_boxplot(propname, btimes, colors, ignore_ns=0.5, ylim=(0, 1), ylabel='Residence time (ns)', nticks=4, starson=False, normdistr=False,
                 savefig=False, prefix=""):
    """
    It makes a boxplot for the column propname
    in all the dataframes of btimes ignoring events shorter than ignore_ns.
    It reports the number of samples on each boxplot.

    If normdistr is True, the dataframes in btimes are resampled to the size of the smallest dataframe.

    If starson is True, it draws black crosses on the events that were alive when the simulation finished.

    ylim and ylabel set the limits and labels of the y axis in the plot, respectively.
    nticks set the number of ticks on the y axis of the plot.
    """
    # toplot = ['DOP', 'DON', 'SER', 'SEN', 'PHC', 'PHN', 'PHE', 'PHA']
    # labels = ['Dop(+)', 'Dop', 'Ser(+)', 'Ser', 'Phe(+)', 'Phe', 'Phe(+/-)', 'Phe(-)']
    toplot = ['PHE', 'DOP', 'SER']
    labels = toplot

    bp = dict(linestyle='-', lw=2, color='k', facecolor='r')
    fp = dict(marker='o', ms=8, ls='none', mec='k', mew=1, alpha=0.1)
    mp = dict(ls='-', lw=1.5, color='k')
    cp = dict(ls='-', lw=2, color='k')
    wp = dict(ls='-', lw=2, color='k')
    nmin = []
    for key in toplot:
        data = btimes[key]
        data_masked = data.loc[data["dt"] > ignore_ns]
        nmin.append(len(data_masked))
    nmin = min(nmin)
    if normdistr:
        labels = [lab + '\n[{}]'.format(nmin) for lab, key in zip(labels, toplot)]
    else:
        labels = [lab + '\n[{}]'.format(len(btimes[key].loc[btimes[key]["dt"] > ignore_ns]))
                  for lab, key in zip(labels, toplot)]
    # fig, ax = plt.subplots(figsize=(len(toplot)*1.5,4), ncols=1, nrows=1, sharex=True)
    fig, ax = plt.subplots(figsize=(6.3, len(toplot) * 0.84), ncols=1, nrows=1, sharex=True)
    # ax.axvspan(-0.5,1.5, color=(0.9,0.95,1.0), zorder=1)
    # ax.axvspan(1.5, 3.5, color=(1.0,0.95,0.9), zorder=1)
    # ax.axvspan(3.5, 7.5, color=(0.98,1.0,0.98), zorder=1)
    ax.tick_params(labelsize=Z)
    # ax.set_ylabel(ylabel, fontsize=Z)
    ax.set_xlabel(ylabel, fontsize=Z)
    for i, (key, lab) in enumerate(zip(toplot, labels)):
        data = btimes[key]
        data_masked = data.loc[data["dt"] > ignore_ns].sample(nmin, random_state=RS)
        stars = data_masked.loc[data_masked["alive"]]
        # bpl = ax.boxplot(data_masked.loc[:,propname], positions=[i], whis=[0,90], widths=0.6, patch_artist=True, boxprops=bp, flierprops=fp, \
        # medianprops=mp, capprops=cp, whiskerprops=wp)
        bpl = ax.boxplot(data_masked.loc[:, propname], positions=[i], whis=[
                         0, 90], widths=0.7, patch_artist=True, boxprops=bp, flierprops=fp, medianprops=mp, capprops=cp, whiskerprops=wp, vert=False)
        bpl['boxes'][0].set_facecolor(colors[key])
        bpl['fliers'][0].set_markerfacecolor(colors[key])
        if starson:
            # ax.scatter([i]*len(stars), stars.loc[:,propname], marker='x', c='k', s=30, zorder=10)
            ax.scatter(stars.loc[:, propname], [i] * len(stars), marker='x', c='k', s=30, zorder=10)
    # ax.set_xticks(np.linspace(0,len(toplot)-1, len(toplot), dtype='int'))
    # ax.set_xticklabels(labels)
    # ax.set_yticks(np.linspace(*ylim, nticks))
    # ax.set_ylim(ylim)
    ax.set_yticks(np.linspace(0, len(toplot) - 1, len(toplot), dtype='int'))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.linspace(*ylim, nticks))
    ax.set_xlim(ylim)
    if savefig:
        plt.savefig(prefix + "_box.png", format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.show()
    plt.close()


def plot_cluster(toplot, btimes, colors, ignore_ns=0.5, xlim=(0, 20), ylim=(1, 2.5), starson=False):
    """
    Scatters dist vs. dt for dataframes the dataframes in btimes with keys in the list toplot.
    The plot ignores events shorter than ignore_ns.
    colors is a dictionary with keys toplot containing the color for each scattered dataframe.

    xlim and ylim set the limits of the plots' axes.

    If starson is True, it draws black crosses on the events that were alive when the simulation finished.
    """
    fig, ax = plt.subplots(figsize=(7, 3), nrows=1, ncols=1)
    ax.tick_params(labelsize=Z)
    ax.set_xlabel('Residence time (ns)', fontsize=Z)
    ax.set_ylabel('Mean distance\nto Au COM (nm)', fontsize=Z)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for key in toplot:
        data = btimes[key]
        data_masked = data.loc[data["dt"] > ignore_ns]
        stars = data_masked.loc[data_masked["alive"]]
        ax.errorbar(data_masked["dt"], data_masked["dist"], alpha=0.3, label=key,
                    fmt='o', mew=0.0, mec='k', color=colors[key], ms=3)
        if starson:
            ax.scatter(stars["dt"], stars["dist"], s=20, c='k', marker='x', zorder=40)
    ax.legend(fontsize=Z, loc='center', bbox_to_anchor=(1.2, 0.5), markerscale=4)
    plt.show()
    plt.close()


def plot_cumevents(toplot, btimes, colors, ignore_ns=0.5, xlim=(0, 5), nticks=3, nbins=80, ylim=None, normed=False, savefig=False, prefix=""):
    """
    It makes a cumulative histogram from the column dt of the
    dataframes in btimes with keys in the list toplot.
    The plot ignores the events shorter than ignore_ns.
    colors is a dictionary with keys toplot containing the color for each scattered dataframe.

    If normed is True, then the cumulative counts are normalized so the maximum is curve is 1.

    If the tuple xlim has xlim[0] > xlim[1], then the histogram is cumulated backwards.

    nbins is the number of bins in which to histogram the data.
    xlim sets the range on which to histogram the data.
    xlim and ylim set the limits of the plots' axes.
    """
    ns = []
    fig, ax = plt.subplots(figsize=(4.5, 3.5), nrows=1, ncols=1)
    ax.tick_params(labelsize=Z)
    ax.set_xlabel("Residence time (ns)", fontsize=Z)
    ax.set_ylabel("Cumulated events", fontsize=Z)
    for key in toplot:
        data = btimes[key]
        data_masked = data.loc[data["dt"] > ignore_ns]
        counts, bins = np.histogram(data_masked["dt"], bins=nbins,
                                    density=False, range=(min(xlim), max(xlim)))
        bins = center_bins(bins)
        if xlim[1] < xlim[0]:
            bins, counts = np.flip(bins), np.flip(counts)
        cumcount = np.cumsum(counts)
        print(key, bins[np.argmin(np.abs(cumcount - 250))])
        if normed:
            cumcount = cumcount / np.max(cumcount)
        ax.errorbar(bins, cumcount, c=colors[key], label=key, lw=2.5)
        ns.append(len(data_masked))
    ax.set_xlim(xlim)
    # ax.set_xlim(0.6)
    if ylim is not None:
        ax.set_ylim(ylim)
        ax.set_yticks(np.linspace(*ylim, 3))
    else:
        ax.set_ylim(0, max(ns) * 1.05)
    ax.set_xticks(np.linspace(*xlim, nticks))
    # ax.legend(fontsize=Z, loc='center', bbox_to_anchor=(1.2,0.5))
    if savefig:
        plt.savefig(prefix + "_cumev.png", format='png', bbox_inches='tight', dpi=300, transparent=True)
        plt.savefig(prefix + "_cumev.svg", format='svg', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_positions(keys, btimes, colors, ignore_ns=0.5, req_sample_size=100, normdistr=False, savefig=False, prefix=""):
    """
    It makes three plots for each dataframe of btimes with a key in the list keys.
    The colors of the three plots is determined by the dictionary colors with keys in the list keys.
    The plots ignore the events shorter than ignore_ns.

    1) The number of events occuring between each different pair of reference-target chemical positions.
    2) The total bound time between each different pair of reference-target chemical positions.
    3) The average binding time per event for each different pair of reference-target chemical positions.

    The size of the points is proportionately scaled
    and it is normalized to the biggest value plotted.
    That is, all plots 1 are normalized to the same maximum. The same for plots 2 and 3.

    Pairs with less than req_sample_size are ommitted and indicated by a black cross.

    If normdistr is True, the dataframes in btimes are resampled to the size of the smallest dataframe.
    """
    n_samples_list, size_list, speeds_list, times_list, naref_list, natarget_list = [], [], [], [], [], []
    for key in keys:
        data = btimes[key]
        data_masked = data.loc[data['dt'] > ignore_ns]
        data_grouped = data_masked.groupby(['aref', 'atarget'])
        sizes = data_grouped['dt'].size()
        times = data_grouped['dt'].sum()
        naref = len(data_masked.index.unique(level='aref'))
        natarget = len(data_masked.index.unique(level='atarget'))
        n_samples_list.append(sizes.loc[sizes > req_sample_size].min())
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
        data_masked = data.loc[data['dt'] > ignore_ns]
        data_grouped = data_masked.groupby(['aref', 'atarget'])
        if normdistr:
            speeds = data_grouped['dt'].sample(
                min(n_samples_min, data_grouped['dt'].size().min()), random_state=RS).groupby(['aref', 'atarget']).mean()
        else:
            speeds = data_grouped['dt'].mean()
        speeds_list.append(speeds.loc[data_grouped.size() > req_sample_size].max())
    speeds_max = max(speeds_list)
    # fig, all_axs = plt.subplots(figsize=(max(3*0.7*naref_max, 12), 1.3*len(keys)*0.7*natarget_max), nrows=len(keys), ncols=3,
    # subplot_kw={'xlim':(-0.5,naref_max-0.5), 'ylim':(-0.5,natarget_max-0.5)},
    # gridspec_kw={'hspace':0.2})
    fig, all_axs = plt.subplots(figsize=(1.3 * len(keys) * 0.7 * natarget_max, max(3 * 0.7 * naref_max, 12)), nrows=len(keys), ncols=3,
                                subplot_kw={'ylim': (-0.5, naref_max - 0.5),
                                            'xlim': (-0.5, natarget_max - 0.5)},
                                gridspec_kw={'hspace': 0.2})
    all_axs[0, 0].set_title('Total number of events', fontsize=Z, pad=20)
    all_axs[0, 1].set_title('Total binding time', fontsize=Z, pad=20)
    for ax in all_axs[-1, :]:
        # ax.set_xlabel("Ligand position", fontsize=Z)
        ax.set_xlabel("Analyte position", fontsize=Z)
    for key, axs, naref, natarget in zip(keys, all_axs, naref_list, natarget_list):
        # axs[0].set_ylabel("{} position".format(key), fontsize=Z)
        axs[0].set_ylabel("Ligand position", fontsize=Z)
        for ax in axs:
            ax.tick_params(labelsize=Z)
        data = btimes[key]
        data_masked = data.loc[data['dt'] > ignore_ns]
        data_grouped = data_masked.groupby(['aref', 'atarget'])
        sizes = data_grouped['dt'].size()
        times = data_grouped['dt'].sum()
        if normdistr:
            speeds = data_grouped['dt'].sample(min(n_samples_min, sizes.min()), random_state=RS).groupby([
                'aref', 'atarget']).mean()
            all_axs[0, 2].set_title("Average binding time\n[{}]".format(
                n_samples_min), fontsize=Z, pad=20)
        else:
            speeds = data_grouped['dt'].mean()
            all_axs[0, 2].set_title("Average binding time", fontsize=Z, pad=20)
        for (n, szs), (m, tms), (o, sps) in zip(sizes.iteritems(), times.iteritems(), speeds.iteritems()):
            # axs[0].errorbar(*n, fmt='o', ms=50*szs/sizes_max, c=colors[key], mec='k', mew=2)
            # axs[1].errorbar(*n, fmt='o', ms=50*tms/times_max, c=colors[key], mec='k', mew=2)
            axs[0].errorbar(n[1], n[0], fmt='o', ms=50 * szs / sizes_max, c=colors[key], mec='k', mew=2)
            axs[1].errorbar(n[1], n[0], fmt='o', ms=50 * tms / times_max, c=colors[key], mec='k', mew=2)
            if szs > req_sample_size:
                # axs[2].errorbar(*n, fmt='o', ms=50*sps/speeds_max, c=colors[key], mec='k', mew=2)
                axs[2].errorbar(n[1], n[0], fmt='o', ms=50 * sps / speeds_max,
                                c=colors[key], mec='k', mew=2)
            else:
                # axs[2].errorbar(*n, fmt='x', ms=8, c='gray', mec='k', mew=2)
                axs[2].errorbar(n[1], n[0], fmt='x', ms=8, c='gray', mec='k', mew=2)
    if savefig:
        plt.savefig(prefix + "_positions.png", format='png',
                    bbox_inches='tight', dpi=300, transparent=True)
        plt.savefig(prefix + "_positions.svg", format='svg',
                    bbox_inches='tight', dpi=300, transparent=False)
        svg2emf(prefix + "_positions.svg")
    plt.show()
    plt.close()


def plot_barpositions(keys, btimes, colors, ignore_ns=0.5, req_sample_size=100, ylim1=None, ylim2=None):
    """
    It plots the total number of events and total bound time
    at each chemical position of the references and targets.
    The plots ignore events shorter than ignore_ns.
    One curve is plotted for each dataframe in the dictionary btimes with keys in the list keys.
    The colors of the curves is determined by the dictionary colors with keys in the list keys.

    Pairs with less than req_sample_size are ommitted and indicated by a black cross.

    ylim1 and ylim2 set the limits of the y axes for
    the total number of events and total bound time, respectively.
    """
    naref_list, natarget_list = [], []
    for key in keys:
        data = btimes[key]
        data_masked = data.loc[data['dt'] > ignore_ns]
        data_grouped = data_masked.groupby(['aref', 'atarget'])
        naref = len(data_masked.index.unique(level='aref'))
        natarget = len(data_masked.index.unique(level='atarget'))
        naref_list.append(naref)
        natarget_list.append(natarget)
    naref_max = max(naref_list)
    natarget_max = max(natarget_list)
    fig, axs = plt.subplots(figsize=(max(2 * naref_max, 12), 9), nrows=2,
                            ncols=2, gridspec_kw={'hspace': 0.4, 'wspace': 0.4})
    for ax in axs[:, 0]:
        ax.set_ylabel("Total number\nof events", fontsize=Z)
    for ax in axs[:, 1]:
        ax.set_ylabel("Total binding\ntime (ns)", fontsize=Z)
    for ax in axs[0, :]:
        ax.set_xlabel("Thiol position", fontsize=Z)
        ax.set_xlim(-0.5, naref_max - 0.5)
    for ax in axs[-1, :]:
        ax.set_xlabel("Analyte position", fontsize=Z)
        ax.set_xlim(-0.5, natarget_max - 0.5)
    for key in keys:
        data = btimes[key]
        data_masked = data.loc[data['dt'] > ignore_ns]
        for col, ax in zip(['aref', 'atarget'], axs):
            grouped = data_masked.groupby(col)['dt']
            sizes = grouped.size()
            times = grouped.sum()
            ax[0].errorbar(sizes.index, sizes, c=colors[key], fmt='o-', ms=10, mec='k', mew=1.5, lw=1.5)
            ax[1].errorbar(times.index, times, c=colors[key], fmt='o-', ms=10, mec='k', mew=1.5, lw=1.5)
    for ax in axs.flatten():
        ax.tick_params(labelsize=Z)
        ax.set_ylim((0, ax.get_ylim()[1]))
    plt.show()
    plt.close()
