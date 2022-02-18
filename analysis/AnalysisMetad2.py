import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.optimize import curve_fit
from scipy.stats import gamma
from scipy import stats
import matplotlib.font_manager as fm

matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2
matplotlib.use('Agg')


def plot_transition(distances,  out_dir, id):
    plt.figure(figsize=(6, 5))
    h = sns.kdeplot(distances,label=r"$Transition\ Distance$")
    plt.xlabel(r"$Distance$")
    plt.legend(loc='best')
    out_path = os.path.join(out_dir, f"transition_{id}.png")
    plt.savefig(out_path)
    plt.close()
    return 1


def estimate_height(rate, kbT, rate_0):
    height = kbT*np.log(rate/rate_0)
    return height


def read_trajectory(src_dir, force_list, file_template="colvar.out", num_runs=20, f_stop=-2.00, f_start=0.0, n_force=11):
    data = {}
    if len(force_list)==0:
        force_list = np.linspace(f_stop, f_start, n_force)
    for i in force_list:
        key = "%3.1f" % i
        data[key] = []
    for k in data:
        file_name = file_template.replace('_FORCE_', k)
        #runs = []
        for i in range(1, num_runs + 1):
            final = file_name.replace('RUN', str(i))
            fullpath = os.path.join(src_dir, final) 
            temp = np.genfromtxt(fullpath, comments='#', dtype=float)
            data[k].append(temp)
        
    return data


def plot_distance_distribution(runs, id, force, out_dir):
    plt.figure(figsize=(6, 5))
    for i,run in enumerate(runs):
        h = sns.kdeplot(run[:, 2], label=f'run: {i+1}')
    plt.xlabel(r"$Distance\ \AA$")
    plt.legend(ncol=3, loc='best')
    out_path = os.path.join(out_dir, f"rDistribution_{force}{id}.png")
    plt.savefig(out_path)
    plt.close()
    return 1


def plot_distance_trajectory(runs, id, force, out_dir, check_escape=False, dx=16):
    if check_escape==False:
        plt.figure(figsize=(6, 5))
        for i,run in enumerate(runs):
            lab = np.abs(float(force))
            plt.plot(run[:, 0], run[:, 2], label=F'run: {i+1}')
            plt.xlabel(r"$Time$")
            plt.ylabel(r"$Distance\ \AA$")
        plt.legend(ncol=3, loc='best')
        out_path = os.path.join(out_dir, f"distTrajectory_{force}{id}.png")
        plt.savefig(out_path)
        plt.close()
    else:
        print(check_escape)
        plt.figure(figsize=(6, 5))
        for i,run in enumerate(runs):
            lab = np.abs(float(force))
            if run[:, 1][-1] > dx:
                plt.plot(run[:, 0], run[:, 2], label=F'run: {i+1}')
                plt.xlabel(r"$Time$")
                plt.ylabel(r"$Distance\ \AA$")
        plt.legend(ncol=3, loc='best')
        out_path = os.path.join(out_dir, f"distTrajectory_{force}.png")
        plt.savefig(out_path)
        plt.close()
    return 1


def read_metad(src_dir, force_list, file_template="metad.out", num_runs=20):
    acc = {}
    for i in force_list:
        key = "%3.1f" % i
        acc[key] = []
    for k in acc:
        file_name = file_template.replace('_FORCE_', k)
        info = []
        for i in range(1, num_runs + 1):
            final = file_name.replace('RUN', str(i))
            fullpath = os.path.join(src_dir, final)
            with open(fullpath) as f:
                for line in f:
                    pass
            info.append([np.float64(f) for f in line.split()])
        acc[k] = np.array(info)
    return acc


def get_escape_times(data, output_data, id):
    mean_product = []
    escape_times = {}
    for k in data:
        runs = data[k]
        product = []
        last = []
        alpha = []
        out_file = open(F"{output_data}/last_times_acc{k}{id}.dat", 'w')
        out_file.write("#force\t last_time\t alpha\t product\n")
        for run in runs:
            if run[0] > 100000000:
                scaled = (run[0]-10000) * run[2] * 1e-6
            else:
                scaled = (run[0]) * run[2] * 1e-6
            product.append(scaled)
            alpha.append(run[2])
            last.append(run[0])
            # print("%s %f %f %f" % (k, run[0], run[2], scaled))
            out_file.write("%s %f %f %f \n" % (k, run[0], run[2], scaled))
        out_file.close()
        mean_product.append(np.mean(product))
        escape_times[k] = product
    return mean_product, escape_times


def plot_kde_hist_cumulative(times_vector,id, out_dir, force):
    plt.figure(figsize=(5, 4))
    sns.kdeplot(times_vector, common_norm=True, cumulative=True)
    a = plt.hist(times_vector, density=True, bins=20, cumulative=True)
    plt.xscale('log')
    plt.savefig(f"{out_dir}/kde_hist_cumulative{force}{id}.png")
    plt.close()
    return 1

# Function to calculate the TCDF with constants rate
def func(x, a):
    return 1 - np.exp(-a * x)


# Function to fit line
def line(x, a, b):
    return a * x + b


def get_ecdf(time_list, min_time, max_time, n_bins):
    time_domain = np.logspace(np.log10(min_time), np.log10(max_time), n_bins)
    N = len(time_list)
    a, b = np.histogram(time_list, time_domain)
    # mod = np.append(a,[0])
    ECDF = np.cumsum(a) / N
    # new = np.append(ECDF,[ECDF[-1]])
    if len(time_domain[:-1]) == len(ECDF):
        # print(len(mod), len(a[:-1]), len(ECDF))
        return time_domain[:-1], ECDF
    else:
        return "error"


def compute_cdfs(time_list):
    min_t = np.min(time_list)/10  
    max_t = np.max(time_list)*10
    nbins = 500
    x, y = get_ecdf(time_list, min_t, max_t, nbins)
    test_mean = np.mean(time_list)
    test_sigma = np.std(time_list, ddof=1)
    test_m = np.median(time_list)
    means = test_mean
    medians = test_m
    uncertainty = test_sigma / np.sqrt(len(time_list))
    stdvs = test_sigma
    mean_sigma_ratio = test_mean / test_sigma
    log2_median_ratio = np.log(2) * test_mean / test_m
    guess = 1.0 / test_mean
    # print(np.mean(time_list),min_t,max_t)
    pars, cov = curve_fit(f=func, xdata=x, ydata=y, p0=[guess], maxfev=1000000)
    stdevs = np.sqrt(np.diag(cov))
    f_stdvs = stdevs[0]
    tau = np.power(pars[0], -1)
    tau_mean_ratio = (tau / test_mean)
    tcdf = gamma.cdf(x, 1, scale=tau)
    cdfs = [x, y, tcdf]
    data = [tau, stdvs, f_stdvs, means, medians,
            uncertainty, mean_sigma_ratio, log2_median_ratio, tau_mean_ratio]
    return cdfs, data

def calculate_error_bars(list_of_taus, num_r):
    bars = []
    for tau in list_of_taus:
        mean_exp = tau
        bootstraps = []
        for b in range(9999):
            rvs = gamma.rvs(1, scale=mean_exp, size=num_r)
            bootstraps.append(rvs)
        b_matrix = np.matrix(bootstraps)
        averages = np.mean(b_matrix, axis=1)
        arr_avg = np.array(averages)
        flat_avg = arr_avg.flatten()
        taus_exp = flat_avg
        tau_mean = np.mean(taus_exp) 
        tau_std = np.std(taus_exp)
        rate_fit = 1/taus_exp
        rate_mean = np.mean(rate_fit)
        rate_std = np.std(rate_fit)
        bars.append([tau_mean, tau_std, rate_mean, rate_std])
    return np.array(bars)


def plot_cdfs(x, y, tcdf, id, force, model_name, out_dir):

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.step(x, y, c='k', label=F'ECDF: F={np.abs(float(force))} {model_name}')
    ax.plot(x, tcdf, label=F'TCDF: F={np.abs(float(force))} {model_name}', c='r')
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
    ax.set_xlim(min(x), max(x))
    ax.tick_params(axis='both', labelsize=14, width=2)
    plt.legend(fontsize=14, loc='upper left', frameon=False)
    fig.tight_layout()
    out_path = os.path.join(out_dir,f'F{force}_{model_name}{id}.png')
    plt.savefig(out_path)
    plt.close()
    return 1


def do_KS2test(time_list, tau):
    size = np.int64(len(time_list) * 1e6)
    rvs1 = gamma.rvs(1, scale=tau, size=size)  # loc=min(time_list))
    ks_stat, p = stats.ks_2samp(rvs1, time_list)
    return ks_stat, p


def plot_escape_time_log(force_list, tau_list, model_name, out_dir, id, error):
    fig, ax = plt.subplots(figsize=(8, 6))
    lab = F"{model_name}"
    bars = error[:,1]
    ax.plot(force_list, tau_list, c='k',marker='s',label=lab, linestyle='', mec='r')
    ax.errorbar(force_list, tau_list, bars, c='r', linestyle='', capsize=3)
    ax.tick_params(axis='both', labelsize=16, width=2)
    ax.legend(fontsize=16, loc='best')
    ax.set_ylabel(r"${\tau}\ ({\mu}s)$", fontsize=16)
    ax.set_yscale('log')
    plt.xlabel(r"$Force\ (pN)$", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=.92)
    out_path = os.path.join(out_dir,f'{model_name}{id}_logOfFittedTaus.png')
    plt.savefig(out_path)
    plt.close()
    return 1


def plot_escape_time(force_list, tau_list, model_name, out_dir, id, error, log=False):
    force_list = np.array(force_list) 
    tau_list = np.array(tau_list)
    if log:
        plot_escape_time_log(force_list, tau_list, model_name, out_dir, id, error)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2.0)
        lab = F"{model_name}"
        bars = error[:,1]
        ax.plot(force_list, tau_list, c='k',marker='s',label=lab, linestyle='', mec='r')
        ax.errorbar(force_list, tau_list, bars, c='r', linestyle='', capsize=3)
        ax.tick_params(axis='both', labelsize=16, width=2)
        ax.legend(fontsize=14, loc='best')
        ax.set_ylabel(r"${\tau}\ ({\mu}s)$", fontsize=16)
        plt.xlabel(r"$Force\ (pN)$", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=.92)
        out_path = os.path.join(out_dir, f'{model_name}{id}_FittedTaus.png')
        plt.savefig(out_path)
        plt.close()
    return 1


def plot_rates_log(force_list, tau_list, model_name, out_dir, id, theo_y, error):
    fig, ax = plt.subplots(figsize=(8, 6))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)
    lab = F"{model_name}"
    r_stds = error[:,3]
    theo_x = np.linspace(0.0,np.max(force_list),100)
    ax.plot(force_list, 1/tau_list, c='k',marker='s',label=lab, linestyle='', mec='r')
    ax.errorbar(force_list, 1/tau_list, r_stds, c='r', linestyle='', capsize=3)
    ax.plot(theo_x, theo_y,  c='k', linestyle='--', linewidth=1.25, label="Bell's model")
    ax.tick_params(axis='both', labelsize=16, width=2)
    ax.legend(fontsize=16, loc='best')
    ax.set_ylabel(r"$rate\ ({\mu}s^{-1})$", fontsize=16)
    ax.set_yscale('log')
    plt.xlabel(r"$Force\ (pN)$", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=.92)
    out_path = os.path.join(out_dir, f'{model_name}{id}_logOfRatesvsForce.png')
    plt.savefig(out_path)
    plt.close()
    return 1


def plot_rates(force_list, tau_list, model_name, out_dir, id, error, log=False, kbT=2.494):
    force_list = np.array(force_list) 
    tau_list = np.array(tau_list)
    bell = bell_params(force_list, tau_list)
    dx = bell[0]*kbT
    k_0 = np.exp(bell[1])
    r_sqr = bell[2]
    theo_x = np.linspace(0.0,np.max(force_list),100)
    theo_y = slip_rate_nm(theo_x, k_0, dx, kbT)[0]
    if log:
        plot_rates_log(force_list, tau_list, model_name, out_dir, id, theo_y,error)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2.0)
        lab = F"{model_name}"
        r_stds = error[:,3]
        ax.plot(force_list, 1/tau_list, c='k',marker='s',label=lab, linestyle='', mec='r')
        ax.errorbar(force_list, 1/tau_list, r_stds, c='r', linestyle='', capsize=3)
        ax.plot(theo_x, theo_y,  c='k', linestyle='--', linewidth=1.25, label="$R^2$ = %.2f"%(r_sqr))
        ax.tick_params(axis='both', labelsize=16, width=2)
        ax.legend(fontsize=16, loc='best')
        ax.set_ylabel(r"$rate\ ({\mu}s^{-1})$", fontsize=16)
        plt.xlabel(r"$Force\ (pN)$", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=.92)
        out_path = os.path.join(out_dir, f'{model_name}{id}_ratesvsForce.png')
        plt.savefig(out_path)
        plt.close()
    return 1


def plot_catch_tau_rates(force_list, tau_list, model_name, out_dir, id, error):
    force_list = np.array(force_list) 
    tau_list = np.array(tau_list)
    rates = 1/tau_list
    data = catch_params(force_list[1:], tau_list[1:])
    k1c_0 = data[0]
    x1c = data[1]
    k1s_0 = data[2]
    x1s = data[3]
    theo_x = np.linspace(0.0, np.max(force_list), 100)
    theo_y = catch_rates_nm(k1c_0,-x1c,k1s_0, x1s, theo_x)[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    print(k1c_0,-x1c,k1s_0, x1s)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)

    lab = F"{model_name}"
    r_stds = error[:,3]
    ax.plot(force_list, 1/tau_list, c='k',marker='s',label=lab, linestyle='', mec='r')
    ax.errorbar(force_list, 1/tau_list, r_stds, c='r', linestyle='', capsize=3)
    ax.plot(theo_x, theo_y,  c='k', linestyle='--', linewidth=1.5, label="catch-slip")
    ax.tick_params(axis='both', labelsize=16, width=2)
    ax.legend(fontsize=16)
    ax.set_ylabel(r"$rate\ ({\mu}s^{-1})$", fontsize=16)
    plt.xlabel(r"$Force\ (pN)$", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=.92)
    out_path = os.path.join(out_dir, f'{model_name}{id}_ratesvsForce.png')
    plt.savefig(out_path)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 6))

    t_stds = error[:,1]
    ax.plot(force_list, tau_list, c='k',marker='s',label=lab, linestyle='', mec='r')
    ax.errorbar(force_list, tau_list, t_stds, c='r', linestyle='', capsize=3)
    ax.plot(theo_x, 1/theo_y,  c='k', linestyle='--', linewidth=1.5, label="Two-pathway model")
    ax.tick_params(axis='both', labelsize=16, width=2)
    ax.legend(fontsize=16)
    ax.set_ylabel(r"$\tau \ ({\mu}s)$", fontsize=16)
    plt.xlabel(r"$Force\ (pN)$", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=.92)
    out_path = os.path.join(out_dir, f'{model_name}{id}_tausvsForce.png')
    plt.savefig(out_path)
    plt.close()
    
    return 1


def catch_rates(k1c_0, x1c, k1s_0, x1s, force, kbT=2.494):
    force_kJ = force*(1/1.66)  #41 pN = 2.478 kJ/(mol*A)
    rate = k1c_0*np.exp(x1c*force_kJ/kbT) + k1s_0*np.exp(x1s*force_kJ/kbT)
    return rate


def catch_rates_fit_nm(force, k1c_0, x1c, k1s_0, x1s, kbT):
    rate = k1c_0*np.exp(-x1c*force/kbT) + k1s_0*np.exp(x1s*force/kbT)
    return rate


def catch_rates_nm(k1c_0, x1c, k1s_0, x1s, force, kbT=2.494):
    force_kJ = force*(1/1.66)  #41 pN = 2.478 kJ/(mol*A)
    rate = k1c_0*np.exp(x1c*force_kJ/kbT) + k1s_0*np.exp(x1s*force_kJ/kbT)
    tau = 1/rate
    return (rate, tau)


def slip_rate_fit_nm(force,k1s_0,xd, kbT):
    return k1s_0*np.exp(xd*force/kbT)


def slip_rate_nm(force, k1s_0, xd, kbT=2.494):
    force_kJ = force*(1/1.66)
    rate = k1s_0*np.exp(xd*force_kJ/kbT)
    tau = 1/rate
    return (rate, tau)


def catch_params(forces_pN, tau_list, units_e='kj', units_l='nm'):
    tau_list = np.array(tau_list)
    forces_pN = np.array(forces_pN)
    
    if units_e == 'kj' and units_l == 'nm':
        forces = forces_pN*(0.60241)
    elif units_e =='kcal' and units_l == 'a':
        forces = forces_pN*0.0143929254
    x = forces
    y = 1/tau_list
    pars, cov = curve_fit(f=catch_rates_fit_nm, xdata=x, ydata=y, maxfev=1000000) #get parameters 
    return pars


def bell_params(forces_pN, tau_list, units_e='kj', units_l='nm'):
    tau_list = np.array(tau_list)
    forces_pN = np.array(forces_pN)
    
    if units_e == 'kj' and units_l == 'nm':
        forces = forces_pN*(0.60241)
    elif units_e =='kcal' and units_l == 'a':
        forces = forces_pN*0.0143929254
    log_tau = np.log(1/tau_list)
    popt, pcov = curve_fit(line, forces, log_tau)
    residuals = log_tau - line(forces, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_tau-np.mean(log_tau))**2)
    r_squared = 1 - (ss_res / ss_tot)
    slope = popt[0]
    intercept = popt[1]
    params_lines = [slope, intercept, r_squared]

    return params_lines


if __name__ == '__main__':
    time_list = [389267.8260916,
                 280615.1612349,
                 352991.755287,
                 254401.5943329,
                 290813.855694,
                 127531.5566096,
                 225731.5920222,
                 350577.3045656,
                 269457.00861759996,
                 314554.8179661,
                 472361.6314188,
                 299550.19261675,
                 346090.13722035,
                 313239.37392205006,
                 218508.85789239997,
                 434531.22000145,
                 224990.1612792,
                 156855.9787566,
                 168118.58454685,
                 301677.0178401]
    plot_kde_hist_cumulative(time_list, out_dir="./", force=0.0)
    cdfs, data = compute_cdfs(time_list)
    time_domain = cdfs[0]
    ecdf = cdfs[1]
    tcdf = cdfs[2]
    tau = data[0]
    print(tau)
    plot_cdfs(x=time_domain, y=ecdf, tcdf=tcdf, force=0.0, model_name='sample', out_dir='./')
    s, p = do_KS2test(time_list, tau)
    print(p)

