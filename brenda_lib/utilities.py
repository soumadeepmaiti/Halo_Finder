import sys
import numpy as np

############################
##### Memory utilities #####
############################

def get_memory():
    ''' Returns the memory usage of the current process.
        Does not work with multiple processes (only gives memory used by the current one.'''
        
    import psutil
    return (psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)

def mem_from_npart(n, u='gb', per_CPU=True, num_CPU=64, PMGRID_fac=2):
    ''' Returns an estimate of the memory needed by Gadget for running with a given number of particles.
        PartAllocFactor and TreeAllocFactor are hardcoded for a general case.
        
        :param n: The number of particles
        :type  n: int
        
        :param u: The units of the output (b, Kb, Mb, Gb, Tb). Default: Gb
        :type  u: str

        :param per_CPU: Wether the output should be the memory per CPU or the total one
        :type  per_CPU: bool

        :param num_CPU: Number of CPU used. Only needed if per_CPU=True
        :type  num_CPU: int

        :param PMGRID_fac: Ratio between the PM grid size and the number of particles, per side
        :type  PMGRID_fac: float
        
        :return: The amount of memory needed to run an Nbody simulation
        :rtype : float '''

    PartAllocFactor = 2.
    TreeAllocFactor = 0.7
    PMGRID = PMGRID_fac*n
    m1 = PartAllocFactor*(68+TreeAllocFactor*64)
    m2 = PMGRID**3 * 12 - 16
    memtot = m1*(n**3) + m2
    cf = {'b':1., 'Kb':1024., 'Mb':(1024.)**2, 'Gb':(1024.)**3, 'Tb':(1024.)**4}
    if per_CPU:
        nCPU = num_CPU*int(np.ceil(n/768))
        return memtot/(nCPU*cf[u])
    else:
        return memtot/cf[u]

#############################
##### Plotting utilities ####
#############################

def set_Ariel_like_plots(set_lim = False, labels = False):
    import matplotlib.pyplot as plt
    plt.rcParams['axes.linewidth'] = 2.1
    plt.tick_params(reset = True, which='both', bottom=True, top=True, right=True, left=True, 
                    direction = 'in', length=8.5, width=2.3, labelsize = 18,
                    labelleft=labels, labeltop=False, labelright=False, labelbottom=labels)
    plt.tick_params(which='major', bottom=True, top=True, right=True, left=True, 
                    direction = 'in', length=17, width=2.3)
    if set_lim:
        plt.xlim(0.006314601343965478, 1)
        plt.ylim(3.15e1, 2e3)

def A_plot_kwargs(ls = '-'):
    if ls == '--':
        return {'ls':'--', 'dashes':(6.5, 5.5), 'linewidth':2.2}
    else:
        return {'linewidth':2.2}

# Colors used in Ariel's evolution mapping paper (reverse-engineered)
Acolors=['black', '#377Bf7', '#96CBFA', '#377D1D', '#A0FC47', '#B480F8', '#F2A837', '#763734', '#EB5326']

############################
#### Nd array utilities ####
############################

def sort_by_columns(arr, cols):
    ''' Returns arr sorted by the column in cols (in order).
    
    :param arr: The nd array to be sorted
    :type  arr: numpy ndarray
    
    :param cols: A list of the column numbers that we want to order the array by (in order)
    :type  cols: tuple or list or array of int
    
    :return: The sorted array
    :rtype : numpy ndarray '''

    keys = [arr[:, i] for i in cols[::-1]]
    return arr[np.lexsort(keys)]
    
def nd_argmax(x):
    ''' Returns the index of the maximum value of an nd array '''

    shape = np.shape(x)
    max_idx = np.argmax(x)
    return np.unravel_index(max_idx, shape)

def nd_argmin(x):
    ''' Returns the index of the minimum value of an nd array '''

    shape = np.shape(x)
    min_idx = np.argmin(x)
    return np.unravel_index(min_idx, shape)

def mask_zeros(Pk, masking_var='pk', to_mask=['k', 'pk', 'shotnoise']):
    if type(Pk) == dict:
        mask = (Pk[masking_var]!=0)
        for masked_qty in to_mask:
            Pk[masked_qty] = Pk[masked_qty][mask]
    elif type(Pk) == tuple or type(Pk) == list or type(Pk) == np.ndarray:
        mask = (Pk[1]!=0)
        result = [Pk[0][mask]]
        for x in Pk[1:]:
            result.append(x[mask])
        return result
    else:
        print('ERROR: Please use as an argument a tuple () or a dict {}')

def safe_diff(X, Y):
    diff = np.zeros(len(X))
    for i in range(len(diff)):
        if Y[i]==0:
            diff[i]=0
        else:
            diff[i]=(X[i]-Y[i])/Y[i]
    return diff

def myfind_BinA(A, B):
    return (A[:, None] == B).argmax(axis=0)

def eval_same_bin(A, B, extend='left', min_common_elements=20):
    xA, yA = A
    xB, yB = B
    x = {0: xA, 1: xB}
    y = {0: yA, 1: yB}
    
    #First, let's check if the x arrays have a sufficient number of common elements 
    x_inters, xA_idx, xB_idx = np.intersect1d(xA, xB, return_indices=True, assume_unique=True)
    if len(xA_idx) >= min_common_elements:
        print('I am returning the elements of the arrays evaluated on the common x values ({} points)'.format(len(xA_idx)))
        return [x_inters, np.array(yA)[xA_idx], np.array(yB)[xB_idx]]
    
    #If not, let's decide which array we should use for the binning and which one should be interpolated
    if xA[0] < xB[0]:
        left_ext_array = 0
    else:
        left_ext_array = 1
    if xA[-1] > xB[-1]:
        right_ext_array = 0
    else:
        right_ext_array = 1
        
    from scipy.interpolate import interp1d        
    #If one of the two arrays contains the other, then interpolate the former on the binning of the latter
    if left_ext_array == right_ext_array:
        mask = slice(None, None)
        ext_array = left_ext_array
    #If not, then choose whichever let's you have more info on the left
    elif extend == 'left':
        interp_idx = np.searchsorted(x[not left_ext_array], x[left_ext_array][-1])
        mask = slice(None, interp_idx)
        ext_array = left_ext_array
    #Or on the right
    else:
        interp_idx = np.searchsorted(x[left_ext_array], x[not left_ext_array][0])
        mask = slice(interp_idx, None)
        ext_array = right_ext_array
    print("I am interpolating the array in input pos {}".format(ext_array))
    output = [np.array(x[not ext_array][mask]), None, None]
    output[(ext_array)+1] = interp1d(x[ext_array], y[ext_array])(x[not ext_array][mask])
    output[(not ext_array)+1] = np.array(y[not ext_array][mask])
    return output

#############################
####### Miscellaneous #######
#############################

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def ask_continue(text=None, abort=True):
    if (text!=None):
        print(text)
    ans = input('Do you want to continue anyway [y/n]? ')
    if ans=='y':
        return True
    elif ans=='n':
        if abort:
            sys.exit()
        return False
    else:
        print('Please insert only y or n')
        return ask_continue(abort=abort)


#############################
##### Reading utilities #####               These routines require certain files to run properly
#############################


def read_Ariel_table(nsims = 9):
    from scipy.interpolate import interp1d
    Ariel_tables = {}
    for i in range(nsims):
        try:
            z, sigma12, g, f, _, dg_ds12 = np.loadtxt('/cobra/u/mesposito/Projects/sigma12-in-z-space/tables/evolution_final_caso%d.dat' %i, unpack=True, skiprows=12)
        except:
            z, sigma12, g, f, _, dg_ds12 = np.loadtxt('/raven/u/mesposito/Projects/velocity-power-spectrum/Ariel_cosmo_calc/evolution_final_caso%d.dat' %i, unpack=True, skiprows=12)
        Ariel_tables['Columbus_{}'.format(i)] = {'z': z, 'a': 1/(1+z), 's12': sigma12, 
                                                 's12_a': interp1d(1/(1+z), sigma12), 
                                                 'g': g, 'f': f, 'f_a': interp1d(1/(1+z), f), 
                                                 'dg_ds12': dg_ds12, 'f_s12': interp1d(sigma12, f)}
    return Ariel_tables


def make_cosmo_suite(fname='cosmology_Aletheia.dat', cosmo_suite_name='Aletheia', use_sigma8=True):
    try:
        f = open(fname)
    except:
        print('I tried to load the file {}, but I could not find it.'.format(fname))
        print('If you want to import the list of cosmologies,')
        print('please call "make_cosmo_suite(fname=path_to/cosmology_columbus.dat)"')
        return None
    lines = f.readlines()
    f.close()
    labels = (lines.pop(0)).split()
    cosmo_suite = [{labels[i]: (lines[n].split()[i] if cosmo_suite_name in lines[n].split()[i]
                 else float(lines[n].split()[i])) for i in range(len(labels))} for n in range(len(lines))]
    cosmo_dict = {}
    for cosmo in cosmo_suite:
        for snapnum in range(5):
            cosmo_params ={}
#             cosmo_params['L'] = cosmo['Lbox']
            cosmo_params['omega_cdm'] = cosmo['OmC']
            cosmo_params['omega_de'] = cosmo['OmL']
            cosmo_params['omega_baryon'] = cosmo['OmB']
            cosmo_params['hubble'] = cosmo['h']
            cosmo_params['ns'] = cosmo['n_s']
            if use_sigma8:
                cosmo_params['A_s'] = None
                cosmo_params['sigma8'] = cosmo['sigma8']
                cosmo_params['ReNormalizeInputSpectrum'] = True
            else:
                cosmo_params['A_s'] = cosmo['A_s']
                cosmo_params['sigma8'] = None    
                cosmo_params['ReNormalizeInputSpectrum'] = False           
            cosmo_params['expfactor'] = 1/(1+cosmo['z(%d)' %snapnum]) #Warning!! May differ from the actual expfactor (check header)
            cosmo_params['tau'] = 0.0952 #
            w0 = cosmo['w0']
            wa = cosmo['wa']
            if w0!=-1 or (wa!=0):
                cosmo_params['de_model'] = 'w0wa'
                cosmo_params['w0'] = w0
                cosmo_params['wa'] = 0.0 if np.isnan(wa) else wa
            cosmo_dict[cosmo['Name']+'_%d' %snapnum] = cosmo_params
    return cosmo_dict


def abacus_cosmo_dict(cosmo_label = '000', z = 0):
    from configobj import ConfigObj
    fname = '/ptmp/mesposito/AbacusSummit/Cosmologies/abacus_cosm{}/CLASS.ini'.format(cosmo_label)
    cosmo = ConfigObj(fname)
    cosmo_params = {}
    h = float(cosmo['h'])
    omega_b = float(cosmo['omega_b'])
    omega_cdm = float(cosmo['omega_cdm'])
    omega_ncdm = float(cosmo['omega_ncdm'])
    cosmo_params['neutrino_mass'] = omega_ncdm*93.14
    cosmo_params['hubble'] = h
    cosmo_params['omega_cdm'] = omega_cdm/h**2
    cosmo_params['omega_de'] = 1-(omega_cdm+omega_b+omega_ncdm)/h**2
    cosmo_params['omega_baryon'] = omega_b/h**2
    cosmo_params['ns'] = float(cosmo['n_s'])
    cosmo_params['A_s'] = float(cosmo['A_s'])
    cosmo_params['sigma8'] = None    
    cosmo_params['ReNormalizeInputSpectrum'] = False           
    cosmo_params['expfactor'] = 1/(1+z)
    cosmo_params['tau'] = 0.0952 #
    w0 = float(cosmo['w0_fld'])
    wa = float(cosmo['wa_fld'])
    if w0!=-1 or (wa!=0):
        cosmo_params['de_model'] = 'w0wa'
        cosmo_params['w0'] = w0
        cosmo_params['wa'] = 0.0 if np.isnan(wa) else wa
    return cosmo_params

