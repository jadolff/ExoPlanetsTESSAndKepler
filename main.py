from constants import *
from load.lc import read_lc, add_zeros, quarter_indexes
from load.planet import read as read_known_planets
from load.star import StarInfo_init

from gaussianization import gaussianize
from fourier.MF import stationary
from planet import transit, tauprior, nst, inject
from false_alarms.pipeline import fa_search
from false_alarms import spurious_transits

from pipeline import nonstationary, ttv, joint, scratch_structure
from pipeline import vetting
from pipeline.inv_scr import get_data_transform
from detection.cpu.pipeline import run as detect
from detection.cpu.pipeline import prepare as prepare_for_detection
from detection.cpu.scan import q_in_dt_units
from detection.cpu import template

# New Imports
from TESS.load_tess import StarInfo_init_tess, read_lc_tess, add_zeros_tess, read_known_planets_tess, dt_tess, q_in_dt_units_tess
import TESS.transit_tess as transit_tess


def bit_info(numbers, bit):
    return np.floor_divide(numbers - np.floor_divide(numbers, 2**(bit))* (2**bit), 2**(bit-1))



def plot_residuals(Time, Flux, FGP, plotdir):
    
    residuals = Flux - FGP
    plt.figure(figsize = (10, 10))
    plt.subplot(3, 1, 1)
    plt.title('Full view')
    plt.plot(Time, residuals, '.')
    plt.ylabel("residuals")
    plt.xlabel('time [days]')
    plt.close()
    
    plt.subplot(3, 1, 2)
    plt.plot(Time, Flux, '.')
    plt.plot(Time, FGP, color='cyan')
    plt.ylabel("residuals")
    plt.xlabel('time [days]')
    t0 = Time[np.argmax(residuals)] 
    plt.xlim(t0 - 5, t0 + 5)
    
    plt.subplot(3, 1, 3)
    plt.plot(Time, Flux, '.')
    plt.plot(Time, FGP, color='cyan')
    plt.ylabel("residuals")
    plt.xlabel('time [days]')
    t0 = Time[np.argmin(residuals)] 
    plt.xlim(t0 - 5, t0 + 5)
    
    plt.savefig(plotdir + 'residuals.png')
    plt.close()


def LOAD_KEP(kepid, folder, hyperparams, batch_num) :

    # Saving Pictures
    kepid_folder_name = str(kepid) + ('_' + str(batch_num) if (batch_num != 0) else '')
    plotdir = (scratch + folder + '/plots/' + kepid_folder_name + '/') if hyperparams.show else None
    if hyperparams.show:
        scratch_structure.make_dir(plotdir)

    # Loading Data
    star = StarInfo_init(kepid, 'star_info_gaia')#'star_info_withplanets')
    
    time, flux, flags, quarter_beginnings = read_lc(star.kepid, injected = False)
    
    average, flux_sigma = gaussianize.rescale(flux) #flux_sigma: we rescale the flux to unit variance of the Gaussian part of the noise distribution flux_sigma * our_flux = original_flux (which is measured in units of star's flux)
    star = star._replace(sigma_star_flux_units = flux_sigma)
    flux = (flux - average) / flux_sigma
    t_start = time[0]
    time -= t_start
    
    Time, Flux, invVar = add_zeros(time, flux, hyperparams)
    max_num_periods = (int)(2 * (Time[-1] / hyperparams.period_min)) # factor of two just to be sure

    quarters = quarter_indexes(Time, quarter_beginnings)

    planet_props = read_known_planets(star, t_start, hyperparams.test_recovery)
    
    spline = transit.prepare_shape(0, star.u1, star.u1) #spline for the planet transit shape (will be used for the search, assuming small planets)
    spline_prior = tauprior.load(star.err_q) # spline for the shape of the transit duration prior

    if hyperparams.show:
        plt.figure(figsize = (10, 5))
        plt.plot(Time, Flux, '.')
        plt.ylim(-5, 5)
        plt.savefig(plotdir + 'flux_normalized.png')
        plt.close()

    return star, t_start, Time, Flux, invVar, max_num_periods, quarters, planet_props, spline, spline_prior

def LOAD_TESS(tessid, folder, hyperparams, batch_num) :

    # Saving Pictures
    tessid_folder_name = str(tessid) + ('_' + str(batch_num) if (batch_num != 0) else '')
    plotdir = (scratch + folder + '/plots/' + tessid_folder_name + '/') if hyperparams.show else None
    if hyperparams.show:
        scratch_structure.make_dir(plotdir)

    # Loading Data
    star = StarInfo_init_tess(tessid, 'star_info_gaia')#'star_info_withplanets')
    
    time, flux, flags, quarter_beginnings = read_lc_tess(star.kepid, injected = False)
    
    average, flux_sigma = gaussianize.rescale(flux) #flux_sigma: we rescale the flux to unit variance of the Gaussian part of the noise distribution flux_sigma * our_flux = original_flux (which is measured in units of star's flux)
    star = star._replace(sigma_star_flux_units = flux_sigma)
    flux = (flux - average) / flux_sigma
    t_start = time[0]
    time -= t_start
    
    Time, Flux, invVar = add_zeros_tess(time, flux, hyperparams)
    max_num_periods = (int)(2 * (Time[-1] / hyperparams.period_min)) # factor of two just to be sure

    quarters = quarter_indexes(Time, quarter_beginnings)

    planet_props = read_known_planets_tess(star, t_start, hyperparams.test_recovery)
    
    spline = transit_tess.prepare_shape(0, star.u1, star.u1) #spline for the planet transit shape (will be used for the search, assuming small planets)
    spline_prior = tauprior.load(star.err_q) # spline for the shape of the transit duration prior

    if hyperparams.show:
        plt.figure(figsize = (10, 5))
        plt.plot(Time, Flux, '.')
        plt.ylim(-5, 5)
        plt.savefig(plotdir + 'flux_normalized.png')
        plt.close()

    return star, t_start, Time, Flux, invVar, max_num_periods, quarters, planet_props, spline, spline_prior


def mainn(kepid, folder, hyperparams, 
          nst_seed= None, inj_seed = None,
          inj_params= None, batch_num = 0, TESS = False):


    #kepid, batch = job_map(id_job)
    #batch_props = hyperparams.batch_props.iloc[batch]
    kepid_folder_name = str(kepid) + ('_' + str(batch_num) if (batch_num != 0) else '')
    plotdir = (scratch + folder + '/plots/' + kepid_folder_name + '/') if hyperparams.show else None
    if hyperparams.show:
        scratch_structure.make_dir(plotdir)
    
    ### load data ###
    if TESS :
        star, t_start, Time, Flux, invVar, max_num_periods, quarters, planet_props, spline, spline_prior = LOAD_TESS(kepid, folder, hyperparams, batch_num)
        dt = dt_tess
    else : 
        star, t_start, Time, Flux, invVar, max_num_periods, quarters, planet_props, spline, spline_prior = LOAD_KEP(kepid, folder, hyperparams, batch_num)

    ### jointly fit FGP, planets and gaussianize outliers ###
    planet_props, Flux, model, Pk, FGP, bandwidth, _, sharp_freq = joint.iteration(Time, Flux, invVar, planet_props, sharp_freq= None)
    
    
    if len(planet_props) != 0:
        ### find TTVs ###
        planet_props = ttv.pipeline(Time, Flux, invVar, Pk, planet_props)

        [ttv.plots(Time, Flux, invVar, model, FGP, planet_props[k], np.zeros(5000), plotdir + 'init' + str(k), maintitle= 'Known planet') for k in range(len(planet_props))]

        ### redo the fit ###
        planet_props, Flux, model, Pk, FGP, bandwidth, mask_planets, _ = joint.iteration(Time, Flux, invVar, planet_props, Pk, sharp_freq)


        if hyperparams.show:
            plt.plot(Time, model+FGP, color= 'tab:blue')
            plt.plot(Time, FGP, color= 'cyan')
            plt.plot(Time, Flux, '.', color= 'black')
            plt.close()
        
        # save the known planets results
        var_noise = np.average(Pk[int(3 * dt * len(Flux)) + 1:]) / len(Flux) # average power spectrum above the FGP frequency cutoff

        planet_props = ttv.fit_quality(Time, Flux - model - FGP, invVar, var_noise, planet_props)
        planet_props = ttv.laplace(Time, Flux, invVar, Pk, planet_props)
        
        [ttv.plots(Time, Flux, invVar, model, FGP, planet_props[k], np.zeros(5000), plotdir + str(k), maintitle= 'Known planet') for k in range(len(planet_props))]
        [planet_props[iplanet].save(scratch + folder + '/known_planets/'+str(kepid)+'_' + str(iplanet)) for iplanet in range(len(planet_props))]

        # remove planets
        Flux[mask_planets] = FGP[mask_planets]  #alternative used to be: Flux = Flux - model
        invVar[mask_planets] = 0.    

    ### optionally invert or scramble the data ###
    data_transform = get_data_transform(hyperparams)
    # Here kepid just serves as a random seed
    Flux, FGP = data_transform(Flux, kepid, quarters), data_transform(FGP, kepid, quarters)
    invVar = np.abs(data_transform(invVar, kepid, quarters)) # if there is inversion, we do not want to invert invVar
    # Pk ??
    ### optionally inject planets ###
    if inj_seed != None:
        rng = np.random.default_rng(inj_seed)
        # generate the properties of the injected planets
        props_inj = inject.props(rng, *inj_params, star, do_poisson= False)
        
        # inject the planets into the light curve
        flux_inj, props_inj = inject.lc(props_inj, Time, invVar, spline, np.zeros(max_num_periods))

        Flux += flux_inj

        # save the properties of the injected planets
        if len(props_inj) != 0:
            props_inj.to_csv(scratch + folder + '/inj/batch' + str(batch_num) + '/' +str(kepid)+'.csv', sep = '\t', index= False)

        
    ### nonstationary noise ###
    local_nonstationarity, step_nonstationarity, rb_nonstationarity= nonstationary.pipeline(Flux, invVar, bandwidth, hyperparams, plotdir)
    non_stationarity = [np.max(local_nonstationarity), np.argmax(local_nonstationarity) * dt, step_nonstationarity[0], rb_nonstationarity[0]]
    df = pd.DataFrame([[star.kepid, t_start, star.sigma_star_flux_units, np.max(Pk) / np.min(Pk), *non_stationarity, len(sharp_freq) != 0], ], 
                      columns = ['kepid', 't_start', 'sigma_star_flux_units', 'PSD_condition_number', 'local_nonstat', 'phase_local_nonstat', 'step_nonstat', 'rb_nonstat', 'has_sharp_peak'])
    df.to_csv(scratch + folder + '/stars/'+str(kepid)+'.csv', sep = '\t', index= False)

    ### individual transits and false alarms ###

    Flux, invVar, fp_info, fp_scores = fa_search(Time, Flux, invVar, Pk, FGP, spline, star, hyperparams, plotdir)

    if len(fp_info) > 0:
        planet_props, Flux, model, Pk, FGP, bandwidth, _, _ = joint.iteration(Time, Flux, invVar, [], Pk, sharp_freq)
    

    if hyperparams.show:
        plot_residuals(Time, Flux, FGP, plotdir)   
    
    
    if len(fp_info) != 0:
        fp_info['kepid'] = kepid
        fp_info['id_event'] = np.arange(len(fp_info))
        fp_info.to_csv(scratch + folder + '/drops/'+str(kepid)+'.csv', sep = '\t', index= False)
    

    spurious_scenarios = [[spurious_transits.localized_glitch(fp_scores, 0.01),
                           spurious_transits.localized_glitch(fp_scores, 0.1),
                           spurious_transits.localized_glitch(fp_scores, 0.2)],
                          [spurious_transits.close_to_gap(invVar),],
                          [spurious_transits.overlap_with_larger_planets(ttv.mask_planets(Time, planet_props)),]
                          ]    

    ### detection and vetting

    
    exit()
    
    def func(nst_seed):
        
        if nst_seed != None:
            period_min_nst = 2.
            rng = np.random.default_rng(nst_seed)
            if TESS :
                func_for_delta = nst.transit_duration_limited(rng, max_num_periods, period_min_nst/dt, q_in_dt_units_tess(star.q), 
                                                                                                    num_transit_durations_min= 2., frac_period_max= 0.4)
            else : 
                func_for_delta = nst.transit_duration_limited(rng, max_num_periods, period_min_nst/dt, q_in_dt_units(star.q), 
                                                                                                    num_transit_durations_min= 2., frac_period_max= 0.4)
        else:
            func_for_delta = nst.basic(max_num_periods)
            
            
        if hyperparams.do_nonstat_mf:
            bandwidth_short = nonstationary.bandwidth_long_to_short(bandwidth, len(Time), hyperparams.stft_width)
            psd = nonstationary.get_psd(Flux, invVar, bandwidth_short, hyperparams.stft_width, hyperparams.stft_sep, sharp_freq)
            mf_update = nonstationary.get_update(hyperparams.stft_width, hyperparams.stft_sep, len(Time), func_for_delta)
            do_conv = lambda Flux, Pk: nonstationary.convolve(Flux, template_bank.templates, Pk, hyperparams.stft_width, hyperparams.stft_sep)

        else:
            psd = Pk
            mf_update = stationary
            do_conv = lambda Flux, Pk: template.convs(Flux, template_bank.templates, Pk)
        
        # scan for periodic events
        template_bank, period_prior_normalization = prepare_for_detection(invVar, psd, star, spline_prior, hyperparams, func_for_delta_given= func_for_delta, do_nonstat_mf= hyperparams.do_nonstat_mf)
        if template_bank == None: # time span of the data is so short that even the shortest period cannot support three transits
            return 
        
        candidates_all = detect(Flux, invVar, psd, template_bank, hyperparams, kepid, hyperparams.do_nonstat_mf)
        
        # vet the detections
        spurious = spurious_transits.SpuriousTransits(spurious_scenarios, func_for_delta, len(Time))
        vetting.main(Time, Flux, invVar, FGP, psd, template_bank, period_prior_normalization, candidates_all, star, spline, spline_prior, spurious, batch_num, func_for_delta, hyperparams, folder, mf_update, do_conv, plotdir)

    func(nst_seed)


if __name__ == '__main__':
    
    from pipeline.hyperparameters import Hyperparams
    
    ### run a single star ###
    
    #kepid = np.array(pd.read_csv(home + 'load/datasets/star_info_withplanets.csv', index_col= False, sep='\t')['kepid'], dtype= int)[0]
    #kepid = np.array(pd.read_csv(home + 'load/datasets/star_info_gaia.csv', index_col= False, sep='\t')['kepid'], dtype= int)[951]
    
    hyp = Hyperparams(
        do_nonstat_mf = False, 
        test_recovery= False,
        period_min = 2.,
        #cutoff_fa_eliminate= 4.
        scan = 1
        )


    #kepids = [5184911, ]#[3852872,8555967,8880317,10253977,5802205,3629119,10844823,9650579,11714107][1:2]

    # koi_for_nst = pd.read_csv(home + 'load/datasets/test_recovery_steve.csv')
    # kepids = koi_for_nst['kepid'].drop_duplicates().to_list()[:1]

    # kepid = 12403119 #np.array(pd.read_csv(home + '/load/datasets/star_info_gaia.csv', index_col= False, sep='\t')['kepid'], dtype= int)[100]

    #kepid = [7026522, 9896456][0]
    # kepid = 11773022

    tessid = 100100827

    mainn(tessid, 'individual', hyp, TESS=True)

    #mainn(None, 'individual', hyperparams, None)
    # num_realizations = 100
    # kepids = koi_for_nst['kepid'].drop_duplicates().to_list()
    # job_map= job_map_nst(kepids, num_realizations)
    # id = 1002
    # hyperparams.scan = 1
    # hyperparams.cutoff_eliminate = np.inf
    # mainn(id, 'individual', hyperparams, job_map= job_map)
    
    
    print('Done.')