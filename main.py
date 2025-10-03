# Some numpy fix
import numpy as np
if not hasattr(np, "int"):
    np.int = int
import time as clock

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
    planet_props = read_known_planets(star, t_start, hyperparams.test_recovery)    
    quarters = quarter_indexes(Time, quarter_beginnings)
    max_num_periods = (int)(2 * (Time[-1] / hyperparams.period_min)) # factor of two just to be sure

    spline = transit.prepare_shape(0, star.u1, star.u1) #spline for the planet transit shape (will be used for the search, assuming small planets)
    spline_prior = tauprior.load(star.err_q) # spline for the shape of the transit duration prior

    if hyperparams.show:
        plt.figure(figsize = (10, 5))
        plt.plot(Time, Flux, '.')
        plt.ylim(-5, 5)
        plt.savefig(plotdir + 'flux_normalized.png')
        plt.close()

    return star, t_start, Time, Flux, invVar, max_num_periods, quarters, planet_props, spline, spline_prior

def LOAD_TESS(tessid, folder, hyperparams, batch_num, n_bins) :

    # Saving Pictures
    tessid_folder_name = str(tessid) + ('_' + str(batch_num) if (batch_num != 0) else '')
    plotdir = (scratch + folder + '/plots/' + tessid_folder_name + '/') if hyperparams.show else None
    if hyperparams.show:
        scratch_structure.make_dir(plotdir)

    # Loading Data
    star = StarInfo_init_tess(tessid, 'star_info_gaia')#'star_info_withplanets')

    # We read in each section Seperately
    all_time, all_flux, all_flags, quarter_beginnings = read_alternative(tessid, n_bins, injected = False)
    num_transits = len(all_time)

    # Do Total case
    time, flux, flags = np.concatenate(all_time), np.concatenate(all_flux), np.concatenate(all_flags)
    average, flux_sigma = gaussianize.rescale(flux) #flux_sigma: we rescale the flux to unit variance of the Gaussian part of the noise distribution flux_sigma * our_flux = original_flux (which is measured in units of star's flux)
    star = star._replace(sigma_star_flux_units = flux_sigma)
    flux = (flux - average) / flux_sigma
    t_start = time[0]
    time -= t_start
    
    Time, Flux, invVar = add_zeros_tess(time, flux, hyperparams, zero_paddling=2000 // n_bins)
    max_num_periods = (int)(2 * (Time[-1] / hyperparams.period_min)) # factor of two just to be sure

    quarters = quarter_indexes(Time, quarter_beginnings)

    planet_props = read_known_planets_tess(star, t_start, hyperparams.test_recovery)

    # Split Up Into individual Transits for Easier Search Later
    Times, Fluxes, invVars, planet_propes, index_start = [], [], [], [], []
    for i in range(num_transits) :
        average, flux_sigma = gaussianize.rescale(all_flux[i]) 
        all_flux[i] = (all_flux[i] - average) / flux_sigma
        t_start_i = all_time[i][0]
        all_time[i] -= t_start_i
        
        Time_i, Flux_i, invVar_i = add_zeros_tess(all_time[i], all_flux[i], hyperparams, zero_paddling=2000 // n_bins)
        planet_props_i = read_known_planets_tess(star, t_start_i, hyperparams.test_recovery)

        Times.append(Time_i); Fluxes.append(Flux_i); invVars.append(invVar_i)
        planet_propes.append(planet_props_i)

        index_start.append( np.where( abs(Time + t_start - t_start_i) < 0.001 * n_bins )[0][0] )
    seperate_transists = [Times, Fluxes, invVars, planet_propes, index_start]
        
    # Spline
    spline = transit.prepare_shape(0, star.u1, star.u2) #spline for the planet transit shape (will be used for the search, assuming small planets)
    spline_prior = tauprior.load(star.err_q) # spline for the shape of the transit duration prior

    if hyperparams.show:
        plt.figure(figsize = (10, 5))
        plt.plot(Time, Flux, '.')
        plt.ylim(-5, 5)
        plt.savefig(plotdir + 'flux_normalized.png')
        plt.close()

    return star, t_start, Time, Flux, invVar, max_num_periods, quarters, planet_props, spline, spline_prior, seperate_transists


def mainn(kepid, folder, hyperparams, 
          nst_seed= None, inj_seed = None,
          inj_params= None, batch_num = 0, start_time = 0.0, bins = 1):

    # What is the relvant survey
    survey = hyperparams.survey
    TESS = (survey == "TESS")

    ## TO DO: Add survey specific folder structure 
    kepid_folder_name = str(kepid) + ('_' + str(batch_num) if (batch_num != 0) else '')
    plotdir = (scratch + folder + '/plots/' + kepid_folder_name + '/') if hyperparams.show else None
    if hyperparams.show:
        scratch_structure.make_dir(plotdir)
    
    ### load data ###
    if TESS :
        star, t_start, Time, Flux, invVar, max_num_periods, quarters, planet_props, spline, spline_prior, seperate_transists = LOAD_TESS(kepid, folder, hyperparams, batch_num, bins)
    else : 
        star, t_start, Time, Flux, invVar, max_num_periods, quarters, planet_props, spline, spline_prior = LOAD_KEP(kepid, folder, hyperparams, batch_num)

    print(f"Size of Input Flux Array is {len(Flux)}")

    time_now = clock.time()
    print(f"Reading data and preprocessing took {int(time_now - start_time)} seconds.")

    ### jointly fit FGP, planets and gaussianize outliers ###
    # For TESS, fit each quarter seperately
    if TESS : 

        model = np.zeros_like(Flux); FGP = np.zeros_like(Flux)
        for i in range( len(seperate_transists[0]) ) :
            Time_i = seperate_transists[0][i]
            Flux_i = seperate_transists[1][i]
            invVar_i = seperate_transists[2][i]
            planet_props_i = seperate_transists[3][i]
            index_start = seperate_transists[4][i]

            planet_props_i, Fluxi, modeli, Pk, FGPi, bandwidth, _, _ = joint.iteration(Time_i, Flux_i, invVar_i, planet_props_i, sharp_freq= None)

            # Change Model, GP and Flux
            N = len(Fluxi)

            # Take Care of edge case
            n = np.min([index_start + N, len(Flux)])
            if n == len(Flux) :
                k = len(Flux) - index_start
            else :
                k = N

            Flux[ index_start : n] = Fluxi[:k]
            model[index_start : n] = modeli[:k] 
            FGP[  index_start : n] = FGPi[:k]

        # Get Pk and Bandwidth
        from fourier import GP, notch
        mask_planets = ttv.mask_planets(Time, planet_props)
        invVar_noplanets = np.copy(invVar)
        invVar_noplanets[mask_planets] = 0

        mask = invVar_noplanets < 0.5
        flux_for_FGP = np.copy(Flux)
        flux_for_FGP[mask] = FGP[mask]
        PSD = np.square(np.abs(np.fft.rfft(flux_for_FGP)))
        bandwidth = GP.automatic_bandwidth(PSD)
        Pk = GP.getPSD(flux_for_FGP, bandwidth)

        # Get peaks
        N = len(Flux)
        freq = np.arange(len(PSD)) / (N * dt)
        PSD_for_peaks = PSD[freq < 20] # Do a frequency cutoff for speed
        peaks = notch.sharp_peaks(PSD_for_peaks)
        has_peaks = (len(peaks) != 0)
        if has_peaks:
            sharp_freq = peaks/Time[-1]
            print("Power Spectrum has sharp peaks.")
        else:
            sharp_freq = np.array([])

        # Notch Filter
        Pk *= notch.filter(len(Flux), sharp_freq)

    else :
        planet_props, Flux, model, Pk, FGP, bandwidth, mask_planets, sharp_freq = joint.iteration(Time, Flux, invVar, planet_props, sharp_freq= None)

    time_now2 = clock.time()
    print(f"Initial Fit took {int(time_now2 - time_now)} seconds.")

    ### Some Plots ###
    # Gaussian Process
    planet_period, _, _ = planet_props[0].params

    plt.figure(figsize=(16,8), dpi=550)
    plt.plot(Time, Flux, label="Observed Flux", alpha=0.33)
    plt.plot(Time, FGP, label="GP of Stellar Variability", color="red")
    plt.plot(Time, model, label="Fit of Planet Transit", alpha=0.5)
    plt.xlabel("t [days]"); plt.legend(); 
    plt.xlim([0.0, 5 * planet_period]); plt.ylim([Flux.min() * 1.05, 3])
    plt.savefig(scratch + folder + '/plots/' + str(kepid) + '/flux_plot_zoom.png')
    plt.close()

    plt.figure(figsize=(16,8), dpi=550)
    plt.plot(Time, Flux, label="Observed Flux")
    plt.plot(Time, FGP, label="GP of Stellar Variability")
    plt.plot(Time, model, label="Fit of Planet Transit")
    plt.xlabel("t [days]"); plt.legend()
    plt.savefig(scratch + folder + '/plots/' + str(kepid) + '/flux_plot.png')
    plt.close()

    # Plot Power Spectrum
    N = len(Flux)
    lenPk = N // 2
    freq = np.arange(lenPk) / (N * dt)

    from fourier import GP
    noise = Flux - FGP - model
    PSD_noise = np.square(np.abs(np.fft.rfft(noise)))
    bandwidth = GP.automatic_bandwidth(PSD_noise)
    Pk_noise = GP.getPSD(noise, bandwidth) 
    Pk_noise = np.ones_like(freq) * np.mean(Pk_noise[250:])

    plt.figure(figsize=(16,8), dpi=550)
    plt.plot(freq, Pk, label="source + noise")
    plt.plot(freq, Pk_noise, label="noise")
    plt.plot(freq,  abs(Pk - Pk_noise), label="source")
    plt.yscale('log'); plt.xlim([0, 15]); plt.legend()
    plt.ylim([1e2, 1e7])
    plt.ylabel(r"$P(\nu)$"); plt.xlabel(r"$\nu$ [1/days]")
    plt.savefig(scratch + folder + '/plots/' + str(kepid) + '/powerspectrum.png')
    plt.close()


    Pk_raw = np.square(np.abs(np.fft.rfft(Flux - model)))
    plt.figure(figsize=(16,8), dpi=550)
    plt.plot(freq, Pk_raw[1:], label="source + noise")
    plt.yscale('log'); plt.xlim([0, 15]); plt.legend()
    plt.ylim([1e2, 1e7])
    plt.ylabel(r"$P(\nu)$"); plt.xlabel(r"$\nu$ [1/days]")
    plt.savefig(scratch + folder + '/plots/' + str(kepid) + '/raw_powerspectrum.png')
    plt.close()

    # Plot Noise
    mask = invVar > 0.5
    noise = noise[mask]

    from scipy.stats import norm
    x_axis = np.linspace(-3, 3, 1000)
    plt.figure(figsize=(16,8), dpi=550)
    plt.hist(noise, density=True, bins=int(np.sqrt(len(noise))), label="Stellar Noise")
    plt.plot(x_axis, norm.pdf(x_axis), label="Standard Gaussian")
    plt.xlabel(r"$n / \sigma$"); plt.ylabel(r"$p(n)$")
    plt.yscale('log'); plt.legend()
    plt.savefig(scratch + folder + '/plots/' + str(kepid) + '/noise.png')
    plt.close()
    
    ##########################

    if len(planet_props) != 0:
        ### find TTVs ###
        planet_props = ttv.pipeline(Time, Flux, invVar, Pk, planet_props)

        max_num_periods = (int)(2 * (Time[-1] / hyperparams.period_min))
        
        [ttv.plots(Time, Flux, invVar, model, FGP, planet_props[k], np.zeros(max_num_periods), plotdir + 'init' + str(k), maintitle= 'Known planet') for k in range(len(planet_props))]

        ### redo the fit ###
        planet_props, Flux, model, Pk, FGP, bandwidth, mask_planets, _ = joint.iteration(Time, Flux, invVar, planet_props, Pk, sharp_freq)

        plt.figure(figsize=(16,8), dpi=550)
        plt.plot(Time, Flux, label="Observed Flux")
        plt.plot(Time, FGP, label="GP of Stellar Variability")
        plt.plot(Time, model, label="Fit of Planet Transit")
        plt.xlabel("t [days]"); plt.legend()
        plt.savefig(scratch + folder + '/plots/' + str(kepid) + '/second_flux_plot.png')
        plt.close()

        if hyperparams.show:
            plt.plot(Time, model+FGP, color= 'tab:blue')
            plt.plot(Time, FGP, color= 'cyan')
            plt.plot(Time, Flux, '.', color= 'black')
            plt.close()
        
        # save the known planets results
        var_noise = np.average(Pk[int(3 * dt * len(Flux)) + 1:]) / len(Flux) # average power spectrum above the FGP frequency cutoff
        planet_props = ttv.fit_quality(Time, Flux - model - FGP, invVar, var_noise, planet_props)
        planet_props = ttv.laplace(Time, Flux, invVar, Pk, planet_props)
        
        [ttv.plots(Time, Flux, invVar, model, FGP, planet_props[k], np.zeros(max_num_periods), plotdir + str(k), maintitle= 'Known planet') for k in range(len(planet_props))]
        [planet_props[iplanet].save(scratch + folder + '/known_planets/'+str(kepid)+'_' + str(iplanet)) for iplanet in range(len(planet_props))]

        # remove planets
        Flux[mask_planets] = FGP[mask_planets]
        invVar[mask_planets] = 0.    

    time_now3 = clock.time()
    print(f"TTVs and Second Fit took {int(time_now3 - time_now2)} seconds.")

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
    # local_nonstationarity, step_nonstationarity, rb_nonstationarity= nonstationary.pipeline(Flux, invVar, bandwidth, hyperparams, plotdir)
    # non_stationarity = [np.max(local_nonstationarity), np.argmax(local_nonstationarity) * dt, step_nonstationarity[0], rb_nonstationarity[0]]
    df = pd.DataFrame([[star.kepid, t_start, star.sigma_star_flux_units, np.max(Pk) / np.min(Pk), len(sharp_freq) != 0], ], 
                      columns = ['kepid', 't_start', 'sigma_star_flux_units', 'PSD_condition_number', 'has_sharp_peak'])
    df.to_csv(scratch + folder + '/stars/'+str(kepid)+'.csv', sep = '\t', index= False)

    ### individual transits and false alarms ###

    ### NOTE: I removed the plot command
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

    time_now4 = clock.time()
    print(f"Spurious Scenarious and False Alarms took {int(time_now4 - time_now3)} seconds.")

    ### detection and vetting

    def func(nst_seed):
        
        if nst_seed != None:
            period_min_nst = 2.
            rng = np.random.default_rng(nst_seed)
            func_for_delta = nst.transit_duration_limited(rng, max_num_periods, period_min_nst/dt, q_in_dt_units(star.q), 
                                                                                                    num_transit_durations_min= 2., frac_period_max= 0.4)
        else:
            func_for_delta = nst.basic(max_num_periods)
        time_nst = clock.time()
        print(f"nst took {int(time_now4 - time_nst)} seconds.")    
            
        if hyperparams.do_nonstat_mf:
            bandwidth_short = nonstationary.bandwidth_long_to_short(bandwidth, len(Time), hyperparams.stft_width)
            psd = nonstationary.get_psd(Flux, invVar, bandwidth_short, hyperparams.stft_width, hyperparams.stft_sep, sharp_freq)
            mf_update = nonstationary.get_update(hyperparams.stft_width, hyperparams.stft_sep, len(Time), func_for_delta)
            do_conv = lambda Flux, Pk: nonstationary.convolve(Flux, template_bank.templates, Pk, hyperparams.stft_width, hyperparams.stft_sep)

        else:
            psd = Pk
            mf_update = stationary
            do_conv = lambda Flux, Pk: template.convs(Flux, template_bank.templates, Pk)

        time_nonstat_mf = clock.time()
        print(f"non-stationary noise took {int(time_nst - time_nonstat_mf)} seconds.")
        
        # scan for periodic events
        template_bank, period_prior_normalization = prepare_for_detection(invVar, psd, star, spline_prior, hyperparams, func_for_delta_given= func_for_delta, do_nonstat_mf= hyperparams.do_nonstat_mf)
        if template_bank == None: # time span of the data is so short that even the shortest period cannot support three transits
            return 
        
        candidates_all = detect(Flux, invVar, psd, template_bank, hyperparams, kepid, hyperparams.do_nonstat_mf)
        
        time_detection = clock.time()
        print(f"detection took {int(time_detection - time_nonstat_mf)} seconds.")

        # vet the detections
        spurious = spurious_transits.SpuriousTransits(spurious_scenarios, func_for_delta, len(Time))
        vetting.main(Time, Flux, invVar, FGP, psd, template_bank, period_prior_normalization, candidates_all, star, spline, spline_prior, spurious, batch_num, func_for_delta, hyperparams, folder, mf_update, do_conv, plotdir)
        
        time_vetting = clock.time()
        print(f"vetting took {int(time_vetting - time_detection)} seconds.")

    func(nst_seed)

    time_now5 = clock.time()
    print(f"detection and vetting took  {int(time_now5 - time_now4)} seconds = {(time_now5 - time_now4) / 60.0 :.3} minutes.")


if __name__ == '__main__':
    
    ## Decide over how many bins to average
    n_bins = 10

    from pipeline.hyperparameters import Hyperparams
    
    ### run a single star ###    
    hyp = Hyperparams(
        do_nonstat_mf = False, 
        test_recovery= False,
        period_min = 2.,
        #cutoff_fa_eliminate= 4.
        scan = 1,

        # Specify Survey
        survey="TESS",

        # Also Adjust some Stuff
        period_density = 1000000 // n_bins,
        stft_sep = 2*(720 // n_bins),
        stft_width = (720 // n_bins)*(720 // n_bins)
        )

    ### Survey Specific Stuff ###
    # Get new dt
    import constants
    constants.dt = constants.dt_dic[hyp.survey] * n_bins

    # Load Everything with new dt
    # Usual Imports
    from constants import *
    import constants

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
    from TESS.load_tess import StarInfo_init_tess, add_zeros_tess, read_known_planets_tess, read_alternative

    ### Run Job ###
    tessids = np.array([310003988, 102264230, 458686847, 368805700, 610976842, 268766053,
       143143769,  16288184, 152476657, 377873569,  15445551, 239154970,
       165987272, 201642601,  63211674,  61538902,  69679391, 120757718,
        27990610, 285272237, 111991770, 346338552,  35022727, 350020859,
       293612446, 236887394, 299220166, 452808876, 138819293, 149918151,
       432549364, 159725995, 252479260, 204317710, 293457754,  26017005,
       448589187,  25375553,  27774415,  70440470, 283722336, 239154970,
       346626688, 159725995,  14570099,  85593751,  67666096,   9155187,
       284326455,  15445551,  50712784,  53750200, 164892194,  47911178,
       272213425, 237320326, 103751498, 380589029,  42821097, 346338552,
        38087018, 233948455,  31858843, 427332229, 308172249, 349827430,
        55315929, 310003988, 190998418, 136916387,  63452790,  61538902,
       271354351,  22529346, 232038798, 166836920, 287563610, 257774438,
       189625051,  19028197,  13349647, 332022997, 437261733,  26017005,
        32487566, 270468559, 267572272, 347329162,  44745133, 267572272,
       420779000,  66818296, 373693175,  54002556,  32949762, 272213425,
       166836920,  28230919, 162922904,  32487566,  33521996, 101721385,
       120610833, 191284318, 164458714, 270380593, 264508014, 158170594,
       347329162, 115524421, 429302040, 122441491, 233948455,  68577662,
       266593143, 458478250, 281459670, 281541555, 237320326, 273231214,
        53750200, 137899948, 271354351, 165987272, 204317710, 293612446,
        97409519,   7020254, 201604954,   8400842,  48506505, 356473034,
       164458714, 129979528,  69679391, 164892194, 286865921, 358516596,
       358516596,  13349647, 299158887,  13349647, 281731203,  27990610,
       157266693,  15445551, 176899385, 138644215,  27916356, 404340025,
       209752908, 336732616, 159725995, 176868951, 270380593, 120960812,
       233602827,   4616072, 366631954, 330687113, 243014114, 248075138,
       369455629, 192826603,  98545929, 368805700,  14614418, 251848941,
        98283926, 120255950, 248111245,  92226327, 115524421, 458686847,
       238176110,  27318774, 243200602, 112604564,  50712784, 350020859,
       204376737, 240681314, 268403451, 158623531, 255930614, 176220787,
       158388163,  32499655, 272213425, 236445129, 178367144,  36352297,
       283722336, 189625051,  73717937, 116156517,  13021029,   1129033,
       399954349, 436875934])    
    # noises = []
    # n = 1
    # for tessid in tessids :
    #     noise = mainn(tessid, 'individual', hyp)
    #     noises.append(noise)
    #     print('\n')
    #     print(f'{(100.0 * n) / (len(tessids))} % done                  ', end='\n')
    #     print('\n')
    #     n += 1
    # noises = np.concatenate(noises)
    # np.savetxt('noise.txt', noises)
    
    start_time = clock.time()

    # tessid = 100100827 ## Standard Test
    # tessid = 36352297 
    tessid = int( input("TESS id? ") )
    mainn(tessid, 'individual', hyp, start_time=start_time, bins=n_bins)

    end_time = clock.time()

    print(f"This took {int((end_time - start_time))} seconds = {(end_time - start_time) / 60.0:.3} minutes.")
    print('Done.')