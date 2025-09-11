import math
import numpy as np
from scipy import integrate, interpolate, optimize
from batman.transitmodel import _quadratic_ld

from detection.cpu.scan import fold_lc
from fourier.MF import MatchedFilter, rfft, stationary
from planet.nst import basic
from constants import *
dt_tess = 0.001388882


class Info():
    
    def __init__(self, snr, params, spline, success):
        self.snr = snr
        self.params = params # [period, phase, tau]
        self.spline = spline
        self.success = success # optimization success
        # later on, goodness of fit is added
    
    def __str__(self):
        return 'SNR= {0:.3}, params= ({1:.4}, {2:.3}, {3:.4})'.format(self.snr, *self.params)
    
    def save(self, name):
        np.savez(name, params = self.params, snr = self.snr, residuals = self.residuals, success = self.success, cov= self.cov)
    

class TTVInfo():
    
    def __init__(self, params, ttvs, found, spline, success):
        self.params = params # [period, phase, tau]
        self.ttvs = ttvs #[[t, tau, snr, optimization success] for each (found) transit]
        self.found = found # bool array, e.g. [1001110] corresponds to the planet which should have 7 transits but 2nd, 3rd and 7th transits were not found.
        self.spline = spline
        self.snr = np.sqrt(np.sum(np.square([ttvs[i][2] for i in range(len(ttvs))]))) # total snr
        self.success = success
        # later on, goodnes of fit and covariance matrix of the ttvs will be added
    
    
    def __str__(self):
        return 'SNR= {0:.3}, params= ({1:.4}, {2:.3}, {3:.3}), found= {4}, success= {5}, ttvs = {6}'.format(self.snr, *self.params, self.found, self.success, self.ttvs)
    
    
    def save(self, name):
        np.savez(name, params = self.params, 
                 ttvs = self.ttvs, found = self.found, snr = self.snr,
                 residuals = self.residuals[0], residuals_ttv = self.residuals[1], cov = self.cov, success= self.success)
    
    
    def optimization_parameters(self):
        """ TTV_data[i] = [t_i, tau_i, snr_i] . 
        Gets it into the form used by optimization:
         [t1, t2, ..., tn, tau1, tau2, ... taun]
        """
        return (self.ttvs[:, :2].T).reshape(len(self.ttvs)*2)
     
    
    def update_ttvs(self, params, snr, success):
        self.success = success
        self.snr = snr
        
        n = len(params)//2
        self.ttvs= np.transpose([params[:n], params[n:], self.ttvs[:, 2]])
    
    
    def get_ttvs(self):
        """convert TTV data to TTVs (in days) for each transit"""
        delt = []
        remainder = self.found
        counter, k = 0, 0
        period, phase = self.params[:2]

        while remainder > 0: # go over all transits
            isfound = remainder % 2 # True if this transit was found and is in the ttv data
            remainder = remainder // 2
            if isfound:
                delt.append((self.ttvs[counter][0] - (k * period + phase)))
                counter += 1
            else:
                delt.append(0.)
            
            k += 1
                
        return delt
    



def intensity(relative_distance, u1, u2):
    """relative tangential distance is in units of radius of star, normed so that depth of transit is 1"""
    mi = np.sqrt(1 - np.square(relative_distance)) # cosinus of angle between observer, center of a star and a point on a surface
    return (1 - u1 * (1 - mi) - u2 * np.square(1 - mi)) #1 at the centre of transit (relative distance = 0)


def intensity_1term(relative_distance, u1, u2):
    mi = np.sqrt(1 - np.square(relative_distance)) # cosinus of angle between observer, center of a star and a point on a surface
    return - (1 - mi)


def intensity_2term(relative_distance, u1, u2):
    mi = np.sqrt(1 - np.square(relative_distance)) # cosinus of angle between observer, center of a star and a point on a surface
    return - np.square(1 - mi)


def limb_darkening(relative_time, r, u1, u2, I):
    """relative_time = abs(time-phase)/time_half_transit
       r = R_planet/R_star
       I = intensity function, e.g. quadratic limb darkening law"""

    if relative_time > (1+r): #not transiting
        return 0.0

    elif relative_time > (1-r): #partially transiting
        return integrate.quad(lambda d: 2 * math.acos((-r**2 + relative_time**2 + d**2) / (2 * relative_time * d)) * d * I(d, u1, u2), relative_time - r, 1)[0]/(math.pi*r**2)

    elif relative_time > r: #fully transiting, but not covering the center of the primary
        return integrate.quad(lambda d: 2 * math.acos((-r**2 + relative_time**2 + d**2) / (2 * relative_time * d)) * d * I(d, u1, u2), relative_time - r, relative_time + r)[0]/(math.pi*r**2) #because A is a value at minimum

    else: #fully transiting and convering the center of the primary
        integral1 = integrate.quad(lambda d: 2 * math.acos((-r**2 + relative_time**2 + d**2) / (2 * relative_time * d)) * d * I(d, u1, u2), r-relative_time, relative_time + r)[0]/(math.pi*r**2)
        return integral1 + central_integral(u1, u2, r - relative_time) * np.square(r - relative_time)/ (r**2)

    # elif relative_time > 0.4: #smoothly interpolates between calculating the integral over planet's shadow and just assuming the integrand is constant over the shadow
    #     f1 = integrate.quad(lambda d: 2 * math.acos((-r**2 + relative_time**2 + d**2) / (2 * relative_time * d)) * d * I(d, u1, u2), relative_time - r, relative_time + r)[0]/(math.pi*r**2) #because A is a value at minimum
    #     f2 = I(relative_time, u1, u2)
    #     alpha = (1 - np.cos(np.pi * (relative_time - 0.4) / 0.1)) * 0.5
    #     return alpha * f1 + (1-alpha) *f2
    # else: #it is useless to compute integral as a difference is 1.5 promile (for small r)
    #    h = 1 - (2.0 /3.0) * (1- np.power(1 - np.square(r), 1.5)) / np.square(r)
    #    return 1 - u1 * h - 2* u2 * (h - 0.25* r**2)



def central_integral(u1, u2, d_given):
    """limb darkening of a centered shadow with radius d (in star's radius units)"""
    d = min(d_given, 1.0)
    h = 1.0 - (2.0 / 3.0) * (1 - np.power(1 - d**2, 1.5)) / d**2

    return 1 - u1 * h - 2 * u2 * (h - 0.25 * d ** 2)


# def integrated_limb_darkening(t, delta_t, r):
#     intervals = np.linspace(t-delta_t/2, t+delta_t/2, 10)
#     return integrate.simps([limb_darkening(x, r) for x in intervals], intervals)/delta_t


def prepare_shape(r, u1, u2):
    """"fits profile with spline before fitting so that profile is computed faster when fitting."""
    relative_times = np.linspace((-1 - r) * 1.1, (1 + r) * 1.1, 1000)

    if r > 0.1:
        #We switch to batman for light curve computation as our routine is really developed for small r (and because batman has some large errors close to the transit borders if r is small!)
        fluxes = -(_quadratic_ld._quadratic_ld(relative_times, r, u1, u2, 1) - 1)
        norm = min(1.0, r**2) * central_integral(u1, u2, r) / central_integral(u1, u2, 1.0)
        fluxes /= norm

    elif r == 0:
        fluxes = [0.0 if abs(t) > 1 else intensity(abs(t), u1, u2) for t in relative_times]

    else:
        fluxes = [limb_darkening(abs(t), r, u1, u2, intensity) for t in relative_times]
        norm = central_integral(u1, u2, r)
        fluxes /= norm

    spline = interpolate.splrep(relative_times, fluxes)

    return (r, spline)



def prepare_shape_derivative(r, u1, u2):
    """"splines for derivative of the light curve with respect to u1 and u2"""

    relative_times = np.linspace((-1 - r) * 1.1, (1 + r) * 1.1, 1000)

    if r == 0:
        fluxes1 = [0.0 if abs(t) > 1 else intensity_1term(abs(t), u1, u2) for t in relative_times]
        fluxes2 = [0.0 if abs(t) > 1 else intensity_2term(abs(t), u1, u2) for t in relative_times]


    if r < 0.1:
        fluxes1 = np.array([limb_darkening(abs(t), r, u1, u2, intensity_1term) for t in relative_times]) / central_integral(u1, u2, r)
        fluxes2 = np.array([limb_darkening(abs(t), r, u1, u2, intensity_2term) for t in relative_times]) / central_integral(u1, u2, r)

    if r > 0.1:
        #We switch to batman for light curve computation as our routine is really developed for small r (and because batman has some large errors close to the transit borders if r is small!)
        norm = min(1.0, r**2) * central_integral(u1, u2, r) #/ central_integral(u1, u2, 1.0)

        fluxes0 = -(_quadratic_ld._quadratic_ld(relative_times, r, 0.0, 0.0, 1) - 1) / norm

        fluxes1 = -(_quadratic_ld._quadratic_ld(relative_times, r, 1.0, 0.0, 1) - 1) * (2.0/3.0) / norm
        fluxes1 -= fluxes0

        fluxes2 = -(_quadratic_ld._quadratic_ld(relative_times, r, 0.0, 1.0, 1) - 1) * (5.0/6.0) / norm
        fluxes2 -= fluxes0


    spline1 = interpolate.splrep(relative_times, fluxes1)
    spline2 = interpolate.splrep(relative_times, fluxes2)

    return (r, spline1), (r, spline2)




def lc(time, period, phase_given, time_half_transit, spline_object, func_for_delta= None, der = 0):
    """
       Args:
            time -> times at which we want to get light curve af a transiting planet. It must be a collection of equally spaced time intervals starting with zero: [0, dt, 2dt, ...].
            period -> planet's period
            phase -> time of the first transit
            time_half_transit -> half of the transit duration
            spline_object -> object which contains information about the (precomputed) transit shape. Can be obtained by prepare_shape(r, u1, u2)
            nst_random_numbers 0 -> used to perturb times of transit. The n-th transit it loacted at time = phase + period*(n+delta[n])
            der -> which derivtive of the light curve to return,
                0: returns the light curve
                1: it's Jacobian (with respect to period, phase and time_half_duration)
                2: the Hessian
        Returns:
            if der = 0: light curve (shape = (len(time)))    
            if der = 1: Jacobian of the light curve (shape = (3, len(time)))
            if der = 2: Hessian of the light curve (shape = (3, 3, len(time)))
    """
    
    
    ### process the inputs
    phase = np.remainder(phase_given, period)
    r, spline = spline_object[0], spline_object[1]
    
    if func_for_delta == None:
        deltas = basic(len(time))(period/dt_tess)
    else:
        deltas = func_for_delta(period/dt_tess) * dt_tess

    
    ### functions for computing the lightcurve
    def f_point(t, k):
        """t = time[i] - phase[k]"""
        return -interpolate.splint((t - dt_tess * 0.5) / time_half_transit, (t + dt_tess * 0.5) / time_half_transit, spline) * time_half_transit / dt_tess

    def J_point(t, k):
        tplus, tminus = (t + dt_tess * 0.5) / time_half_transit, (t-dt_tess*0.5)/time_half_transit
        Uint = interpolate.splint(tminus, tplus, spline)
        Uplus, Uminus = interpolate.splev(tplus, spline), interpolate.splev(tminus, spline)
        J1 = -(Uplus - Uminus)/dt_tess
        J0 = J1*k
        J2 = (-(tplus*Uplus-tminus*Uminus) + Uint) / dt_tess
        return -np.array([J0, J1, J2])


    def H_point(t, k):
        tplus, tminus = (t + dt_tess * 0.5) / time_half_transit, (t - dt_tess * 0.5) / time_half_transit
        DUplus, DUminus = interpolate.splev(tplus, spline, der=1), interpolate.splev(tminus, spline, der=1)

        H3 = (DUplus - DUminus) / (time_half_transit * dt_tess)  # H22
        H0 = H3 * k ** 2  # H11
        H1 = H3 * k  # H12
        H4 = (tplus * DUplus - tminus * DUminus) / (time_half_transit * dt_tess)  # H23
        H2 = H4 * k  # H13
        H5 = (DUplus * tplus ** 2 - DUminus * tminus ** 2) / (time_half_transit * dt_tess)  # H33

        return -np.array([H0, H1, H2, H3, H4, H5])

    
    if der == 0:
        results = np.zeros(len(time)) # light curve for a planet transit
        point_function = f_point
    elif der == 1:
        results = np.zeros((len(time), 3)) # Jacobian of the light curve
        point_function = J_point
    else:
        results = np.zeros((len(time), 6))  # Hessian folded like: H11, H12, H13, H22, H23, H33
        point_function = H_point
        
    
    ### prepate the output form
    def output_form(results):
        if der == 0:
            return results

        elif der == 1:
            return np.transpose(results)

        else: #der == 2
            return np.array([[results[:, 0], results[:, 1], results[:, 2]],
                            [results[:, 1], results[:, 3], results[:, 4]],
                            [results[:, 2], results[:, 4], results[:, 5]]])

    ### check for invalid values of the parameters
    if period < 0.0:
        #print(period, phase_given, time_half_transit)
        #raise ValueError('Invalid period in transit.lc')
        return output_form(results)
    elif time_half_transit < 0.0:
        #print(period, phase_given, time_half_transit)
        #raise ValueError('Invalid time_half_transit in transit.lc')
        return output_form(results)

    ### build the light curve, transit by transit    
    try:
        number_of_transits = 2 + int((time[-1] - (phase-time_half_transit*(1+r)))/period) # 1 + int() is the ceil function. We add +1 just to be on the safe side.
    except ValueError:
        print(period, phase, time_half_transit)
        raise ValueError
    

    for index_transit in range(number_of_transits): # iterate over transits
        
        # extent of the transit
        phase_transit = loc(index_transit, period, phase, deltas[index_transit if index_transit < len(deltas) else len(deltas)])
   
        imin = np.clip(int((phase_transit - time_half_transit * (1+r)) / dt_tess)-1, 0, len(time)-1) # make sure that the index does not fall out of range
        imax = np.clip(int((phase_transit + time_half_transit * (1+r)) / dt_tess)+2, 0, len(time)-1)
        
        for i in range(imin, imax): # each point in the transit
            results[i] = point_function(time[i] - phase_transit, index_transit)
    
    
    return output_form(results)

        
def loc(k, period, phase, delta):
    return phase + k * period + delta


def template(lenTime, time_half_duration, spline):
    """prepares template with lenght len(invVar) of shape [right part of U shape, 0, 0, 0, ..., 0, left part of U shape]."""

    integer_time_half_duration = int(time_half_duration/dt_tess)+1
    template = [- interpolate.splint((t - dt_tess*0.5)/time_half_duration, (t +dt_tess*0.5)/time_half_duration, spline[1])*time_half_duration/dt_tess for t in (np.arange(-integer_time_half_duration, integer_time_half_duration+1))*dt_tess]
    template = np.concatenate((template[integer_time_half_duration:], np.zeros(lenTime - len(template)), template[:integer_time_half_duration]))
    
    return template




def bank(lenflux, spline, taus):
    """prepares multiple planet templates, corresponding to the width_arr"""

    templates = np.array([template(lenflux, tau, spline) for tau in taus])

    borders = (1.3 * taus / dt_tess).astype(int)
    return {'name': "planet", 
            'templates': templates, 
            'both signs': False, 
            'borders': borders}




def lc_tdv(time, params, spline_object):
    """params = [t1, t2, ... tn, tau1, tau2, ... taun]"""
    num_transits = len(params) //2

    def point_function(t, tau):
        """t = T[i] - phase[k]"""
        return -interpolate.splint((t - dt_tess * 0.5) / tau, (t + dt_tess * 0.5) / tau, spline) * tau / dt_tess

    results = np.zeros(len(time))  # light curve for a planet transit

    rr, spline = spline_object[0], spline_object[1]

    for i_transit in range(num_transits):
        phase = params[i_transit]
        tau = params[num_transits + i_transit]
        for i in range(int((phase - tau * (1 + rr)) / dt_tess) - 1, int((phase + tau * (1 + rr)) / dt_tess) + 2):  # +1 because we want to include index after the end of the transit and +1 because of the indexing in python
            results[i] = point_function(time[i] - phase, tau)

    return results



def grad_lc_tdv(time, params, spline_object):
    """returns the gradient of ttv_planet light curve with respect to params
        returned shape is (len(params), len(time))"""
    num_transits = len(params) //2

    J = np.zeros((len(params), len(time)))  # Jacobian of the light curve

    def point_function(t, tau):
        tplus, tminus = (t + dt_tess * 0.5) / tau, (t - dt_tess * 0.5) / tau
        Uint = interpolate.splint(tminus, tplus, spline)
        Uplus, Uminus = interpolate.splev(tplus, spline), interpolate.splev(tminus, spline)
        Jt = -(Uplus - Uminus) / dt_tess  # time of transit
        Jtau = (-(tplus * Uplus - tminus * Uminus) + Uint) / dt_tess  # transit duration
        return -Jt, -Jtau

    rr, spline = spline_object[0], spline_object[1]

    for i_transit in range(num_transits):
        phase, tau = params[i_transit], params[num_transits + i_transit]

        for i in range(int((phase - tau * (1 + rr)) / dt_tess) - 1, int((phase + tau * (1 + rr)) / dt_tess) + 2):  # +1 because we want to include index after the end of the transit and +1 because of the indexing in python
            J[i_transit, i], J[num_transits + i_transit, i] = point_function(time[i] - phase, tau)

    return J




def periodic_mf(Time, Flux, invVar, Pk, spline, func_for_delta= None, opt_method= 'lm', maxiter= 40, mf_update= stationary):
    """initializes the general MatchedFilter object by passing it the given data and the periodic light curve model"""
    
    mf = MatchedFilter(Time, Flux, invVar, Pk,
                        lambda t, z: lc(t, *z, spline, func_for_delta),
                        lambda t, z: lc(t, *z, spline, func_for_delta, der = 1),
                        lambda t, z: lc(t, *z, spline, func_for_delta, der = 2),
                        opt_method= opt_method,
                        maxiter= maxiter,
                        update = mf_update
                        )
    
    mf.get_bounds = lambda z: optimize.Bounds(lb= np.array([z[0] - z[2], max(0., z[1] - 1.3 * z[2]), 0.7 * dt_tess]), # optimized for period_min = 2 days, so tau_k = q cbrt(period) = 0.05 = 2.5 dt (for typical q= 0.04). We pick 0.7 dt to allow also for tau < tauK.
                                              ub= np.array([z[0] + z[2], min(z[1] + 1.3 * z[2], z[0]), 0.25 * z[0]]))
    
    return mf


def single_mf(Time, Flux, invVar, Pk, spline, opt_method = 'lm', maxiter= 40, mf_update= stationary):
    """initializes the general MatchedFilter object by passing it the given data and the single-transit light curve model"""
    
    mf = MatchedFilter(Time, Flux, invVar, Pk,
                        lambda t, z: lc(t, 2 * Time[-1], *z, spline),
                        lambda t, z: lc(t, 2 * Time[-1], *z, spline, der = 1)[1:, :],
                        lambda t, z: lc(t, 2 * Time[-1], *z, spline, der = 2)[1:, 1:, :],
                        opt_method= opt_method,
                        maxiter= maxiter,
                        update = mf_update
                        )
    
    mf.get_bounds = lambda z: optimize.Bounds(lb= np.array([z[0] - 1.3 * z[1], 0.7 * dt_tess]), 
                                              ub= np.array([z[0] + 1.3 * z[1], np.inf]))
    
    return mf
    


def plot_lc(Time, residual_flux, invVar, model, props, ttvs, savename, maintitle= None):
    """flux= Flux - FGP"""
    
    period, phase, tau = props.params
    span = max(3 * tau, 8*dt_tess)
    xlims = (phase - span, phase + span)
    
    ### folds ###
    w = np.ones(len(invVar))
    f_data = fold_lc(period/dt_tess, residual_flux, w, invVar, ttvs/dt_tess)[0]
    f_model = fold_lc(period/dt_tess, model, w, invVar, ttvs/dt_tess)[0]
    t = np.arange(len(f_data)) * dt_tess
        
    ### title ###
    plt.figure(figsize = (13, 8))
    
    sub = 'snr = {0:.2}, period = {1:.4}, tau = {2:.2}'.format(props.snr, period, tau)
    plt.suptitle(sub if maintitle == None else (maintitle + '\n' + sub))

    ### local view ###
    plt.subplot(2, 2, 1)
    plt.plot(t, f_data, '.-', color='black')
    plt.plot(t, f_model, color= 'teal')
    plt.xlim(*xlims)

    plt.xlabel('phase [days]')
    plt.ylabel('folded flux')
    
    ### global view ###
    plt.subplot(2, 1, 2)
    plt.plot(t, f_data, '.-', color='black')
    plt.plot(t, f_model, color= 'teal')

    ylims = plt.gca().get_ylim()
    plt.plot(np.ones(2) * phase, ylims, '--', color = 'teal', alpha = 0.5) # location of the transit
    plt.plot(np.ones(2) * np.mod(phase + period * 0.5, period), ylims, '--', color = 'xkcd:ruby', alpha = 0.5) # most probable location of the potential secondary transit
    plt.xlabel('phase [days]')
    plt.ylabel('folded flux')
    plt.ylim(ylims)
    
    ### local view (individual transits) ###
    plt.subplot(2, 2, 2)
    num_trans = 1 + int((Time[-1] - phase - tau) / period)
    for itrans in range(num_trans):
        plt.plot(Time - itrans * period, residual_flux, '.-', alpha= 0.4, label='{0}'.format(np.round(phase + itrans * period, 2)))
    plt.plot(t, f_model, color = 'teal', lw = 4) 
    if num_trans < 7:
        plt.legend()
        
    plt.xlabel('phase [days]')
    plt.xlim(*xlims)
    
    plt.savefig(savename)
    plt.close()
    
    

def plot_transit_contribution(itrans, Time, flux, invVar, params):

    period, phase, tau = params
    span = max(3 * tau, 8*dt_tess)
    xlims = (-span, span)
    t_trans = phase + itrans * period
    mask_transit = np.abs(Time - t_trans) < (period * 0.5)
    invVar2 = np.copy(invVar)
    invVar2[mask_transit] = 0.
    
    ### folds ###
    w = np.ones(len(invVar))
    f_all = fold_lc(period/dt_tess, flux, w, invVar, np.zeros(2000))[0]
    f_without = fold_lc(period/dt_tess, flux, w, invVar2, np.zeros(2000))[0]
    t = np.arange(len(f_all)) * dt_tess
    
    # all transits
    
    def get_xy(x, y):
        mask = (x > xlims[0]) & (x < xlims[1])
        return x[mask], y[mask]
    
    plt.plot(*get_xy(t - phase, f_all), '.-', color='black', label= 'both')
    plt.plot(*get_xy(Time - t_trans, flux), '.-', color= 'xkcd:ruby', label= 'just transit')
    plt.plot(*get_xy(t - phase, f_without), '.-', color= 'teal', label= 'everything else')

    plt.xlabel('time [days]')
    plt.ylabel('flux')
    plt.legend()  
      
    return invVar2