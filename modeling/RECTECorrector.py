import numpy as np
from lmfit import Model, Parameters
from scipy.interpolate import interp1d

from RECTE import RECTE


def rampProfile(crate, slope, dTrap_s, dTrap_f, trap_pop_s,
                trap_pop_f, tExp, expTime):
    """Ramp profile for single directional scan

    And RECTE model parameters: number of traps, trapping coeeficient
    and trap life time

    :param crate: average count rate in electron/second
    :param slope: visit-long slope
    :param dTrap_s: extra trapped slow charges between orbits
    :param dTrap_f: extra trapped fast charges between orbits
    :param trap_pop_s: initially trapped slow charges
    :param trap_pop_f: initially trapped fast charges
    :param tExp: beginning of each exposure
    :param expTime: exposure time
    :returns: observed counts
    :rtype: numpy.array

    """

    tExp = (tExp - tExp[0])
    cRates = crate * (1 + tExp * slope / 1e7) / expTime
    obsCounts = RECTE(
        cRates,
        tExp,
        expTime,
        trap_pop_s,
        trap_pop_f,
        dTrap_s=[dTrap_s],
        dTrap_f=[dTrap_f],
        dt0=[0],
        lost=0,
        mode='scanning')
    return obsCounts


def RECTECorrector1(t, orbits, orbits_transit, counts, p, expTime,
                    include_transit=False):
    """correct the ackbar model for one directional scan observations

    :param t: time stamps of the exposures
    :param orbits: orbit number of the exposures
    :param orbits_transit: orbits in which transits/eclipses occur
    :param counts: observed counts
    :param p: Parameters objects to fit
    :param expTime: exposure time
    :returns: RECTE profile for correciting the light curve, best fit
    count rate array, ackbar output, slope
    :rtype: tuple of four numpy array

    """
    p = p.copy()
    p.add('crate', value=counts.mean(), vary=True)
    p.add('slope', value=0, min=-3, max=3, vary=True)
    rampModel = Model(rampProfile, independent_vars=['tExp', 'expTime'])
    t0 = t - t[0]  # make the first element in time array 0
    weights = np.ones_like(t)
    weights[np.in1d(orbits, orbits_transit)] = 0
    # weights[orbits == 3] = 0
    fitResult = rampModel.fit(
        counts,
        tExp=t0,
        expTime=expTime,
        params=p,
        weights=weights,
        method='nelder')
    fitResult = rampModel.fit(
        counts,
        tExp=t0,
        expTime=expTime,
        params=fitResult.params,
        weights=weights,
        method='powell')
    print(fitResult.best_values)
    RECTE_in = fitResult.params['crate'].value * (
        1 + t0 * fitResult.params['slope'] / 1e7)
    RECTE_out = fitResult.best_fit
    correctTerm = RECTE_out / RECTE_in
    return correctTerm, fitResult.params['crate'].value *\
        (1 + t0*fitResult.params['slope']/1e7), RECTE_out, \
        (1 + t0*fitResult.params['slope']/1e7)


def rampProfile2(crate1, slope1, crate2, slope2, dTrap_s, dTrap_f,
                 trap_pop_s, trap_pop_f, tExp, expTime, scanDirect):
    """Ramp profile for bi-directional scan And ackbar model

    :param crate1: average count rate in electron/second for the upward direction
    :param slope1: visit-long slope for upward direction
    :param crate2: average count rate in electron/second for downward direction
    :param slope2: visit-long slope for downward direction
    :param dTrap_s: extra trapped slow charges between orbits
    :param dTrap_f: extra trapped fast charges between orbits
    :param trap_pop_s: initially trapped slow charges
    :param trap_pop_f: initially trapped fast charges
    :param tExp: beginning of each exposure
    :param expTime: exposure time
    :param scanDirect: scan direction (0 or 1) for each exposure
    :returns: observed counts
    :rtype: numpy.array

    """
    tExp = (tExp - tExp[0])
    upIndex, = np.where(scanDirect == 0)
    downIndex, = np.where(scanDirect == 1)
    cRates = np.zeros_like(tExp, dtype=float)
    cRates[upIndex] = (
        crate1 * (1 + tExp * slope1 / 1e7) / expTime)[upIndex]
    cRates[downIndex] = (
        crate2 * (1 + tExp * slope2 / 1e7) / expTime)[downIndex]
    obsCounts = RECTE(
        cRates,
        tExp,
        expTime,
        trap_pop_s,
        trap_pop_f,
        dTrap_f=[dTrap_f],
        dTrap_s=[dTrap_s],
        dt0=[0],
        lost=0,
        mode='scanning')
    return obsCounts


def RECTECorrector2(t,
                    orbits,
                    orbits_transit,
                    counts,
                    p,
                    expTime,
                    scanDirect):
    """correct the ackbar model for one directional scan observations

    :param t: time stamps of the exposures
    :param orbits: orbit number of the exposures
    :param orbits_transit: orbits in which transits/eclipses occur
    :param counts: observed counts
    :param p: Parameters objects to fit
    :param expTime: exposure time
    :param scanDirect: scan direction (0 or 1) for each exposure
    :returns: RECTE profile for correciting the light curve, best fit
    count rate array, ackbar output, slope
    :rtype: tuple of four numpy array

    """
    upIndex, = np.where(scanDirect == 0)
    downIndex, = np.where(scanDirect == 1)
    p = p.copy()
    p.add('crate1', value=counts.mean(), vary=True)
    p.add('crate2', value=counts.mean(), vary=True)
    p.add('slope1', value=0, min=-5, max=5, vary=True)
    p.add('slope2', value=0, min=-5, max=5, vary=True)
    rampModel2 = Model(
        rampProfile2, independent_vars=['tExp', 'expTime', 'scanDirect'])
    # model fit, obtain crate, and transit parameter,
    # but ignore transit para for this time
    t0 = t - t[0]  # make the first element in time array 0
    weights = np.ones_like(t)
    # if not inlucde the transit orbit,
    weights[np.in1d(orbits, orbits_transit)] = 0
    fitResult = rampModel2.fit(
        counts,
        tExp=t0,
        expTime=expTime,
        scanDirect=scanDirect,
        weights=weights,
        params=p,
        method='powell')

    counts_fit = np.zeros_like(counts, dtype=float)
    counts_fit[upIndex] = (fitResult.params['crate1'].value * (
        1 + t0 * fitResult.params['slope1'] / 1e7))[upIndex]
    counts_fit[downIndex] = (fitResult.params['crate2'].value * (
        1 + t0 * fitResult.params['slope2'] / 1e7))[downIndex]
    RECTE_out = fitResult.best_fit
    RECTE_in = np.zeros_like(RECTE_out)
    RECTE_in[upIndex] = fitResult.params['crate1'].value * (
        1 + t0[upIndex] * fitResult.params['slope1'] / 1e7)
    RECTE_in[downIndex] = fitResult.params['crate2'].value * (
        1 + t0[downIndex] * fitResult.params['slope2'] / 1e7)
    correctTerm = RECTE_out / RECTE_in
    slopes = np.zeros_like(RECTE_out)
    slopes[upIndex] = 1 + t0[upIndex] * fitResult.params['slope1'] / 1e7
    slopes[downIndex] = 1 + t0[downIndex] * fitResult.params['slope2'] / 1e7
    crates = np.zeros_like(RECTE_out)
    crates[upIndex] = fitResult.params['crate1'] * slopes[upIndex]
    crates[downIndex] = fitResult.params['crate2'] * slopes[downIndex]
    return correctTerm, crates, RECTE_out, slopes


if __name__ == '__main__':
    pass
