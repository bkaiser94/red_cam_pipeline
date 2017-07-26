def superExtract(*args, **kw):
    """
    Optimally extract curved spectra, following Marsh 1989.

    :INPUTS:
       data : 2D Numpy array
         Appropriately calibrated frame from which to extract
         spectrum.  Should be in units of ADU, not electrons!

       variance : 2D Numpy array
         Variances of pixel values in 'data'.

       gain : scalar
         Detector gain, in electrons per ADU

       readnoise : scalar
         Detector readnoise, in electrons.

    :OPTIONS:
       trace : 1D numpy array
         location of spectral trace.  If None, :func:`traceorders` is
         invoked.

       goodpixelmask : 2D numpy array
         Equals 0 for bad pixels, 1 for good pixels

       npoly : int
         Number of profile polynomials to evaluate (Marsh's
         "K"). Ideally you should not need to set this -- instead,
         play with 'polyspacing' and 'extract_radius.' For symmetry,
         this should be odd.

       polyspacing : scalar
         Spacing between profile polynomials, in pixels. (Marsh's
         "S").  A few cursory tests suggests that the extraction
         precision (in the high S/N case) scales as S^-2 -- but the
         code slows down as S^2.

       pord : int
         Order of profile polynomials; 1 = linear, etc.

       bkg_radii : 2-sequence
         inner and outer radii to use in computing background

       extract_radius : int
         radius to use for both flux normalization and extraction

       dispaxis : bool
         0 for horizontal spectrum, 1 for vertical spectrum

       bord : int >= 0
         Degree of polynomial background fit.

       bsigma : int >= 0
         Sigma-clipping threshold for computing background.

       tord : int >= 0
         Degree of spectral-trace polynomial (for trace across frame
         -- not used if 'trace' is input)

       csigma : int >= 0
         Sigma-clipping threshold for cleaning & cosmic-ray rejection.

       finite : bool
         If true, mask all non-finite values as bad pixels.

       qmode : str ('fast' or 'slow')
         How to compute Marsh's Q-matrix.  Valid inputs are
         'fast-linear', 'slow-linear', 'fast-nearest,' 'slow-nearest,'
         and 'brute'.  These select between various methods of
         integrating the nearest-neighbor or linear interpolation
         schemes as described by Marsh; the 'linear' methods are
         preferred for accuracy.  Use 'slow' if you are running out of
         memory when using the 'fast' array-based methods.  'Brute' is
         both slow and inaccurate, and should not be used.
         
       nreject : int
         Number of outlier-pixels to reject at each iteration. 

       retall : bool
         If true, also return the 2D profile, background, variance
         map, and bad pixel mask.
             
    :RETURNS:
       object with fields for:
         spectrum

         varSpectrum

         trace


    :EXAMPLE:
      ::

        import spec
        import numpy as np
        import pylab as py

        def gaussian(p, x):
           if len(p)==3:
               p = concatenate((p, [0]))
           return (p[3] + p[0]/(p[1]*sqrt(2*pi)) * exp(-(x-p[2])**2 / (2*p[1]**2)))

        # Model some strongly tilted spectral data:
        nx, nlam = 80, 500
        x0 = np.arange(nx)
        gain, readnoise = 3.0, 30.
        background = np.ones(nlam)*10
        flux =  np.ones(nlam)*1e4
        center = nx/2. + np.linspace(0,10,nlam)
        FWHM = 3.
        model = np.array([gaussian([flux[ii]/gain, FWHM/2.35, center[ii], background[ii]], x0) for ii in range(nlam)])
        varmodel = np.abs(model) / gain + (readnoise/gain)**2
        observation = np.random.normal(model, np.sqrt(varmodel))
        fitwidth = 60
        xr = 15

        output_spec = spec.superExtract(observation, varmodel, gain, readnoise, polyspacing=0.5, pord=1, bkg_radii=[10,30], extract_radius=5, dispaxis=1)

        py.figure()
        py.plot(output_spec.spectrum.squeeze() / flux)
        py.ylabel('(Measured flux) / (True flux)')
        py.xlabel('Photoelectrons')
        


    :TO_DO:

      Introduce even more array-based, rather than loop-based,
      calculations.  For large spectra computing the C-matrix takes
      the most time; this should be optimized somehow.

    :SEE_ALSO:

    """

    # 2012-08-25 20:14 IJMC: Created.
    # 2012-09-21 14:32 IJMC: Added error-trapping if no good pixels
    #                      are in a row. Do a better job of extracting
    #                      the initial 'standard' spectrum.

    from scipy import signal
    from pylab import *
    from superextract_tools import bfixpix, traceorders, polyfitr, baseObject,message



    # Parse inputs:
    frame, variance, gain, readnoise = args[0:4]

    frame    = gain * np.array(frame, copy=False)
    variance = gain**2 * np.array(variance, copy=False)
    variance[variance<=0.] = readnoise**2

    # Parse options:
    if kw.has_key('verbose'):
        verbose = kw['verbose']
    else:
        verbose = False
    if verbose: from time import time


    if kw.has_key('goodpixelmask'):
        goodpixelmask = kw['goodpixelmask']
        if isinstance(goodpixelmask, str):
            goodpixelmask = pyfits.getdata(goodpixelmask).astype(bool)
        else:
            goodpixelmask = np.array(goodpixelmask, copy=True).astype(bool)
    else:
        goodpixelmask = np.ones(frame.shape, dtype=bool)


    if kw.has_key('dispaxis'):
        dispaxis = kw['dispaxis']
    else:
        dispaxis = 0

    if dispaxis==0:
        frame = frame.transpose()
        variance = variance.transpose()
        goodpixelmask = goodpixelmask.transpose()


    if kw.has_key('pord'):
        pord = kw['pord']
    else:
        pord = 2

    if kw.has_key('polyspacing'):
        polyspacing = kw['polyspacing']
    else:
        polyspacing = 1

    if kw.has_key('bkg_radii'):
        bkg_radii = kw['bkg_radii']
    else:
        bkg_radii = [15, 20]
        if verbose: message("Setting option 'bkg_radii' to: " + str(bkg_radii))

    if kw.has_key('extract_radius'):
        extract_radius = kw['extract_radius']
    else:
        extract_radius = 10
        if verbose: message("Setting option 'extract_radius' to: " + str(extract_radius))

    if kw.has_key('npoly'):
        npoly = kw['npoly']
    else:
        npoly = 2 * int((2.0 * extract_radius) / polyspacing / 2.) + 1

    if kw.has_key('bord'):
        bord = kw['bord']
    else:
        bord = 1
        if verbose: message("Setting option 'bord' to: " + str(bord))

    if kw.has_key('tord'):
        tord = kw['tord']
    else:
        tord = 3
        if verbose: message("Setting option 'tord' to: " + str(tord))

    if kw.has_key('bsigma'):
        bsigma = kw['bsigma']
    else:
        bsigma = 3
        if verbose: message("Setting option 'bsigma' to: " + str(bsigma))

    if kw.has_key('csigma'):
        csigma = kw['csigma']
    else:
        csigma = 5
        if verbose: message("Setting option 'csigma' to: " + str(csigma))

    if kw.has_key('qmode'):
        qmode = kw['qmode']
    else:
        #qmode = 'fast'
        qmode = 'fast-linear' #Best option if the machine can handle it
        if verbose: message("Setting option 'qmode' to: " + str(qmode))

    if kw.has_key('nreject'):
        nreject = kw['nreject']
    else:
        nreject = 100
        if verbose: message("Setting option 'nreject' to: " + str(nreject))

    if kw.has_key('finite'):
        finite = kw['finite']
    else:
        finite = True
        if verbose: message("Setting option 'finite' to: " + str(finite))


    if kw.has_key('retall'):
        retall = kw['retall']
    else:
        retall = False


    if finite:
        goodpixelmask *= (np.isfinite(frame) * np.isfinite(variance))

    variance[True-goodpixelmask] = frame[goodpixelmask].max() * 1e9
    nlam, fitwidth = frame.shape
   

    # Define trace (Marsh's "X_j" in Eq. 9)
    if kw.has_key('trace'):
        trace = kw['trace']
        xyfits = [np.arange(0,len(trace)),trace]
    else:
        trace = None

    if trace is None:
        trace = tord
    if not hasattr(trace, '__iter__'):
        #if verbose: print "Tracing not fully tested; dispaxis may need adjustment."
        #pdb.set_trace()
        tracecoef, xyfits = traceorders(frame, pord=trace, nord=1, verbose=verbose, plotalot=verbose-1, g=gain, rn=readnoise, badpixelmask=True-goodpixelmask, dispaxis=dispaxis, fitwidth=min(fitwidth, 80),retfits=True)
        trace = np.polyval(tracecoef.ravel(), np.arange(nlam))

    #xxx = np.arange(-fitwidth/2, fitwidth/2)
    #backgroundAperture = (np.abs(xxx) > bkg_radii[0]) * (np.abs(xxx) < bkg_radii[1])
    #extractionAperture = np.abs(xxx) < extract_radius
    #nextract = extractionAperture.sum()
    #xb = xxx[backgroundAperture]

    #trace.reshape(nlam,1) is the center of the fitted profile of the star
    # xxx is the distance of the frame on either side of the fitted profile.
    xxx = np.arange(fitwidth) - trace.reshape(nlam,1)
    backgroundApertures = (np.abs(xxx) > bkg_radii[0]) * (np.abs(xxx) <= bkg_radii[1])
    extractionApertures = np.abs(xxx) <= extract_radius
    nextracts = extractionApertures.sum(1)

    #Step3: Sky Subtraction
    background = 0. * frame
    #background_rms = np.zeros(nlam)
    #bkgrndmask = goodpixelmask
    for ii in range(nlam):
        if goodpixelmask[ii, backgroundApertures[ii]].any():
            fit,fit_chisq,fit_niter = polyfitr(xxx[ii,backgroundApertures[ii]], frame[ii, backgroundApertures[ii]], bord, bsigma, w=(goodpixelmask/variance)[ii, backgroundApertures[ii]], verbose=verbose-1,plotall=False,diag=True)
            #If you want to plot the fit to the background you can use this. Or set plotall=True above
            #if ii == 1100:
            #print ii
            #    plt.clf()
            #    plt.plot(xxx[ii,:],frame[ii,:],'k^')
            #    plt.plot(xxx[ii,backgroundApertures[ii]],frame[ii, backgroundApertures[ii]],'b^')
            #    plt.plot(xxx[ii,:],np.polyval(fit,xxx[ii,:]))
            #    plt.show()
            background[ii, :] = np.polyval(fit, xxx[ii])
            thisrow = backgroundApertures[ii]
            #background_rms[ii] = fit_chisq
            #print background_rms[ii], fit_chisq
            #plt.show()
            #Save values
            if ii == 1100:
                background_column_pixels = xxx[ii,:]
                background_column_values = frame[ii,:]
                background_fit_pixels = xxx[ii,backgroundApertures[ii]]
                background_fit_values = frame[ii, backgroundApertures[ii]]
                background_fit_polynomial = np.polyval(fit,xxx[ii,:])
        else:
            background[ii] = 0.
    #plt.clf()
    #plt.plot(range(nlam),background_rms,'b^')
    #plt.show()
    background_at_trace = np.array([np.interp(0, xxx[j], background[j]) for j in range(nlam)])
    
    # (my 3a: mask any bad values)
    badBackground = True - np.isfinite(background)
    background[badBackground] = 0.
    if verbose and badBackground.any():
        print "Found bad background values at: ", badBackground.nonzero()
    #Subtract the background here
    skysubFrame = frame - background

    # Interpolate and fix bad pixels for extraction of standard
    # spectrum -- otherwise there can be 'holes' in the spectrum from
    # ill-placed bad pixels.
    fixSkysubFrame = bfixpix(skysubFrame, True-goodpixelmask, n=8, retdat=True)

    #Step4: Extract 'standard' spectrum and its variance
    standardSpectrum = np.zeros((nlam, 1), dtype=float)
    varStandardSpectrum = np.zeros((nlam, 1), dtype=float)
    for ii in range(nlam):
        thisrow_good = extractionApertures[ii] #* goodpixelmask[ii] * 
        standardSpectrum[ii] = fixSkysubFrame[ii, thisrow_good].sum()
        varStandardSpectrum[ii] = variance[ii, thisrow_good].sum()


    spectrum = standardSpectrum.copy()
    varSpectrum = varStandardSpectrum

    # Define new indices (in Marsh's appendix):
    N = pord + 1
    mm = np.tile(np.arange(N).reshape(N,1), (npoly)).ravel()
    nn = mm.copy()
    ll = np.tile(np.arange(npoly), N)
    kk = ll.copy()
    pp = N * ll + mm
    qq = N * kk + nn

    jj = np.arange(nlam)  # row (i.e., wavelength direction)
    ii = np.arange(fitwidth) # column (i.e., spatial direction)
    jjnorm = np.linspace(-1, 1, nlam) # normalized X-coordinate
    jjnorm_pow = jjnorm.reshape(1,1,nlam) ** (np.arange(2*N-1).reshape(2*N-1,1,1))

    # Marsh eq. 9, defining centers of each polynomial:
    constant = 0.  # What is it for???
    poly_centers = trace.reshape(nlam, 1) + polyspacing * np.arange(-npoly/2+1, npoly/2+1) + constant


    # Marsh eq. 11, defining Q_kij    (via nearest-neighbor interpolation)
    #    Q_kij =  max(0, min(S, (S+1)/2 - abs(x_kj - i)))
    if verbose: tic = time() 
    if qmode=='fast-nearest': # Array-based nearest-neighbor mode.
        if verbose: tic = time()
        Q = np.array([np.zeros((npoly, fitwidth, nlam)), np.array([polyspacing * np.ones((npoly, fitwidth, nlam)), 0.5 * (polyspacing+1) - np.abs((poly_centers - ii.reshape(fitwidth, 1, 1)).transpose(2, 0, 1))]).min(0)]).max(0)

    elif qmode=='slow-linear': # Code is a mess, but it works.
        invs = 1./polyspacing
        poly_centers_over_s = poly_centers / polyspacing
        xps_mat = poly_centers + polyspacing
        xms_mat = poly_centers - polyspacing
        Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
        for i in range(fitwidth):
            ip05 = i + 0.5
            im05 = i - 0.5
            for j in range(nlam):
                for k in range(npoly):
                    xkj = poly_centers[j,k]
                    xkjs = poly_centers_over_s[j,k]
                    xps = xps_mat[j,k] #xkj + polyspacing
                    xms = xms_mat[j,k] # xkj - polyspacing

                    if (ip05 <= xms) or (im05 >= xps):
                        qval = 0.
                    elif (im05) > xkj:
                        lim1 = im05
                        lim2 = min(ip05, xps)
                        qval = (lim2 - lim1) * \
                            (1. + xkjs - 0.5*invs*(lim1+lim2))
                    elif (ip05) < xkj:
                        lim1 = max(im05, xms)
                        lim2 = ip05
                        qval = (lim2 - lim1) * \
                            (1. - xkjs + 0.5*invs*(lim1+lim2))
                    else:
                        lim1 = max(im05, xms)
                        lim2 = min(ip05, xps)
                        qval = lim2 - lim1 + \
                            invs * (xkj*(-xkj + lim1 + lim2) - \
                                        0.5*(lim1*lim1 + lim2*lim2))
                    Q[k,i,j] = max(0, qval)


    elif qmode=='fast-linear': # Code is a mess, but it's faster than 'slow' mode
        invs = 1./polyspacing
        xps_mat = poly_centers + polyspacing
        Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
        for j in range(nlam):
            xkj_vec = np.tile(poly_centers[j,:].reshape(npoly, 1), (1, fitwidth))
            xps_vec = np.tile(xps_mat[j,:].reshape(npoly, 1), (1, fitwidth))
            xms_vec = xps_vec - 2*polyspacing

            ip05_vec = np.tile(np.arange(fitwidth) + 0.5, (npoly, 1))
            im05_vec = ip05_vec - 1
            ind00 = ((ip05_vec <= xms_vec) + (im05_vec >= xps_vec))
            ind11 = ((im05_vec > xkj_vec) * (True - ind00))
            ind22 = ((ip05_vec < xkj_vec) * (True - ind00))
            ind33 = (True - (ind00 + ind11 + ind22)).nonzero()
            ind11 = ind11.nonzero()
            ind22 = ind22.nonzero()

            n_ind11 = len(ind11[0])
            n_ind22 = len(ind22[0])
            n_ind33 = len(ind33[0])

            if n_ind11 > 0:
                ind11_3d = ind11 + (np.ones(n_ind11, dtype=int)*j,)
                lim2_ind11 = np.array((ip05_vec[ind11], xps_vec[ind11])).min(0)
                Q[ind11_3d] = ((lim2_ind11 - im05_vec[ind11]) * invs * \
                                   (polyspacing + xkj_vec[ind11] - 0.5*(im05_vec[ind11] + lim2_ind11)))
            
            if n_ind22 > 0:
                ind22_3d = ind22 + (np.ones(n_ind22, dtype=int)*j,)
                lim1_ind22 = np.array((im05_vec[ind22], xms_vec[ind22])).max(0)
                Q[ind22_3d] = ((ip05_vec[ind22] - lim1_ind22) * invs * \
                                   (polyspacing - xkj_vec[ind22] + 0.5*(ip05_vec[ind22] + lim1_ind22)))
            
            if n_ind33 > 0:
                ind33_3d = ind33 + (np.ones(n_ind33, dtype=int)*j,)
                lim1_ind33 = np.array((im05_vec[ind33], xms_vec[ind33])).max(0)
                lim2_ind33 = np.array((ip05_vec[ind33], xps_vec[ind33])).min(0)
                Q[ind33_3d] = ((lim2_ind33 - lim1_ind33) + invs * \
                                   (xkj_vec[ind33] * (-xkj_vec[ind33] + lim1_ind33 + lim2_ind33) - 0.5*(lim1_ind33*lim1_ind33 + lim2_ind33*lim2_ind33)))
            

    elif qmode=='brute': # Neither accurate, nor memory-frugal.
        oversamp = 4.
        jj2 = np.arange(nlam*oversamp, dtype=float) / oversamp
        trace2 = np.interp(jj2, jj, trace)
        poly_centers2 = trace2.reshape(nlam*oversamp, 1) + polyspacing * np.arange(-npoly/2+1, npoly/2+1, dtype=float) + constant
        x2 = np.arange(fitwidth*oversamp, dtype=float)/oversamp
        Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
        for k in range(npoly):
            Q[k,:,:] = an.binarray((np.abs(x2.reshape(fitwidth*oversamp,1) - poly_centers2[:,k]) <= polyspacing), oversamp)

        Q /= oversamp*oversamp*2

    else:  # 'slow' Loop-based nearest-neighbor mode: requires less memory
        if verbose: tic = time()
        Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
        for k in range(npoly):
            for i in range(fitwidth):
                for j in range(nlam):
                    Q[k,i,j] = max(0, min(polyspacing, 0.5*(polyspacing+1) - np.abs(poly_centers[j,k] - i)))

    if verbose: print '%1.2f s to compute Q matrix (%s mode)' % (time() - tic, qmode)
        

    # Some quick math to find out which data columns are important, and
    #   which contain no useful spectral information:
    Qmask = Q.sum(0).transpose() > 0
    Qind = Qmask.transpose().nonzero()
    Q_cols = [Qind[0].min(), Qind[0].max()]
    nQ = len(Qind[0])
    Qsm = Q[:,Q_cols[0]:Q_cols[1]+1,:]

    # Prepar to iteratively clip outliers
    newBadPixels = True
    iter = -1
    if verbose: print "Looking for bad pixel outliers."
    while newBadPixels:
        iter += 1
        if verbose: print "Beginning iteration %i" % iter


        # Compute pixel fractions (Marsh Eq. 5):
        #     (Note that values outside the desired polynomial region
        #     have Q=0, and so do not contribute to the fit)
        E = (skysubFrame / spectrum).transpose()
        invEvariance = (spectrum**2 / variance).transpose()
        weightedE = (skysubFrame * spectrum / variance).transpose() # E / var_E
        invEvariance_subset = invEvariance[Q_cols[0]:Q_cols[1]+1,:]

        # Define X vector (Marsh Eq. A3):
        if verbose: tic = time()
        X = np.zeros(N * npoly, dtype=float)
        X0 = np.zeros(N * npoly, dtype=float)
        for q in qq:
            X[q] = (weightedE[Q_cols[0]:Q_cols[1]+1,:] * Qsm[kk[q],:,:] * jjnorm_pow[nn[q]]).sum() 
        if verbose: print '%1.2f s to compute X matrix' % (time() - tic)

        # Define C matrix (Marsh Eq. A3)
        if verbose: tic = time()
        C = np.zeros((N * npoly, N*npoly), dtype=float)

        buffer = 1.1 # C-matrix computation buffer (to be sure we don't miss any pixels)
        for p in pp:
            qp = Qsm[ll[p],:,:]
            for q in qq:
                #  Check that we need to compute C:
                if np.abs(kk[q] - ll[p]) <= (1./polyspacing + buffer):
                    if q>=p: 
                        # Only compute over non-zero columns:
                        C[q, p] = (Qsm[kk[q],:,:] * qp * jjnorm_pow[nn[q]+mm[p]] * invEvariance_subset).sum() 
                    if q>p:
                        C[p, q] = C[q, p]


        if verbose: print '%1.2f s to compute C matrix' % (time() - tic)

        ##################################################
        ##################################################
        # Just for reference; the following is easier to read, perhaps, than the optimized code:
        if False: # The SLOW way to compute the X vector:
            X2 = np.zeros(N * npoly, dtype=float)
            for n in nn:
                for k in kk:
                    q = N * k + n
                    xtot = 0.
                    for i in ii:
                        for j in jj:
                            xtot += E[i,j] * Q[k,i,j] * (jjnorm[j]**n) / Evariance[i,j]
                    X2[q] = xtot

            # Compute *every* element of C (though most equal zero!)
            C = np.zeros((N * npoly, N*npoly), dtype=float)
            for p in pp:
                for q in qq:
                    if q>=p:
                        C[q, p] = (Q[kk[q],:,:] * Q[ll[p],:,:] * (jjnorm.reshape(1,1,nlam)**(nn[q]+mm[p])) / Evariance).sum()
                    if q>p:
                        C[p, q] = C[q, p]
        ##################################################
        ##################################################

        # Solve for the profile-polynomial coefficients (Marsh Eq. A)4: 
        if np.abs(np.linalg.det(C)) < 1e-10:
            Bsoln = np.dot(np.linalg.pinv(C), X)
        else:
            Bsoln = np.linalg.solve(C, X)

        Asoln = Bsoln.reshape(N, npoly).transpose()

        # Define G_kj, the profile-defining polynomial profiles (Marsh Eq. 8)
        Gsoln = np.zeros((npoly, nlam), dtype=float)
        for n in range(npoly):
            Gsoln[n] = np.polyval(Asoln[n,::-1], jjnorm) # reorder polynomial coef.

        # Compute the profile (Marsh eq. 6) and normalize it:
        if verbose: tic = time()
        profile = np.zeros((fitwidth, nlam), dtype=float)
        for i in range(fitwidth):
            profile[i,:] = (Q[:,i,:] * Gsoln).sum(0)

        #Normalize the profile here
        if profile.min() < 0:
            profile[profile < 0] = 0. 
        profile /= profile.sum(0).reshape(1, nlam)
        profile[True - np.isfinite(profile)] = 0.
        if verbose: print '%1.2f s to compute profile' % (time() - tic)

        #Plot the profile and estimated fraction. This mimics Marsh's Figure 2.
        #print skysubFrame.shape
        #for x in [1525,1526,1527,1528,1529,1530,1531,1532,1533,1534,1535]:
        #    plt.clf()
        #    plt.figure(1)
        #    plt.plot(skysubFrame[x,:],'k')
        #    #plt.show()
        #    #plt.clf()
        #    plt.figure(2)
        #    plt.plot(profile[:,x],'b')
        #    plt.plot(E[:,x],'g')
        #    plt.show()

        #Step6: Revise variance estimates 
        modelSpectrum = spectrum * profile.transpose()
        modelData = modelSpectrum + background
        variance0 = np.abs(modelData) + readnoise**2
        variance = variance0 / (goodpixelmask + 1e-9) # De-weight bad pixels, avoiding infinite variance

        outlierVariances = (frame - modelData)**2/variance
        if outlierVariances.max() > csigma**2:
            newBadPixels = True
            # Base our nreject-counting only on pixels within the spectral trace:
            maxRejectedValue = max(csigma**2, np.sort(outlierVariances[Qmask])[-nreject])
            worstOutliers = (outlierVariances>=maxRejectedValue).nonzero()
            goodpixelmask[worstOutliers] = False
            numberRejected = len(worstOutliers[0])
            #pdb.set_trace()
        else:
            newBadPixels = False
            numberRejected = 0
        if verbose: print "Rejected %i pixels on this iteration " % numberRejected
            
        # Optimal Spectral Extraction: (Horne, Step 8)
        fixSkysubFrame = bfixpix(skysubFrame, True-goodpixelmask, n=8, retdat=True)
        spectrum = np.zeros((nlam, 1), dtype=float)
        varSpectrum = np.zeros((nlam, 1), dtype=float)
        goodprof =  profile.transpose() * goodpixelmask #Horne: M*P
       
        for ii in range(nlam):
            thisrow_good = extractionApertures[ii]
            denom = (goodprof[ii, thisrow_good] * profile.transpose()[ii, thisrow_good] / variance0[ii, thisrow_good]).sum() #Horne: sum(M*P**2/V)
            if denom==0:
                spectrum[ii] = 0.
                varSpectrum[ii] = 9e9
            else:
                spectrum[ii] = (goodprof[ii, thisrow_good] * fixSkysubFrame[ii, thisrow_good] / variance0[ii, thisrow_good]).sum() / denom #Horne: sum(M*P*(D-S)/V) / sum(M*P**2/V)
                varSpectrum[ii] = goodprof[ii, thisrow_good].sum() / denom #Horne: sum(M*P) / sum(M*P**2/V)
           

    ret = baseObject()
    ret.spectrum = spectrum
    ret.raw = standardSpectrum
    ret.varSpectrum = varSpectrum
    ret.trace = trace
    ret.tracepos = xyfits
    ret.units = 'electrons'
    ret.background = background_at_trace #background_at_trace 
    ret.backgroundcolumnpixels = background_column_pixels
    ret.backgroundcolumnvalues = background_column_values 
    ret.backgroundfitpixels = background_fit_pixels
    ret.backgroundfitvalues = background_fit_values
    ret.backgroundfitpolynomial = background_fit_polynomial

    ret.function_name = 'spec.superExtract'

    if retall:
        ret.profile_map = profile
        ret.extractionApertures = extractionApertures
        ret.background_map = background
        ret.variance_map = variance0
        ret.goodpixelmask = goodpixelmask
        ret.function_args = args
        ret.function_kw = kw

    return  ret
