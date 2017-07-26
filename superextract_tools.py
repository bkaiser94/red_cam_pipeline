import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import pdb
from pylab import plot, legend, title, figure, arange, cm


def imshow(data, x=[], y=[], aspect='auto', interpolation='nearest', cmap=None, vmin=[], vmax=[]):
    """ Version of pylab's IMSHOW with my own defaults:
    ::

      imshow(data, aspect='auto', interpolation='nearest', cmap=cm.gray, vmin=[], vmax=[])

    Other IMSHOW options are default, but a new one exists: 
          x=  and y=  let you set the axes values by passing in the x and y coordinates."""
    #2008-07-25 18:30 IJC: Created to save a little bit of time and do axes.

    if cmap==None:
        cmap = cm.gray

    def getextent(data, x, y):
        """ Gets the extent of the data for plotting.  Subfunc of IMSHOW."""
        dsh = data.shape

        if len(x)==0:
            x = arange(dsh[1])
        if len(y)==0:
            y = arange(dsh[0])

        dx = 1.0* (x.max() - x.min()) / (len(x) - 1)
        xextent = [x.min() - dx/2.0, x.max() + dx/2.0]
        xextent = [x[0] - dx/2.0, x[-1] + dx/2.0]

        dy = 1.0* (y.max() - y.min()) / (len(y) - 1)
        yextent = [y.max() + dy/2.0, y.min() - dy/2.0]
        yextent = [y[-1] + dy/2.0, y[0] - dy/2.0]

        extent = xextent + yextent
        
        return extent

    def getclim(data, vmin, vmax):
        if vmin.__class__==list:
            vmin = data.min()
        if vmax.__class__==list:
            vmax = data.max()
        return [vmin, vmax]
    
    #------------- Start the actual routine -------------

    extent = getextent(data, x,y)
    clim   = getclim(data, vmin, vmax)
    imshow(data, aspect=aspect, interpolation=interpolation, cmap=cmap, 
              vmin=clim[0], vmax=clim[1], extent=extent)
    

# ===========================================================================

def bfixpix(data, badmask, n=4, retdat=False):
    """Replace pixels flagged as nonzero in a bad-pixel mask with the
    average of their nearest four good neighboring pixels.

    :INPUTS:
      data : numpy array (two-dimensional)

      badmask : numpy array (same shape as data)

    :OPTIONAL_INPUTS:
      n : int
        number of nearby, good pixels to average over

      retdat : bool
        If True, return an array instead of replacing-in-place and do
        _not_ modify input array `data`.  This is always True if a 1D
        array is input!

    :RETURNS: 
      another numpy array (if retdat is True)

    :TO_DO:
      Implement new approach of Popowicz+2013 (http://arxiv.org/abs/1309.4224)
    """
    # 2010-09-02 11:40 IJC: Created
    #2012-04-05 14:12 IJMC: Added retdat option
    # 2012-04-06 18:51 IJMC: Added a kludgey way to work for 1D inputs
    # 2012-08-09 11:39 IJMC: Now the 'n' option actually works.

    if data.ndim==1:
        data = np.tile(data, (3,1))
        badmask = np.tile(badmask, (3,1))
        ret = bfixpix(data, badmask, n=2, retdat=True)
        return ret[1]


    nx, ny = data.shape

    badx, bady = np.nonzero(badmask)
    nbad = len(badx)

    if retdat:
        data = np.array(data, copy=True)
    
    for ii in range(nbad):
        thisloc = badx[ii], bady[ii]
        rad = 0
        numNearbyGoodPixels = 0

        while numNearbyGoodPixels<n:
            rad += 1
            xmin = max(0, badx[ii]-rad)
            xmax = min(nx, badx[ii]+rad)
            ymin = max(0, bady[ii]-rad)
            ymax = min(ny, bady[ii]+rad)
            x = np.arange(nx)[xmin:xmax+1]
            y = np.arange(ny)[ymin:ymax+1]
            yy,xx = np.meshgrid(y,x)
            #print ii, rad, xmin, xmax, ymin, ymax, badmask.shape
            
            rr = abs(xx + 1j*yy) * (1. - badmask[xmin:xmax+1,ymin:ymax+1])
            numNearbyGoodPixels = (rr>0).sum()
        
        closestDistances = np.unique(np.sort(rr[rr>0])[0:n])
        numDistances = len(closestDistances)
        localSum = 0.
        localDenominator = 0.
        for jj in range(numDistances):
            localSum += data[xmin:xmax+1,ymin:ymax+1][rr==closestDistances[jj]].sum()
            localDenominator += (rr==closestDistances[jj]).sum()

        #print badx[ii], bady[ii], 1.0 * localSum / localDenominator, data[xmin:xmax+1,ymin:ymax+1]
        data[badx[ii], bady[ii]] = 1.0 * localSum / localDenominator

    if retdat:
        ret = data
    else:
        ret = None

    return ret
    
# ===========================================================================


def traceorders(filename,g,rn, pord=5, dispaxis=0, nord=1, verbose=False, ordlocs=None, stepsize=20, fitwidth=20, plotalot=False, medwidth=6, xylims=None, uncertainties=None, badpixelmask=None, retsnr=False, retfits=False):
    """
    Trace spectral orders for a specified filename.

    filename : str OR 2D array
        full path and filename to a 2D echelleogram FITS file, _OR_
        a 2D numpy array representing such a file.

    OPTIONAL INPUTS:
    pord : int
        polynomial order of spectral order fit

    dispaxis : int
        set dispersion axis: 0 = horizontal and  = vertical
    
    nord : int
        number of spectral orders to trace.

    ordlocs : (nord x 2) numpy array
        Location to "auto-click" and 

    verbose: int
        0,1,2; whether (and how much) to print various verbose debugging text

    stepsize : int
        number of pixels to step along the spectrum while tracing

    fitwidth : int
        number of pixels to use when fitting a spectrum's cross-section
        
    medwidth : int
        number of columns to average when fitting profiles to echelleograms
        
    plotalot : bool
        Show pretty plot?  If running in batch mode (using ordlocs)
        default is False; if running in interactive mode (ordlocs is
        None) default is True.

    xylims : 4-sequence
        Extract the given subset of the data array: [xmin, xmax, ymin, ymax]

    uncertainties : str OR 2D array
        full path and filename to a 2D uncertainties FITS file, _OR_
        a 2D numpy array representing such a file.
        
        If this is set, 'g' and 'rn' below are ignored.  This is
        useful if, e.g., you are analyzing data which have already
        been sky-subtracted, nodded on slit, or otherwise altered.
        But note that this must be the same size as the input data!

    g : scalar > 0
      Detector gain, electrons per ADU (for setting uncertainties)
    
    rn : scalar > 0
      Detector read noise, electrons (for setting uncertainties)
    
    retsnr : bool
      If true, also return the computed S/N of the position fit at
      each stepped location.
    
    retfits : bool
      If true, also return the X,Y positions at each stepped location.
    

    :RETURNS:
       (nord, pord) shaped numpy array representing the polynomial
         coefficients for each order (suitable for use with np.polyval)

    :NOTES:

      If tracing fails, a common reason can be that fitwidth is too
      small.  Try increasing it!
    """
    # 2010-08-31 17:00 IJC: If best-fit PSF location goes outside of
    #     the 'fitwidth' region, nan values are returned that don't
    #     cause the routine to bomb out quite so often.
    # 2010-09-08 13:54 IJC: Updated so that going outside the
    #     'fitwidth' region is determined only in the local
    #     neighborhood, not in relation to the (possibly distant)
    #     initial guess position.
    # 2010-11-29 20:56 IJC: Added medwidth option
    # 2010-12-17 09:32 IJC: Changed color scaling options; added xylims option.

    # 2012-04-27 04:43 IJMC: Now perform _weighted_ fits to the measured traces!

    #global gain
    #global readnoise

    gain = g
    readnoise = rn

    if verbose < 0:
        verbose = 0

    if not g==gain:
        message("Setting gain to " + str(g))
        gain = g

    if not rn==readnoise:
        message("Setting readnoise to " + str(rn))
        readnoise = rn

    if ordlocs is not None:
        ordlocs = np.array(ordlocs, copy=False)
        if ordlocs.ndim==1:
            ordlocs = ordlocs.reshape(1, ordlocs.size)
        autopick = True
    else:
        autopick = False


    plotalot = (not autopick) or plotalot
    if isinstance(filename, np.ndarray):
        ec = filename.copy()
    else:
        try:
            ec = pyfits.getdata(filename)
        except:
            message("Could not open filename %s" % filename)
            return -1


    if isinstance(uncertainties, np.ndarray):
        err_ec = uncertainties.copy()
    else:
        try:
            err_ec = pyfits.getdata(uncertainties)
        except:
            err_ec = np.sqrt(ec * gain + readnoise**2)

    if dispaxis<>0:
        ec = ec.transpose()
        err_ec = err_ec.transpose()
        if verbose: message("Took transpose of echelleogram to rotate dispersion axis")
    else:
        pass

    if xylims is not None:
        try:
            ec = ec[xylims[0]:xylims[1], xylims[2]:xylims[3]]
            err_ec = err_ec[xylims[0]:xylims[1], xylims[2]:xylims[3]]
        except:
            message("Could not extract subset: ", xylims)
            return -1

    if badpixelmask is None:
        badpixelmask = np.zeros(ec.shape, dtype=bool)
    else:
        if not hasattr(badpixelmask, 'shape'):
            badpixelmask = pyfits.getdata(badpixelmask)
        if xylims is not None:
            badpixelmask = badpixelmask[xylims[0]:xylims[1], xylims[2]:xylims[3]]

        
    err_ec[badpixelmask.nonzero()] = err_ec[np.isfinite(err_ec)].max() * 1e9

    try:
        ny, nx = ec.shape
    except:
        message("Echellogram file %s does not appear to be 2D: exiting." % filename)
        return -1

    if plotalot:
        f = plt.figure()
        ax = plt.axes()
        plt.imshow(ec, interpolation='nearest',aspect='auto')
        sortedvals = np.sort(ec.ravel())
        plt.clim([sortedvals[nx*ny*.01], sortedvals[nx*ny*.99]])
        #plt.imshow(np.log10(ec-ec.min()+np.median(ec)),interpolation='nearest',aspect='auto')
        ax.axis([0, nx, 0, ny])

    orderCoefs = np.zeros((nord, pord+1), float)
    position_SNRs = []
    #xyfits = np.zeros([len(xPositions),2])
    if not autopick:

        ordlocs = np.zeros((nord, 2),float)
        for ordernumber in range(nord):
            message('Selecting %i orders; please click on order %i now.' % (nord, 1+ordernumber))
            plt.figure(f.number)
            guessLoc = pickloc(ax, zoom=fitwidth)
            ordlocs[ordernumber,:] = guessLoc
            ax.plot([guessLoc[0]],[guessLoc[1]], '*k')
            ax.axis([0, nx, 0, ny])
            plt.figure(f.number)
            plt.draw()
            if verbose:
                message("you picked the location: ")
                message(guessLoc)

    for ordernumber in range(nord):
        guessLoc = ordlocs[ordernumber,:]
        xInit, yInit, err_yInit = fitPSF(ec, guessLoc, fitwidth=fitwidth,verbose=verbose, medwidth=medwidth, err_ec=err_ec)
        if plotalot:
            ax.plot([xInit],[yInit], '*k')
            ax.axis([0, nx, 0, ny])
            plt.figure(f.number)
            plt.draw()

        if verbose:
            message("Initial (fit) position: (%3.2f,%3.2f)"%(xInit,yInit))

        # Prepare to fit PSFs at multiple wavelengths.

        # Determine the other positions at which to fit:
        xAbove = np.arange(1, np.ceil(1.0*(nx-xInit)/stepsize))*stepsize + xInit
        xBelow = np.arange(-1,-np.ceil((1.+xInit)/stepsize),-1)*stepsize + xInit
        #The edges of the 2D array might contain overscan, so don't fit those for the
        # trace. So we now shorten xAbove and xBelow by one.
        xAbove = xAbove[:-1]
        xBelow = xBelow[:-1]
        nAbove = len(xAbove)
        nBelow = len(xBelow)
        nToMeasure = nAbove + nBelow + 1
        iInit = nBelow

        if verbose:
            message("Going to measure PSF at the following %i locations:"%nToMeasure )
            message(xAbove)
            message(xBelow)
        
        # Measure all positions "above" the initial selection:
        yAbove = np.zeros(nAbove,float)
        err_yAbove = np.zeros(nAbove,float)
        lastY = yInit
        for i_meas in range(nAbove):
            guessLoc = xAbove[i_meas], lastY

            thisx, thisy, err_thisy = fitPSF(ec, guessLoc, fitwidth=fitwidth, verbose=verbose-1, medwidth=medwidth, err_ec=err_ec)
            if abs(thisy - yInit)>fitwidth/2:
                thisy = yInit
                err_thisy = yInit
                lastY = yInit
            else:
                lastY = thisy.astype(int)
            yAbove[i_meas] = thisy
            err_yAbove[i_meas] = err_thisy
            if verbose:
                print thisx, thisy
            if plotalot and not np.isnan(thisy):
                #ax.plot([thisx], [thisy], 'xk')
                ax.errorbar([thisx], [thisy], [err_thisy], fmt='xk')

        # Measure all positions "below" the initial selection:
        yBelow = np.zeros(nBelow,float)
        err_yBelow = np.zeros(nBelow,float)
        lastY = yInit
        for i_meas in range(nBelow):
            guessLoc = xBelow[i_meas], lastY
            thisx, thisy, err_thisy = fitPSF(ec, guessLoc, fitwidth=fitwidth, verbose=verbose-1, medwidth=medwidth, err_ec=err_ec)
            if abs(thisy-lastY)>fitwidth/2:
                thisy = np.nan
            else:
                lastY = thisy.astype(int)
            yBelow[i_meas] = thisy
            err_yBelow[i_meas] = err_thisy
            if verbose:
                print thisx, thisy
            if plotalot and not np.isnan(thisy):
                ax.errorbar([thisx], [thisy], [err_thisy], fmt='xk')
    
        # Stick all the fit positions together:
        yPositions = np.concatenate((yBelow[::-1], [yInit], yAbove))
        err_yPositions = np.concatenate((err_yBelow[::-1], [err_yInit], err_yAbove))
        xPositions = np.concatenate((xBelow[::-1], [xInit], xAbove))

        if verbose:
            message("Measured the following y-positions:")
            message(yPositions)

        theseTraceCoefs = polyfitr(xPositions, yPositions, pord, 3, \
                                       w=1./err_yPositions**2, verbose=verbose)
        orderCoefs[ordernumber,:] = theseTraceCoefs
        # Plot the traces
        if plotalot:
            ax.plot(np.arange(nx), np.polyval(theseTraceCoefs,np.arange(nx)), '-k')
            ax.plot(np.arange(nx), np.polyval(theseTraceCoefs,np.arange(nx))+fitwidth/2, '--k')
            ax.plot(np.arange(nx), np.polyval(theseTraceCoefs,np.arange(nx))-fitwidth/2, '--k')
            ax.axis([0, nx, 0, ny])   
            plt.figure(f.number)
            plt.draw()

        if retsnr:
            position_SNRs.append(yPositions / err_yPositions)
        if retfits:
            xyfits = np.array([xPositions,yPositions])
            #xyfits.append((xPositions, yPositions))
    
    # Prepare for exit and return:
    ret = (orderCoefs,)
    if retsnr:
        ret = ret + (position_SNRs,)
    #if retfits:
    #    ret = ret + (xyfits,)
    if len(ret)==1:
        ret = ret[0]
    return  ret, xyfits

# ===========================================================================


def message(text):
    """Display a message; for now, with text."""
    from sys import stdout

    print text
    stdout.flush()
# ===========================================================================


def pickloc(ax=None, zoom=10):
    """
    :INPUTS:
       ax   : (axes instance) -- axes in which to pick a location

       zoom : int -- zoom radius for target confirmation
            : 2-tuple -- (x,y) radii for zoom confirmation.
    """
    # 2011-04-29 19:26 IJC: 
    # 2011-09-03 20:59 IJMC: Zoom can now be a tuple; x,y not cast as int.

    pickedloc = False
    if ax is None:
        ax = plt.gca()

    axlimits = ax.axis()

    if hasattr(zoom, '__iter__') and len(zoom)>1:
        xzoom, yzoom = zoom
    else:
        xzoom = zoom
        yzoom = zoom

    while not pickedloc:
        ax.set_title("click to select location")
        ax.axis(axlimits)

        x = None
        while x is None:  
            selectevent = plt.ginput(n=1,show_clicks=False)
            if len(selectevent)>0: # Prevent user from cancelling out.
                x,y = selectevent[0]

        #x = x.astype(int)
        #y = y.astype(int)

        
        if zoom is not None:
            ax.axis([x-xzoom,x+xzoom,y-yzoom,y+yzoom])

        ax.set_title("you selected xy=(%i,%i)\nclick again to confirm, or press Enter/Return to try again" %(x,y)  )
        plt.draw()

        confirmevent = plt.ginput(n=1,show_clicks=False)
        if len(confirmevent)>0:  
            pickedloc = True
            loc = confirmevent[0]

    return loc

# ===========================================================================


def fitPSF(ec, guessLoc, fitwidth=20, verbose=False, sigma=5, medwidth=6, err_ec=None):
    """
    Helper function to fit 1D PSF near a given region.  Assumes
    spectrum runs horizontally across the frame!

    ec : 2D numpy array
        echellogram array, with horizontal dispersion direction
    guessLoc : 2-tuple
       A slight misnomer for this (x,y) tuple: y is a guess and will
         be fit, but x is the coordinate at which the fitting takes
         place.
    fitwidth : int
       width of cross-dispersion direction to use in fitting
    medwidth : int
       number of columns to average over when fitting a profile
    verbose : bool
       verbosity/debugging printout flag

    sigma : scalar
       sigma scale for clipping bad values
    """
    # 2010-08-24 22:00 IJC: Added sigma option
    # 2010-11-29 20:54 IJC: Added medwidth option
    # 2011-11-26 23:17 IJMC: Fixed bug in computing "badval"
    # 2012-04-27 05:15 IJMC: Now allow error estimates to be passed in.
    # 2012-04-28 08:48 IJMC: Added better guessing for initial case.

    if verbose<0:
        verbose = False

    ny, nx = ec.shape

    x = guessLoc[0].astype(int)
    y = guessLoc[1].astype(int)

    # Fit the PSF profile at the initial, selected location:
    ymin = max(y-fitwidth/2, 0)
    ymax = min(y+fitwidth/2, ny)
    xmin = max(x-medwidth/2, 0)
    xmax = min(x+medwidth/2, nx)
    if verbose:
        message("Sampling: ec[%i:%i,%i:%i]"%(ymin,ymax,xmin,xmax))

    firstSeg = np.median(ec[ymin:ymax,xmin:xmax],1)
    if verbose:
        print firstSeg

    if err_ec is None:
        ey = stdr(firstSeg, sigma)
        badval = abs((firstSeg-np.median(firstSeg))/ey) > sigma
        err = np.ones(firstSeg.shape, float)
        err[badval] = 1e9
    else:
        err = np.sqrt((err_ec[ymin:ymax,xmin:xmax]**2).mean(1))
        err[True - np.isfinite(err)] = err[np.isfinite(err)].max() * 1e9

    guessAmp = (wmean(firstSeg, 1./err**2) - np.median(firstSeg)) * fitwidth
    if not np.isfinite(guessAmp):
        pdb.set_trace()

    #VERY important to give fitGaussian initial guess. Otherwise will fail.
    fit, efit = fitGaussian(firstSeg, verbose=verbose, err=err, guess=[guessAmp[0], 5, fitwidth/2., np.median(firstSeg)])
    #fit, efit = fitGaussian(firstSeg, verbose=verbose, err=err, guess=None)
    ##################
    ###Use the following if you want to see each fit plotted
    ###blah = np.arange(1.0*len(firstSeg))
    ###plt.clf()
    ###plt.plot(firstSeg)
    ###plt.plot(gaussian(fit,blah))
    ###plt.show()
    ##################
    newY = ymin+fit[2]
    err_newY = efit[2]
    if verbose:
        message("Initial position: (%3.2f,%3.2f)"%(x,newY))
    return x, newY, err_newY

# ===========================================================================


def wmean(a, w, axis=None, reterr=False):
    """wmean(a, w, axis=None)

    Perform a weighted mean along the specified axis.

    :INPUTS:
      a : sequence or Numpy array
        data for which weighted mean is computed

      w : sequence or Numpy array
        weights of data -- e.g., 1./sigma^2

      reterr : bool
        If True, return the tuple (mean, err_on_mean), where
        err_on_mean is the unbiased estimator of the sample standard
        deviation.

    :SEE ALSO:  :func:`wstd`
    """
    # 2008-07-30 12:44 IJC: Created this from ...
    # 2012-02-28 20:31 IJMC: Added a bit of documentation
    # 2012-03-07 10:58 IJMC: Added reterr option

    newdata    = np.array(a, subok=True, copy=True)
    newweights = np.array(w, subok=True, copy=True)

    if axis==None:
        newdata    = newdata.ravel()
        newweights = newweights.ravel()
        axis = 0

    ash  = list(newdata.shape)
    wsh  = list(newweights.shape)

    nsh = list(ash)
    nsh[axis] = 1

    if ash<>wsh:
        warn('Data and weight must be arrays of same shape.')
        return []
    
    wsum = newweights.sum(axis=axis).reshape(nsh) 
    
    weightedmean = (a * newweights).sum(axis=axis).reshape(nsh) / wsum
    if reterr:
        # Biased estimator:
        #e_weightedmean = sqrt((newweights * (a - weightedmean)**2).sum(axis=axis) / wsum)

        # Unbiased estimator:
        #e_weightedmean = sqrt((wsum / (wsum**2 - (newweights**2).sum(axis=axis))) * (newweights * (a - weightedmean)**2).sum(axis=axis))
        
        # Standard estimator:
        e_weightedmean = np.sqrt(1./newweights.sum(axis=axis))

        ret = weightedmean, e_weightedmean
    else:
        ret = weightedmean

    return ret

# ===========================================================================


def fitGaussian(vec, err=None, verbose=False, guess=None):
    """Fit a Gaussian function to an input data vector.

    Return the fit, and uncertainty estimates on that fit.
    
    SEE ALSO: :func:`analysis.gaussian`"""
    # 2012-12-20 13:28 IJMC: Make a more robust guess for the centroid.

    xtemp = np.arange(1.0*len(vec))

    if guess is None: # Make some educated guesses as to the parameters:
        pedestal = (0.8*np.median(vec) + 0.2*(vec[0]+vec[1]))
        area = (vec-pedestal).sum()
        centroid = ((vec-pedestal)**2*xtemp).sum()/((vec-pedestal)**2).sum()
        if centroid<0:
            centroid = 1.
        elif centroid>len(vec):
            centroid = len(vec)-2.
        #pdb.set_trace()
        sigma = area/vec[int(centroid)]/np.sqrt(2*np.pi)
        if sigma<=0:
            sigma = .01

        guess = [area,sigma,centroid,pedestal]

    if err is None:
        err = np.ones(vec.shape, dtype=float)

    badvals = True - (np.isfinite(xtemp) * np.isfinite(err) * np.isfinite(vec))
    vec[badvals] = np.median(vec[True - badvals])
    err[badvals] = vec[True - badvals].max() * 1e9

    if verbose: 
        print 'Gaussian guess parameters>>', guess

    if not np.isfinite(xtemp).all():
        pdb.set_trace()
    if not np.isfinite(vec).all():
        pdb.set_trace()
    if not np.isfinite(err).all():
        pdb.set_trace()

    try:
        fit, fitcov = optimize.leastsq(egaussian, guess, args=(xtemp, vec, err), full_output=True)[0:2]
    except:
        pdb.set_trace()

    if fitcov is None: # The fitting was really bad!
        fiterr = np.abs(fit)
    else:
        fiterr = np.sqrt(np.diag(fitcov))

    #pdb.set_trace()
    if verbose:
        print 'Best-fit parameters>>', fit
        f = plt.figure()
        ax = plt.axes()
        plt.plot(xtemp, vec, 'o', \
                     xtemp, gaussian(fit, xtemp), '-', \
                     xtemp, gaussian(guess, xtemp), '--')

    return fit, fiterr

# ===========================================================================


def egaussian(p, x, y, e=None):
    """ Compute the deviation between the values y and the gaussian defined by p, x:

    p is a three- or four-component array, list, or tuple.

    Returns:   y - p3 - p0/(p1*sqrt(2pi)) * exp(-(x-p2)**2 / (2*p1**2))

    if an error array, e (typ. one-sigma) is entered, the returned value is divided by e.

    SEE ALSO:  :func:`gaussian`"""
    # 2008-09-11 15:19 IJC: Created
    # 2009-09-02 15:20 IJC: Added weighted case
    # 2011-05-18 11:46 IJMC: Moved to analysis.

    if e==None:
        e=np.ones(x.shape)
    fixval(e,y.max()*1e10)

    z = (y - gaussian(p, x))/e
    fixval(z,0)

    return z

# ===========================================================================


def gaussian(p, x):
    """ Compute a gaussian distribution at the points x.

        p is a three- or four-component array, list, or tuple:

        y =  [p3 +] p0/(p1*sqrt(2pi)) * exp(-(x-p2)**2 / (2*p1**2))

        p[0] -- Area of the gaussian
        p[1] -- one-sigma dispersion
        p[2] -- central offset (mean location)
        p[3] -- optional constant, vertical offset

        NOTE: FWHM = 2*sqrt(2*ln(2)) * p1  ~ 2.3548*p1

        SEE ALSO:  :func:`egaussian`"""
    #2008-09-11 15:11 IJC: Created for LINEPROFILE
    # 2011-05-18 11:46 IJC: Moved to analysis.
    # 2013-04-11 12:03 IJMC: Tried to speed things up slightly via copy=False
    # 2013-05-06 21:42 IJMC: Tried to speed things up a little more.

    if not isinstance(x, np.ndarray):
        x = array(x, dtype=float, copy=False)

    if len(p)==3:
        p = array(p, copy=True)
        p = concatenate((p, [0]))
    #elif len(p)==4:
    #    p = array(p, copy=False)

    return  p[3] + p[0]/(p[1]*np.sqrt(2*np.pi)) * np.exp(-(x-p[2])**2 / (2*p[1]**2))

# ===========================================================================


def fixval(arr, repval, retarr=False):
    """Fix non-finite values in a numpy array, and replace them with repval.

    :INPUT:
       arr -- numpy array, with values to be replaced.

       repval -- value to replace non-finite elements with

    :OPTIONAL INPUT:

       retarr -- if False, changes values in arr directly (more
       efficient).  if True, returns a fixed copy of the input array,
       which is left unchanged.

    :example:
     ::

       fixval(arr, -1)
       """
    # 2009-09-02 14:07 IJC: Created
    # 2012-12-23 11:49 IJMC: Halved run time.

    if retarr:
        arr2 = arr.ravel().copy()
    else:
        arr2 = arr.ravel()

    finiteIndex = np.isfinite(arr2)
    if not finiteIndex.any():
        badIndex = find((1-finiteIndex))
        arr2[badIndex] = repval

    if retarr:
        return arr2.reshape(arr.shape)
    else:
        return

# ===========================================================================


def polyfitr(x, y, N, s, fev=100, w=None, diag=False, clip='both', \
                 verbose=False, plotfit=False, plotall=False, eps=1e-13, catchLinAlgError=False):
    """Matplotlib's polyfit with weights and sigma-clipping rejection.

    :DESCRIPTION:
      Do a best fit polynomial of order N of y to x.  Points whose fit
      residuals exeed s standard deviations are rejected and the fit is
      recalculated.  Return value is a vector of polynomial
      coefficients [pk ... p1 p0].

    :OPTIONS:
        w:   a set of weights for the data; uses CARSMath's weighted polynomial 
             fitting routine instead of numpy's standard polyfit.

        fev:  number of function evaluations to call before stopping

        'diag'nostic flag:  Return the tuple (p, chisq, n_iter)

        clip: 'both' -- remove outliers +/- 's' sigma from fit
              'above' -- remove outliers 's' sigma above fit
              'below' -- remove outliers 's' sigma below fit

        catchLinAlgError : bool
          If True, don't bomb on LinAlgError; instead, return [0, 0, ... 0].

    :REQUIREMENTS:
       :doc:`CARSMath`

    :NOTES:
       Iterates so long as n_newrejections>0 AND n_iter<fev. 


     """
    # 2008-10-01 13:01 IJC: Created & completed
    # 2009-10-01 10:23 IJC: 1 year later! Moved "import" statements within func.
    # 2009-10-22 14:01 IJC: Added 'clip' options for continuum fitting
    # 2009-12-08 15:35 IJC: Automatically clip all non-finite points
    # 2010-10-29 09:09 IJC: Moved pylab imports inside this function
    # 2012-08-20 16:47 IJMC: Major change: now only reject one point per iteration!
    # 2012-08-27 10:44 IJMC: Verbose < 0 now resets to 0
    # 2013-05-21 23:15 IJMC: Added catchLinAlgError

    #from CARSMath import polyfitw
    
    from numpy.linalg import LinAlgError

    if verbose < 0:
        verbose = 0
    
    xx = np.array(x, copy=False)
    yy = np.array(y, copy=False)
    noweights = (w==None)
    if noweights:
        ww = np.ones(xx.shape, float)
    else:
        ww = np.array(w, copy=False)

    ii = 0
    nrej = 1

    if noweights:
        goodind = np.isfinite(xx)*np.isfinite(yy)
    else:
        goodind = np.isfinite(xx)*np.isfinite(yy)*np.isfinite(ww)
    
    xx2 = xx[goodind]
    yy2 = yy[goodind]
    ww2 = ww[goodind]

    while (ii<fev and (nrej<>0)):
        if noweights:
            p = np.polyfit(xx2,yy2,N)
            residual = yy2 - np.polyval(p,xx2)
            stdResidual = np.std(residual)
            clipmetric = s * stdResidual
        else:
            if catchLinAlgError:
                try:
                    p = polyfitw(xx2,yy2, ww2, N)
                except LinAlgError:
                    p = np.zeros(N+1, dtype=float)
            else:
                p = polyfitw(xx2,yy2, ww2, N)

            p = p[::-1]  # polyfitw uses reverse coefficient ordering
            residual = (yy2 - np.polyval(p,xx2)) * np.sqrt(ww2)
            clipmetric = s

        if clip=='both':
            worstOffender = abs(residual).max()
            #pdb.set_trace()
            if worstOffender <= clipmetric or worstOffender < eps:
                ind = np.ones(residual.shape, dtype=bool)
            else:
                ind = abs(residual) < worstOffender
        elif clip=='above':
            worstOffender = residual.max()
            if worstOffender <= clipmetric:
                ind = np.ones(residual.shape, dtype=bool)
            else:
                ind = residual < worstOffender
        elif clip=='below':
            worstOffender = residual.min()
            if worstOffender >= -clipmetric:
                ind = np.ones(residual.shape, dtype=bool)
            else:
                ind = residual > worstOffender
        else:
            ind = np.ones(residual.shape, dtype=bool)
    
        xx2 = xx2[ind]
        yy2 = yy2[ind]
        if (not noweights):
            ww2 = ww2[ind]
        ii = ii + 1
        nrej = len(residual) - len(xx2)
        if plotall:
            allx = np.arange(x[0],x[-1],1)
            figure()
            plot(x,y, '.', xx2,yy2, 'x', allx, np.polyval(p, allx), '--')
            legend(['data', 'fit data', 'fit'])
            title('Iter. #' + str(ii) + ' -- Close all windows to continue....')
            plt.show()
        
        if verbose:
            print str(len(x)-len(xx2)) + ' points rejected on iteration #' + str(ii)

    if (plotfit or plotall):
        figure()
        plot(x,y, '.', xx2,yy2, 'x', allx, np.polyval(p, allx), '--')
        legend(['data', 'fit data', 'fit'])
        title('Close window to continue....')

    if diag:
        #chisq = ( (residual)**2. / yy2 ).sum()
        chisq = ( (residual)**2. ).sum()
        p = (p, chisq, ii)

    return p

# ===========================================================================


def polyfitw(x, y, w, ndegree, return_fit=0):
   """
   Performs a weighted least-squares polynomial fit with optional error estimates.

   Inputs:
      x: 
         The independent variable vector.

      y: 
         The dependent variable vector.  This vector should be the same 
         length as X.

      w: 
         The vector of weights.  This vector should be same length as 
         X and Y.

      ndegree: 
         The degree of polynomial to fit.

   Outputs:
      If return_fit==0 (the default) then polyfitw returns only C, a vector of 
      coefficients of length ndegree+1.
      If return_fit!=0 then polyfitw returns a tuple (c, yfit, yband, sigma, a)
         yfit:  
            The vector of calculated Y's.  Has an error of + or - Yband.

         yband: 
            Error estimate for each point = 1 sigma.

         sigma: 
            The standard deviation in Y units.

         a: 
            Correlation matrix of the coefficients.

   Written by:   George Lawrence, LASP, University of Colorado,
                 December, 1981 in IDL.
                 Weights added, April, 1987,  G. Lawrence
                 Fixed bug with checking number of params, November, 1998, 
                 Mark Rivers.  
                 Python version, May 2002, Mark Rivers
   """

   n = min(len(x), len(y)) # size = smaller of x,y
   m = ndegree + 1         # number of elements in coeff vector
   a = np.zeros((m,m),dtype=float)  # least square matrix, weighted matrix
   b = np.zeros(m,dtype=float)    # will contain sum w*y*x^j
   z = np.ones(n,dtype=float)     # basis vector for constant term

   a[0,0] = np.sum(w)
   b[0] = np.sum(w*y)

   for p in range(1, 2*ndegree+1):     # power loop
      z = z*x   # z is now x^p
      if (p < m):  b[p] = np.sum(w*y*z)   # b is sum w*y*x^j
      sum = np.sum(w*z)
      for j in range(max(0,(p-ndegree)), min(ndegree,p)+1):
         a[j,p-j] = sum

   #a = LinearAlgebra.inverse(a)
   a = np.linalg.inv(a)
   #c = N.matrixmultiply(b, a)
   c = np.dot(b, a)
   if (return_fit == 0):
      return c     # exit if only fit coefficients are wanted

   # compute optional output parameters.
   yfit = np.zeros(n,dtype=float)+c[0]   # one-sigma error estimates, init
   for k in range(1, ndegree +1):
      yfit = yfit + c[k]*(x**k)  # sum basis vectors
   var = np.sum((yfit-y)**2 )/(n-m)  # variance estimate, unbiased
   sigma = np.sqrt(var)
   yband = np.zeros(n,dtype=float) + a[0,0]
   z = np.ones(n,dtype=float)
   for p in range(1,2*ndegree+1):     # compute correlated error estimates on y
      z = z*x		# z is now x^p
      sum = 0.
      for j in range(max(0, (p - ndegree)), min(ndegree, p)+1):
         sum = sum + a[j,p-j]
      yband = yband + sum * z      # add in all the error sources
   yband = yband*var
   yband = np.sqrt(yband)
   return c, yfit, yband, sigma, a

# ===========================================================================


class baseObject:
    """Empty object container.
    """
    def __init__(self):
        return


# ===========================================================================

def lampextract(frame,trace,extract_radius):
    nlam,width = frame.shape

    #trace.reshape(nlam,1) is the center of the fitted profile of the star
    # xxx is the distance of the frame on either side of the fitted profile.
    xxx = np.arange(width) - trace.reshape(nlam,1)
    extractionApertures = np.abs(xxx) <= extract_radius

    standardSpectrum = np.zeros((nlam, 1), dtype=float)
    for ii in range(nlam):
        thisrow_good = extractionApertures[ii]
        standardSpectrum[ii] = frame[ii, thisrow_good].sum()

    return standardSpectrum
