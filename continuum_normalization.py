'''
Written by J. T. Fuchs, UNC, 2016. Some initial work done by E. Dennihy, UNC.

Continuum normalizes a ZZ Ceti spectrum to match a DA model. The response function is determined by dividing the observed spectrum by the model spectrum. The response function is fitted with a polynomial. The observed spectrum is then divided by this fit to deliver the continuum normalized spectrum.

To run: (Red filename is optional)
python model_calibration.py bluefilename redfilename
python model_calibration.py wtfb.wd1401-147_930_blue.ms.fits wtfb.wd1401-147_930_red.ms.fits

:INPUTS:
     bluefilename: string, filename of wavelength calibrated ZZ Ceti blue spectrum

:OPTIONAL:
     redfilename: string, filename of wavelength calibrated ZZ Ceti red spectrum

:OUTPUTS:
     continuum normalized spectrum: '_flux_model' added to filename. Name of model used written to header. 
     
     normalization_ZZCETINAME_DATE.txt: File for diagnostics. ZZCETINAME is name of the ZZ Ceti spectrum supplied. DATE is the current date and time. Columns are: blue wavelengths, blue response all data, blue masked wavelengths, blue masked response data, blue response fit, red wavelengths, red response all data, red masked wavelengths, red masked response data, red response fit

'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
#import pyfits as fits
import astropy.io.fits as fits
import spectools as st
import os
import sys
import datetime
from scipy.interpolate import UnivariateSpline


#=============================================


def normalize_now(filenameblue,filenamered,redfile,plotall=True,extinct_correct=False):
    #Read in the observed spectrum
    obs_spectrablue,airmass,exptime,dispersion = st.readspectrum(filenameblue)
    datalistblue = fits.open(filenameblue)

    if redfile:
        obs_spectrared, airmassred,exptimered,dispersionred = st.readspectrum(filenamered)
    
    #Extinction correction
    if extinct_correct:
        print 'Extinction correcting spectra.'
        plt.clf()
        plt.plot(obs_spectrablue.warr,obs_spectrablue.opfarr)
        obs_spectrablue.opfarr = st.extinction_correction(obs_spectrablue.warr,obs_spectrablue.opfarr,airmass)
        obs_spectrablue.farr = st.extinction_correction(obs_spectrablue.warr,obs_spectrablue.farr,airmass)
        obs_spectrablue.sky = st.extinction_correction(obs_spectrablue.warr,obs_spectrablue.sky,airmass)
        obs_spectrablue.sigma = st.extinction_correction(obs_spectrablue.warr,obs_spectrablue.sigma,airmass)
        plt.plot(obs_spectrablue.warr,obs_spectrablue.opfarr)
        plt.show()
        
        if redfile:
            plt.clf()
            plt.plot(obs_spectrared.warr,obs_spectrared.opfarr)
            obs_spectrared.opfarr = st.extinction_correction(obs_spectrared.warr,obs_spectrared.opfarr,airmassred)
            obs_spectrared.farr = st.extinction_correction(obs_spectrared.warr,obs_spectrared.farr,airmassred)
            obs_spectrared.sky = st.extinction_correction(obs_spectrared.warr,obs_spectrared.sky,airmassred)
            obs_spectrared.sigma = st.extinction_correction(obs_spectrared.warr,obs_spectrared.sigma,airmassred)
            plt.plot(obs_spectrared.warr,obs_spectrared.opfarr)
            plt.show()
    

    #Read in measured FWHM from header. This is used to convolve the model spectrum.
    FWHMpix = datalistblue[0].header['specfwhm']
    FWHM = FWHMpix * (obs_spectrablue.warr[-1] - obs_spectrablue.warr[0])/len(obs_spectrablue.warr)

    #Read in DA model
    cwd = os.getcwd()
    os.chdir('/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/modelfitting/Koester_08')
    dafile = 'da12500_800.dk'
    mod_wav, mod_spec = np.genfromtxt(dafile,unpack=True,skip_header=33)
    os.chdir(cwd) #Move back to directory with observed spectra


    #Convolve the model to match the seeing of the spectrum
    intlambda = np.divide(range(31000),10.) + 3660.0
    lowlambda = np.min(np.where(mod_wav > 3600.))
    highlambda = np.min(np.where(mod_wav > 6800.))
    shortlambdas = mod_wav[lowlambda:highlambda]
    shortinten = mod_spec[lowlambda:highlambda]
    interp = inter.InterpolatedUnivariateSpline(shortlambdas,shortinten,k=1)
    intflux = interp(intlambda)
    sig = FWHM / (2. * np.sqrt(2.*np.log(2.)))
    gx = np.divide(range(360),10.)
    gauss = (1./(sig * np.sqrt(2. * np.pi))) * np.exp(-(gx-18.)**2./(2.*sig**2.))
    gf = np.divide(np.outer(intflux,gauss),10.)
    length = len(intflux) - 360.
    cflux = np.zeros(length)
    clambda = intlambda[180:len(intlambda)-180]
    x  = 0
    while x < length:
        cflux[x] = np.sum(np.diagonal(gf,x,axis1=1,axis2=0),dtype='d')
        x += 1
    interp2 = inter.InterpolatedUnivariateSpline(clambda,cflux,k=1)
    cflux2blue = interp2(obs_spectrablue.warr)
    cflux2blue /= 10**13. #Divide by 10**13 to scale
    if redfile:
        cflux2red = interp2(obs_spectrared.warr)
        cflux2red /= 10**13. #Divide by 10**13 to scale


    #plt.clf()
    #plt.plot(obs_spectrablue.warr,obs_spectrablue.opfarr,'b')
    #plt.plot(obs_spectrablue.warr,cflux2blue,'r')
    #if redfile:
    #    plt.plot(obs_spectrared.warr,obs_spectrared.opfarr,'b')
    #    plt.plot(obs_spectrared.warr,cflux2red,'r')
    #plt.show()

    #The response function is the observed spectrum divided by the model spectrum.
    response_blue = obs_spectrablue.opfarr/cflux2blue
    if redfile:
        response_red = obs_spectrared.opfarr/cflux2red

    '''
    plt.clf()
    plt.plot(obs_spectrablue.warr,response_blue,'k')
    if redfile:
        plt.plot(obs_spectrared.warr,response_red,'k')
    plt.show()
    '''

    #We want to mask out the Balmer line features, and the telluric line in the red spectrum. Set up the wavelength ranges to mask here.
    #balmer_features_blue = [[3745,3757],[3760,3780],[3784,3812],[3816,3856],[3865,3921],[3935,4021],[4040,4191],[4223,4460],[4691,5010]] #Keeping ends
    balmer_features_blue = [[3400,3700],[3745,3757],[3760,3780],[3784,3812],[3816,3856],[3865,3921],[3935,4021],[4040,4191],[4223,4460],[4691,5010],[5140,5500]] #Discarding ends
    balmer_features_red = [[6350,6780],[6835,6970]]

    balmer_mask_blue = obs_spectrablue.warr == obs_spectrablue.warr
    for wavrange in balmer_features_blue:
        inds = np.where((obs_spectrablue.warr > wavrange[0]) & (obs_spectrablue.warr < wavrange[1]))
        balmer_mask_blue[inds] = False

    if redfile:
        balmer_mask_red = obs_spectrared.warr == obs_spectrared.warr
        for wavrange in balmer_features_red:
            indxs = np.where((obs_spectrared.warr > wavrange[0]) & (obs_spectrared.warr < wavrange[1]))
            balmer_mask_red[indxs] = False

    spec_wav_masked_blue = obs_spectrablue.warr[balmer_mask_blue]
    response_masked_blue = response_blue[balmer_mask_blue]

    if redfile:    
        spec_wav_masked_red = obs_spectrared.warr[balmer_mask_red]
        response_masked_red = response_red[balmer_mask_red]


    #Fit the response function with a polynomial. The order of polynomial is specified first. 
    response_poly_order_blue = 7.
    response_fit_blue_poly = np.polyfit(spec_wav_masked_blue,response_masked_blue,response_poly_order_blue)
    response_fit_blue = np.poly1d(response_fit_blue_poly)


    if redfile:
        response_poly_order_red = 3.
        response_fit_red_poly = np.polyfit(spec_wav_masked_red,response_masked_red,response_poly_order_red)
        response_fit_red = np.poly1d(response_fit_red_poly)

    #Save response function
    #np.savetxt('response_model_no_extinction.txt',np.transpose([obs_spectrablue.warr,response_fit_blue(obs_spectrablue.warr),obs_spectrared.warr,response_fit_red(obs_spectrared.warr)]))
    #plt.clf()
    #plt.plot(obs_spectrablue.warr,response_fit_blue(obs_spectrablue.warr)/response_fit_blue(obs_spectrablue.warr)[1000])
    #plt.show()
    #exit()
    if plotall:
        plt.clf()
        plt.plot(obs_spectrablue.warr,response_blue,'r')
        plt.plot(spec_wav_masked_blue,response_masked_blue,'g.')
        plt.plot(obs_spectrablue.warr,response_fit_blue(obs_spectrablue.warr),'k--')
        #plt.show()
        
        #plt.clf()
        if redfile:
            plt.plot(obs_spectrared.warr,response_red,'r')
            plt.plot(spec_wav_masked_red,response_masked_red,'g.')
            plt.plot(obs_spectrared.warr,response_fit_red(obs_spectrared.warr),'k--')
        plt.show()

    #Divide by the fit to the response function to get the continuum normalized spectra. Divide every extension by the same polynomial
    fcorr_wd_blue_opfarr = obs_spectrablue.opfarr / response_fit_blue(obs_spectrablue.warr)
    fcorr_wd_blue_farr = obs_spectrablue.farr / response_fit_blue(obs_spectrablue.warr)
    fcorr_wd_blue_sky = obs_spectrablue.sky / response_fit_blue(obs_spectrablue.warr)
    fcorr_wd_blue_sigma = obs_spectrablue.sigma / response_fit_blue(obs_spectrablue.warr)


    if redfile:
        fcorr_wd_red_opfarr = obs_spectrared.opfarr / response_fit_red(obs_spectrared.warr)
        fcorr_wd_red_farr = obs_spectrared.farr / response_fit_red(obs_spectrared.warr)
        fcorr_wd_red_sky = obs_spectrared.sky / response_fit_red(obs_spectrared.warr)
        fcorr_wd_red_sigma = obs_spectrared.sigma / response_fit_red(obs_spectrared.warr)
    
    if plotall:
        plt.clf()
        plt.plot(obs_spectrablue.warr,fcorr_wd_blue_opfarr,'b')
        if redfile:
            plt.plot(obs_spectrared.warr,fcorr_wd_red_opfarr,'r')
        plt.show()
    #exit()

    #Save parameters for diagnostics
    if redfile:
        bigarray = np.zeros([len(obs_spectrablue.warr),12])
        bigarray[0:len(obs_spectrablue.warr),0] = obs_spectrablue.warr
        bigarray[0:len(response_blue),1] = response_blue
        bigarray[0:len(spec_wav_masked_blue),2] = spec_wav_masked_blue
        bigarray[0:len(response_masked_blue),3] = response_masked_blue
        bigarray[0:len(response_fit_blue(obs_spectrablue.warr)),4] = response_fit_blue(obs_spectrablue.warr)
        bigarray[0:len(fcorr_wd_blue_opfarr),5] = fcorr_wd_blue_opfarr
        bigarray[0:len(obs_spectrared.warr),6] = obs_spectrared.warr
        bigarray[0:len(response_red),7] = response_red
        bigarray[0:len(spec_wav_masked_red),8] = spec_wav_masked_red
        bigarray[0:len(response_masked_red),9] = response_masked_red
        bigarray[0:len(response_fit_red(obs_spectrared.warr)),10] = response_fit_red(obs_spectrared.warr)
        bigarray[0:len(fcorr_wd_red_opfarr),11] = fcorr_wd_red_opfarr
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        endpoint = '_930'
        with open('continuum_normalization_' + filenameblue[5:filenameblue.find(endpoint)] + '_' + now + '.txt','a') as handle:
            header = str(filenameblue) + ',' + str(filenamered) + ',' + dafile + '\n Columns structured as blue then red. If no red file, only blue data given. Columns are: blue wavelengths, blue response all data, blue masked wavelengths, blue masked response data, blue response fit, blue continuum-normalize flux, red wavelengths, red response all data, red masked wavelengths, red masked response data, red response fit, red continuum-normalized flux'
            np.savetxt(handle,bigarray,fmt='%f',header=header)
    if not redfile:
        bigarray = np.zeros([len(obs_spectrablue.warr),6])
        bigarray[0:len(obs_spectrablue.warr),0] = obs_spectrablue.warr
        bigarray[0:len(response_blue),1] = response_blue
        bigarray[0:len(spec_wav_masked_blue),2] = spec_wav_masked_blue
        bigarray[0:len(response_masked_blue),3] = response_masked_blue
        bigarray[0:len(response_fit_blue(obs_spectrablue.warr)),4] = response_fit_blue(obs_spectrablue.warr)
        bigarray[0:len(fcorr_wd_blue_opfarr),5] = fcorr_wd_blue_opfarr
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        endpoint = '_930'
        with open('continuum_normalization_' + filenameblue[5:filenameblue.find(endpoint)] + '_' + now + '.txt','a') as handle:
            header = str(filenameblue) + ',' + ',' + dafile + '\n Columns structured as blue then red. If no red file, only blue data given. Columns are: blue wavelengths, blue response all data, blue masked wavelengths, blue masked response data, blue response fit, blue continuum-normalized flux'
            np.savetxt(handle,bigarray,fmt='%f',header=header)
    
    
    #Save the continuum normalized spectra here.
    Ni = 4. #Number of extensions
    Nx1 = len(fcorr_wd_blue_opfarr)
    if redfile:
        Nx2 = len(fcorr_wd_red_opfarr)
    Ny = 1. #All 1D spectra

    #Update header
    header1 = st.readheader(filenameblue)
    header1.set('STANDARD',dafile,'DA Model for Continuum Calibration')
    header1.set('RESPPOLY',response_poly_order_blue,'Polynomial order for Response Function')
    header1.set('DATENORM',datetime.datetime.now().strftime("%Y-%m-%d"),'Date of Continuum Normalization')


    data1 = np.empty(shape = (Ni,Ny,Nx1))
    data1[0,:,:] = fcorr_wd_blue_opfarr 
    data1[1,:,:] = fcorr_wd_blue_farr
    data1[2,:,:] = fcorr_wd_blue_sky
    data1[3,:,:] = fcorr_wd_blue_sigma 

    #Check that filename does not already exist. Prompt user for input if it does.
    loc1 = filenameblue.find('.ms.fits')
    newname1 = filenameblue[0:loc1] + '_flux_model_short.ms.fits'
    clob = False
    mylist = [True for f in os.listdir('.') if f == newname1]
    exists = bool(mylist)

    if exists:
        print 'File %s already exists.' % newname1
        nextstep = raw_input('Do you want to overwrite or designate a new name (overwrite/new)? ')
        if nextstep == 'overwrite':
            clob = True
            exists = False
        elif nextstep == 'new':
            newname1 = raw_input('New file name: ')
            exists = False
        else:
            exists = False
    print 'Writing ', newname1
    newim1 = fits.PrimaryHDU(data=data1,header=header1)
    newim1.writeto(newname1,clobber=clob)


    #Save the red file if it exists.
    if redfile:
        header2 = st.readheader(filenamered)
        header2.set('STANDARD',dafile,'DA Model for Continuum Calibration')
        header2.set('RESPPOLY',response_poly_order_red,'Polynomial order for Response Function')
        header2.set('DATENORM',datetime.datetime.now().strftime("%Y-%m-%d"),'Date of Continuum Normalization')
        data2 = np.empty(shape = (Ni,Ny,Nx2))
        data2[0,:,:] = fcorr_wd_red_opfarr 
        data2[1,:,:] = fcorr_wd_red_farr
        data2[2,:,:] = fcorr_wd_red_sky
        data2[3,:,:] = fcorr_wd_red_sigma 
    
    
        loc2 = filenamered.find('.ms.fits')
        newname2 = filenamered[0:loc2] + '_flux_model_short.ms.fits'
        clob = False
        mylist = [True for f in os.listdir('.') if f == newname2]
        exists = bool(mylist)

        if exists:
            print 'File %s already exists.' % newname2
            nextstep = raw_input('Do you want to overwrite or designate a new name (overwrite/new)? ')
            if nextstep == 'overwrite':
                clob = True
                exists = False
            elif nextstep == 'new':
                newname2 = raw_input('New file name: ')
                exists = False
            else:
                exists = False
        print 'Writing ', newname2
        newim2 = fits.PrimaryHDU(data=data2,header=header2)
        newim2.writeto(newname2,clobber=clob)


#################################
#############################

if __name__ == '__main__':
    if len(sys.argv) == 3:
        script, filenameblue, filenamered = sys.argv
        redfile = True
    elif len(sys.argv) == 2:
        script, filenameblue = sys.argv
        filenamered = None
        redfile = False
    else:
        print '\n Incorrect number of arguments. \n'
    
    normalize_now(filenameblue,filenamered,redfile,extinct_correct=False)

        
