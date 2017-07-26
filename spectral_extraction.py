'''
Spectral Extraction from a 2D image.
Uses superextract written by Ian Crossfield and modified by JTF.
superextract is based on optimal spectral extraction as detailed by Marsh (1989) and Horne (1986)
Crossfield's version can be found at www.lpl.arizona.edu/!ianc/python/index.html
Dependencies: superextract.py and superextrac_tools.py

For best results, first bias-subract, flat-field, and trim the 2D image before running this.

To run:
python spec_extract.py filename_of_2d_spectrum
python spec_extract.py tnb.0526.WD1422p095_930_blue.fits

:INPUTS:
    spectral filename : string
        Name of file you want to extract.

:OPTIONS:
    lamp file : string, optional
        You will be prompted for this. If you want to extract a lamp using the same trace as the spectral filename, provide this. 

:OUTPUTS:
    Extracted 1D spectrum. .ms.fits is added to the end of the file. User will be prompted before overwriting existing image. Extensions in order are optimally extracted spectrum, raw extracted spectrum, background, sigma spectrum

    extraction_params.txt: text file containing date/time of extraction, extraction radius, background radius, and new filename

    extraction_ZZCETINAME_DATE.txt: File for diagnostics. ZZCETINAME is name of the ZZ Ceti spectrum supplied. DATE is the current date and time. Columns are: Measured FWHM, pixel value of each FWHM, fit to FWHM measurements, all pixel values, profile pixels, profile position, fit profile positions

extract_radius and  bkg_radii computed automatically. FWHM used for extract_radius
output file name and lamp output filename done automatically

To Do:
- 

'''
import sys
import os
import numpy as np
#import pyfits as fits
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import datetime
import mpfit
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit

import spectools as st
import superextract
from superextract_tools import lampextract
from ReduceSpec_tools import gauss, fitgauss
from pylab import *


#===========================================
def SigClip(data_set, lo_sig, hi_sig):
    # Sigma Cliping Function  #
    # Input is set of counts for a particular pixel, 
    # along with low and high sigma factors. 
    # Output is a list containg only the data that is with the sigma factors.
    # Only a single rejection iteration is made. 
    Avg = np.mean(data_set)
    St_Div = np.std(data_set)
    min_val = Avg-lo_sig*St_Div
    max_val = Avg+hi_sig*St_Div
    cliped_data = []
    for val in data_set:
        if min_val <= val <= max_val:
            cliped_data.append( val )
        else:
            try:
                cliped_data.append(cliped_data[-1])
            except:
                cliped_data.append(data_set[1])
    return cliped_data 

#===========================================
def line(x,m,b):
    y = m*x + b
    return y
#===========================================


#===========================================
#Primary Program
#===========================================
def extract_now(specfile,lamp,FWHMfile,tracefile,trace_exist=False):
    #Open file and read gain and readnoise
    datalist = fits.open(specfile)
    data = datalist[0].data
    data = np.transpose(data[0,:,:])
    
    #Since we have combined multiple images, to keep our statistics correct, we need to multiply the values in ADU by the number of images
    try:
        nimages = float(datalist[0].header['NCOMBINE'])
    except:
        nimages = 1.
    
    data = nimages * data
    
    #gain = datalist[0].header['GAIN']
    gain = 1.33 #from 2017-06-07
    rdnoise = np.sqrt(nimages) * datalist[0].header['RDNOISE']

    #Calculate the variance of each pixel in ADU
    varmodel = ((nimages*rdnoise**2.) + np.absolute(data)*gain)/gain
    
    #Fit a Gaussian every 10 pixels to determine FWHM for convolving in the model fitting, unless this file already exists
    fitpixel = np.arange(3,len(data[:,100]),10)
    allfwhm = np.zeros(len(fitpixel))
    for x in fitpixel:
        #forfit = data[x,2:]
        forfit = np.median(np.array([data[x-2,2:],data[x-1,2:],data[x,2:],data[x+1,2:],data[x+2,2:]]),axis=0)
        guess = np.zeros(4)
        guess[0] = np.mean(forfit)
        guess[1] = np.amax(forfit)
        guess[2] = np.argmax(forfit)
        guess[3] = 3.
        error_fit = np.ones(len(forfit))
        xes = np.linspace(0,len(forfit)-1,num=len(forfit))
        fa = {'x':xes,'y':forfit,'err':error_fit}
        fitparams = mpfit.mpfit(fitgauss,guess,functkw=fa,quiet=True)
        allfwhm[np.argwhere(fitpixel == x)] = 2.*np.sqrt(2.*np.log(2.))*fitparams.params[3]
        #plt.clf()
        #plt.plot(forfit)
        #plt.plot(xes,gauss(xes,fitparams.params))
        #plt.show()
        
    fwhmclipped = SigClip(allfwhm,3,3)
    
    if not FWHMfile:
    #Fit using a line, but give user the option to fit with a different order
        order = 1
        repeat = 'yes'
        while repeat == 'yes':
            fwhmpolyvalues = np.polyfit(fitpixel,fwhmclipped,order)
            allpixel = np.arange(0,len(data[:,100]),1)
            fwhmpoly = np.poly1d(fwhmpolyvalues)
            plt.clf()
            plt.plot(fitpixel,fwhmclipped,'^')
            plt.plot(allpixel,fwhmpoly(allpixel),'g')
            plt.title(specfile)
            plt.show()
            repeat = raw_input('Do you want to try again (yes/no)? ')
            if repeat == 'yes':
                order = raw_input('New order for polynomial: ')
        
        
        locfwhm = specfile.find('.fits')
        print '\n Saving FWHM file.'
        np.save(specfile[0:locfwhm] + '_poly',fwhmpoly(allpixel))
    diagnostics = np.zeros([len(data[:,100]),12])
    diagnostics[0:len(allfwhm),0] = fwhmclipped
    diagnostics[0:len(fitpixel),1] = fitpixel
    if not FWHMfile:
        diagnostics[0:len(allpixel),2] = fwhmpoly(allpixel)
        diagnostics[0:len(allpixel),3] = allpixel
    else:
        fwhm_fit = np.load(FWHMfile)
        allpixel = np.arange(0,len(data[:,100]),1)
        diagnostics[0:len(fwhm_fit),2] = fwhm_fit
        diagnostics[0:len(allpixel),3] = allpixel
    
    #===============================
    #Section to prepare inputs for extraction
    #===============================
    
    
    #Fit a column of the 2D image to determine the FWHM in pixels
    if 'blue' in specfile.lower():
        #Average over 5 rows to deal with any remaining cosmic rays
        forfit = np.mean(np.array([data[1198,:],data[1199,:],data[1200,:],data[1201,:],data[1202,:]]),axis=0)
    elif 'red' in specfile.lower():
        forfit = np.mean(np.array([data[998,:],data[999,:],data[1000,:],data[1001,:],data[1002,:]]),axis=0)
    

    guess = np.zeros(4)
    guess[0] = np.mean(forfit)
    guess[1] = np.amax(forfit)
    guess[2] = np.argmax(forfit)
    guess[3] = 3.
    
    error_fit = np.ones(len(forfit))
    xes = np.linspace(0,len(forfit)-1,num=len(forfit))
    fa = {'x':xes,'y':forfit,'err':error_fit}
    fitparams = mpfit.mpfit(fitgauss,guess,functkw=fa,quiet=True)
    
    #The guassian gives us sigma, convert to FWHM
    fwhm = 2.*np.sqrt(2.*np.log(2.))*fitparams.params[3]
    extraction_rad = 5. * np.round(fwhm,decimals=1) #Extract up to 5 times FWHM
    
    
    #Check to make sure background region does not go within 10 pixels of edge
    background_radii = [35,60]
    #First check this against the bottom
    if fitparams.params[2] - background_radii[1] < 10.:
        background_radii[1] = fitparams.params[2] - 10.
        background_radii[0] -= 60 - background_radii[1]
    #Then check against the top
    hold = background_radii[1]
    if fitparams.params[2] + background_radii[1] > 190.:
        background_radii[1] = 190. - fitparams.params[2]
        background_radii[0] -= hold - background_radii[1]
    #Ensure that the closest point is at least 20 pixels away.
    if background_radii[0] < 20.:
        background_radii[0] = 20.
    background_radii[0] = np.round(background_radii[0],decimals=1)
    background_radii[1] = np.round(background_radii[1],decimals=1)
    #plt.plot(data[1200,:])
    #plt.plot(xes,gauss(xes,fitparams.params))
    #plt.show()
    #extraction_rad = 10.
    #background_radii = [40,60]
    
    
    #Extract the spectrum using superextract
    print 'Starting extraction.'
    if trace_exist:
        trace = np.load(tracefile)
        output_spec = superextract.superExtract(data,varmodel,gain,rdnoise,trace=trace,pord=2,tord=2,bord=1,bkg_radii=background_radii,bsigma=2.,extract_radius=extraction_rad,dispaxis=1,verbose=False,csigma=5.,polyspacing=1,retall=False)
    else:
        output_spec = superextract.superExtract(data,varmodel,gain,rdnoise,pord=2,tord=2,bord=1,bkg_radii=background_radii,bsigma=2.,extract_radius=extraction_rad,dispaxis=1,verbose=False,csigma=5.,polyspacing=1,retall=False)
    #pord = order of profile polynomial. Default = 2. This seems appropriate, no change for higher or lower order.
    #tord = degree of spectral-trace polynomial, 1 = line
    #bord = degree of polynomial background fit
    #bkg_radii = inner and outer radii to use in computing background. Goes on both sides of aperture.  
    #bsigma = sigma-clipping thresholf for computing background
    #extract_radius: radius for spectral extraction. Setting this to be 5*FWHM
    #csigma = sigma-clipping threshold for cleaning & cosmic-ray rejection. Default = 5.
    #qmode: how to compute Marsh's Q-matrix. 'fast-linear' default and preferred.
    #nreject = number of outlier-pixels to reject at each iteration. Default = 100
    #polyspacing = Marsh's S: the spacing between the polynomials. This should be <= 1. Default = 1. Best to leave at 1. S/N decreases dramatically if greater than 1. If less than one, slower but final spectrum is the same. Crossfield note: A few cursory tests suggests that the extraction precision (in the high S/N case) scales as S^-2 -- but the code slows down as S^2.
    #Verbose=True if you want lots of output

    ###########
    # In superextract, to plot a 2D frame at any point, use the following
    #   plt.clf()
    #   plt.imshow(np.transpose(frame),aspect='auto',interpolation='nearest')
    #   plt.show()
    ##########
    
    print 'Done extracting. Starting to save.'
    if not trace_exist:
        print 'Saving the trace.'
        np.save(specfile[0:locfwhm] + '_trace',output_spec.trace)
    sigSpectrum = np.sqrt(output_spec.varSpectrum)
    #plt.clf()
    #plt.imshow(data)
    #plt.plot(output_spec.spectrum,'b')
    #plt.plot(output_spec.raw,'g')
    #plt.plot(output_spec.varSpectrum,'r')
    #plt.plot(sigSpectrum,'r')
    #plt.plot(output_spec.trace,'m')
    #plt.plot(output_spec.tracepos[0],output_spec.tracepos[1],'b^')
    #plt.plot(output_spec.background,'k')
    #plt.plot(output_spec.profile_map,'b')
    #plt.show()
    #plt.clf()
    #plt.plot(output_spec.extractionApertures)
    #plt.show()
    #plt.clf()
    #plt.plot(output_spec.background_map)
    #plt.show()
    
    #Save diagnostic info
    diagnostics[0:len(output_spec.tracepos[0]),4] = output_spec.tracepos[0]
    diagnostics[0:len(output_spec.tracepos[1]),5] = output_spec.tracepos[1]
    diagnostics[0:len(output_spec.trace),6] = output_spec.trace
    diagnostics[0:len(output_spec.backgroundcolumnpixels),7] = output_spec.backgroundcolumnpixels
    diagnostics[0:len(output_spec.backgroundcolumnvalues),8] = output_spec.backgroundcolumnvalues
    diagnostics[0:len(output_spec.backgroundfitpixels),9] = output_spec.backgroundfitpixels
    diagnostics[0:len(output_spec.backgroundfitvalues),10] = output_spec.backgroundfitvalues
    diagnostics[0:len(output_spec.backgroundfitpolynomial),11] = output_spec.backgroundfitpolynomial
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    endpoint = '.fits'
    with open('extraction_' + specfile[4:specfile.find(endpoint)] + '_' + now + '.txt','a') as handle:
        header = 'Columns are: 1) Measured FWHM, 2) pixel value of each FWHM, 3) fit to FWHM measurements, 4) all pixel values, 5) profile pixels, 6) profile position, 7) fit profile positions, 8) Pixels values of cut at pixel 1200, 9) Values along column 1200, 10) Pixels of fit to background, 11) Values used for fit, 12) polynomial fit to background'
        np.savetxt(handle,diagnostics,fmt='%f',header=header)
    
    #Compute the extracted signal to noise and save to header
    if 'blue' in specfile.lower():
        low_pixel, high_pixel = 1125., 1175.
    elif 'red' in specfile.lower():
        low_pixel, high_pixel = 825., 875.
    shortspec = output_spec.spectrum[:,0][low_pixel:high_pixel]
    shortsigma = sigSpectrum[:,0][low_pixel:high_pixel]
    shortpix = np.linspace(low_pixel,high_pixel,num=(high_pixel-low_pixel),endpoint=False)
    guessline = np.zeros(2)
    guessline[0] = (shortspec[-1] - shortspec[0]) / (shortpix[-1]-shortpix[0])
    guessline[1] = shortspec[0]-guessline[0]*shortpix[0]
    par, cov = curve_fit(line,shortpix,shortspec,guessline)
    bestline = line(shortpix,par[0],par[1])
    
    signal = np.mean(shortspec)
    noise = np.sqrt(np.sum((shortspec-bestline)**2.) / float(len(bestline))) #Noise from RMS
    #noise = np.mean(shortsigma) #noise from sigma spectrum
    sn_res_ele = signal/noise * np.sqrt(fwhm)
    print 'Signal to Noise is: ', sn_res_ele
    
    #plt.clf()
    #plt.plot(shortpix,shortspec,'b')
    #plt.plot(shortpix,bestline,'g')
    #plt.show()
    
    #exit()

    #Get the image header and add keywords
    header = st.readheader(specfile)
    header.set('BANDID1','Optimally Extracted Spectrum')
    header.set('BANDID2','Raw Extracted Spectrum')
    header.set('BANDID3','Mean Background')
    header.set('BANDID4','Sigma Spectrum')
    header.set('DISPCOR',0) #Dispersion axis of image
    fwhmsave = np.round(fwhm,decimals=4)
    snrsave = np.round(sn_res_ele,decimals=4)
    header.set('SPECFWHM',fwhmsave,'FWHM of spectrum in pixels') #FWHM of spectrum in pixels
    header.set('SNR',snrsave,'Signal to Noise per resolution element') 
    header.set('DATEEXTR',datetime.datetime.now().strftime("%Y-%m-%d"),'Date of Spectral Extraction')
    
    #Save the extracted image
    Ni = 4. #Number of extensions
    Nx = 1. #All 1D spectra
    Ny = len(output_spec.spectrum[:,0])
    spectrum = np.empty(shape = (Ni,Nx,Ny))
    spectrum[0,:,:] = output_spec.spectrum[:,0]
    spectrum[1,:,:] = output_spec.raw[:,0]
    spectrum[2,:,:] = output_spec.background
    spectrum[3,:,:] = sigSpectrum[:,0]
    
    #Save the extracted spectra with .ms.fits in the filename
    #Ask to overwrite if file already exists or provide new name
    loc = specfile.find('.fits')
    newname = specfile[0:loc] + '.ms.fits'
    clob = False

    mylist = [True for f in os.listdir('.') if f == newname]
    exists = bool(mylist)

    if exists:
        print 'File %s already exists.' % newname
        nextstep = raw_input('Do you want to overwrite or designate a new name (overwrite/new)? ')
        if nextstep == 'overwrite':
            clob = True
            exists = False
        elif nextstep == 'new':
            newname = raw_input('New file name: ')
            exists = False
        else:
            exists = False
    
    
    newim = fits.PrimaryHDU(data=spectrum,header=header)
    newim.writeto(newname,clobber=clob)
    print 'Wrote %s to file.' % newname
    
    #Save parameters to a file for future reference. 
    # specfile,date of extraction, extration_rad,background_radii,newname,newname2
    f = open('extraction_params.txt','a')
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    newinfo = specfile + '\t' + now + '\t' + str(extraction_rad) + '\t' + str(background_radii) + '\t' + newname
    f.write(newinfo + "\n")
    f.close()
    
    ###########################
    #Extract a lamp spectrum using the trace from above
    ##########################
    
    
    if lamp != 'no':
        lamplist = fits.open(lamp)
        lampdata = lamplist[0].data
        lampdata = lampdata[0,:,:]
        lampdata = np.transpose(lampdata)

        #extraction radius will be the FWHM the star
        #But since the Fe lamps are taken binned by 1 in spectral direction, we need to adjust the trace to match.
        #We do that by expanding out the trace to match the length of the Fe lamps, then interpolate and read off values for every pixel.
        bin2size = np.arange(1,len(output_spec.trace)+1)
        bin1size = np.arange(1,len(lampdata)+1)
        ratio = float(len(bin1size)) / float(len(bin2size))
        interpolates = InterpolatedUnivariateSpline(ratio*bin2size,output_spec.trace,k=1)
        newtrace = interpolates(bin1size)
        
        #Do the extraction here.
        lamp_radius = np.ceil(fwhm) #Make sure that extraction radius is a whole number, otherwise you'll get odd structures.
        lampspec = lampextract(lampdata,newtrace,lamp_radius)
        
        #Save the 1D lamp
        lampheader = st.readheader(lamp)
        lampheader.set('BANDID2','Raw Extracted Spectrum')
        lampheader.set('REF',newname,'Reference Star used for trace')
        lampheader.set('DATEEXTR',datetime.datetime.now().strftime("%Y-%m-%d"),'Date of Spectral Extraction')

        Ni = 1. #We are writing just 1 1D spectrum
        Ny = len(lampspec[:,0])
        lampspectrum = np.empty(shape = (Ni,Ny))
        lampspectrum[0,:] = lampspec[:,0]
        
        #Save the extracted spectra with .ms.fits in the filename
        #Ask to overwrite if file already exists or provide new name
        loc2 = lamp.find('.fits')
        loc3 = newname.find('_930')
        newname2 = lamp[0:loc2] + '_' + newname[5:loc3]  + '.ms.fits'
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


        lampim = fits.PrimaryHDU(data=lampspectrum,header=lampheader)
        lampim.writeto(newname2,clobber=clob)
        print 'Wrote %s to file.' % newname2
        
        #Save parameters to a file for future reference. 
        # specfile,date of extraction, extration_rad,background_radii,newname,newname2
        background_radii2 = [0,0] #We do not extract a background for the lamp
        f = open('extraction_params.txt','a')
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        newinfo2 = lamp + '\t' + now + '\t' + str(extraction_rad) + '\t' + str(background_radii2) + '\t' + newname2
        f.write(newinfo2 + "\n")
        f.close()
        
        #######################
        # End lamp extraction
        #######################

        
#Read in file from command line
if __name__ == '__main__':
    if len(sys.argv) == 3:
        script, specfile, tracefile = sys.argv
        trace_exist = True
        lamp = 'no'
        FWHMfile = None
    if len(sys.argv) == 2:
        script, specfile = sys.argv
        trace_exist = False
        tracefile = False
        FWHMfile = None
        lampcheck = raw_input('Do you want to extract a lamp too? (yes/no) ')
        if lampcheck == 'yes':
            lamp = raw_input('Enter lamp name: ')
        else:
            lamp = 'no'
    extract_now(specfile,lamp,FWHMfile,tracefile,trace_exist)







#To unpack these values, use the following
#arr = np.genfromtxt('extraction_params.txt',dtype=None,delimiter='\t')
#names, date, fwhm, back, newname = [], [],np.zeros(len(arr)),[],[]
#for m in np.arange(len(arr)):
#    names.append(arr[m][0])
#    date.append(arr[m][1])
#    fwhm[m] = arr[m][2]    
#    back.append(arr[m][3])
#    newname.append(arr[m][4])
    
