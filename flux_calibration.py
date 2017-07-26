"""

Written by JT Fuchs in July 2015
Based off pySALT redution routine specsens.py by S. Crawford
And reading the darned IRAF documentation

flux_calibration.py performs flux calibration on a 1D and wavelength-calibrated spectrum

To run file:
python flux_calibration.py spec_list --flux_list listflux.txt --stan_list liststandards.txt --extinct False
python flux_calibration.py GD1212.ms.fits --usemaster True


:INPUTS:
    spec_list: either single *.fits file or text file containing list of files to flux calibrate.

:OPTIONS:
    --flux_list: string, file containing standard star fluxes. These are typically m*.dat.

    --stan_list: string, file with list of 1D standard star spectra

    --usemaster: boolean, Option to use master response function instead of single star observation. Default: False

    --extinct: boolean, Option to extinction correct spectra. Default: True

:OUTPUTS: 
        flux calibrated files (_flux is added to the filename). User will be prompted if file will overwrite existing file.

        sensitivity_params.txt:  File is updated everytime spec_sens.py is run. Contains information used in the flux calibration. Columns are: input observed spectrum, date/time program was run, observed standard spectrum used for calibration, flux calibration file (m*dat), pixel regions excluded in fit, order of polynomial to flux standard, width in Angstroms used for rebinning, output spectrum filename

        sens_fits_DATE.txt: File for diagnostics. Columns are: wavelength, observed flux, polynomial fit, and residuals for each standard listed above. There are extra zeros at the bottom of some columns. 



Each list should have the names of the stars, with blue and red exposures next to each other.
The ordering of the standard star flux files should match the order of the standard star list.
Example:

liststandard:
wtfb.LTT3218_930_blue.ms.fits
wtfb.LTT3218_930_red.ms.fits
wnb.GD50_930_blue.ms.fits
wnb.GD50_930_red.ms.fits

listflux:
mltt3218.dat
mgd50.dat

spec_list:
wnb.WD0122p0030_930_blue.ms.fits
wnb.WD0122p0030_930_red.ms.fits
wnb.WD0235p069_930_blue.ms.fits
wnb.WD0235p069_930_red.ms.fits

#####

Counting variables are fruits and vegetables.


"""

import os
import sys
import numpy as np
#import pyfits as fits
import astropy.io.fits as fits
import spectools as st
import datetime
from glob import glob
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import argparse

#=============================================
#To help with command line interpretation
def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    if v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#=============================================
#These functions are to help with excluding regions from the sensitivity function
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def onclick(event):
    global ix,iy
    ix, iy = event.xdata,event.ydata
    global coords
    ax.axvline(x=ix,color='k',linewidth='3')
    fig.canvas.draw()
    coords.append((ix,iy))


#=============================================


def flux_calibrate_now(stdlist,fluxlist,speclist,extinct_correct=False,masterresp=False):
    if extinct_correct:
        extinctflag = 0
    else:
        extinctflag = -1
    if masterresp: #Use the master response function
        #Read in master response function and use that.
        cwd = os.getcwd()
        os.chdir('/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/standards/response_curves/')
        standards = sorted(glob('*resp*.npy'))

        master_response_blue_in = np.load(standards[0])
        master_response_blue_in_pol = np.poly1d(master_response_blue_in)
        master_response_blue_out = np.load(standards[1])
        master_response_blue_out_pol = np.poly1d(master_response_blue_out)
        master_response_red_in = np.load(standards[2])
        master_response_red_in_pol = np.poly1d(master_response_red_in)
        master_response_red_out = np.load(standards[3])
        master_response_red_out_pol = np.poly1d(master_response_red_out)

        os.chdir(cwd)

        airstd = np.ones([4])
        #airstd[0] = 1.1

        #For saving files correctly
        stdflux = np.array(['mmaster.dat'])
        #standards = np.array([masterlist])
        allexcluded = [[None] for i in range(len(standards))]
        orderused = np.zeros([len(standards)])
        size = 0.

        #Find shift for each night
        #For blue setup: use mean of 4530-4590
        #for red setup: use mean of 6090-6190
        try:
            flux_tonight_list = np.genfromtxt('response_curves.txt',dtype=str)
            print 'Found response_curves.txt file.'
            print flux_tonight_list
            if flux_tonight_list.size == 1:
                flux_tonight_list = np.array([flux_tonight_list])
            for x in flux_tonight_list:
                #print x
                if 'blue' in x.lower():
                    wave_tonight, sens_tonight = np.genfromtxt(x,unpack=True)
                    blue_low_index = np.min(np.where(wave_tonight > 4530.))
                    blue_high_index = np.min(np.where(wave_tonight > 4590.))
                    blue_mean_tonight = np.mean(sens_tonight[blue_low_index:blue_high_index])
                elif 'red' in x.lower():
                    wave_tonight, sens_tonight = np.genfromtxt(x,unpack=True)
                    red_low_index = np.min(np.where(wave_tonight > 6090.))
                    red_high_index = np.min(np.where(wave_tonight > 6190.))
                    red_mean_tonight = np.mean(sens_tonight[red_low_index:red_high_index])
        except:
            print 'No response_curves.txt file found.'
            blue_mean_tonight = None
            red_mean_tonight = None
            flux_tonight_list = ['None','None']
        
    else: #Use the standard star fluxes in the typical manner
        #Read in each standard star spectrum 
        standards = np.genfromtxt(stdlist,dtype=str)
        if standards.size ==1:
            standards = np.array([standards])
        stdflux = np.genfromtxt(fluxlist,dtype=str)
        if stdflux.size == 1:
            stdflux = np.array([stdflux]) #save stdflux explicitly as an array so you can index if only 1 element
        #Check that the files are set up correctly to avoid mixing standards.
        #This checks that the files in liststandard have similar characters to those in listflux and the correct order. But might break if flux file doesn't match. E.G. mcd32d9927.dat is often called CD-32_9927 in our system. 
        '''
        onion = 0
        for stanspec in standards:
            quickcheck = stdflux[onion//2].lower()[1:-4] in stanspec.lower()
            if not quickcheck:
                print 'Check your standard star and flux files. They are mixed up.'
                sys.exit()
            onion += 1
        '''
        orderused = np.zeros([len(standards)])
        senspolys = []
        airstd = np.zeros([len(standards)])
        allexcluded = [[None] for i in range(len(standards))]
        
        #Calculating the sensitivity function of each standard star
        cucumber = 0
        for stdspecfile in standards:
            print stdspecfile
            #Read in the observed spectrum of the standard star
            obs_spectra,airmass,exptime,dispersion = st.readspectrum(stdspecfile) #obs_spectra is an object containing opfarr,farr,sky,sigma,warr
            airstd[cucumber] = airmass
            #plt.clf()
            #plt.plot(obs_spectra.warr,obs_spectra.opfarr)
            #plt.show()
        
            #Do the extinction correction
            if extinct_correct:
                print 'Extinction correcting spectra.'
                plt.clf()
                plt.plot(obs_spectra.warr,obs_spectra.opfarr)
                obs_spectra.opfarr = st.extinction_correction(obs_spectra.warr,obs_spectra.opfarr,airmass)
                plt.plot(obs_spectra.warr,obs_spectra.opfarr)
                #plt.show()

            #Change to the standard star directory
            cwd = os.getcwd()
            os.chdir('/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/standards')

            #read in the standard file
            placeholder = cucumber // 2
            stdfile = stdflux[placeholder]
            std_spectra = st.readstandard(stdfile)
            os.chdir(cwd)
            #plt.clf()
            #plt.plot(std_spectra.warr,std_spectra.magarr,'.')
            #plt.show()
            #Only keep the part of the standard file that overlaps with observation.
            lowwv = np.where(std_spectra.warr >= np.min(obs_spectra.warr))
            lowwv = np.asarray(lowwv)
            highwv = np.where(std_spectra.warr <= np.max(obs_spectra.warr))
            highwv = np.asarray(highwv)
            index = np.intersect1d(lowwv,highwv)
        
            std_spectra.warr = std_spectra.warr[index]
            std_spectra.magarr = std_spectra.magarr[index]
            std_spectra.wbin = std_spectra.wbin[index]
        
            #Convert from AB mag to fnu, then to fwave (ergs/s/cm2/A)
            stdzp = 3.68e-20 #The absolute flux per unit frequency at an AB mag of zero
            std_spectra.magarr = st.magtoflux(std_spectra.magarr,stdzp)
            std_spectra.magarr = st.fnutofwave(std_spectra.warr, std_spectra.magarr)

            #plt.clf()
            #plt.plot(std_spectra.warr,std_spectra.magarr,'.')
            #plt.show()
            #np.savetxt('hz4_stan.txt',np.transpose([std_spectra.warr,std_spectra.magarr]))
            #exit()
        
            #We want to rebin the observed spectrum to match with the bins in the standard file. This makes summing up counts significantly easier.
            #Set the new binning here.
            print 'Starting to rebin: ',stdspecfile 
            low = np.rint(np.min(obs_spectra.warr)) #Rounds to nearest integer
            high = np.rint(np.max(obs_spectra.warr))
            size = 0.05 #size in Angstroms you want each bin
        
            num = (high - low) / size + 1. #number of bins. Must add one to get correct number.
            wavenew = np.linspace(low,high,num=num) #wavelength of each new bin

            #Now do the rebinning using Ian Crossfield's rebinning package
            binflux = st.resamplespec(wavenew,obs_spectra.warr,obs_spectra.opfarr,200.) #200 is the oversampling factor
            print 'Done rebinning. Now summing the spectrum into new bins to match', stdfile
            #plt.clf()
            #plt.plot(obs_spectra.warr,obs_spectra.opfarr)
            #plt.plot(wavenew,binflux)
            #plt.show()
        
            #Now sum the rebinned spectra into the same bins as the standard star file
            counts = st.sum_std(std_spectra.warr,std_spectra.wbin,wavenew,binflux)
            #plt.clf()
            #plt.plot(std_spectra.warr,std_spectra.magarr)
            #plt.plot(obs_spectra.warr,obs_spectra.opfarr,'b')
            #plt.plot(std_spectra.warr,counts,'g+')
            #plt.show()
            
            #Calculate the sensitivity function
            sens_function = st.sensfunc(counts,std_spectra.magarr,exptime,std_spectra.wbin,airmass)
            #plt.clf()
            #plt.plot(std_spectra.warr,sens_function)
            #plt.show()
            #sys.exit()
            #Fit a low order polynomial to this function so that it is smooth.
            #The sensitivity function is in units of 2.5 * log10[counts/sec/Ang / ergs/cm2/sec/Ang]
            #Choose regions to not include in fit, first by checking if a mask file exists, and if not the prompt for user interaction.
            if 'blue' in stdspecfile.lower():
                std_mask = stdfile[0:-4] + '_blue_maskasdf.dat'
            if 'red' in stdspecfile.lower():
                std_mask = stdfile[0:-4] + '_red_maskasdf.dat'
            std_mask2 = glob(std_mask)
            if len(std_mask2) == 1.:
                print 'Found mask file.\n'
                mask = np.ones(len(std_spectra.warr))
                excluded_wave = np.genfromtxt(std_mask) #Read in wavelengths to exclude
                #print excluded_wave
                #print type(excluded_wave)
                #Find index of each wavelength
                excluded = []
                for x in excluded_wave:
                    #print x
                    #print np.where(std_spectra.warr == find_nearest(std_spectra.warr,x))
                    pix_val = np.where(std_spectra.warr == find_nearest(std_spectra.warr,x))
                    excluded.append(pix_val[0][0])
                #print excluded
                lettuce = 0
                while lettuce < len(excluded):
                    mask[excluded[lettuce]:excluded[lettuce+1]+1] = 0
                    lettuce += 2
                excluded =  np.array(excluded).tolist()
                allexcluded[cucumber] = excluded
                indices = np.where(mask !=0.)
                lambdasfit = std_spectra.warr[indices]
                fluxesfit = sens_function[indices]
            else:
                print 'No mask found. User interaction required.\n'
                
                global ax, fig, coords
                coords = []
                plt.clf()
                fig = plt.figure(1)
                ax = fig.add_subplot(111)
                ax.plot(std_spectra.warr,sens_function)
                cid = fig.canvas.mpl_connect('button_press_event',onclick)
                print 'Please click on both sides of regions you want to exclude. Then close the plot.'
                plt.title('Click both sides of regions you want to exclude. Then close the plot.')
                plt.show(1)
        
        
                #Mask our the regions you don't want to fit
                #We need make sure left to right clicking and right to left clicking both work.
                mask = np.ones(len(std_spectra.warr))
                excluded = np.zeros(len(coords))
                lettuce = 0
                if len(coords) > 0:
                    while lettuce < len(coords):
                        x1 = np.where(std_spectra.warr == (find_nearest(std_spectra.warr,coords[lettuce][0])))
                        excluded[lettuce] = np.asarray(x1)
                        lettuce += 1
                        x2 = np.where(std_spectra.warr == (find_nearest(std_spectra.warr,coords[lettuce][0])))
                        if x2 < x1:
                            x1,x2 = x2,x1
                        mask[x1[0][0]:x2[0][0]+1] = 0 #have to add 1 here to the second index so that we exclude through that index. Most important for when we need to exclude the last point of the array.
                        excluded[lettuce-1] = np.asarray(x1)
                        excluded[lettuce] = np.asarray(x2)
                        lettuce += 1

                excluded =  np.array(excluded).tolist()
                allexcluded[cucumber] = excluded
                indices = np.where(mask !=0.)
                lambdasfit = std_spectra.warr[indices]
                fluxesfit = sens_function[indices]
        
                #Save masked wavelengths
                lambdasnotfit = std_spectra.warr[excluded]
                #print lambdasnotfit
                #print stdfile
                if 'blue' in stdspecfile.lower():
                    std_mask_name = stdfile[0:-4] + '_blue_mask.dat'
                if 'red' in stdspecfile.lower():
                    std_mask_name = stdfile[0:-4] + '_red_mask.dat'
                np.savetxt(std_mask_name,np.transpose(np.array(lambdasnotfit)))
                #exit()

            ##Move back to directory with observed spectra
            #os.chdir(cwd) 
        
        
            #Make sure they are finite
            ind1 = np.isfinite(lambdasfit) & np.isfinite(fluxesfit)
            lambdasfit = lambdasfit[ind1]
            fluxesfit = fluxesfit[ind1]

            print 'Fitting the sensitivity funtion now.'
            order = 4
            repeat = 'yes'
            while repeat == 'yes':
                p = np.polyfit(lambdasfit,fluxesfit,order)
                f = np.poly1d(p)
                smooth_sens = f(lambdasfit)
                residual = fluxesfit - smooth_sens
                plt.close()
                plt.ion()
                g, (ax1,ax2) = plt.subplots(2,sharex=True)
                ax1.plot(lambdasfit,fluxesfit,'b+')
                ax1.plot(lambdasfit,smooth_sens,'r',linewidth=2.0)
                ax1.set_ylabel('Sensitivity Function')
                ax2.plot(lambdasfit,residual,'k+')
                ax2.set_ylabel('Residuals')
                ax1.set_title('Current polynomial order: %s' % order)
                g.subplots_adjust(hspace=0)
                plt.setp([a.get_xticklabels() for a in g.axes[:-1]],visible=False)
                plt.show()
                plt.ioff()
                #Save this sensitivity curve
                '''
                try:
                    temp_file = fits.open(stdspecfile)
                    ADCstat = temp_file[0].header['ADCSTAT']
                except:
                    ADCstat = 'none'
                    pass
                if 'blue' in stdspecfile.lower():
                    resp_name = 'senscurve_' + stdfile[1:-4] + '_' + str(np.round(airstd[cucumber],decimals=3))  + '_' + ADCstat  + '_' + cwd[60:70] + '_blue.txt'
                elif 'red' in stdspecfile.lower():
                    resp_name = 'senscurve_' + stdfile[1:-4] + '_' + str(np.round(airstd[cucumber],decimals=3))  + '_' + ADCstat  + '_' + cwd[60:70] + '_red.txt'
                print resp_name
                #exit()
                np.savetxt(resp_name,np.transpose([lambdasfit,fluxesfit]))
                '''
                repeat = raw_input('Do you want to try again (yes/no)? ')
                if repeat == 'yes':
                    order = raw_input('New order for polynomial: ')

            orderused[cucumber] = order
            senspolys.append(f)

            #Save arrays for diagnostic plots
            if cucumber == 0:
                bigarray = np.zeros([len(lambdasfit),4.*len(standards)])
                artichoke = 0
            bigarray[0:len(lambdasfit),artichoke] = lambdasfit
            bigarray[0:len(fluxesfit),artichoke+1] = fluxesfit
            bigarray[0:len(smooth_sens),artichoke+2] = smooth_sens
            bigarray[0:len(residual),artichoke+3] = residual
            artichoke += 4
                   
            cucumber += 1

        #Save fit and residuals into text file for diagnostic plotting later.
        #Need to save lambdasfit,fluxesfit,smooth_sens,residual for each standard
        #List of standards is found as standards
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        with open('sens_fits_' + now + '.txt','a') as handle:
            header = str(standards) + '\n Set of four columns correspond to wavelength, observed flux, polynomial fit, \n and residuals for each standard listed above. \n You will probably need to strip zeros from the bottoms of some columns.'
            np.savetxt(handle,bigarray,fmt='%f',header = header)    

    #Outline for next steps:
    #Read in both red and blue files
    #compute airmass and compare to airstd
    #choose best standard and flux calibrate both blue and red
    #save files and write to sensitivity_params.txt
    
    if speclist[-4:] == 'fits':
        specfile = np.array([speclist])
    else:
        specfile = np.genfromtxt(speclist,dtype=str)
        if specfile.size ==1:
            specfile = np.array([specfile])
    
    length = len(specfile)
    airwd = np.zeros([length])
    bean = 0
    #if length == 1:
    #    redfile = False
    #else:
    #    redfile = True

    avocado = 0
    while avocado < length:
        #Read in the blue and red spectra we want to flux calibrate. Save the airmass
        WD_spectra1,airmass1,exptime1,dispersion1 = st.readspectrum(specfile[avocado])
        if (len(specfile) >= 1) and (avocado+1 < length):
            if 'red' in specfile[avocado+1]:
                redfile = True
            else:
                redfile = False
        else:
            redfile = False
        if redfile:
            WD_spectra2,airmass2,exptime2,dispersion2 = st.readspectrum(specfile[avocado+1])
                
        #Extinction correct WD
        if extinct_correct:
            print 'Extinction correcting spectra.'
            #plt.clf()
            #plt.plot(WD_spectra1.warr,WD_spectra1.opfarr)
            WD_spectra1.opfarr = st.extinction_correction(WD_spectra1.warr,WD_spectra1.opfarr,airmass1)
            WD_spectra1.farr = st.extinction_correction(WD_spectra1.warr,WD_spectra1.farr,airmass1)
            WD_spectra1.sky = st.extinction_correction(WD_spectra1.warr,WD_spectra1.sky,airmass1)
            WD_spectra1.sigma = st.extinction_correction(WD_spectra1.warr,WD_spectra1.sigma,airmass1)
            #plt.plot(WD_spectra1.warr,WD_spectra1.opfarr)
            #plt.show()

            if redfile:
                #plt.clf()
                #plt.plot(WD_spectra2.warr,WD_spectra2.opfarr)
                WD_spectra2.opfarr = st.extinction_correction(WD_spectra2.warr,WD_spectra2.opfarr,airmass2)
                WD_spectra2.farr = st.extinction_correction(WD_spectra2.warr,WD_spectra2.farr,airmass2)
                WD_spectra2.sky = st.extinction_correction(WD_spectra2.warr,WD_spectra2.sky,airmass2)
                WD_spectra2.sigma = st.extinction_correction(WD_spectra2.warr,WD_spectra2.sigma,airmass2)


                #zaplt.plot(WD_spectra2.warr,WD_spectra2.opfarr)
                #plt.show()
        airwd[avocado] = airmass1
        if redfile:
            airwd[avocado+1] = airmass2
        #Compare the airmasses to determine the best standard star
        tomato = 0
        while tomato < len(airstd):
            if redfile:
                diff = np.absolute(np.mean([airwd[avocado],airwd[avocado+1]]) - np.mean([airstd[tomato],airstd[tomato+1]]))
            else:
                diff = np.absolute(airwd[avocado] - airstd[tomato])
            if tomato == 0:
                difference = diff
                choice = tomato
            if diff < difference:
                difference = diff
                choice = tomato
            tomato += 2
    
        #To get the flux calibration, perform the following
        #Flux = counts / (Exptime * dispersion * 10**(sens/2.5))
        #Get the sensitivity function at the correct wavelength spacing
        if masterresp:
            header_temp = st.readheader(specfile[avocado])
            ADCstatus = header_temp['ADCSTAT']
            if ADCstatus == 'IN':
                sens_wave1_unscale = master_response_blue_in_pol(WD_spectra1.warr)
                blue_low_index = np.min(np.where(WD_spectra1.warr > 4530.))
                blue_high_index = np.min(np.where(WD_spectra1.warr > 4590.))
                blue_mean_stan = np.mean(sens_wave1_unscale[blue_low_index:blue_high_index])
                if blue_mean_tonight == None:
                    sens_wave1 = sens_wave1_unscale
                else:
                    sens_wave1 = sens_wave1_unscale + (blue_mean_tonight - blue_mean_stan)
                choice = 0
            else:
                sens_wave1_unscale = master_response_blue_out_pol(WD_spectra1.warr)
                blue_low_index = np.min(np.where(WD_spectra1.warr > 4530.))
                blue_high_index = np.min(np.where(WD_spectra1.warr > 4590.))
                blue_mean_stan = np.mean(sens_wave1_unscale[blue_low_index:blue_high_index])
                if blue_mean_tonight == None:
                    sens_wave1 = sens_wave1_unscale
                else:
                    sens_wave1 = sens_wave1_unscale + (blue_mean_tonight - blue_mean_stan)
                choice = 1
            if redfile:
                header_temp = st.readheader(specfile[avocado+1])
                ADCstatus = header_temp['ADCSTAT']
                if ADCstatus == 'IN':
                    sens_wave2_unscale = master_response_red_in_pol(WD_spectra2.warr)
                    red_low_index = np.min(np.where(WD_spectra2.warr > 6090.))
                    red_high_index = np.min(np.where(WD_spectra2.warr > 6190.))
                    red_mean_stan = np.mean(sens_wave2_unscale[red_low_index:red_high_index])
                    if red_mean_tonight == None:
                        sens_wave2 = sens_wave2_unscale
                    else:
                        sens_wave2 = sens_wave2_unscale + (red_mean_tonight - red_mean_stan)
                    choice2 = 2
                else:
                    sens_wave2_unscale = master_response_red_out_pol(WD_spectra2.warr)
                    red_low_index = np.min(np.where(WD_spectra2.warr > 6090.))
                    red_high_index = np.min(np.where(WD_spectra2.warr > 6190.))
                    red_mean_stan = np.mean(sens_wave2_unscale[red_low_index:red_high_index])
                    if red_mean_tonight == None:
                        sens_wave2 = sens_wave2_unscale
                    else:
                        sens_wave2 = sens_wave2_unscale + (red_mean_tonight - red_mean_stan)
                    choice2 = 3
        else:
            sens_wave1 = senspolys[choice](WD_spectra1.warr)
            if redfile:
                sens_wave2 = senspolys[choice+1](WD_spectra2.warr)

        #Perform the flux calibration. We do this on the optimal extraction, non-variance weighted aperture, the sky spectrum, and the sigma spectrum.
        print 'Doing the final flux calibration.'
        #np.savetxt('response_g60-54_extinction_2016-03-17.txt',np.transpose([WD_spectra1.warr,(exptime1 * dispersion1 * 10.**(sens_wave1/2.5))]))#,WD_spectra2.warr,(exptime2 * dispersion2 * 10.**(sens_wave2/2.5))]))
        #exit()
        star_opflux1 = st.cal_spec(WD_spectra1.opfarr,sens_wave1,exptime1,dispersion1)
        star_flux1 = st.cal_spec(WD_spectra1.farr,sens_wave1,exptime1,dispersion1)
        sky_flux1 = st.cal_spec(WD_spectra1.sky,sens_wave1,exptime1,dispersion1)
        sigma_flux1 = st.cal_spec(WD_spectra1.sigma,sens_wave1,exptime1,dispersion1)

        if redfile:
            star_opflux2 = st.cal_spec(WD_spectra2.opfarr,sens_wave2,exptime2,dispersion2)
            star_flux2 = st.cal_spec(WD_spectra2.farr,sens_wave2,exptime2,dispersion2)
            sky_flux2 = st.cal_spec(WD_spectra2.sky,sens_wave2,exptime2,dispersion2)
            sigma_flux2 = st.cal_spec(WD_spectra2.sigma,sens_wave2,exptime2,dispersion2)
        
        #plt.clf()
        #plt.plot(WD_spectra.warr,star_opflux)
        #plt.show()

        #Save final spectra if using master response
        if masterresp:
            if avocado == 0:
                diagnostic_array = np.zeros([len(WD_spectra1.warr),2*length])
            diagnostic_array[0:len(WD_spectra1.warr),bean] = WD_spectra1.warr
            bean += 1
            diagnostic_array[0:len(star_opflux1),bean] = star_opflux1
            bean += 1
            if redfile:
                diagnostic_array[0:len(WD_spectra2.warr),bean] = WD_spectra2.warr
                bean += 1
                diagnostic_array[0:len(star_opflux2),bean] = star_opflux2
                bean += 1
        #if avocado == (length -1 ) or (redfile == True and avocado == (length-2)):
        #    print 'Saveing diagnostic file.'
        #    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        #    with open('flux_fits_' + now + '.txt','a') as handle:
        #        header = str(specfile) + '\n Each star is formatted as wavelength, flux'
        #        np.savetxt(handle,diagnostic_array,fmt='%.10e',header=header)


        print 'Saving the final spectrum.'
        
        #Save the flux-calibrated spectrum and update the header
        header1 = st.readheader(specfile[avocado])
        header1.set('EX-FLAG',extinctflag) #Extiction correction? 0=yes, -1=no
        header1.set('CA-FLAG',0) #Calibrated to flux scale? 0=yes, -1=no
        header1.set('BUNIT','erg/cm2/s/A') #physical units of the array value
        header1.set('STANDARD',str(standards[choice]),'Flux standard used') #flux standard used for flux-calibration
        if masterresp:
            header1.set('STDOFF',str(flux_tonight_list[0]),'Night offset used')
        
        if redfile:
            header2 = st.readheader(specfile[avocado+1])
            header2.set('EX-FLAG',extinctflag) #Extiction correction? 0=yes, -1=no
            header2.set('CA-FLAG',0) #Calibrated to flux scale? 0=yes, -1=no
            header2.set('BUNIT','erg/cm2/s/A') #physical units of the array value
            if masterresp:
                header2.set('STANDARD',str(standards[choice2]),'Flux standard used') #flux standard used for flux-calibration
                header1.set('STDOFF',str(flux_tonight_list[1]),'Night offset used')
            else:
                header2.set('STANDARD',str(standards[choice+1]),'Flux standard used') #flux standard used for flux-calibration

        #Set up size of new fits image
        Ni = 4. #Number of extensions
        Nx1 = len(star_flux1)
        if redfile:
            Nx2 = len(star_flux2)
        Ny = 1. #All 1D spectra

        data1 = np.empty(shape = (Ni,Ny,Nx1))
        data1[0,:,:] = star_opflux1
        data1[1,:,:] = star_flux1
        data1[2,:,:] = sky_flux1
        data1[3,:,:] = sigma_flux1
    
        if redfile:
            data2 = np.empty(shape = (Ni,Ny,Nx2))
            data2[0,:,:] = star_opflux2
            data2[1,:,:] = star_flux2
            data2[2,:,:] = sky_flux2
            data2[3,:,:] = sigma_flux2

        #Add '_flux' to the end of the filename
        loc1 = specfile[avocado].find('.ms.fits')
        if masterresp:
            newname1 = specfile[avocado][0:loc1] + '_flux_' + stdflux[0][1:-4]  + '.ms.fits'
        else:
            newname1 = specfile[avocado][0:loc1] + '_flux_' + stdflux[choice//2][1:-4]  + '.ms.fits'
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
        print 'Saving: ', newname1
        newim1 = fits.PrimaryHDU(data=data1,header=header1)
        newim1.writeto(newname1,clobber=clob)

        if redfile:
            loc2 = specfile[avocado+1].find('.ms.fits')
            if masterresp:
                newname2 = specfile[avocado+1][0:loc2] + '_flux_' + stdflux[0][1:-4] + '.ms.fits'
            else:
                newname2 = specfile[avocado+1][0:loc2] + '_flux_' + stdflux[choice//2][1:-4] + '.ms.fits'
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

            newim2 = fits.PrimaryHDU(data=data2,header=header2)
            newim2.writeto(newname2,clobber=clob)
            print 'Saving: ', newname2

        #Finally, save all the used parameters into a file for future reference.
        # specfile,current date, stdspecfile,stdfile,order,size,newname
        f = open('sensitivity_params.txt','a')
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        if masterresp:
            newinfo1 = specfile[avocado] + '\t' + now + '\t' + standards[choice] + '\t' + stdflux[0] + '\t' + str(allexcluded[choice]) + '\t' + str(orderused[choice]) + '\t' + str(size) + '\t' + newname1
        else:
            newinfo1 = specfile[avocado] + '\t' + now + '\t' + standards[choice] + '\t' + stdflux[choice//2] + '\t' + str(allexcluded[choice]) + '\t' + str(orderused[choice]) + '\t' + str(size) + '\t' + newname1
        if redfile:
            if masterresp:
                newinfo2 = specfile[avocado+1] + '\t' + now + '\t' + standards[choice2] + '\t' + stdflux[0] + '\t' + str(allexcluded[choice+1]) + '\t' + str(orderused[choice+1]) + '\t' + str(size) + '\t' + newname2
            else:
                newinfo2 = specfile[avocado+1] + '\t' + now + '\t' + standards[choice+1] + '\t' + stdflux[choice//2] + '\t' + str(allexcluded[choice+1]) + '\t' + str(orderused[choice+1]) + '\t' + str(size) + '\t' + newname2
            f.write(newinfo1 + "\n" + newinfo2 + "\n")
        else:
            f.write(newinfo1 + "\n")
        f.close()

        if redfile:
            avocado += 2
        else:
            avocado += 1

    print 'Done flux calibrating the spectra.'





#Run from command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec_list')
    parser.add_argument('--flux_list',default=None)
    parser.add_argument('--stan_list',default=None)
    parser.add_argument('--usemaster',type=str2bool,nargs='?',const=False,default=False,help='Activate nice mode.')
    parser.add_argument('--extinct',type=str2bool,nargs='?',const=True,default=True,help='Activate nice mode.')
    args = parser.parse_args()
    #print args.stand_list
    flux_calibrate_now(args.stan_list,args.flux_list,args.spec_list,extinct_correct=args.extinct,masterresp=args.usemaster)
