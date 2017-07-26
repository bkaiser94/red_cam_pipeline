# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 20:48:10 2015

@author: jmeza
"""


# ===========================================================================
# Packages ==================================================================
# ===========================================================================

import numpy as np
#import pyfits as fits
import astropy.io.fits as fits
import mpfit
import os
import datetime
import matplotlib.pyplot as plt
import cosmics
from glob import glob
from astropy.convolution import convolve, convolve_fft, Box2DKernel

# ===========================================================================
# Lesser Functions Used by Main Functions ===================================
# ===========================================================================

def init():
    global diagnostic
    diagnostic = np.zeros([2071,28])

def save_diagnostic():
    global now
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    header = 'Reduction done on ' + now + '\n Zeros in a whole column typically mean blue/red setup not included. Will need to strip zeros from end. \n Columns are: 0) average from bias, 1) average from scaled bias, 2) standard deviation of bias \n 3) Blue flat field average, 4) Blue flat field standard deviation, 5) Blue flat field scaled average, 6) Blue flat field scaled standard deviation \n 7) Red flat field average, 8) Red flat field standard deviation, 9) Red flat field scaled average, 10) Red flat field scaled standard deviation \n 11)Blue Pixels for polynomial fit over littrow ghost, 12) Blue values for polynomial fit over littrow ghost, 13)Polynomial fit mask out littrow ghost  \n 14) Cut along row 100 for blue flat, 15) Cut along row 100 for red flat, 16) junk zeros \n 17) Range of pixels used to find the littrow ghost, 18) Range of values used to find the littrow ghost, 19) Range of pixels used to fit the littrow ghost, 20) Gaussian fit to small number of pixels to find the center of the littrow ghost, 21) The upper and lower edges of the masked region saved to the header \n  22) Combined blue flat pixel values, 23) Combined blue flat values, 24) Polynomial fit to combined blue flat \n 25) Combined red flat pixel values, 26) Combined red flat values, 27) Polynomial fit to combined red flat '
    with open('reduction_' + now + '.txt','a') as handle:
        np.savetxt(handle,diagnostic,fmt='%f',header=header)

def gauss(x,p): #single gaussian
    return p[0] +  p[1]*np.exp(-(((x-p[2])/(np.sqrt(2)*p[3])))**2.)

def fitgauss(p,fjac=None,x=None,y=None,err=None):
    #Parameter values are passed in p
    #fjac = None just means partial derivatives will not be computed
    model = gauss(x,p)
    status = 0
    return([status,(y-model)/err])

def gaussslope(x,p): #single gaussian
    return p[0] +  p[1]*x + p[2]*np.exp(-(((x-p[3])/(np.sqrt(2)*p[4])))**2.)

def fitgaussslope(p,fjac=None,x=None,y=None,err=None):
    #Parameter values are passed in p
    #fjac = None just means partial derivatives will not be computed
    model = gaussslope(x,p)
    status = 0
    return([status,(y-model)/err])

def adcstat(specname):
    hdu = fits.getheader(specname)
    adc_stat = hdu['ADCSTAT']
    print 'ADC status during observations was ', adc_stat
    return adc_stat


def checkspec(listcheck):
    #Calculates the FWHM and profile postion for two points on each spectrum
    #If these values deviate by more than given values, prints warning.
    #Saves all values in a text file.
    listcheck = np.genfromtxt(listcheck,dtype=str)
    print '\n \n Now checking FWHM and center of spectral profile for stability.'
    #Max values acceptable
    maxcendev = 2. #Deviation from center of gaussian
    maxfwhmdev = 0.5 #deviation of fwhm

    fwhm1 = np.zeros(len(listcheck))
    fwhm2 = np.zeros(len(listcheck))
    center1 = np.zeros(len(listcheck))
    center2 = np.zeros(len(listcheck))
    peak1 = np.zeros(len(listcheck))
    peak2 = np.zeros(len(listcheck))
    global now
    newfilename = 'FWHM_records_' + now + '.txt'
    mylist = [True for f in os.listdir('.') if f == newfilename]
    exists = bool(mylist)
    f = open('FWHM_records_' + now + '.txt','a')
    if not exists:
        header = '#Columns: filename, Column of 2D image checked, FWHM of Gaussian fit to that column, Center position of Gaussian fit to that column, Peak of Gaussian fit to that column, Second column checked, FWHM of second column, Center position of second column, peak of gaussian fit to that column.'
        f.write(header+ "\n")
    n = 0.
    for specfile in listcheck:
        datalist = fits.open(specfile)
        data = datalist[0].data
        data = data[0,:,:]
        data = np.transpose(data)

        #Fit a column of the 2D image to determine the center and FWHM 
        #forfit1 = data[550,2:] #column 550 and 1750 are good for both setups
        forfit1 = np.mean(np.array([data[548,:],data[549,:],data[550,:],data[551,:],data[552,:]]),axis=0)
        forfit1 = forfit1[2:] #We have not trimmed yet, so get rid of the bottom rows
        guess1 = np.zeros(4)
        guess1[0] = np.mean(forfit1)
        guess1[1] = np.amax(forfit1)
        guess1[2] = np.argmax(forfit1)
        guess1[3] = 3.
        error_fit1 = np.ones(len(forfit1))
        xes1 = np.linspace(2,len(forfit1)-1,num=len(forfit1))
        fa1 = {'x':xes1,'y':forfit1,'err':error_fit1}
        fitparams1 = mpfit.mpfit(fitgauss,guess1,functkw=fa1,quiet=True)
        fwhm1[n] = 2.*np.sqrt(2.*np.log(2.))*fitparams1.params[3]
        center1[n] = fitparams1.params[2]
        peak1[n] = fitparams1.params[1]
        #print np.round(fwhm1[n],decimals=1),np.round(center1[n],decimals=1),np.round(peak1[n],decimals=1)
        #plt.clf()
        #plt.plot(xes1,forfit1)
        #plt.plot(xes1,gauss(xes1,fitparams1.params))
        #plt.show()

        #forfit2 = data[1750,2:] #column 550 and 1750 are good for both setups
        forfit2 = np.mean(np.array([data[1748,:],data[1749,:],data[1750,:],data[1751,:],data[1752,:]]),axis=0)
        forfit2 = forfit2[2:]
        guess2 = np.zeros(4)
        guess2[0] = np.mean(forfit2)
        guess2[1] = np.amax(forfit2)
        guess2[2] = np.argmax(forfit2)
        guess2[3] = 3.

        error_fit2 = np.ones(len(forfit2))
        xes2 = np.linspace(2,len(forfit2)-1,num=len(forfit2))
        fa2 = {'x':xes2,'y':forfit2,'err':error_fit2}
        fitparams2 = mpfit.mpfit(fitgauss,guess2,functkw=fa2,quiet=True)

        fwhm2[n] = 2.*np.sqrt(2.*np.log(2.))*fitparams2.params[3]
        center2[n] = fitparams2.params[2]
        peak2[n] = fitparams2.params[1]
        #print np.round(fwhm2[n],decimals=1),np.round(center2[n],decimals=1),np.round(peak2[n],decimals=1)
        #plt.clf()
        #plt.plot(xes2,forfit2)
        #plt.plot(xes2,gauss(xes2,fitparams2.params))
        #plt.show()

        info = specfile + '\t' + '550' + '\t' + str(np.round(fwhm1[n],decimals=2)) + '\t' + str(np.round(center1[n],decimals=2)) + '\t' + str(np.round(peak1[n],decimals=2))  + '\t' + '1750' + '\t' + str(np.round(fwhm2[n],decimals=2)) + '\t' + str(np.round(center2[n],decimals=2)) + '\t' + str(np.round(peak2[n],decimals=2))
        f.write(info+ "\n")

        n += 1
    f.close()

    #Check if values deviate by more than a certain amount

    if (np.max(fwhm1) - np.min(fwhm1)) > maxfwhmdev:
        print 'WARNING!!! Left FWHM varying significantly. Values are %s' % fwhm1
    elif (np.max(fwhm2) - np.min(fwhm2)) > maxfwhmdev:
        print 'WARNING!!! Right FWHM varying significantly. Values are %s' % fwhm2
    else:
        print 'FWHM is stable.'

    if (np.max(center1) - np.min(center1)) > maxcendev:
        print 'WARNING!!! Left profile center varying significantly. Values are %s' % center1
    elif (np.max(center2) - np.min(center2)) > maxcendev:
        print 'WARNING!!! Right profile center varying significantly. Values are %s' % center2
    else:
        print 'Profile center is stable.'
        
# ============================================================================

def Read_List( lst ):
    # This function reads a list of images and decomposes them into a python
    # list of image names. 
    list_file = open(lst,'r')
    im_list = list_file.read()
    list_file.close()
    im_list = im_list.split()
    return im_list
    
def List_Combe(img_list):
    # This is meant to combe trough list names to identify seperate sublist of
    # stars / flats / standars 
    sub_lists= [] # list of sub_list of images 
    sl= [] # sub_list of images
    sl.append(img_list[0]) # place first image in sublist
    i= 0; # image counter  
    #img_list[0][0] is a string, so need to check that agaisnt strings. Use a shorter cutpoint if these are RAW images. This will help eliminate problems with short filenames.
    if (img_list[0][0] == '0') or (img_list[0][0] == '1') or (img_list[0][0] == '2'):
        cutpoint = 5
    else:
        cutpoint = 10
    while i < len(img_list)-1: # run trough all images 
        if img_list[i+1].__contains__(img_list[i][cutpoint:]) == True: #Old = 4
            sl.append(img_list[i+1]) # place it in the sub_list 
        else:
            # if the images dont match: 
            sub_lists.append(sl) # write the sublist to the list of sublist 
            sl= [] # clear the sublist
            sl.append(img_list[i+1]) # append the image to the new list 
        i= i+1 # image counter
    sub_lists.append(sl) # append the last sublist to the list of sublist 
    return sub_lists # return the list of sub_list of images
    
def check_file_exist(name):
    # This function is to be called before wirting a file. 
    # This function checks if the file name already exist.
    # If it does it appends a number to the begining until 
    # the name no longer matches the files in the directory. 
    
    # List of files in directory
    listDirFiles = [f for f in os.listdir('.') if f.endswith('.fits')]
    # If "name" is in the derectory append a number i until it doent match 
    # If name is not in directory then we simply return name
    if listDirFiles.__contains__(name):
        i= 2
        while listDirFiles.__contains__(name):
            name= str(i) + name
            i= i+1
    return name

def Fix_Header( header ):
    # This function deletes the header cards that contain the badly coded 
    # degree symbol '\xb0'. If they are not deleted pyfits won't write the
    # headers. 
    bad_key = ['param0', 'param61', 'param62', 'param63']
    for p in bad_key:
        if p in header: 
            bad_str = header.comments[p]
            if '\xb0' in bad_str:
                del header[p]      

def decimal_dec(hdu_str):
    # Read header strings in "hh:mm:ss" or "dd:mm:ss" fromat 
    # and outputs the value as a decimal. 
    val_list = [float(n) for n in hdu_str.split(':')]
    #if val_list[0] < 0 :
    if str(val_list[0])[0] == '-':
        sng = -1
        val_list[0] = sng*val_list[0]
    else:
        sng = 1
    val_deci =  sng*(val_list[0]+((val_list[1]+(val_list[2]/60.0))/60.0))
    return val_deci

def decimal_ra(hdu_str):
    # Read header strings in "hh:mm:ss" or "dd:mm:ss" fromat 
    # and outputs the value as a decimal. 
    val_list = [float(n) for n in hdu_str.split(':')]
    if val_list[0] < 0 :
        sng = -1.
        val_list[0] = sng*val_list[0]
    else:
        sng = 1.
    val_deci =  15.*sng*(val_list[0]+((val_list[1]+(val_list[2]/60.0))/60.0))

    return val_deci

def SigClip(data_set, lo_sig, hi_sig):
    # Sigma Cliping Function  #
    # Input is set of counts for a particular pixel, 
    # along with low and high sigma factors. 
    # Output is a list containg only the data that is with the sigma factors.
    # Only a single rejection iteration is made. 
    Avg = np.median(data_set)
    #remove_max = np.delete(data_set,data_set.argmax())
    #St_Dev = np.std(remove_max)
    St_Dev = np.std(data_set)
    min_val = Avg-lo_sig*St_Dev
    max_val = Avg+hi_sig*St_Dev
    cliped_data = []
    #masked_data = []
    for val in data_set:
        if min_val <= val <= max_val:
            cliped_data.append( val )
        #else:
        #    masked_data.append( val)
    return cliped_data#, masked_data
        
def RaDec2AltAz(ra, dec, lat, lst ):
    # Input: RA in decimal hours; DEC in decimal deg; 
    # LAT in decimal deg; LST in decimal hours; 
    # Output: ALT, AZ, HA in decimal deg. 
     
    # Compute Hour Angle
        ha = lst-ra # hour angle in deg
        if ha < 0 :
            ha = ha+360.
        if ha > 360:
            ha = ha-360.
    # Convert Qunataties to Radians 
        ra = ra*(np.pi/180.0) 
        dec = dec*(np.pi/180.0) 
        lat = lat*(np.pi/180.0) 
        ha = ha*(np.pi/180.0)
    # Calculate Altitiude 
        a =  np.sin(dec)*np.sin(lat)
        b = np.cos(dec)*np.cos(lat)*np.cos(ha)
        alt = np.arcsin( a+b ) # altitude in radians 
    # Calculate Azimuth 
        a = np.sin(dec)-np.sin(lat)*np.sin(alt)
        b = np.cos(lat)*np.cos(alt)
        az = np.arccos( a/b ) # azumuth in radians
        if np.sin(ha) > 0:
            az = (2.*np.pi) - az 
    # Convert Alt, Az, and Ha to decimal deg
        alt = alt*(180.0/np.pi)
        az = az*(180.0/np.pi)
        ha = ha*(180.0/np.pi)
        return alt, az, ha
        
def AirMass(alt, scale):
    # Calculates instantaneus airmass to be called by SetAirMass() #
    # This comes from Allen, Astrophysical Quantities, page 125.
    # See also http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?setairmass
    # Input: 
    #   scale = atmospheric scale factor (defalut 750)
    #   alt = altitude of star in degrees.  
    # Output: 
    #   AM = airmass from given altitude and scale factor
    x = scale*np.sin(np.pi*alt/180.)
    AM = np.sqrt( x**2. + 2.*scale + 1. ) - x
    return AM
        
def EffectiveAirMass(AM_st, AM_mid, AM_end):
    # Calculate effective airmass to be called by SetAirMass() and Imcombine()
    # This comes from Stetson, 'Some Factors Affecting the Accuracy of Stellar 
    # Photometry with CCDs,' DAO preprint, September 1988 and uses Simpson's rule.
    # See also http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?setairmass
    # Input: airmass at start, middel, and end of an exposure. 
    # Output: Effective Airmass 
    AM_eff = (AM_st + 4.*AM_mid + AM_end)/6.  
    return AM_eff

def Trim_Spec(img):
    # Trims Overscan region and final row of of image #
    # The limits of the 2x2 binned trim are: [:, 1:199, 9:2054]
    # The limits of the 1x2 trim are: [:, 1:199, 19:4111]
    print "\n====================\n"  
    print 'Triming Image: %s\n' % img
    img_head= fits.getheader(img) 
    img_data= fits.getdata(img)    
    Fix_Header(img_head)
    try:
        length = float(img_head['PARAM17'])
    except:
        length = float(img_head['PG3_1'])
    if length == 2071.:
        img_head.append( ('CCDSEC', '[9:2055,1:200]' ,'Original Pixel Indices'),
                   useblanks= True, bottom= True )
        NewHdu = fits.PrimaryHDU(data= img_data[:, 1:200, 9:2055], header= img_head)
        new_file_name= check_file_exist('t'+img)
        NewHdu.writeto(new_file_name, output_verify='warn', clobber= True )
        return (new_file_name)
    elif length == 4142.:
        img_head.append( ('CCDSEC', '[19:4111,1:200]' ,'Original Pixel Indices'),
                   useblanks= True, bottom= True )
        NewHdu = fits.PrimaryHDU(data= img_data[:, 1:200, 19:4111], header= img_head)
        new_file_name= check_file_exist('t'+img)
        NewHdu.writeto(new_file_name, output_verify='warn', clobber= True )
        return (new_file_name)
    else:
        print 'WARNING. Image not trimmed. \n'

def Add_Scale (img_block):
    # Function to be called by Imcombine. 
    # The function is meant to additively sclae a set of images, (zeros in particular). 
    # The input is a numpy block of pixel values (see imcombine). 
    # The function calculates the average number of 
    # counts of the region [25:75, 1700:1800] of the first image. 
    # Then scales the rest of the images by adding the diffrence between the 
    # average counts of the first image and its own.
    # Returns a scaled image block, and a list of scale values. 
    print("Scaling Counts Additively.\n")
    ni, ny, nx = np.shape(img_block)
    Cavg= [] # Average Counts 
    Sval= []  # Scale Values
    for i in range(0,ni):
        Cavg.append( np.mean(img_block[i, 25:75, 1700:1800]) )
        Sval.append( Cavg[0]-Cavg[i] )
        img_block[i]= img_block[i] + Sval[i]
    try:
        diagnostic[0:len(Cavg),0] = np.array(Cavg)
    except:
        pass
    return img_block, Sval
    
def Mult_Scale (img_block,index):
    # Function to be called by Imcombine. 
    # The function is meant to multiplicative sclae a set of images, (flats in particular). 
    # The input is a numpy block of pixel values (see imcombine). 
    # The function calculates the average number of 
    # counts of the region [25:75, 1700:1800] of the first image. 
    # Then scales the rest of the images by multiplying by the ratio between the 
    # average counts of the first image and its own.
    # Returns a scaled image block, and a list of scale values. 
    print("Scaling Counts Multiplicatively.\n")
    ni, ny, nx = np.shape(img_block)
    Cavg= [] # Average Counts 
    Cstd = [] #Standard deviation 
    Sval= []  # Scale Values
    for i in range(0,ni):
        Cavg.append( np.mean(img_block[i, 25:75, 1700:1800]) )
        Cstd.append( np.std(img_block[i,25:75,1700:1800]))
        Sval.append( Cavg[0]/Cavg[i] )
        img_block[i]= img_block[i]*Sval[i]
    try:
        if index == 1:
            diagnostic[0:len(Cavg),3] = np.array(Cavg)
            diagnostic[0:len(Cstd),4] = np.array(Cstd)
        elif index == 2:
            diagnostic[0:len(Cavg),7] = np.array(Cavg)
            diagnostic[0:len(Cstd),8] = np.array(Cstd)
    except:
        pass
    return img_block, Sval
        
    
# ===========================================================================
# Main Functions ============================================================
# ===========================================================================

def lacosmic(img):
    print ''
    print 'Finding cosmic rays in ', img
    datalist = fits.open(img)
    data = datalist[0].data
    data2 = data[0,:,:]
    array = data2
    header = fits.getheader(img)
    Fix_Header(header) 
    gain = 1.33 #datalist[0].header['GAIN'] #1.33 from 2017-06-07
    rdnoise = datalist[0].header['RDNOISE']

    c = cosmics.cosmicsimage(array, gain=gain, readnoise=rdnoise, sigclip = 5.0, sigfrac = 0.5, objlim = 4.0,satlevel=45000.0,verbose=True)
    c.run(maxiter=4)

    maskname = img[0:img.find('.fits')] + '_mask.fits'
    mask_array = np.expand_dims(c.mask,axis=0)
    mask_array = np.cast['uint8'](mask_array)
    mask_im = fits.PrimaryHDU(data=mask_array,header=header) 
    mask_im.writeto(maskname,clobber=True)
    print 'Mask image: ', maskname

    cleanname = 'c' + img
    data_array = np.expand_dims(c.cleanarray,axis=0)
    header.set('MASK',maskname,'Mask of cosmic rays')
    clean_im = fits.PrimaryHDU(data=data_array,header=header)
    clean_im.writeto(cleanname,clobber=True)
    print 'Clean image: ', cleanname
    return cleanname, maskname

def Bias_Subtract( img_list, zero_img ):
    # This function takes in a list of images and a bias image 'zero_img'
    # and performs a pixel by pixel subtration using numpy.
    # The function writes the bias subtracted images as 'b.Img_Name.fits'.
    # The output is a list of names for the bias subtrated images. 
    print "\n====================\n"  
    print 'Bias Subtracting Images: \n' 
        
    zero_data = fits.getdata(zero_img)
    bias_sub_list = []
    for img in img_list:
        print img
        hdu = fits.getheader(img)
        Fix_Header(hdu) 
        img_data = fits.getdata(img)
        img_data[ np.isnan(img_data) ] = 0
        b_img_data = np.subtract(img_data, zero_data)
        print 'b.'+"%s Mean: %.3f StDev: %.3f" % (img, np.mean(b_img_data), np.std(img_data))
        hdu.set( 'DATEBIAS', datetime.datetime.now().strftime("%Y-%m-%d"), 'Date of Bias Subtraction' )
        hdu.append( ('BIASSUB', zero_img ,'Image Used to Bias Subtract.'),
                   useblanks= True, bottom= True )
        NewHdu = fits.PrimaryHDU(b_img_data, hdu)
        bias_sub_name= check_file_exist('b.'+img)
        NewHdu.writeto(bias_sub_name, output_verify='warn', clobber= True)
        bias_sub_list.append( bias_sub_name )
    return bias_sub_list

# ===========================================================================

def Norm_Flat_Avg( flat ):
    # Takes average value of all the pixels and devides the entier flat by 
    # that value using numpy. 
    print "\n====================\n" 
    print 'Normalizing %s By Dividing Each Pixel By Average Value:' % ( flat )
    # Read Data, take average, and divide # 
    flat_data = fits.getdata(flat)
    flat_data[ np.isnan(flat_data) ] = 0
    # Calculate Average of the flat excluding bottom row and overscan regions # 
    avg_flat = np.average( flat_data[:, 1:200, 9:2055] )
    norm_flat_data = np.divide( flat_data, float(avg_flat) )
    print 'Average Value: %s\n' % avg_flat
    # Copy Header, write changes, and write file #
    hdu = fits.getheader(flat)
    Fix_Header(hdu)
    hdu.append( ('NORMFLAT', avg_flat,'Average Used to Normalize the Flat.'), 
               useblanks= True, bottom= True )
    NewHdu = fits.PrimaryHDU(data= norm_flat_data, header= hdu)
    norm_flat_name= check_file_exist('n'+flat)
    NewHdu.writeto(norm_flat_name, output_verify='warn', clobber= True )
    
    print 'Flat: %s Mean: %.3f StDev: %.3f' % (norm_flat_name, np.mean(norm_flat_data), np.std(norm_flat_data)) 
    return (norm_flat_name)

# ============================================================================    

def Norm_Flat_Poly( flat , order):
    print "\n====================\n" 
    print 'Normalizing %s By Fitting Polynomial to center rows [95:105]:' % ( flat )
    # Decide Order # 
    #if flat.lower().__contains__("blue")== True:
    #    order= 3;
    #elif flat.lower().__contains__("red")== True:
    #    order= 3; 
    #else:
    #    print ("Could not identifiy blue or red flat")
    #    order= raw_input("Fit Order?>>>")
    print "Fit Order: %s" % order
    #See in littrow ghost file already exists for blue files
    if flat.lower().__contains__("blue")== True:
        littrow_exist = glob('littrow_ghost.txt')
        if len(littrow_exist) == 1:
            print 'littrow_ghost.txt file already exists. Using that for mask.'
            littrow_ghost = np.genfromtxt('littrow_ghost.txt')
            litt_low = int(littrow_ghost[0])
            litt_hi = int(littrow_ghost[1])
        else:
            print 'Finding and saving littrow ghost location'
            littrow_ghost = find_littrow(flat)
            litt_low = int(littrow_ghost[0])
            litt_hi = int(littrow_ghost[1])
    else:
        #These are dummy values so we can concatenate below 
        litt_low = 100
        litt_hi = 99
    # Read Flat and Average Center Rows # 
    flat_data = fits.getdata(flat)
    flat_data[ np.isnan(flat_data) ] = 0
    fit_data= np.median(flat_data[0][95:105], axis=0) # Median of center Rows ###
    X= range(0,len(fit_data)) # Column Numbers 
    # Fit the data removeing the limits of the overscan regions and littrow ghost. #
    lo= 10;
    hi= 2055;
    xvals = np.concatenate((X[lo:litt_low],X[litt_hi:hi]))
    yvals = np.concatenate((fit_data[lo:litt_low],fit_data[litt_hi:hi]))
    # Calculate Fit # 
    coeff= np.polyfit(xvals, yvals, order ) # coefficents of polynomial fit # 
    profile= np.poly1d(coeff)(X) # Profile Along Dispersion axis # 
    #plt.clf()
    #plt.plot(xvals,yvals,'b.')
    #plt.plot(X,profile,'r')
    #plt.show()
    #Save values for diagnostics
    #if flat.lower().__contains__("blue"):
    #    diagnostic[0:len(X[lo:hi]),22] = X[lo:hi]
    #    diagnostic[0:len(fit_data[lo:hi]),23] = fit_data[lo:hi]
    #    diagnostic[0:len(profile),24] = profile
    if flat.lower().__contains__("red"):
        diagnostic[0:len(X[lo:hi]),25] = X[lo:hi]
        diagnostic[0:len(fit_data[lo:hi]),26] = fit_data[lo:hi]
        diagnostic[0:len(profile),27] = profile
    # Divide each Row by the Profile # 
    for row in flat_data[0]:
        i= 0; 
        while i < len(row): 
            row[i]= row[i]/profile[i]
            i= i+1   
            
    # Copy Header, write changes, and write file #
    hdu = fits.getheader(flat)
    Fix_Header(hdu)
    hdu.append( ('NORMFLAT ', order,'Flat Polynomial Fit Order'), 
               useblanks= True, bottom= True )
    for i in range(0,len(coeff)):
        coeff_str=  "{0:.5e}".format(coeff[i])
        coeff_order = str(len(coeff)-i-1)
        coeff_title = 'NCOEF%s' %coeff_order
        coeff_expla = 'Flat Polynomial Coefficient - Term %s' %coeff_order
        hdu.append((coeff_title,coeff_str,coeff_expla),
                   useblanks= True, bottom= True )
    NewHdu = fits.PrimaryHDU(data= flat_data, header= hdu)
    norm_flat_name= check_file_exist('n'+flat)
    NewHdu.writeto(norm_flat_name, output_verify='warn', clobber= True )
    
    print '\nFlat: %s Mean: %.3f StDev: %.3f' % (norm_flat_name, np.mean(flat_data), np.std(flat_data))
    return (norm_flat_name)

# ============================================================================    

def Norm_Flat_Boxcar( flat ):
    print 'Normalizing ', flat , 'by boxcar smoothing'
    flat_image = fits.getdata(flat)
    flat_data = flat_image[0,:,:] ###
    #See if littrow ghost file already exists for blue files
    if flat.lower().__contains__("blue")== True:
        littrow_exist = glob('littrow_ghost.txt')
        if len(littrow_exist) == 1:
            print 'littrow_ghost.txt file already exists. Using that for mask.'
            littrow_ghost = np.genfromtxt('littrow_ghost.txt')
            litt_low = int(littrow_ghost[0])
            litt_hi = int(littrow_ghost[1])
        else:
            print 'Finding and saving littrow ghost location'
            littrow_ghost = find_littrow(flat)
            litt_low = int(littrow_ghost[0])
            litt_hi = int(littrow_ghost[1])
        image_masked = flat_data.copy()
        rows = image_masked.shape[0]
        columns = np.arange(image_masked.shape[1])
        columns_littrow = np.linspace(litt_low,litt_hi,num=(litt_hi-litt_low)+1)
        for x in np.arange(rows):
            row_data = np.concatenate((flat_data[x,litt_low-15:litt_low+1],flat_data[x,litt_hi:litt_hi+16]))
            columns_fit = np.concatenate((columns[litt_low-15:litt_low+1],columns[litt_hi:litt_hi+16]))
            pol = np.polyfit(columns_fit,row_data,2)
            polp = np.poly1d(pol)
            #if (x > 80) and (x < 90):
            #    plt.plot(columns_fit,row_data,'b+')
            #    plt.plot(columns_fit,polp(columns_fit),'r')
            #    plt.show()
            if x == 100:
                diagnostic[0:len(columns_fit),11] = columns_fit
                diagnostic[0:len(row_data),12] = row_data
                diagnostic[0:len(row_data),13] = polp(columns_fit)
            image_masked[x,litt_low:litt_hi+1] = polp(columns_littrow)
    else:
        #These are dummy values so we can concatenate below 
        litt_low = 100
        litt_hi = 99
        image_masked = flat_data.copy()
    
    print 'Boxcar smoothing ',  flat, ' now.\n'
    kernel_size = 200 #size of boxcar kernel to convolve with image
    boxcar_kernel = Box2DKernel(kernel_size)
    image_pad = np.pad(image_masked,kernel_size,'mean',stat_length=40) #Pad to reduce edge effects
    image_smooth = convolve_fft(image_pad,boxcar_kernel,boundary='fill',fill_value=0)
    image_smooth_unpad = image_smooth[kernel_size:(-1*kernel_size),kernel_size:(-1*kernel_size)]

    image_divided = flat_data / image_smooth_unpad

    #plt.clf()
    #plt.plot(image_divided[100,:])
    #plt.show()

    if flat.lower().__contains__("blue"):
        diagnostic[0:len(image_divided[100,:]),14] = image_divided[100,:]
    if flat.lower().__contains__("red"):
        diagnostic[0:len(image_divided[100,:]),15] = image_divided[100,:]


    # Copy Header, write changes, and write file #
    hdu = fits.getheader(flat)
    Fix_Header(hdu)
    hdu.append( ('FLATTYPE', 'BOXCAR','Kernel used to flatten'), useblanks= True, bottom= True )
    hdu.append(('KERNEL',kernel_size,'Kernel size used'), useblanks= True, bottom= True )
    NewHdu = fits.PrimaryHDU(data= image_divided, header= hdu)
    norm_flat_name= check_file_exist('n'+flat)
    NewHdu.writeto(norm_flat_name, output_verify='warn', clobber= True )
    
    print '\nFlat: %s Mean: %.3f StDev: %.3f' % (norm_flat_name, np.mean(flat_data), np.std(flat_data))
    return (norm_flat_name)

# ============================================================================    

def Norm_Flat_Boxcar_Multiples( flat ,adc_stat=None):
    print 'Normalizing ', flat, 'by using multiple boxcars.'
    flat_image = fits.getdata(flat)
    quartz_data = flat_image[0,:,:] ###
    if adc_stat == None:
        hdu = fits.getheader(flat)
        adc_stat = hdu['ADCSTAT']
    print 'Using ADC status: ', adc_stat
    if adc_stat == 'IN':
        dome_flat_directory = '/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/domeflats/ADC'
        dome_flat_name = 'tb.DomeFlat_930_blue_adc.fits'
    else:
        dome_flat_directory = '/afs/cas.unc.edu/depts/physics_astronomy/clemens/students/group/domeflats/NOADC'
        dome_flat_name = 'tb.DomeFlat_930_blue_noadc.fits'

    print 'Masking littrow ghost.'
    if flat.lower().__contains__("blue")== True:
        littrow_exist = glob('littrow_ghost.txt')
        if len(littrow_exist) == 1:
            print 'littrow_ghost.txt file already exists. Using that for mask.'
            littrow_ghost = np.genfromtxt('littrow_ghost.txt')
            litt_low = int(littrow_ghost[0])
            litt_hi = int(littrow_ghost[1])
        else:
            print 'Finding and saving littrow ghost location'
            littrow_ghost = find_littrow(flat)
            litt_low = int(littrow_ghost[0])
            litt_hi = int(littrow_ghost[1])
        quartzim_masked = quartz_data.copy()
        rows = quartzim_masked.shape[0]
        columns = np.arange(quartzim_masked.shape[1])
        columns_littrow = np.linspace(litt_low,litt_hi,num=(litt_hi-litt_low)+1)
        for x in np.arange(rows):
            row_data = np.concatenate((quartz_data[x,litt_low-15:litt_low+1],quartz_data[x,litt_hi:litt_hi+16]))
            columns_fit = np.concatenate((columns[litt_low-15:litt_low+1],columns[litt_hi:litt_hi+16]))
            pol = np.polyfit(columns_fit,row_data,2)
            polp = np.poly1d(pol)
            #if (x > 80) and (x < 90):
            #    plt.plot(columns_fit,row_data,'b+')
            #    plt.plot(columns_fit,polp(columns_fit),'r')
            #    plt.show()
            if x == 100:
                diagnostic[0:len(columns_fit),11] = columns_fit
                diagnostic[0:len(row_data),12] = row_data
                diagnostic[0:len(row_data),13] = polp(columns_fit)
            quartzim_masked[x,litt_low:litt_hi+1] = polp(columns_littrow)
    else:
        #These are dummy values so we can concatenate below 
        litt_low = 100
        litt_hi = 99
        image_masked = quartz_data.copy()
    print 'Boxcar smoothing quartz flat with kernel of 20'
    quartz_kernel_size = 20 #If this is too small, we don't take out anything. Too large and we take out everything. Goal is to strike middle so that we remove only low frequency stuff. 
    quartz_boxcar_kernel = Box2DKernel(quartz_kernel_size)

    quartz_image_pad = np.pad(quartzim_masked,quartz_kernel_size,'mean',stat_length=10)
    quartzim_smooth = convolve(quartz_image_pad,quartz_boxcar_kernel)
    quartz_image_smooth_unpad = quartzim_smooth[quartz_kernel_size:(-1*quartz_kernel_size),quartz_kernel_size:(-1*quartz_kernel_size)]

    nQuartz20 = quartz_data / quartz_image_smooth_unpad

    #############################
    #Now do the same for the domeflat
    #############################
    print 'Starting dome flat portion'
    getcwd = os.getcwd()
    os.chdir(dome_flat_directory)
    dome = fits.getdata(dome_flat_name)
    domeim = dome[0,:,:]

    #Replace littrow ghost with parabolic fit between edges
    print 'Masking littrow ghost in dome flat'
    domeim_masked = domeim.copy()
    littrow_ghost_red = np.genfromtxt('littrow_ghost_red.txt')
    litt_low_red = int(littrow_ghost_red[0])
    litt_hi_red = int(littrow_ghost_red[1])
    rows = domeim.shape[0]
    columns = np.arange(domeim.shape[1])
    columns_littrow_red = np.linspace(litt_low_red,litt_hi_red,num=(litt_hi_red-litt_low_red)+1)
    columns_fit_red = np.linspace(litt_low_red-15,litt_hi_red+15,num=(litt_hi_red-litt_low_red)+31)
    #print columns_fit_red
    
    for x in np.arange(rows):
        #row_data = image[x,litt_low-15:litt_hi+16]
        row_data_red = np.concatenate((domeim[x,litt_low_red-15:litt_low_red+1],domeim[x,litt_hi_red:litt_hi_red+16]))
        columns_fit_red = np.concatenate((columns[litt_low_red-15:litt_low_red+1],columns[litt_hi_red:litt_hi_red+16]))
        pol = np.polyfit(columns_fit_red,row_data_red,2)
        polp = np.poly1d(pol)
        #if (x > 80) and (x < 90):
        #    plt.plot(columns_fit_red,row_data_red,'b+')
        #    plt.plot(columns_fit_red,polp(columns_fit_red),'r')
        #    plt.show()
        domeim_masked[x,litt_low_red:litt_hi_red+1] = polp(columns_littrow_red)

    #quartz_kernel_size = 20 #If this is too small, we don't take out anything. Too large and we take out everything. Goal is to strike middle so that we remove only low frequency stuff. 
    #quartz_boxcar_kernel = Box2DKernel(quartz_kernel_size)
    #boxcar_kernel = Gaussian2DKernel(kernel_size)
    print 'Boxcar smoothing dome flat with kernel of 20'
    dome_image_pad = np.pad(domeim_masked,quartz_kernel_size,'mean',stat_length=10)
    domeim_smooth = convolve(dome_image_pad,quartz_boxcar_kernel)
    dome_image_smooth_unpad = domeim_smooth[quartz_kernel_size:(-1*quartz_kernel_size),quartz_kernel_size:(-1*quartz_kernel_size)]
    
    os.chdir(getcwd)

    ####################
    # Multiple nQuartz by dome_image_smooth_unpad
    ####################
    print 'Mutliplying the two flats.'
    nQD = np.multiply(nQuartz20,dome_image_smooth_unpad)


    ####################
    # Take nQB, fit a nth order poly, then smooth with boxcar 200
    ####################
    print 'Fitting 5th order polynomial'
    order= 5;
    nnQD = nQD.copy()
    fit_data = np.median(nQD[95:105],axis=0)# Median of center Rows

    X= range(0,len(fit_data)) # Column Numbers 
    # Fit the data removeing the limits of the overscan regions and littrow ghost. #

    # Calculate Fit # 
    coeff= np.polyfit(X[650:], fit_data[650:], order ) # coefficents of polynomial fit # 
    profile= np.poly1d(coeff)(X) # Profile Along Dispersion axis # 
    #plt.clf()
    #plt.plot(X[650:],fit_data[650:],'b')
    #plt.plot(X,profile,'r')
    #plt.plot(X[650:],fit_data[650:]/profile[650:])
    #plt.show()
    for row in nnQD:
        i= 0; 
        while i < len(row): 
            row[i]= row[i]/profile[i]
            i= i+1   
    

    if flat.lower().__contains__("blue"):
        diagnostic[0:len(X[650:]),22] = X[650:]
        diagnostic[0:len(fit_data[650:]),23] = fit_data[650:]
        diagnostic[0:len(profile),24] = profile


    #newim = fits.PrimaryHDU(data=nnQD,header=domehdu.header)
    #newim.writeto('nnQD_blue.fits',clobber=True)
    #exit()

    finalim_masked = nnQD.copy()
    rows = finalim_masked.shape[0]
    columns = np.arange(finalim_masked.shape[1])
    columns_littrow = np.linspace(litt_low,litt_hi,num=(litt_hi-litt_low)+1)
    for x in np.arange(rows):
        row_data = np.concatenate((nnQD[x,litt_low-15:litt_low+1],nnQD[x,litt_hi:litt_hi+16]))
        columns_fit = np.concatenate((columns[litt_low-15:litt_low+1],columns[litt_hi:litt_hi+16]))
        pol = np.polyfit(columns_fit,row_data,2)
        polp = np.poly1d(pol)
        #if (x > 80) and (x < 90):
        #    plt.plot(columns_fit,row_data,'b+')
        #    plt.plot(columns_fit,polp(columns_fit),'r')
        #    plt.show()
        finalim_masked[x,litt_low:litt_hi+1] = polp(columns_littrow)

    print 'Boxcar smoothing with 200'
    kernel_size = 200 #size of boxcar kernel to convolve with image
    boxcar_kernel = Box2DKernel(kernel_size)

    finalimage_pad = np.pad(finalim_masked,kernel_size,'mean',stat_length=40) #Pad to reduce edge effects
    finalimage_smooth = convolve_fft(finalimage_pad,boxcar_kernel,boundary='fill',fill_value=0)
    finalimage_smooth_unpad = finalimage_smooth[kernel_size:(-1*kernel_size),kernel_size:(-1*kernel_size)]
    image_divided = nnQD / finalimage_smooth_unpad

    #newim = fits.PrimaryHDU(data=image_divided,header=domehdu.header)
    #newim.writeto('nnnQD_blue.fits',clobber=True)

    ###############################
    #Do a 200 pixel boxcar on the original quartz flat and use that for the first 760 pixels.
    ###############################
    flat_image = fits.getdata(flat)
    flat_data = flat_image[0,:,:] ###
    # Calculate Fit # 
    fit_data = np.median(flat_data[95:105],axis=0)
    X= range(0,len(fit_data)) # Column Numbers 
    order = 3.
    coeff= np.polyfit(X, fit_data, order ) # coefficents of polynomial fit # 
    profile= np.poly1d(coeff)(X) # Profile Along Dispersion axis # 
    #plt.clf()
    #plt.plot(X[650:],fit_data[650:],'b')
    #plt.plot(X,profile,'r')
    #plt.plot(X[650:],fit_data[650:]/profile[650:])
    #plt.show()
    for row in flat_data:
        i= 0; 
        while i < len(row): 
            row[i]= row[i]/profile[i]
            i= i+1   

    print 'Boxcar smoothing ',  flat, ' now.\n'
    kernel_size = 200 #size of boxcar kernel to convolve with image
    boxcar_kernel = Box2DKernel(kernel_size)

    image_pad = np.pad(flat_data,kernel_size,'mean',stat_length=40) #Pad to reduce edge effects
    image_smooth = convolve_fft(image_pad,boxcar_kernel,boundary='fill',fill_value=0)
    image_smooth_unpad = image_smooth[kernel_size:(-1*kernel_size),kernel_size:(-1*kernel_size)]

    image_divided_quartz = flat_data / image_smooth_unpad
    

    ###############################
    #Stictch the two images together
    ###############################
    print 'Stitching images together.'
    try:
        stitchloc_temp = np.genfromtxt('stitch_location.txt')
        stitchloc = float(stitchloc_temp)
        print 'Found stitch_location.txt file. Using ', stitchloc, ' for stitching location.'
    except:
        stitchloc = 747.
        print 'No file found. Using ', stitchloc, ' for stitching location.'
    leftside = image_divided_quartz[:,:stitchloc]
    rightside = image_divided[:,stitchloc:]

    newimage = np.concatenate((leftside,rightside),axis=1)


    if flat.lower().__contains__("blue"):
        diagnostic[0:len(newimage[100,:]),14] = newimage[100,:]

    # Copy Header, write changes, and write file #
    hdu = fits.getheader(flat)
    Fix_Header(hdu)
    hdu.append( ('FLATTYPE', 'BOXCAR','Kernel used to flatten'), useblanks= True, bottom= True )
    hdu.append(('KERNEL',kernel_size,'Kernel size used'), useblanks= True, bottom= True )
    hdu.append(('DOMEFLAT',dome_flat_name,'Dome Flat used'), useblanks= True, bottom= True )
    hdu.append(('STITCHLO',stitchloc,'Stitch location between flats'), useblanks= True, bottom= True )
    NewHdu = fits.PrimaryHDU(data= newimage, header= hdu)
    norm_flat_name= check_file_exist('n'+flat)
    NewHdu.writeto(norm_flat_name, output_verify='warn', clobber= True )
    
    print '\nFlat: %s Mean: %.3f StDev: %.3f' % (norm_flat_name, np.mean(flat_data), np.std(flat_data))
    return (norm_flat_name)




# ===========================================================================    
    
def Flat_Field( spec_list, flat ):
    # This Function divides each spectrum in spec_list by the flat and writes
    # The new images as fits files. The output is a list of file names of 
    # the flat fielded images. 
    print "\n====================\n" 
    print 'Flat Fielding Images by Dividing by %s\n' % (flat) 
    
    np.seterr(divide= 'warn')
    flat_data = fits.getdata(flat)
    #If flat is a blue spectrum, find the Littrow ghost and add those pixels to the header
    if 'blue' in flat.lower():
        #See if littrow_ghost.txt already exists
        file_exist = glob('littrow_ghost.txt')
        if len(file_exist) == 1:
            littrow_location = np.genfromtxt('littrow_ghost.txt')
            littrow_ghost = [littrow_location[0],littrow_location[1]]
            fit_data = np.median(flat_data[75:85],axis=0)
            low_index = 1210. #Lowest pixel to search within
            high_index = 1710. #highest pixel to search within
            fit_data1 = fit_data[low_index:high_index]
            fit_pix1 = np.linspace(low_index,low_index+len(fit_data1),num=len(fit_data1))
            diagnostic[0:len(fit_pix1),17] = fit_pix1
            diagnostic[0:len(fit_data1),18] = fit_data1
            diagnostic[0,21] = littrow_ghost[0]
            diagnostic[1,21] = littrow_ghost[1]
        else:
            littrow_ghost = find_littrow(flat)
            litt_low = int(littrow_ghost[0])
            litt_hi = int(littrow_ghost[1])
        try:
            hduflat = fits.getheader(flat)
            stitchloc = hduflat['STITCHLO']
            #print stitchloc
        except:
            stitchloc = 'None'
            pass
    else:
        littrow_ghost = 'None'
        stitchloc = 'None'
    f_spec_list = []
    if isinstance(spec_list,str):
        spec_list = [spec_list] #Ensure that spec_list is actually a list
    for spec in spec_list:
        spec_data = fits.getdata(spec)
        f_spec_data = np.divide(spec_data, flat_data)
        f_spec_data[ np.isnan(f_spec_data) ] = 0
        print "f"+"%s Mean: %.3f StDev: %.3f" % (spec, np.mean(f_spec_data), np.std(f_spec_data) ) 
        hdu = fits.getheader(spec)
        Fix_Header(hdu)
        hdu.set('DATEFLAT', datetime.datetime.now().strftime("%Y-%m-%d"), 'Date of Flat Fielding')
        hdu.set('LITTROW',str(littrow_ghost),'Littrow Ghost location in Flat')
        hdu.append( ('FLATFLD', flat,'Image used to Flat Field.'), 
               useblanks= True, bottom= True )
        hdu.append(('STITCHLO',stitchloc,'Stitch location between flats'), useblanks= True, bottom= True )
        NewHdu = fits.PrimaryHDU(data= f_spec_data, header= hdu)
        new_file_name= check_file_exist('f'+spec)
        NewHdu.writeto(new_file_name, output_verify='warn', clobber= True)
        f_spec_list.append(new_file_name)
    return f_spec_list

# ===========================================================================

def find_littrow(flat):
    print 'Finding Littrow Ghost'
    #Do a normalization first.
    flat_data = fits.getdata(flat)
    flat_data[ np.isnan(flat_data) ] = 0
    fit_data= np.median(flat_data[0][95:105], axis=0) # Median of center Rows
    X= range(0,len(fit_data)) # Column Numbers 
    # Fit the data removeing the limits of the overscan regions. #
    lo= 10; #10
    hi= 2055; #2055
    coeff = np.polyfit(X[lo:hi],fit_data[lo:hi],4)
    profile = np.poly1d(coeff)(X)
    #plt.clf()
    #plt.plot(X[lo:hi],fit_data[lo:hi],'bo')
    #plt.plot(X,profile)
    #plt.show()
    for row in flat_data[0]:
        i = 0;
        while i < len(row):
            row[i] = row[i]/profile[i]
            i += 1

    fit_data = np.median(flat_data[0][75:85],axis=0)
    low_index = 1210. #Lowest pixel to search within
    high_index = 1730. #highest pixel to search within
    fit_data1 = fit_data[low_index:high_index]
    fit_pix1 = np.linspace(low_index,low_index+len(fit_data1),num=len(fit_data1))
    max_pixel = np.argmax(fit_data1)
    fit_data2 = fit_data1[max_pixel-30:max_pixel+30]
    guess1 = np.zeros(5)
    guess1[0] = np.mean(fit_data2)
    guess1[1] = (fit_data2[-1]-fit_data2[0])/len(fit_data2)
    guess1[2] = np.amax(fit_data2)
    guess1[3] = np.argmax(fit_data2)
    guess1[4] = 4.
    error_fit1 = np.ones(len(fit_data2))
    xes1 = np.linspace(0,len(fit_data2)-1,num=len(fit_data2))
    fa1 = {'x':xes1,'y':fit_data2,'err':error_fit1}
    fitparams1 = mpfit.mpfit(fitgaussslope,guess1,functkw=fa1,quiet=True)
    center_pixel = low_index+max_pixel-30.+fitparams1.params[3]
    littrow_ghost = [np.rint(center_pixel-9.),np.rint(center_pixel+9.)]
    np.savetxt('littrow_ghost.txt',[np.rint(center_pixel-9.),np.rint(center_pixel+9.)])
    #plt.clf()
    #plt.plot(fit_pix1,fit_data1)
    #plt.axvline(littrow_ghost[0])
    #plt.axvline(littrow_ghost[1])
    #plt.show()
    return littrow_ghost
    

# ===========================================================================

def SetAirMass(img, lat= -30.238, scale= 750):
    # This Function Calculates The Effective Airmass of a single image  
    # Inputs:
    #   img = image name
    #   lat = latitude of observer in decimal degrees. 
    #       (Default Soar lat: '-30:14:16.8' = -30.238 deg)
    #   scale = atmospheric scale factor 750
    # Output: 
    #   AMeff = effective airmass for single image
     
    # Image Info #    
    hdulist = fits.open(img, 'update')
    hdu = hdulist[0]
    
    Fix_Header(hdu.header)
            
    ra = decimal_ra( hdu.header['RA'] ) # hours
    dec = decimal_dec( hdu.header['DEC'] ) # deg
    lst_st = decimal_ra( hdu.header['LST'] ) # start exposure LST in hours
    exp = hdu.header['EXPTIME']  # sec
    lst_mid = lst_st + (exp/2.)/3600. # mid exposure LST in hours
    lst_end = lst_st + (exp)/3600. # end exposure LST in hours

    # Air Mass Calculations # 
    times = [lst_st, lst_mid, lst_end]
    AM = []
    for t in times:
        alt, az, ha = RaDec2AltAz(ra, dec, lat, t )
        airmass = AirMass(alt, scale)
        AM.append( airmass )
    AMeff = EffectiveAirMass(AM[0], AM[1], AM[2])
    
    # Print and write to header # 
    print '\nImage:', img
    print 'Observatory Latitude: %s' % lat
    print 'AM_st   AM_mid  AM_end  AM_eff'
    print '%5.4f  %5.4f  %5.4f  %5.4f' % (AM[0], AM[1], AM[2], AMeff)
    hdu.header.set( 'AIRMASS', np.round(AMeff,6) , 
                   'Calculated Effective Airmass' )
    hdulist.close()
    return AMeff    
 
# =========================================================================== 
 
def imcombine(im_list, output_name, method,  
              lo_sig = 10, hi_sig = 3, overwrite= False, mask=False):
# Image Combination Script # 
# Inputs:
#   im_list = mist be a python list of images or "@listfile"
#   output_name =  name of combined fits image 
#   method = The method to use for combining (median, average, sum)
#   lo_sig = low sigma cliping factor (default = 3 sigma) 
#   hi_sig = high sigma cliping factor (default = 3 sigma)
#   overwrite = if true go ahead and re write existing file 'output_name'
#               if false it will warn you and ask for new output_name. 
#               (default false)
# Output:
#   After succefully combining, calculateing airmass, and writing to fits file, 
#   The return of this function is the name of the combined 
#   image (Output_name).
    print "\n====================\n" 
    print "Combining Images:"
    print "Using %s of count values." % method 
    print "Sigma Cliping Factors (low, high): (%s, %s)\n" % (lo_sig, hi_sig)
    
    # Read image data and put it in a numpy block # 
    Ni = len(im_list)
    for i in range(0, Ni):
        # First size the array to contain the data based on 1st image #
        # Create block with 3 axis:
        #   axis[0] has length of number of images.
        #   axis[1] is the vertical axis of the chip.
        #   axis[2] is the horizontal axis of the chip.
        if i == 0:  
            img_data = fits.getdata(im_list[i])
            #n,Ny,Nx = np.shape(img_data)
            Ny = img_data.shape[-2]
            Nx = img_data.shape[-1]
            img_block = np.ndarray( shape= (Ni,Ny,Nx) )
            img_block[i,:,:] = img_data
            if (not mask) is False:
                mask_data = fits.getdata(mask[i])
                mask_block = np.ndarray(shape= (Ni,Ny,Nx) )
                mask_block[i,:,:] = mask_data
        # Then go ahead and read the rest of the images into the block #   
        else: 
            img_block[i,:,:] = fits.getdata(im_list[i])
            if (not mask) is False:
                mask_block[i,:,:] = fits.getdata(mask[i])
        # set nan values to zero # 
        img_block[ np.isnan(img_block) ] = 0
        
    # If Zero Additive Scale Images # 
    if im_list[0].lower().__contains__("zero"):
        img_block, Scale= Add_Scale(img_block)
    # If Flats Multiplicative Scale Images # 
    elif im_list[0].lower().__contains__("flat"):
        if im_list[0].lower().__contains__("blue"):
            index = 1.
            img_block, Scale= Mult_Scale(img_block,index)
        elif im_list[0].lower().__contains__("red"):
            index = 2.
            img_block, Scale= Mult_Scale(img_block,index)
    # If Not, Dont Scale # 
    else: 
        print "Did Not Scale Images.\n" 
        Scale= np.empty(Ni)
        Scale[:]= np.NaN
    
    # Print Name and Statistics of Each image % 
    avgarr,stdarr = np.zeros(Ni), np.zeros(Ni)
    for i in range(0,Ni):
        Avg= np.mean(img_block[i,25:75,1700:1800])
        Std= np.std(img_block[i,25:75,1700:1800])
        avgarr[i] = Avg
        stdarr[i] = Std
        print ( "%02d: %s ScaleValue:% .3f Mean: %.3f StDev: %.3f" 
                % (i, im_list[i], Scale[i], Avg, Std) )
    
    #Save Values to diagnostic array
    try:
        if im_list[0].lower().__contains__("zero"):
            diagnostic[0:len(avgarr),1] = avgarr
            diagnostic[0:len(stdarr),2] = stdarr
        if im_list[0].lower().__contains__("flat"):
            if im_list[0].lower().__contains__("blue"):
                diagnostic[0:len(avgarr),5] = avgarr
                diagnostic[0:len(stdarr),6] = stdarr
            elif im_list[0].lower().__contains__("red"):
                diagnostic[0:len(avgarr),9] = avgarr
                diagnostic[0:len(stdarr),10] = stdarr
    except:
        pass
    ## Combine the images acording to input "method" using SigmaClip() above ## 
    comb_img = np.ndarray( shape= (1,Ny,Nx), dtype='float32')
    ##mask_img = np.ndarray( shape= (1,Ny,Nx), dtype='float32')
    while True: # Contunualy askes for method if input is wierd # 
        
        if method == 'median':
            for y in range(0,Ny):
                for x in range(0,Nx):
                    counts = img_block[:,y,x]
                    val = np.median( SigClip(counts, lo_sig, hi_sig) )
                    comb_img[0,y,x] = np.float32(val)
            break # exit while loop 
    
        elif method == 'average':
            for y in range(0,Ny):
                for x in range(0,Nx):
                    if (not mask) is False:
                        counts = img_block[:,y,x]
                        masks = mask_block[:,y,x].astype(bool)
                        mx = np.ma.masked_array(counts,masks)
                        val = mx.mean() #We don't want to sigma clip if already masking
                        #if True in masks:
                        #    print counts
                        #    print masks
                        #    print val
                        #    print ''
                    else:
                        counts = img_block[:,y,x]
                        #counts_good, counts_bad = SigClip(counts, lo_sig, hi_sig)
                        val = np.average( SigClip(counts, lo_sig, hi_sig) )
                        #val = np.average(counts_good)
                    comb_img[0,y,x] = np.float32(val)
                    #mask = np.average(counts_bad)
                    #mask_img[0,y,x] = np.float32(mask)
            #mask_image = fits.PrimaryHDU(data=mask_img)
            #mask_image.writeto('Mask.fits')
                                   
            break # exit while loop
        
        elif method == 'sum':
            for y in range(0,Ny):
                for x in range(0,Nx):
                    counts = img_block[:,y,x]
                    val = np.sum( SigClip(counts, lo_sig, hi_sig) )
                    comb_img[0,y,x] = np.float32(val)
            #print img_block[:,100,50]
            #print comb_img[:,100,50]
            break # exit while loop
        
        else:
            # if 'method' input is wanky, ask for method again. 
            print "\nError: Method NOT AVALABLE." 
            print "Available Methods: ('median', 'average', 'sum')"
            print "Enter Valid Method"
            method = raw_input('>>>')
    
    # Set NAN values to zero 
    comb_img[ np.isnan(comb_img) ] = np.float32(0)
    
    ###### Calculate Effetive Airmass for combined image ######
    # The EffAM value is writen into the header in the next section #
    print '\nCalculating Effective Airmass:'    
    
    # if were just combining 2 images #
    if Ni == 2:
        AM0 = SetAirMass(im_list[0])
        AM2 = SetAirMass(im_list[1])
        AM1 = (AM0+AM2)/2
        EffAM = EffectiveAirMass(AM0, AM1, AM2)
        print '\nEffective Airmass of combined image: %5.4f' % EffAM
    # if were combining an odd number of images # 
    elif Ni%2 == 1: 
        images = [ im_list[0], im_list[Ni//2], im_list[-1] ] 
        AM = [ SetAirMass(img) for img in images ] 
        EffAM = EffectiveAirMass( AM[0], AM[1], AM[2] )
        print '\nEffective Airmass of combined image: %5.4f' % EffAM
    # if were combing an even number of images #  
    elif Ni%2 == 0:
        images = [im_list[0], im_list[(Ni//2)-1], im_list[Ni//2], im_list[-1]]
        AM = [ SetAirMass(img) for img in images ]
        EffAM = EffectiveAirMass( AM[0], (AM[1]+AM[2])/2, AM[3])
        print '\nEffective Airmass of combined image: %5.4f' % (EffAM)
    # Otherwise we fail # 
    else:
        print "Eff AirMass calculation failed? This never happens!"
    
    ###### Overwrite Protection loop, just in case ######
    if overwrite == False:
        from os.path import isfile
    
    while overwrite == False: # Outer Loop #  
    # Breaks if file name doesnot exist or overwrite == true # 
        exist = isfile(output_name) # Asks computer if file name exist # 
        
        if exist == False: 
            print "\nWriting combined image to fits file",output_name,"..." 
            break # Break out of outter loop and continue writing # 
        elif exist == True:
            while True: # Inner Loop # 
            # Breaks if user wishes to overwite, abort, or gives new name.
            # loop also checks new names for existance.  
                print "\nFile name",output_name,
                print "already exist do you wish to overwrite?"
                yes_no = raw_input('yes or no ?>>>')
            
                if yes_no == 'no':
                    # If overwrite no: prompt new name or abort # 
                    print"\nEnter new file name or Ctrl-c to Abort "
                    output_name = raw_input('>>>')
                    print "\nNew File Name:= ", output_name
                    break # breaks out of Inner loop only.
                          # Code proceeds to Outer Loop to 
                          # ask computer if new file name exist.     
                
                elif yes_no == 'yes':
                    # If overwrite yes: Break Inner Loop and Outer loop # 
                    overwrite = True
                    print "\nOverwriting Image:", output_name
                    break
                
                else: 
                    # If yes_no input is wierd return to Inner loop
                    # to ask question again. 
                    print "\nInput Not Recognized."
                    
    ###### The following part only runs if above while loop is satisfied ######
    
    # Copy header of first image in im_list and fix degree symbol issue. 
    hdulist = fits.open(im_list[0])
    hdu = hdulist[0]
    # This checks the string and deletes the bad keywords from header. 
    Fix_Header(hdu.header)
    
    # Write Effective Airmass into header # 
    hdu.header.set('AIRMASS',np.round(EffAM,6),'Calculated Effective Airmass')

    #Write date of image combination to header #
    hdu.header.set('DATECOMB', datetime.datetime.now().strftime("%Y-%m-%d"), 'Date of Image combination')
    
    # Write the imcombine information into header #
    N = len(im_list)
    for i in range(0,N):
        num = str(i+1).zfill(3)
        key = 'IMCMB'+num
        hdu.header.append( (key, im_list[i]), useblanks= True, bottom= True )
    hdu.header.append( ('NCOMBINE', N), useblanks= True, bottom = True )
    hdu.header.append( ('COMBTYPE', method,'Operation Used to Combine'),
                      useblanks= True, bottom= True )
    
    # Make sure header BITPIX reflects data encodeing as float 32 ie: -32 
    hdu.header['BITPIX'] = -32
    
    # Write header to new fits file  
    new_file_name= check_file_exist(output_name)
    hdu.writeto(new_file_name, output_verify='warn', clobber= True)
    
    # write combined data to new fits file  # 
    fits.update(output_name, data= comb_img, header= hdu.header, 
                output_verify='warn')
                         
    print ( "\nCombined Image: %s Mean: %.3f StDev: %.3f" 
            % (new_file_name, np.mean(comb_img), np.std(comb_img)) ) 
    return new_file_name           

# ===========================================================================
# ===========================================================================            
# ===========================================================================
