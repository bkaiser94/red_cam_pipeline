'''
Written by JT Fuchs, UNC. 

PURPOSE: This program takes ZZ Ceti observations with Goodman and runs the full pipeline on a night. Uses ReduceSpec.py, spectral_extraction.py, Wavelenght_Calibration.py, continuum_normalization.py, flux_calibration.py, and diagnostics.py (and all dependencies therein).

DIRECTORY FILES THAT SHOULD EXIST:
    listZero - text file containing list of bias images to combine
    
    listFlat - text file containing list of flat field images to combine. If both blue and red set, give all blue files first, then all red files.

    listSpec - text file containing list of spectra to combine. Organize by target. 

    listFe - text file containing list of Iron lamps to combine. If both blue and red set, give all blue files first, then all red files.

'''



import ReduceSpec
import spectral_extraction
import Wavelength_Calibration
import continuum_normalization
import flux_calibration
import diagnostics
from glob import glob
import config
import os



if config.cautious:
    print "config.cautious ==True, so you're gonna have to be paying attention."
#=========================
#Begin Fits Reduction
#=========================
ReduceSpec.reduce_now(['script_name','listZero','listFlat','listSpec','listFe'])

#========================
#Begin Spectral Extraction
#========================
print 'Beginning spectral extraction.'
spec_files = sorted(glob('cftb*fits'))
single_spec_list = []
for x in spec_files:
    if ('cftb.0' in x) or ('cftb.1' in x) or ('cftb.2' in x):
        single_spec_list.append(x)
for x in single_spec_list:
    spec_files.remove(x)
spec_files = sorted(spec_files)

lamp_file_blue = sorted(glob('tFe*blue*fits'))
lamp_file_red = sorted(glob('tFe*red*fits'))

print "lamp_file_blue: ", lamp_file_blue
print "lamp_file_red: ", lamp_file_red


#Search for FWHM and trace file for each spectrum. If it does not exist, these go to None and will be fit and saved during the extraction.
trace_files = []
FWHM_files = []
for x in spec_files:
    trace_name = '*' + x[5:-5] + '*trace.npy'
    new_trace = glob(trace_name)
    if len(new_trace) == 0:
        trace_files.append(None)
    else:
        trace_files.append(new_trace[0])
    fwhm_name = '*' + x[5:-5] + '*poly.npy'
    new_fwhm = glob(fwhm_name)
    if len(new_fwhm) == 0:
        FWHM_files.append(None)
    else:
        FWHM_files.append(new_fwhm[0])


for x in spec_files:
    if 'blue' in x.lower():
        lamp_file = lamp_file_blue[0]
    elif 'red' in x.lower():
        lamp_file = lamp_file_red[0]
    else:
        print "no colors in spec name... again"
        print "using this file:"
        print sorted(glob('tFe*fits'))
        lamp_file_a= sorted(glob('tFe*fits')+glob('t*_fe*fits'))
        print "lamp_file_a: ", lamp_file_a
        lamp_file = lamp_file_a[0]
        print lamp_file
    FWHM_thisfile = FWHM_files[spec_files.index(x)]
    trace_thisfile = trace_files[spec_files.index(x)]
    if trace_thisfile != None:
        trace_exist_file = True
    else:
        trace_exist_file = False
    print ''
    print x, lamp_file,trace_thisfile, FWHM_thisfile
    #Must add in option of not have trace file or FWHM file
    #if no FWHMfile, FWHMfile=None
    spectral_extraction.extract_now(x,lamp_file,FWHMfile=FWHM_thisfile,tracefile=trace_thisfile,trace_exist=trace_exist_file)


#=========================
# Begin Wavelength Calibration
#=========================
print '\n Beginning Wavelength Calibration'
spec_files = sorted(glob('cftb*ms.fits'))
lamp_files = sorted(glob('tFe*ms.fits')+glob("t*_fe*ms.fits"))

def check_offsets():
    print "checking for offsets in this directory: ", os.getcwd()
    offset_file = glob('offsets.txt') #Offset file must be structured as blue, then red
    print "offset_file: ", offset_file
    if len(offset_file) == 0:
        offset_file = None
    else:
        offset_file = offset_file[0]
    return offset_file
#offset_file = glob('offsets.txt') #Offset file must be structured as blue, then red
#if len(offset_file) == 0:
    #offset_file = None
#else:
    #offset_file = offset_file[0]
offset_file = check_offsets() #check for offset files once before going through this.
starting_offset_file= offset_file #This is the determinant as to whether or not we should be 
#print spec_files
#print lamp_files
counter_b = 0
counter_r = 0
#Need to carefully match up the correct lamp and spectrum files. This seems to work well.
#current setup as of 2017-08-15 relies on blue calibration before red
for x in lamp_files:
    if 'blue' in x.lower():
        lamp_color = 'blue'
    elif 'red' in x.lower():
        lamp_color = 'red'
    for y in spec_files:
        ###if (y[5:y.find('_930')] in x) and (y[y.find('_930'):y.find('_930')+8] in x):
        try:
            if (lamp_color in y.lower()) and (y[5:y.find('_930')] in x):
                print x, y, offset_file
                if  (lamp_color== 'blue'):
                    if counter_b > 0:
                        print "recognized counter_b"
                        offset_file =check_offsets() #will now know that there is an offset file after the first run.
                    print "counter_b: ", counter_b
                    counter_b+=1
                if (lamp_color == 'red'):
                    if counter_r > 0:
                        print "recognized counter_r"
                        offset_file = check_offsets()
                    elif (starting_offset_file == None):
                        offset_file = starting_offset_file
                    print "counter_r: ", counter_r
                    counter_r += 1
                if offset_file == None:
                    plotalot = True
                else:
                    plotalot = False
                print x,y, offset_file
                Wavelength_Calibration.calibrate_now(x,y,'no',config.zzceti,offset_file,plotall=plotalot)
        except NameError as nameerror:
            #protects from the lamp_color not getting assigned in those if statements up there, but it also catches errors where one of the variables isn't defined too, doesn't it?
            print "NameError: ", nameerror
            print "still no colors in files for like the 200th time."
            print "Filename that has no colors: ", y
            if offset_file== None:
                plotalot= True
            else:
                plotalot= False
            Wavelength_Calibration.calibrate_now(x,y,'no',config.zzceti,offset_file,plotall=plotalot) #changedthisvalue The 4th arg 'no' is whether or not we're looking at a zzceti, and the 'no' setting is kind of a roll of the dice.

#=========================
#Begin Continuum Normalization
#=========================
print '\n Begin continuum normalization.'
continuum_files = sorted(glob('wcftb*ms.fits'))
#print continuum_files
x = 0
while x < len(continuum_files):
    if x == len(continuum_files)-1:
        #print continuum_files[x]
        continuum_normalization.normalize_now(continuum_files[x],None,False,plotall=False)
        x += 1
    elif continuum_files[x][0:continuum_files[x].find('930')] == continuum_files[x+1][0:continuum_files[x].find('930')]:
        #print continuum_files[x],continuum_files[x+1]
        continuum_normalization.normalize_now(continuum_files[x],continuum_files[x+1],True,plotall=False)
        x += 2
    else:
        #print continuum_files[x]
        continuum_normalization.normalize_now(continuum_files[x],None,False,plotall=False)
        x += 1


#=========================
#Begin Flux Calibration
#=========================
print '\nBegin flux calibration.'
#We should use the same files are for the continuum normalization. But if you want to change that for some reason, adjust below.
'''
continuum_files = sorted(glob('wcftb*ms.fits'))
single_spec_list = []
for x in continuum_files:
    if 'flux' in x:
        single_spec_list.append(x)
for x in single_spec_list:
    continuum_files.remove(x)
continuum_files = sorted(continuum_files)
#print continuum_files
'''
stdlist = None
fluxlist = None
if config.to_flux:
    flux_calibration.flux_calibrate_now(stdlist,fluxlist,continuum_files,extinct_correct=True,masterresp=True)
if not config.to_flux:
    print "Not Flux calibrating since to_flux: " , config.to_flux

#=========================
#Begin Flux Calibration
#=========================
print 'Running diagnostics.'

diagnostics.diagnostic_now()
