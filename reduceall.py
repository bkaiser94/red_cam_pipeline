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
lamp_files = sorted(glob('tFe*ms.fits'))
offset_file = glob('offsets.txt') #Offset file must be structured as blue, then red
if len(offset_file) == 0:
    offset_file = None
else:
    offset_file = offset_file[0]

#print spec_files
#print lamp_files
#Need to carefully match up the correct lamp and spectrum files. This seems to work well.
for x in lamp_files:
    if 'blue' in x.lower():
        lamp_color = 'blue'
    elif 'red' in x.lower():
        lamp_color = 'red'
    for y in spec_files:
        ###if (y[5:y.find('_930')] in x) and (y[y.find('_930'):y.find('_930')+8] in x):
        if (lamp_color in y.lower()) and (y[5:y.find('_930')] in x):
            print x, y, offset_file
            if offset_file == None:
                plotalot = True
            else:
                plotalot = False
            Wavelength_Calibration.calibrate_now(x,y,'no','yes',offset_file,plotall=plotalot)

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
flux_calibration.flux_calibrate_now(stdlist,fluxlist,continuum_files,extinct_correct=True,masterresp=True)

#=========================
#Begin Flux Calibration
#=========================
print 'Running diagnostics.'

diagnostics.diagnostic_now()
