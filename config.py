"""
This file contains the values of the code setup that are used in different modules during data reduction.
"""
cautious = True #when True it will ask if you if the background region is ok, etc. Basically it's the boolean for if you want to be bothered as the program runs
length_headers=  ['PARAM17', 'PG3_1', 'PG5_10'] #the various headers from the various iterations of the camera software that correspond to the serial_pixel_length
binning_headers= ['PARAM18', 'PG3_2', 'PG5_9'] # "" serial_binning headers
red_cam_id = 'Red' 
blue_cam_id = 'Blue'
camera_header= 'INSTCONF' #yes, this is the header that contains the indication of which camera is being used; it is not the 'camera_header'
affirmatives= ['yes', 'y', 'true']
negatives= ['no', 'n', 'false']
blue_cam_lotrim= 9
blue_cam_hightrim= 2055
red_cam_lotrim= 45
red_cam_hightrim= blue_cam_hightrim
zzceti = 'no' #argument for Wavelength_Calibration.py
to_flux = False #argument that determines whether or not flux calibration will occur in the run. Since we don't yet have a red cam master standard, it makes sense to keep this commented.








###################
#def ask_yn(query_string):
    #unacceptable = True
                #while unacceptable:
                    #answer = raw_input("Would you like to try to find a littrow_ghost anyway? >>>")
                    #if answer.lower() in config.affirmatives:
                        #unacceptable = False
                        #response = True #wants to fit a littrow ghost
                    #elif answer.lower() in config.negatives:
                        #unacceptable = False
                        #response = False
