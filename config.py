"""
This file contains the values of the code setup that are used in different modules during data reduction.
"""
cautious = False #when True it will ask if you if the background region is ok, etc. Basically it's the boolean for if you want to be bothered as the program runs
length_headers=  ['PARAM17', 'PG3_1', 'PG5_10'] #the various headers from the various iterations of the camera software that correspond to the serial_pixel_length
binning_headers= ['PARAM18', 'PG3_2', 'PG5_9'] # "" serial_binning headers
red_cam_id = 'Red' 
blue_cam_id = 'Blue'
camera_header= 'INSTCONF' #yes, this is the header that contains the indication of which camera is being used; it is not the 'camera_header'
