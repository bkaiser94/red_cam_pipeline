# -*- coding: utf-8 -*-
"""
Diagnostic Plots for Pipeline
Author: Patrick O'Brien
Date last updated: February 2017
"""
# Import statements
from glob import glob
import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileMerger
import pandas as pd

##### ------------------------------------------------------------------ #####

# Set up flags table
def setup_flags_table(fwhm_file_name):
    arr = np.genfromtxt(fwhm_file_name, dtype=None,delimiter='\t')
    data = [ [0 for col in range(num_flags)] for row in range(len(arr))]
    flags = pd.DataFrame(data, columns = ['Star', 'Exposure','Date', 'Bias1', 'Bias2', 'BlueFlat1','BlueFlat2', 'RedFlat1', 'RedFlat2', 'BluePoly', 'BlueCut', 'RedCut', 'Littrow', 'ExtFWHM', 'ExtProf', 'FitToBack', 'ProfFWHM', 'ProfPos', 'PeakGauss', 'ResponseBlue', 'ResponseRed', 'WaveFitResBlue', 'WaveFitResRed'])
    return flags

##### ------------------------------------------------------------------ #####

# function that takes file names and organizes by the unique star name in each
def unique_star_names(seq, idfun=None): 
       # order preserving
       if idfun is None:
           def idfun(x): return x
       seen = {}
       result = []
       for item in seq:
           marker = idfun(item)
           if marker in seen: continue
           seen[marker] = 1
           result.append(item)
       return result
           
##### ------------------------------------------------------------------ #####
# Calibrations function
def diagnostic_plots_cals(file_name, flags):
    date = str(file_name[10:20])
    pp = PdfPages('cal_plots.pdf')
    pdfs.append('cal_plots.pdf')

    arr = np.genfromtxt(file_name, dtype=None, delimiter=' ')
    
    bias_bs, bias_as, bias_std = [],[],[]
    flat_blue_bs, flat_blue_std_bs, flat_blue_as, flat_blue_std_as = [],[],[], []
    flat_red_bs, flat_red_std_bs, flat_red_as, flat_red_std_as = [],[],[], []
    blue_pix, blue_val, blue_poly = [],[],[]
    blue_cut_row100, red_cut_row100, junk_zeros = [],[],[]
    littrow_pix, littrow_val, fit_pix_lit, fit_lit, masked_edges = [],[],[],[],[]
    com_blue_pix, com_blue_flat, blue_poly_fit = [], [], []
    com_red_pix, com_red_flat, red_poly_fit = [], [], []


    for m in np.arange(len(arr)):
    
        bias_bs.append(arr[m][0])
        bias_as.append(arr[m][1])
        bias_std.append(arr[m][2])
        
        flat_blue_bs.append(arr[m][3])
        flat_blue_std_bs.append(arr[m][4])
        flat_blue_as.append(arr[m][5])
        flat_blue_std_as.append(arr[m][6])
    
        flat_red_bs.append(arr[m][7])
        flat_red_std_bs.append(arr[m][8])
        flat_red_as.append(arr[m][9])
        flat_red_std_as.append(arr[m][10])
        
        blue_pix.append(arr[m][11])
        blue_val.append(arr[m][12])
        blue_poly.append(arr[m][13])
        
        blue_cut_row100.append(arr[m][14])
        red_cut_row100.append(arr[m][15])
        junk_zeros.append(arr[m][16])
    
        littrow_pix.append(arr[m][17])
        littrow_val.append(arr[m][18])
        fit_pix_lit.append(arr[m][19])
        fit_lit.append(arr[m][20])
        masked_edges.append(arr[m][21])
        
        com_blue_pix.append(arr[m][22])
        com_blue_flat.append(arr[m][23])
        blue_poly_fit.append(arr[m][24])

        com_red_pix.append(arr[m][25])
        com_red_flat.append(arr[m][26])
        red_poly_fit.append(arr[m][27])

    bias_bs = np.array(bias_bs)
    bias_bs = np.trim_zeros(bias_bs, 'b')
    
    bias_as = np.array(bias_as)
    bias_as = np.trim_zeros(bias_as, 'b')
    
    bias_std = np.array(bias_std)
    bias_std = np.trim_zeros(bias_std, 'b')
    
    flat_blue_bs = np.array(flat_blue_bs)
    flat_blue_bs = np.trim_zeros(flat_blue_bs, 'b')
    
    flat_blue_as = np.array(flat_blue_as)
    flat_blue_as = np.trim_zeros(flat_blue_as, 'b')
    
    flat_blue_std_bs = np.array(flat_blue_std_bs)
    flat_blue_std_bs = np.trim_zeros(flat_blue_std_bs, 'b')
    
    flat_blue_std_as = np.array(flat_blue_std_as)
    flat_blue_std_as = np.trim_zeros(flat_blue_std_as, 'b')
    
    flat_red_bs = np.array(flat_red_bs)
    flat_red_bs = np.trim_zeros(flat_red_bs, 'b')
    
    flat_red_as = np.array(flat_red_as)
    flat_red_as = np.trim_zeros(flat_red_as, 'b')
    
    flat_red_std_bs = np.array(flat_red_std_bs)
    flat_red_std_bs = np.trim_zeros(flat_red_std_bs, 'b')
    
    flat_red_std_as = np.array(flat_red_std_as)
    flat_red_std_as = np.trim_zeros(flat_red_std_as, 'b')
    
    blue_pix = np.array(blue_pix)
    blue_pix = np.trim_zeros(blue_pix, 'b')
    
    blue_val = np.array(blue_val)
    blue_val = np.trim_zeros(blue_val, 'b')
    
    blue_poly = np.array(blue_poly)
    blue_poly = np.trim_zeros(blue_poly, 'b')
    
    blue_cut_row100 = np.array(blue_cut_row100)
    blue_cut_row100 = np.trim_zeros(blue_cut_row100, 'b')
    
    red_cut_row100 = np.array(red_cut_row100)
    red_cut_row100 = np.trim_zeros(red_cut_row100, 'b')

    bias_bs = np.array(bias_bs)
    bias_bs = np.trim_zeros(bias_bs, 'b')
    
    littrow_pix = np.array(littrow_pix)
    littrow_pix = np.trim_zeros(littrow_pix, 'b')
    
    littrow_val = np.array(littrow_val)
    littrow_val = np.trim_zeros(littrow_val, 'b')
    
    fit_pix_lit = np.array(fit_pix_lit)
    fit_pix_lit = np.trim_zeros(fit_pix_lit, 'b')
    
    fit_lit = np.array(fit_lit)
    fit_lit = np.trim_zeros(fit_lit, 'b')
    
    com_blue_pix = np.array(com_blue_pix)
    com_blue_pix = np.trim_zeros(com_blue_pix, 'b')
    
    com_blue_flat = np.array(com_blue_flat)
    com_blue_flat = np.trim_zeros(com_blue_flat, 'b')
    
    blue_poly_fit = np.array(blue_poly_fit)
    blue_poly_fit = np.trim_zeros(blue_poly_fit)

    com_red_pix = np.array(com_red_pix)
    com_red_pix = np.trim_zeros(com_red_pix, 'b')
    
    com_red_flat = np.array(com_red_flat)
    com_red_flat = np.trim_zeros(com_red_flat, 'b')
    
    red_poly_fit = np.array(red_poly_fit)
    red_poly_fit = np.trim_zeros(red_poly_fit)
    
    edge1 = float(masked_edges[0])
    edge2 = float(masked_edges[1])

    plt.figure()
    plt.errorbar(np.arange(len(bias_bs)), bias_bs, yerr=bias_std, marker="o", linestyle="None")
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.title('Bias - Before Scaling')
    plt.savefig(pp,format='pdf')
    plt.close()
    
    plt.figure()
    plt.errorbar(np.arange(len(bias_as)), bias_as, yerr=bias_std, marker="o", linestyle="None")
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.title('Bias - After Scaling')
    plt.savefig(pp,format='pdf')
    plt.close()
    
    plt.figure()
    plt.errorbar(np.arange(len(flat_blue_bs)), flat_blue_bs, yerr=flat_blue_std_bs, marker="o", linestyle="None")
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.title('Blue Flat - Before Scaling')
    plt.savefig(pp,format='pdf')
    plt.close()
    
    plt.figure()
    plt.errorbar(np.arange(len(flat_blue_as)), flat_blue_as, yerr=flat_blue_std_as, marker="o", linestyle="None")
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.title('Blue Flat - After Scaling')
    plt.savefig(pp,format='pdf')
    plt.close()
    
    if len(flat_red_bs) > 0:
        plt.figure()
        plt.errorbar(np.arange(len(flat_red_bs)), flat_red_bs, yerr=flat_red_std_bs, ecolor='r', marker="o",markerfacecolor='r', linestyle="None")
        plt.xlabel('Number')
        plt.ylabel('Value')
        plt.title('Red Flat - Before Scaling')
        plt.savefig(pp,format='pdf')
        plt.close()
        
        plt.figure()
        plt.errorbar(np.arange(len(flat_red_as)), flat_red_as, yerr=flat_red_std_as, ecolor='r', marker="o",markerfacecolor='r', linestyle="None")
        plt.xlabel('Number')
        plt.ylabel('Value')
        plt.title('Red Flat - After Scaling')
        plt.savefig(pp,format='pdf')
        plt.close()
        
    plt.figure()
    plt.plot(blue_pix, blue_val,'o')
    plt.plot(blue_pix, blue_poly,'g')
    plt.xlabel('Pixel')
    plt.ylabel('Value')
    plt.title('Blue Polynomial Littrow Fit Check')
    plt.savefig(pp,format='pdf')
    plt.close()

    plt.figure()
    plt.plot(np.arange(len(blue_cut_row100)), blue_cut_row100, 'b')
    plt.plot(np.arange(len(blue_cut_row100)), np.ones(len(blue_cut_row100)), 'k--')
    plt.xlabel('Pixel')
    plt.ylabel('Value')
    plt.title('Cut Along Row 100 - Blue')
    plt.savefig(pp, format='pdf')
    plt.close()
    
    plt.figure()
    plt.plot(com_blue_pix, com_blue_flat,'o')
    plt.plot(np.arange(len(blue_poly_fit)),blue_poly_fit,'g')
    plt.xlabel('Pixel')
    plt.ylabel('Value')
    plt.title('Blue Polynomial Check')
    plt.savefig(pp,format='pdf')
    plt.close()
    
    if len(red_cut_row100 > 0):
        plt.figure()
        plt.plot(np.arange(len(red_cut_row100)), red_cut_row100, 'r')
        plt.plot(np.arange(len(red_cut_row100)), np.ones(len(red_cut_row100)), 'k--')
        plt.xlabel('Pixel')
        plt.ylabel('Value')
        plt.title('Cut Along Row 100 - Red')
        plt.savefig(pp, format='pdf')
        plt.close()
        
        plt.figure()
        plt.plot(com_red_pix, com_red_flat,'ro')
        plt.plot(np.arange(len(red_poly_fit)),red_poly_fit,'g')
        plt.xlabel('Pixel')
        plt.ylabel('Value')
        plt.title('Red Polynomial Check')
        plt.savefig(pp,format='pdf')
        plt.close()
    
    plt.figure()
    plt.plot(littrow_pix, littrow_val, 'k-')
    plt.plot(fit_pix_lit, fit_lit, 'r-')
    plt.axvline(x=masked_edges[0])
    plt.axvline(x=masked_edges[1])
    plt.xlabel('Pixel')
    plt.ylabel('Normalized Flux')
    plt.title('Location of Littrow Ghost')
    plt.savefig(pp, format='pdf')
    plt.close()
    
    pp.close()
    
    ###### FLAGS #####
    bias1_flag = 0
    bias2_flag = 0
    blueflat1_flag = 0
    blueflat2_flag = 0
    redflat1_flag = 0
    redflat2_flag = 0
    blue_poly_flag = 0
    blue_cut_flag = 0
    red_cut_flag = 0
    littrow_flag = 0
    unscaled_bias_std = np.std(bias_bs)
    scaled_bias_std = np.std(bias_as)
    unscaled_blue_flat_std = np.std(flat_blue_bs)
    scaled_blue_flat_std = np.std(flat_blue_as)
    
    if all( (np.mean(bias_bs) - 2*unscaled_bias_std) <= x <= (np.mean(bias_bs) + 2*unscaled_bias_std) for x in bias_bs):
        bias1_flag = 0
    else:
        bias1_flag = 1
    if all( (np.mean(bias_as) - 2*scaled_bias_std) <= x <= (np.mean(bias_as) + 2*scaled_bias_std) for x in bias_as):
        bias2_flag = 0
    else:
        bias2_flag = 1
    if all( (np.mean(flat_blue_bs) - 2*unscaled_blue_flat_std) <= x <= (np.mean(flat_blue_bs) + 2*unscaled_blue_flat_std) for x in flat_blue_bs):
        blueflat1_flag = 0
    else:
        blueflat1_flag = 1
    if all( (np.mean(flat_blue_as) - 2*scaled_blue_flat_std) <= x <= (np.mean(flat_blue_as) + 2*scaled_blue_flat_std) for x in flat_blue_as):
        blueflat2_flag = 0
    else:
        blueflat2_flag = 1
    if all( abs((blue_val[x] - blue_poly[x])) < 250 for x in range(len(blue_pix))):
        blue_poly_flag = 0
    else:
        blue_poly_flag = 1
    if all( abs((blue_cut_row100[x] - 1.0)) < 0.1 for x in range(len(blue_cut_row100))):
        blue_cut_flag = 0
    else:
        blue_cut_flag = 1    
    if abs(np.average([edge1, edge2]) - 1400) < 10:
        littrow_flag = 0
    else:
        littrow_flag = 1
    flags['Date'] = date   
    flags['Bias1'] = bias1_flag
    flags['Bias2'] = bias2_flag
    flags['BlueFlat1'] = blueflat1_flag
    flags['BlueFlat2'] = blueflat2_flag
    flags['BluePoly'] = blue_poly_flag
    flags['BlueCut'] = blue_cut_flag
    flags['Littrow'] = littrow_flag
    
    if len(flat_red_bs) > 0:
        unscaled_red_flat_std = np.std(flat_red_bs)
        scaled_red_flat_std = np.std(flat_red_as)
        if all( (np.mean(flat_red_bs) - 2*unscaled_red_flat_std) <= x <= (np.mean(flat_red_bs) + 2*unscaled_red_flat_std) for x in flat_red_bs):
            redflat1_flag = 0
        else:
            redflat1_flag = 1
        if all( (np.mean(flat_red_as) - 2*scaled_red_flat_std) <= x <= (np.mean(flat_red_as) + 2*scaled_red_flat_std) for x in flat_red_as):
            redflat2_flag = 0
        else:
            redflat2_flag = 1
        if all( abs((red_cut_row100[x] - 1.0)) < 0.1 for x in range(len(red_cut_row100))):
            red_cut_flag = 0
        else:
            red_cut_flag = 1   
          
    flags['RedFlat1'] = redflat1_flag
    flags['RedFlat2'] = redflat2_flag
    flags['RedCut'] = red_cut_flag

##### ------------------------------------------------------------------ #####
# FWHM / Profile Position function
def diagnostic_plots_FWHM(file_name, flags):
    date = str(file_name[13:23])
    
    pp = PdfPages('fwhm_plots.pdf')
    pdfs.append('fwhm_plots.pdf')
    def unique_star_names(seq, idfun=None): 
       # order preserving
       if idfun is None:
           def idfun(x): return x
       seen = {}
       result = []
       for item in seq:
           marker = idfun(item)
           # in old Python versions:
           # if seen.has_key(marker)
           # but in new ones:
           if marker in seen: continue
           seen[marker] = 1
           result.append(item)
       return result
    
    arr = np.genfromtxt(file_name, dtype=None,delimiter='\t')
    
    names, col1, fwhm1, pos1, peak1, col2, fwhm2, pos2, peak2 = [],[],[],[],[],[],[],[],[]
    for m in np.arange(len(arr)):
        names.append(str(arr[m][0][10:-5]))
        col1.append(arr[m][1])
        fwhm1.append(arr[m][2])    
        pos1.append(arr[m][3])
        peak1.append(arr[m][4])
        col2.append(arr[m][5])
        fwhm2.append(arr[m][6])   
        pos2.append(arr[m][7])
        peak2.append(arr[m][8])
    fwhm2 = np.array(fwhm2)
    col2 = np.array(col2)
    pos2 = np.array(pos2)
    peak2 = np.array(peak2)

    no_duplicates = sorted(list(set(names)))
    
    cat_pts = []
    fwhm_pts = []
    pos_pts = []
    peak_pts = []
    for i in range(len(names)):
        for j in range(len(no_duplicates)):
            if no_duplicates[j] in names[i]:
                current_fwhm_array = []
                cat_pts.append(j)
                fwhm_pts.append(fwhm2[i])
                pos_pts.append(pos2[i])
                peak_pts.append(peak2[i])

    ##### FLAGS #####
    flags['Star'] = names
    

    for i in range(len(no_duplicates)):        
        star_name = no_duplicates[i]
        num_exposures = names.count(star_name)
        current_fwhm_array = []
        current_pos_array = []
        current_peak_array = []
        star_indexes = flags[flags['Star'] == star_name].index.tolist()

        for j in range(len(fwhm_pts)):
            if cat_pts[j] == i:
                current_fwhm_array.append(fwhm_pts[j])
                current_pos_array.append(pos_pts[j])
                current_peak_array.append(peak_pts[j])

        for k in range(num_exposures):
            flags.set_value(star_indexes[k], 'Exposure', k)
            if abs(current_fwhm_array[k] - np.median(current_fwhm_array)) > fwhm_tol:#*np.std(current_fwhm_array):
                flags.set_value(star_indexes[k], 'ProfFWHM', 1)
                
            if abs(current_pos_array[k] - np.median(current_pos_array)) > pos_tol:#*np.std(current_pos_array):
                flags.set_value(star_indexes[k], 'ProfPos', 1)
                
            if abs(current_peak_array[k] - np.median(current_peak_array)) > peak_tol:#*np.std(current_peak_array):
                flags.set_value(star_indexes[k], 'PeakGauss', 1)
    
    ##### ----- #####    
    x = np.arange(len(no_duplicates))
    jitter = 0.1*np.random.rand(len(cat_pts))
    cat_pts = cat_pts + jitter
    
    plt.figure()
    plt.xticks(x, no_duplicates, rotation=90)
    plt.scatter(cat_pts,fwhm_pts)
    plt.xlabel('Star')
    plt.ylabel('Value')
    plt.title('FWHM')
    plt.tight_layout()
    plt.savefig(pp,format='pdf')
    plt.close()
    
    plt.figure()
    plt.xticks(x, no_duplicates, rotation=90)
    plt.scatter(cat_pts, pos_pts)
    plt.xlabel('Star')
    plt.ylabel('Value')
    plt.title('Profile Position')
    plt.tight_layout()
    plt.savefig(pp,format='pdf')
    plt.close()
    
    plt.figure()
    plt.xticks(x, no_duplicates, rotation=90)
    plt.scatter(cat_pts, peak_pts)
    plt.xlabel('Star')
    plt.ylabel('Value')
    plt.title('Peak Value of Gaussian')
    plt.tight_layout()
    plt.savefig(pp,format='pdf')
    plt.close()
    
    pp.close()   
##### ------------------------------------------------------------------ #####
# Wavelength calibrations function
def diagnostic_plots_wavecal(files, flags):
    
    star_name = str(files[0][8:-21])
    pdf_name = 'wavecal_plots_' + star_name + '.pdf'
    pp = PdfPages(pdf_name)
    pdfs.append(pdf_name)
    blue_arr = []
    red_arr = []
    if len(files) == 1:
        with open(files[0], 'r') as f:
            first_line = f.readline()
            if 'blue' in first_line:
                blue_arr = np.genfromtxt(files[0],dtype=None,delimiter=' ')
            elif 'red' in first_line:
                red_arr = np.genfromtxt(files[0],dtype=None,delimiter=' ')
    elif len(files) > 1:
        with open(files[0], 'r') as f:
            first_line = f.readline()
            if 'blue' in first_line:
                blue_arr = np.genfromtxt(files[0],dtype=None,delimiter=' ')
                red_arr = np.genfromtxt(files[1],dtype=None,delimiter=' ')
            elif 'red' in first_line:
                blue_arr = np.genfromtxt(files[1],dtype=None,delimiter=' ')
                red_arr = np.genfromtxt(files[0],dtype=None,delimiter=' ')

    if len(blue_arr) > 0:
        wave_fit, res, wave1, flux1, lam_fit, wave2, flux2, line_fit = [],[],[],[],[],[],[],[]
        
        for m in np.arange(len(blue_arr)):
        
            wave_fit.append(blue_arr[m][0])
            res.append(blue_arr[m][1])
            
            wave1.append(blue_arr[m][2])
            flux1.append(blue_arr[m][3])
            lam_fit.append(blue_arr[m][4])
            
            wave2.append(blue_arr[m][5])
            flux2.append(blue_arr[m][6])
            line_fit.append(blue_arr[m][7])
        
        wave_fit = np.array(wave_fit)
        wave_fit = np.trim_zeros(wave_fit, 'b')
        
        res = np.array(res)
        res = np.trim_zeros(res, 'b')
        
        wave1 = np.array(wave1)
        wave1 = np.trim_zeros(wave1, 'b')
        
        lam_fit = np.array(lam_fit)
        lam_fit = np.trim_zeros(lam_fit, 'b')
        
        flux1 = np.array(flux1)
        flux1 = np.trim_zeros(flux1, 'b')
        
        wave2 = np.array(wave2)
        wave2 = np.trim_zeros(wave2, 'b')
        
        flux2 = np.array(flux2)
        flux2 = np.trim_zeros(flux2, 'b')
        
        line_fit = np.array(line_fit)
        line_fit = np.trim_zeros(line_fit, 'b')
        
        xmin = np.min(wave_fit)-500
        xmax = np.max(wave_fit)+500
        x = np.linspace(xmin,xmax,1000)
        zeros = np.zeros(len(x))
        plt.figure()
        plt.scatter(wave_fit,res)
        plt.plot(x,zeros,'b--')
        plt.xlim(xmin,xmax)
        plt.xlabel('Wavelength')
        plt.ylabel('Residuals (pixels)')
        plt.title('Wavelength Fit Residuals - Blue - ' + star_name)
        plt.savefig(pp,format='pdf')
        plt.close()       
        
        if len(wave2) != 0:
            x_line = np.linspace(np.min(wave2),np.max(wave2),len(line_fit))
            plt.figure()
            plt.plot(wave2,flux2)
            plt.plot(x_line,line_fit)
            plt.xlabel('Pixels')
            plt.ylabel('Flux')
            plt.title('Zero Point Offset - Blue - ' + star_name)
            plt.savefig(pp,format='pdf')
            plt.close()
        
        ##### FLAGS #####
        star_indexes = flags[flags['Star'] == star_name].index.tolist()
        blue_wave_fit_flag = 0
        if all(-wave_fit_tol <= x <= wave_fit_tol for x in res):
            blue_wave_fit_flag = 0
        else:
            blue_wave_fit_flag = 1
        for k in range(len(star_indexes)):
            flags.set_value(star_indexes[k], 'WaveFitResBlue', blue_wave_fit_flag)
    ### ---------------------------------------------------------------------- ###
    if len(red_arr) > 0:
        wave_fit, res, wave1, flux1, lam_fit, wave2, flux2, line_fit = [],[],[],[],[],[],[],[]
        
        for m in np.arange(len(red_arr)):
        
            wave_fit.append(red_arr[m][0])
            res.append(red_arr[m][1])
            
            wave1.append(red_arr[m][2])
            flux1.append(red_arr[m][3])
            lam_fit.append(red_arr[m][4])
            
            wave2.append(red_arr[m][5])
            flux2.append(red_arr[m][6])
            line_fit.append(red_arr[m][7])
        
        wave_fit = np.array(wave_fit)
        wave_fit = np.trim_zeros(wave_fit, 'b')
        
        res = np.array(res)
        res = np.trim_zeros(res, 'b')
        
        wave1 = np.array(wave1)
        wave1 = np.trim_zeros(wave1, 'b')
        
        lam_fit = np.array(lam_fit)
        lam_fit = np.trim_zeros(lam_fit, 'b')
        
        flux1 = np.array(flux1)
        flux1 = np.trim_zeros(flux1, 'b')
        
        wave2 = np.array(wave2)
        wave2 = np.trim_zeros(wave2, 'b')
        
        flux2 = np.array(flux2)
        flux2 = np.trim_zeros(flux2, 'b')
        
        line_fit = np.array(line_fit)
        line_fit = np.trim_zeros(line_fit, 'b')
        
        xmin = np.min(wave_fit)-500
        xmax = np.max(wave_fit)+500
        x = np.linspace(xmin,xmax,1000)
        zeros = np.zeros(len(x))
        plt.figure()
        plt.scatter(wave_fit,res,color='red')
        plt.plot(x,zeros,'r--')
        plt.xlim(xmin,xmax)
        plt.xlabel('Wavelength')
        plt.ylabel('Residuals (pixels)')
        plt.title('Wavelength Fit Residuals - Red - ' + star_name)
        plt.savefig(pp,format='pdf')
        plt.close()        
        if len(wave2) != 0:
            x_line = np.linspace(np.min(wave2),np.max(wave2),len(line_fit))
            plt.figure()
            plt.plot(wave2,flux2,'r')
            plt.plot(x_line,line_fit,'g')
            plt.xlabel('Pixels')
            plt.ylabel('Flux')
            plt.title('Zero Point Offset - Red - ' + star_name)
            plt.savefig(pp,format='pdf')
            plt.close()
        
        ##### FLAGS #####
        star_indexes = flags[flags['Star'] == star_name].index.tolist()
        red_wave_fit_flag = 0
        if all(-wave_fit_tol <= x <= wave_fit_tol for x in res):
            red_wave_fit_flag = 0
        else:
            red_wave_fit_flag = 1
        for k in range(len(star_indexes)):
            flags.set_value(star_indexes[k], 'WaveFitResRed', red_wave_fit_flag)

    pp.close()

##### ------------------------------------------------------------------ #####
# Continuum function
def diagnostic_plots_continuum(file_name, flags):
    star_name = str(file_name[24:-21])
    pdf_name = 'modelcal_plots_' + star_name + '.pdf'
    pp = PdfPages(pdf_name)
    pdfs.append(pdf_name)

    arr = np.genfromtxt(file_name, dtype=None, delimiter=' ')

    blue_lam, blue_res, blue_masked_lam, blue_masked_res, blue_res_fit, norm_spec_blue = [],[],[],[],[],[]

    for m in np.arange(len(arr)):
    
        blue_lam.append(arr[m][0])
        blue_res.append(arr[m][1])
        
        blue_masked_lam.append(arr[m][2])
        blue_masked_res.append(arr[m][3])
        blue_res_fit.append(arr[m][4])
        
        norm_spec_blue.append(arr[m][5])

    blue_lam = np.array(blue_lam)
    blue_lam = np.trim_zeros(blue_lam, 'b')
    
    blue_res = np.array(blue_res)
    blue_res = np.trim_zeros(blue_res, 'b')
    
    blue_masked_lam = np.array(blue_masked_lam)
    blue_masked_lam = np.trim_zeros(blue_masked_lam, 'b')
    
    blue_masked_res = np.array(blue_masked_res)
    blue_masked_res = np.trim_zeros(blue_masked_res, 'b')
    
    blue_res_fit = np.array(blue_res_fit)
    blue_res_fit = np.trim_zeros(blue_res_fit, 'b')
    
    norm_spec_blue = np.array(norm_spec_blue)
    norm_spec_blue = np.trim_zeros(norm_spec_blue)

    plt.figure()
    plt.plot(blue_lam, blue_res)
    plt.plot(blue_masked_lam, blue_masked_res,'g.')
    plt.plot(blue_lam, blue_res_fit,'r')
    plt.xlabel('Wavelength')
    plt.ylabel('Response (observed/model)')
    plt.title('Response - Blue - ' + star_name)
    plt.savefig(pp,format='pdf')
    plt.close()
    
    plt.figure()
    plt.plot(blue_lam, norm_spec_blue)
    plt.xlabel('Wavelength')
    plt.title('Continuum Normalized Spectrum - Blue')
    plt.savefig(pp,format='pdf')
    plt.close()
    
    ##### FLAGS #####
    star_indexes = flags[flags['Star'] == star_name].index.tolist()
    blue_res_flag = 0
    for i in range(10,len(blue_masked_res)-10):
        wavelength_interval = blue_masked_lam[i-10:i+10]
        res_interval = blue_masked_res[i-10:i+10]
        fit_interval = blue_res_fit[np.where(blue_lam == wavelength_interval)]
        min_res = np.min(res_interval)
        max_res = np.max(res_interval)
        if all( min_res <= x <= max_res for x in fit_interval):
            blue_res_flag += 0
        else:
            blue_res_flag += 1
    
    if blue_res_flag > 0:
        for k in range(len(star_indexes)):
            flags.set_value(star_indexes[k], 'ResponseBlue', 1)

    ### ---------------------------------------------------------------------- ###
    if len(arr[0]) > 6:    
        red_lam, red_res, red_masked_lam, red_masked_res, red_res_fit, norm_spec_red = [],[],[],[],[],[]

        for m in np.arange(len(arr)):
        
            red_lam.append(arr[m][6])
            red_res.append(arr[m][7])
            
            red_masked_lam.append(arr[m][8])
            red_masked_res.append(arr[m][9])
            red_res_fit.append(arr[m][10])
            
            norm_spec_red.append(arr[m][11])
    
        red_lam = np.array(red_lam)
        red_lam = np.trim_zeros(red_lam, 'b')
        
        red_res = np.array(red_res)
        red_res = np.trim_zeros(red_res, 'b')
        
        red_masked_lam = np.array(red_masked_lam)
        red_masked_lam = np.trim_zeros(red_masked_lam, 'b')
        
        red_masked_res = np.array(red_masked_res)
        red_masked_res = np.trim_zeros(red_masked_res, 'b')
        
        red_res_fit = np.array(red_res_fit)
        red_res_fit = np.trim_zeros(red_res_fit, 'b')
        
        norm_spec_red = np.array(norm_spec_red)
        norm_spec_red = np.trim_zeros(norm_spec_red, 'b')

        plt.figure()
        plt.plot(red_lam, red_res)
        plt.plot(red_masked_lam, red_masked_res,'g.')
        plt.plot(red_lam, red_res_fit,'r')
        plt.xlabel('Wavelength')
        plt.ylabel('Response (observed/model)')
        plt.title('Response - Red - ' + star_name)
        plt.savefig(pp,format='pdf')
        plt.close()
        
        plt.figure()
        plt.plot(red_lam, norm_spec_red,'r')
        plt.xlabel('Wavelength')
        plt.title('Continuum Normalized Spectrum - Red')
        plt.savefig(pp,format='pdf')
        plt.close()   
        
        ##### FLAGS #####
        star_indexes = flags[flags['Star'] == star_name].index.tolist()
        red_res_flag = 0
        for i in range(10,len(blue_masked_res)-10):
            wavelength_interval = blue_masked_lam[i-10:i+10]
            res_interval = blue_masked_res[i-10:i+10]
            fit_interval = blue_res_fit[np.where(blue_lam == wavelength_interval)]
            min_res = np.min(res_interval)
            max_res = np.max(res_interval)
            if all( min_res <= x <= max_res for x in fit_interval):
                red_res_flag += 0
            else:
                red_res_flag += 1
    
        if red_res_flag > 0:
            for k in range(len(star_indexes)):
                flags.set_value(star_indexes[k], 'ResponseRed', 1)
    pp.close()
  
##### ------------------------------------------------------------------ #####
# Extraction function
def diagnostic_plots_extraction(file_name, flags):
    star_name = str(file_name)[11:-21]
    date = str(file_name)[-20:-10]

    pdf_name = 'extraction_plots_' + star_name + '.pdf'
    pp = PdfPages(pdf_name)
    pdfs.append(pdf_name)

    arr = np.genfromtxt(file_name, dtype=None, delimiter=' ')

    meas_FWHM, pix_FWHM, fit_FWHM, all_pix = [],[],[],[]
    prof_pix, prof_pos, fit_prof_pos = [],[],[]
    pix_val_1200, val_1200, pixel_back_fit, val_fit, poly_fit_back = [],[],[],[],[]
    
    for m in np.arange(len(arr)):
        meas_FWHM.append(arr[m][0])
        pix_FWHM.append(arr[m][1])
        fit_FWHM.append(arr[m][2])
        all_pix.append(arr[m][3])
        
        prof_pix.append(arr[m][4])
        prof_pos.append(arr[m][5])
        fit_prof_pos.append(arr[m][6])
        
        pix_val_1200.append(arr[m][7])
        val_1200.append(arr[m][8])
        pixel_back_fit.append(arr[m][9])
        val_fit.append(arr[m][10])
        poly_fit_back.append(arr[m][11])

    meas_FWHM = np.array(meas_FWHM)
    meas_FWHM = np.trim_zeros(meas_FWHM, 'b')

    pix_FWHM = np.array(pix_FWHM)
    pix_FWHM = np.trim_zeros(pix_FWHM, 'b')

    fit_FWHM = np.array(fit_FWHM)
    fit_FWHM = np.trim_zeros(fit_FWHM, 'b')
    
    all_pix = np.array(all_pix)
    all_pix = np.trim_zeros(all_pix, 'b')

    prof_pix = np.array(prof_pix)
    prof_pix = np.trim_zeros(prof_pix, 'b')
    
    prof_pos = np.array(prof_pos)
    prof_pos = np.trim_zeros(prof_pos, 'b')

    fit_prof_pos = np.array(fit_prof_pos)
    fit_prof_pos = np.trim_zeros(fit_prof_pos, 'b')
    
    pix_val_1200 = np.array(pix_val_1200)
    pix_val_1200 = np.trim_zeros(pix_val_1200, 'b')
    
    val_1200 = np.array(val_1200)
    val_1200 = np.trim_zeros(val_1200, 'b')
    
    pixel_back_fit = np.array(pixel_back_fit)
    pixel_back_fit = np.trim_zeros(pixel_back_fit, 'b')
    
    val_fit = np.array(val_fit)
    val_fit = np.trim_zeros(val_fit, 'b')
    
    poly_fit_back = np.array(poly_fit_back)  
    poly_fit_back = np.trim_zeros(poly_fit_back, 'b')
    
    plt.figure()
    plt.scatter(pix_FWHM,meas_FWHM)
    plt.plot(np.arange(len(fit_FWHM)),fit_FWHM)
    plt.xlabel('Pixel')
    plt.ylabel('FWHM')
    plt.title('Extraction FWHM - ' + star_name)
    plt.savefig(pp,format='pdf')
    plt.close()
    
    plt.figure()
    plt.scatter(prof_pix, prof_pos)
    plt.plot(np.arange(len(fit_prof_pos)),fit_prof_pos)
    plt.xlabel('Pixel')
    plt.ylabel('Profile Position')
    plt.title('Extraction Profile - ' + star_name)
    plt.savefig(pp,format='pdf')
    plt.close()
    
    plt.figure()
    plt.scatter(pix_val_1200, val_1200, color='k', marker='^')
    plt.scatter(pixel_back_fit, val_fit, color='b', marker='^')
    plt.plot(pix_val_1200, poly_fit_back, 'b-')
    plt.xlabel('Pixel')
    plt.title('Fit to Background at Column 1200')
    plt.savefig(pp,format='pdf')
    plt.close()
    
    pp.close()
    
    ##### FLAGS #####
    star_indexes = flags[flags['Star'] == star_name].index.tolist()
    ext_FWHM_flag = 0
    ext_profile_flag = 0
    background_fit_flag= 0
    meas_FWHM_std = np.std(meas_FWHM)
    max_back_ind = np.argmax(val_1200)
    fit_at_max = poly_fit_back[max_back_ind]
    avg_poly_fit = np.average(val_fit)
    
    if all( (np.mean(meas_FWHM) - ext_FWHM_num_sigma*meas_FWHM_std) <= x <= (np.mean(meas_FWHM) + ext_FWHM_num_sigma*meas_FWHM_std) for x in meas_FWHM):
        ext_FWHM_flag = 0
    else:
        ext_FWHM_flag = 1
    if all( abs(prof_pos[x] - fit_prof_pos[int(prof_pix[x])]) < ext_prof_tol for x in range(len(prof_pos)) ):
        ext_profile_flag = 0
    else:
        ext_profile_flag = 1
    if abs(fit_at_max - avg_poly_fit) < background_fit_tol:
        background_fit_flag = 0
    else:
        background_fit_flag = 1
    
    for k in range(len(star_indexes)):    
        flags.set_value(star_indexes[k], 'ExtFWHM', ext_FWHM_flag)
        flags.set_value(star_indexes[k], 'ExtProf', ext_profile_flag)
        flags.set_value(star_indexes[k], 'FitToBack', background_fit_flag)

##### ------------------------------------------------------------------ #####
# Final spectra function
def diagnostic_plots_spectra(file_name, flags):
    date = str(file_name)[9:19]

    pdf_name = 'final_spectra_plots.pdf'
    pp = PdfPages(pdf_name)
    pdfs.append(pdf_name)
    
    spectra_names = []
    with open(file_name, 'r') as f:
        first_line = f.readline()
        second_line = f.readline()
        names = first_line[3:-1] + ' ' + second_line[3:-2]
        names = names.split(' ')
        for n in names:
            spectra_names.append(n[7:-9])
    star_count = len(spectra_names)
    lam_arrs = [[] for i in range(star_count)]   
    flux_arrs = [[] for i in range(star_count)]

    arr = np.genfromtxt(file_name, dtype=None, delimiter=' ')
    for m in np.arange(len(arr)):
        lams_to_append = arr[m][0::2]
        flux_to_append = arr[m][1::2]
        
        for i in range(len(lam_arrs)):
            lam_arrs[i].append(lams_to_append[i])
            flux_arrs[i].append(flux_to_append[i])
        
    for i in range(len(spectra_names)):
        plt.figure()
        plt.plot(lam_arrs[i], flux_arrs[i])
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Flux')
        plt.title('Final Spectrum - ' + spectra_names[i])
        plt.savefig(pp,format='pdf')
        plt.close()
                    
    pp.close()


def diagnostic_now():
    
    original_date = os.getcwd()[-10::]

    ##### Flags #####
    # thresholds and constants
    global pdfs, num_flags, num_sigma, fwhm_tol, pos_tol, peak_tol, ext_FWHM_num_sigma,ext_prof_tol,background_fit_tol,wave_fit_tol
    num_flags = 23
    num_sigma = 2
    fwhm_tol = 1
    pos_tol = 5
    peak_tol = 500
    ext_FWHM_num_sigma = 2
    ext_prof_tol = 5
    background_fit_tol = 5
    wave_fit_tol = 0.15

    pdfs = glob('diagnostics_plots.pdf')
    for f in pdfs:
        os.remove(f)

    
    pdfs = []


    # Use the FWHM_records file to determine how many total exposures there were for the given date
    fwhm_files = glob('FWHM*.txt')
    file_name = str(fwhm_files[0])
    flags = setup_flags_table(file_name)


    ##### ------------------------------------------------------------------ #####    
    # Sort file names by type
    cal_files = glob('reduction*.txt')
    fwhm_files = glob('FWHM*.txt')
    wave_cal_files = glob('wavecal*.txt')
    model_cal_files = glob('continuum_normalization*.txt')
    extraction_files = glob('extraction_*_*.txt')
    spectra_files = glob('flux_fits*.txt')

    ##### ------------------------------------------------------------------ #####
    # Calibrations
    for i in range(len(cal_files)): # Repeat copy of data below
        file_name = str(cal_files[i])
        diagnostic_plots_cals(file_name, flags)
    
    ##### ------------------------------------------------------------------ #####
    # FWHM
    for i in range(len(fwhm_files)):    # First line not commented out
        file_name = str(fwhm_files[i])
        diagnostic_plots_FWHM(file_name, flags)

    ##### ------------------------------------------------------------------ #####
    # Wavelength Calibrations
    star_names = []
    for i in range(len(wave_cal_files)):
        star_names.append(wave_cal_files[i][8:-21])
        with open(wave_cal_files[i], 'r') as f:
            first_line = f.readline()

    unique_names = unique_star_names(star_names)

    for sub in unique_names:
        file_names = [x for x in wave_cal_files if str(sub) in x]
        diagnostic_plots_wavecal(file_names, flags)

    ##### ------------------------------------------------------------------ #####
    # Model Calibrations
    star_names = []
    for i in range(len(model_cal_files)):
        star_names.append(model_cal_files[i][24:-21])
        with open(model_cal_files[i], 'r') as f:
            first_line = f.readline()

    unique_names = unique_star_names(star_names)

    for sub in unique_names:
        file_name = [x for x in model_cal_files if str(sub) in x]
        diagnostic_plots_continuum(file_name[0], flags)

    ##### ------------------------------------------------------------------ #####
    # Extraction
    for i in range(len(extraction_files)):
        file_name = str(extraction_files[i])
        diagnostic_plots_extraction(file_name, flags)
    
    ######------------------------------------------------------------------ #####
    for i in range(len(spectra_files)):
        file_name = str(spectra_files[i])
        diagnostic_plots_spectra(file_name, flags) 
 
    ######------------------------------------------------------------------ #####
    # Merge all pdfs of plots
    #pdfs = glob('*.pdf')
    outfile = PdfFileMerger()

    for f in pdfs:
        outfile.append(open(f, 'rb'))
        os.remove(f)
    
    outfile.write(open('diagnostic_plots.pdf', 'wb'))
    flags.to_csv('diagnostics_flags.csv')


#Run from command line
if __name__ == '__main__':
     diagnostic_now()

