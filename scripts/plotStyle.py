#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-06-28 (Y/M/D) at 18:17
@author: Anup K. Prasad
email: anupkprasad121@gmail.com
"""
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

def setPlotStyle():
    # Restore matplotlib defaults
    plt.rcdefaults()
    plt.rcParams["figure.dpi"] = 600

    # Font and line settings for single-column plots
    rc_fonts = {
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1,
    }

    mpl.rcParams.update(rc_fonts)
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"

    # Tick width and size settings
    mpl.rcParams['xtick.major.size'] = 1.5
    mpl.rcParams['xtick.major.width'] = 1
    mpl.rcParams['xtick.minor.size'] = 1
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.major.size'] = 1.5
    mpl.rcParams['ytick.major.width'] = 1
    mpl.rcParams['ytick.minor.size'] = 1
    mpl.rcParams['ytick.minor.width'] = 1

    # Load custom fonts (Arial from external directory)
    font_dirs = ['/home/anup/myScripts/utils/arialFonts/']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    # Set font family and math text settings
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    plt.rcParams['mathtext.rm'] = 'Arial'

