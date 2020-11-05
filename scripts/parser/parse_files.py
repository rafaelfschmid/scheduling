#!/usr/bin/python3.6

import os
import sys
#import calc_local_functions as calc_functions
import parse_functions
#import config_generator

#if(len(sys.argv) == 2):
	#if(sys.argv[1] == "clear"):
		#parse_functions.delete_output_dir("output")
#else:	
dirFiles = sys.argv[1]
pathFiles = 'times/' + dirFiles
print('Processing all machines into directory: ', pathFiles, '".')
vecMap = parse_functions.parse_strategies(pathFiles)
bestStrategies, bestValues = parse_functions.calc_best_strategy(vecMap)



csvAllFiles = "output/csv/" + dirFiles + "All.txt"
parse_functions.create_output_dir("output/csv/")
parse_functions.create_csv_all(vecMap, csvAllFiles)

csvFile = "output/csv/" + dirFiles + ".csv"
parse_functions.create_output_dir("output/csv/")
parse_functions.create_csv(bestStrategies, csvFile)

#texFile = "output/tex/" + dirFiles + ".tex"
#create_output_dir("output/tex/")
#gen_functions.create_tex(bestStrategies, texFile, machine.upper())

scurveFile = "output/scurves/" + dirFiles + ".eps"
parse_functions.create_output_dir("output/scurves/")
scurves = parse_functions.calc_scurves(vecMap, bestValues)
parse_functions.create_scurve(scurves, scurveFile)