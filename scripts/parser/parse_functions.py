import os
#import math
import config_executor

def parse_strategies(dirFiles):
	vecMap = {}
	entries = os.scandir(dirFiles)
	for entry in entries:
		if(not entry.is_file()):
			continue
		
		if(not entry.name.endswith('.time')):
			continue

		print("Parsing file: " + entry.path)
		f = open(entry.path)

		lines = f.readlines()
		f.close()

		mapFile = {}
		i = 1
		while (i < len(lines)):
			length = int(lines[i].strip())
			seg = int(lines[i+1].strip())
			i = i+2

			s=0.0
			count = 0
			while (i < len(lines) and lines[i] != " \n"):
				s += float(lines[i].strip())
				i += 1
				count += 1
				
			s = s/count

			if(not seg in mapFile):
				mapFile[seg] = {}

			mapFile[seg][length] = s
			i += 1

		vecMap[entry.name.split('.')[0]] = mapFile

	return vecMap


def calc_best_strategy(vecMap):
	print("Caculating best strategies...")
	
	bestStrategies = {}
	bestValues = {}

	r = config_executor.restrictions['global']
	
	seg = r.segInf
	while(seg <= r.segSup):

		bestStrategies[seg] = {}
		bestValues[seg] = {}

		length = r.lenInf
		while(length <= r.lenSup):
			if((length/seg > 1) and (length*seg <= 268435456)):
				minValue = float("inf") #vecMap['bbsegsort'][seg][length]
				minChoice = '--'
			
				for strategy in vecMap:

					if(not seg in vecMap[strategy]):
						continue

					if(not length in vecMap[strategy][seg]):
						continue
					
					if(vecMap[strategy][seg][length] < minValue):
						minValue = vecMap[strategy][seg][length]
						minChoice = strategy

				bestValues[seg][length] = minValue
				bestStrategies[seg][length] = minChoice
			length *= 2

		seg *= 2

	return bestStrategies, bestValues

def create_output_dir(outputDir):
	if (not os.path.exists(outputDir)):
		os.makedirs(outputDir, exist_ok=True)
		print("Creating output directories: " + outputDir)

def delete_output_dir(outputDir):
	import shutil
	if (os.path.exists(outputDir)):
		shutil.rmtree(outputDir)
		print("Deleting output directory: " + outputDir)

def create_output_dir(outputDir):
	if (not os.path.exists(outputDir)):
		os.makedirs(outputDir, exist_ok=True)
		print("Creating output directories: " + outputDir)

def removing_existing_file(filename):
	filename = filename.replace('//','/')
	if os.path.exists(filename):
		os.remove(filename)
		print("Removing file: " + filename)


def create_csv(bestStrategies, csvFile):
	removing_existing_file(csvFile)
	print("Creating csv file: " + csvFile)
	f = open(csvFile, 'w')

	caption = "Teste"
	
	f.write(caption)
	
	r = config_executor.restrictions['global']
	length = r.lenInf
	while length <= r.lenSup:
		f.write(";"+str(length))
		length *= 2
	f.write("\n")

	seg=r.segInf
	while seg <= r.segSup:
		length = r.lenInf
		#f.write(str(int(math.log(seg,2))))
		f.write(str(seg))

		while length <= r.lenSup:
			if(length/seg <= 1):
				f.write(";--")
			else:
				if(length in bestStrategies[seg]):
					f.write(";" + bestStrategies[seg][length])
				else:
					f.write(";--")
			length *= 2
		
		seg *= 2
		f.write("\n")

	f.close()


def create_csv_all(vecMap, csvFile):
	removing_existing_file(csvFile)
	print("Creating csv file: " + csvFile)
	f = open(csvFile, 'w')

	r = config_executor.restrictions['global']

	for strategy in vecMap:
		f.write(strategy)

		length = r.lenInf
		while length <= r.lenSup:
			f.write(";"+str(length))
			length *= 2
		f.write("\n")

		seg=r.segInf
		while seg <= r.segSup:
			length = r.lenInf
			#f.write(str(int(math.log(seg,2))))
			f.write(str(seg))

			while length <= r.lenSup:
				if(length/seg <= 1):
					f.write(";--")
				else:
					if(length in vecMap[strategy][seg]):
						f.write(";" + str(vecMap[strategy][seg][length]))#.replace('.', ','))
					else:
						f.write(";--")
				length *= 2
			
			seg *= 2
			f.write("\n")

		f.write("\n")

	f.close()


def calc_scurves(vecMap, bestValues):

	scurves = {}
	for strategy in vecMap:

		c = []
		for seg in vecMap[strategy]:
			for length in vecMap[strategy][seg]:
				if(length in bestValues[seg]):
					c.append(vecMap[strategy][seg][length]/bestValues[seg][length])

		scurves[strategy] = sorted(c)

	return scurves


def create_scurve(scurves, scurveFile):
	removing_existing_file(scurveFile)
	print("Creating scurve file: " + scurveFile)
	
	import matplotlib.pylab as plt

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.set_ylim([0, 9])

	for i in range(1,10,1):
		plt.axhline(y = i, color ="black", linestyle ="--", linewidth=0.1, dashes=(5, 10)) 

	for strategy in scurves:
		length = len(scurves[strategy])
		middle = int(length/2-1)
		markers_on = [0, middle, length-1]
		plt.plot(scurves[strategy], config_executor.symbols[strategy], color=config_executor.colors[strategy], markevery=markers_on, label=config_executor.abbreviations[strategy], dashes=(5, 5))

	plt.ylabel('Normalized Times')
	#plt.locator_params(nbins=3)
	plt.xticks([]) # hide axis x
	plt.legend() # show line names
	
	plt.savefig(scurveFile, format='eps')
	#plt.show()
