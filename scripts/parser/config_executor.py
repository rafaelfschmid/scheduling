from typing import NamedTuple

class Restriction(NamedTuple):
	segInf: int
	segSup: int
	lenInf: int
	lenSup: int

restrictions = {}
restrictions['global'] 		= Restriction(segInf=2,segSup=8192,lenInf=1024,lenSup=65536)


# Abbreviations of each strategy, used in csv and scurve
abbreviations = {'hybridmerge':'HM','hybridradix':'HR','gpuradixblocks':'GR','gpumergeblocks':'GM','minsort':'MS', '--':'--'}

# Symbols to be plotted in scurve
symbols = {'hybridmerge':'o--','hybridradix':'*--','gpuradixblocks':'v--','gpumergeblocks':'d--','minsort':'P--'}

# Colors of each strategy
colors = {'hybridmerge':'green','hybridradix':'blue','gpuradixblocks':'red','gpumergeblocks':'purple','minsort':'brown'}