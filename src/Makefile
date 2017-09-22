libs=-I"../lib"
#caching global memory
#caching=-Xptxas -dlcm=ca
params=-std=c++11 --expt-extended-lambda -lcuda
gen=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
segment=2

all: min minsort hybridmerge hybridradix gpuradix gpumerge gpumintaskrow gpumintaskcol

min: min.cu
	nvcc $(gen) -O3 min.cu -o min.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

minsort: minsort.cu
	nvcc $(gen) -O3 minsort.cu -o minsort.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

hybridmerge: hybridmerge.cu
	nvcc $(gen) -O3 hybridmerge.cu -o hybridmerge.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

hybridradix: hybridradix.cu
	nvcc $(gen) -O3 hybridradix.cu -o hybridradix.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

gpuradix: gpuradix.cu
	nvcc $(gen) gpuradix.cu -o gpuradix.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

gpumerge: gpumerge.cu
	nvcc $(gen) gpumerge.cu -o gpumerge.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

gpumintaskrow: gpumintaskrow.cu
	nvcc $(gen) gpumintaskrow.cu -o gpumintaskrow.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

gpumintaskcol: gpumintaskcol.cu
	nvcc $(gen) gpumintaskcol.cu -o gpumintaskcol.exe $(libs) $(params) -DELAPSED_TIME=$(TIME) -DBLOCK_SIZE=512

gpumintaskstruct: gpumintaskstruct.cu
	nvcc $(gen) gpumintaskstruct.cu -o gpumintaskstruct.exe $(libs) $(params) -DELAPSED_TIME=$(TIME) -DBLOCK_SIZE=512

gpuminmachine: gpuminmachine.cu
	nvcc $(gen) gpuminmachine.cu -o gpuminmachine.exe $(libs) $(params) -DELAPSED_TIME=$(TIME) -DBLOCK_SIZE=32

gputranspose: gputranspose.cu
	nvcc $(gen) gputranspose.cu -o gputranspose.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

generate: gen_equal.cpp
	g++ -O3 gen_equal.cpp -o equal.exe -DEXP_BITS_SIZE=$(BITS) -DRAND

parser: parser.cpp
	g++ -O3 parser.cpp -o parser.exe -std=c++11

parsermean: parsermean.cpp
	g++ -O3 parsermean.cpp -o parsermean.exe -std=c++11

