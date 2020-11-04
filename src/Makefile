libs=-I"../lib" -I"$(CUDA_HOME)/include"
#caching global memory
#caching=-Xptxas -dlcm=ca
ccompiler=-ccbin g++-4.9
params=-std=c++11 --expt-extended-lambda -lcuda -D_FORCE_INLINES
gen=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 #-gencode arch=compute_60,code=sm_60

all: min minsort hybridmerge hybridradix gpuradix gpumerge gpumintaskcol gpumergeblocks gpuradixblocks

min: min.cu
	nvcc $(ccompiler) $(gen) -O3 min.cu -o min.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

minsort: minsort.cu
	nvcc $(ccompiler) $(gen) -O3 minsort.cu -o minsort.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

hybridmerge: hybridmerge.cu
	nvcc $(ccompiler) $(gen) -O3 hybridmerge.cu -o hybridmerge.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

hybridradix: hybridradix.cu
	nvcc $(ccompiler) $(gen) -O3 hybridradix.cu -o hybridradix.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

gpuradix: gpuradix.cu
	nvcc $(ccompiler) $(gen) gpuradix.cu -o gpuradix.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

gpuradixblocks: gpuradixblocks.cu
	nvcc $(ccompiler) $(gen) gpuradixblocks.cu -o gpuradixblocks.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

gpumerge: gpumerge.cu
	nvcc $(ccompiler) $(gen) gpumerge.cu -o gpumerge.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

gpumergeblocks: gpumergeblocks.cu
	nvcc $(ccompiler) $(gen) gpumergeblocks.cu -o gpumergeblocks.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

gpumintaskrow: gpumintaskrow.cu
	nvcc $(ccompiler) $(gen) gpumintaskrow.cu -o gpumintaskrow.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)

gpumintaskcol: gpumintaskcol.cu
	nvcc $(ccompiler) $(gen) gpumintaskcol.cu -o gpumintaskcol.exe $(libs) $(params) -DELAPSED_TIME=$(TIME) -DBLOCK_SIZE=512

gpumintaskstruct: gpumintaskstruct.cu
	nvcc $(ccompiler) $(gen) gpumintaskstruct.cu -o gpumintaskstruct.exe $(libs) $(params) -DELAPSED_TIME=$(TIME) -DBLOCK_SIZE=512

gpuminmachine: gpuminmachine.cu
	nvcc $(gen) gpuminmachine.cu -o gpuminmachine.exe $(libs) $(params) -DELAPSED_TIME=$(TIME) -DBLOCK_SIZE=32

gputranspose: gputranspose.cu
	nvcc $(gen) gputranspose.cu -o gputranspose.exe $(libs) $(params) -DELAPSED_TIME=$(TIME)


