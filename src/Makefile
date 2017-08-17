libs=-I"../../lib"
#caching global memory
#caching=-Xptxas -dlcm=ca
params=-std=c++11 --expt-extended-lambda -lcuda
gen=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
segment=2

all: minmin maxmin

minmin: minmin.cpp
	g++ -O3 minmin.cpp -o minmin.exe -std=c++11

maxmin: maxmin.cpp
	g++ -O3 maxmin.cpp -o maxmin.exe -std=c++11

gpuminmin: gpuminmin.cu
	for block in 2 4 8 16 32 ; do \
		nvcc $(gen) gpuminmin.cu -o gpuminmin$${block}.exe $(libs) $(params) -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DBLOCK_SIZE=$$block ; \
	done

generate: gen_equal.cpp
	g++ -O3 gen_equal.cpp -o equal.exe -DEXP_BITS_SIZE=$(BITS) -DRAND

parser: parser.cpp
	g++ -O3 parser.cpp -o parser.exe -std=c++11

