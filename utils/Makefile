BITS=12
SEGS=1048576

all: generate

generate: gen_equal.cpp
	g++ -O3 gen_equal.cpp -o equal.exe -DEXP_BITS_SIZE=$(BITS) -DRAND

parser: parser.cpp
	g++ -O3 parser.cpp -o parser.exe -std=c++11

parsermean: parsermean.cpp
	g++ -O3 parsermean.cpp -o parsermean.exe -std=c++11


