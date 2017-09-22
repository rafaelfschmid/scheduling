#!/bin/bash
prog=$1 #program to run
input=$2 #input files dir

for i in 1 2 4 8 16 32 ; do
	((t = $i * 1024))
	((m = $i * 32))
	echo ${t}x${m}		

	for j in `seq 1 20` ; do
		./$prog $t $m < $input/etc_c_${t}x${m}_hihi_${j}.dat
	done 
	echo " "
done

