#!/bin/bash
prog1=$1 #program to generate results
dir1=$2 #input files dir
dir2=$3 #result files dir

for i in 1 2 4 8 16 ; do
	((t = $i * 1024))
	((m = $i * 32))

	for j in `seq 1 20` ; do
		./${prog1} $t $m < ${dir1}/etc_c_${t}x${m}_hihi_${j}.dat > ${dir2}/${t}x${m}_${j}.out
		echo etc_c_${t}x${m}_hihi_${j}.dat
	done 
	echo " "
done


