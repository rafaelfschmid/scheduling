#!/bin/bash
input=$1 #input files dir

for i in 1 2 4 8 16 32 ; do
	((t = $i * 1024))
	((m = $i * 32))
	#echo ${t}x${m}		

	for j in `seq 1 20` ; do
		file=$input/etc_c_${t}x${m}_hihi_${j}.dat

		if !(test -s $file)
		then
			echo $file "NOT FOUND"
		fi

	done 
done

