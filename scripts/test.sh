#!/bin/bash
prog1=$1 #program to test
dir1=$2 #test files dir
dir2=$3 #result files dir
dir3=$4 #errors files dir

echo $prog1

for i in 1 2 4 8 16 ; do
	((t = $i * 1024))
	((m = $i * 32))

	for j in `seq 1 20` ; do
		echo etc_c_${t}x${m}_hihi_${j}.dat
		filename=etc_c_${t}x${m}_hihi_${j}.dat

		./$prog1 ${t} ${m} < $dir1/$filename > $dir2/"test.out"

		if ! cmp -s $dir2/"test.out" $dir2/${t}x${m}_${j}".out"; then
			mkdir -p $dir3
			cat $dir2/"test.out" > $dir3/$prog1"_"${t}x${m}_${j}".out"
			echo "There are something wrong."
			#break;
		else
			echo "Everthing ok."		
		fi
	
	done 
	echo " "
done



