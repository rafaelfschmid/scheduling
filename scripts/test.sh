#!/bin/bash
prog1=$1 #program to test
dir1=$2 #test files dir
dir2=$3 #result files dir
dir3=$4 #errors files dir

echo $prog1

for filename in `ls -tr $dir1`; do
	file=$filename
	file=$(echo $file| cut -d'/' -f 3)
	c=$(echo $file| cut -d'.' -f 1)
	echo $c".in"

	./$prog1 < $dir1/$filename > $dir2/"test.out"

	if ! cmp -s $dir2/"test.out" $dir2/$c".out"; then
		mkdir -p $dir3
		cat $dir2/"test.out" > $dir3/$prog1"_"$c".out"
		echo "There are something wrong."
		#break;
	else
		echo "Everthing ok."		
	fi
done


