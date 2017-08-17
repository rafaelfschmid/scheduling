#!/bin/bash
prog=$1 #program to test
dir1=$2 #test files dir

for filename in `ls -tr $dir1`; do
	file=$filename
	file=$(echo $file| cut -d'/' -f 3)
	c=$(echo $file| cut -d'.' -f 1)

	echo $c
#	for b in `seq 1 10`; do
		./$prog < $dir1/$filename 
#	done
	echo " "
done

