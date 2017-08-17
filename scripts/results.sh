#!/bin/bash
prog1=$1 #program to generate results
dir1=$2 #input files dir
dir2=$3 #result files dir

for filename in `ls -tr $dir1`; do
	file=$filename
	file=$(echo $file| cut -d'/' -f 3)
	c=$(echo $file| cut -d'.' -f 1)
	echo $c".in"

	./$prog1 < $dir1/$filename > $dir2/$c".out"
done
