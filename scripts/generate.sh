#!/bin/bash
dir=$1

c=32
while [ $c -le 2048 ]
do
	./equal.exe $c > $dir/$c".in"
	((c=$c*2))
	sleep 1.0
done
