#!/bin/bash
input=$1 #arquivos de entrada
time=$2  #caminho dos tempos

for block in 2 4 8 16 32 64 128 256 512 1024; do
	echo "blockuni 2 " $block
	./../scripts/exec.sh blocksortinguni${block}_2.exe $input > $time/blocksortinguni${block}_2.out
 	./../scripts/exec.sh blocksortingshareduni${block}_2.exe $input > $time/blocksortingshareduni${block}_2.out 
done

for block in 4 8 16 32 64 128 256 512 1024; do
	echo "blockuni 4 " $block
	./../scripts/exec.sh blocksortinguni${block}_4.exe $input > $time/blocksortinguni${block}_4.out 
	./../scripts/exec.sh blocksortingshareduni${block}_4.exe $input > $time/blocksortingshareduni${block}_4.out 

done

for block in 8 16 32 64 128 256 512 1024; do
	echo "blockuni 8 " $block
	./../scripts/exec.sh blocksortinguni${block}_8.exe $input > $time/blocksortinguni${block}_8.out 
	./../scripts/exec.sh blocksortingshareduni${block}_8.exe $input > $time/blocksortingshareduni${block}_8.out 

done

for block in 16 32 64 128 256 512 1024; do
	echo "blockuni 16 " $block
	./../scripts/exec.sh blocksortinguni${block}_16.exe $input > $time/blocksortinguni${block}_16.out 
	./../scripts/exec.sh blocksortingshareduni${block}_16.exe $input > $time/blocksortingshareduni${block}_16.out 

done

for block in 32 64 128 256 512 1024; do
	echo "blockuni 32 " $block
	./../scripts/exec.sh blocksortinguni${block}_32.exe $input > $time/blocksortinguni${block}_32.out 
	./../scripts/exec.sh blocksortingshareduni${block}_32.exe $input > $time/blocksortingshareduni${block}_32.out 

done

for block in 64 128 256 512 1024; do
	echo "blockuni 64 " $block
	./../scripts/exec.sh blocksortinguni${block}_64.exe $input > $time/blocksortinguni${block}_64.out 
	./../scripts/exec.sh blocksortingshareduni${block}_64.exe $input > $time/blocksortingshareduni${block}_64.out 

done

for block in 128 256 512 1024; do
	echo "blockuni 128 " $block
	./../scripts/exec.sh blocksortinguni${block}_128.exe $input > $time/blocksortinguni${block}_128.out 
	./../scripts/exec.sh blocksortingshareduni${block}_128.exe $input > $time/blocksortingshareduni${block}_128.out 

done

for block in 256 512 1024; do
	echo "blockuni 256 " $block
	./../scripts/exec.sh blocksortinguni${block}_256.exe $input > $time/blocksortinguni${block}_256.out 
	./../scripts/exec.sh blocksortingshareduni${block}_256.exe $input > $time/blocksortingshareduni${block}_256.out 

done

for block in 512 1024; do
	echo "blockuni 512 " $block
	./../scripts/exec.sh blocksortinguni${block}_512.exe $input > $time/blocksortinguni${block}_512.out 
	./../scripts/exec.sh blocksortingshareduni${block}_512.exe $input > $time/blocksortingshareduni${block}_512.out 

done

for block in 1024; do
	echo "blockuni 1024 " $block
	./../scripts/exec.sh blocksortinguni${block}_1024.exe $input > $time/blocksortinguni${block}_1024.out 
	./../scripts/exec.sh blocksortingshareduni${block}_1024.exe $input > $time/blocksortingshareduni${block}_1024.out 

done


