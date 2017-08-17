#!/bin/bash
input=$1 #arquivos de entrada
time=$2  #caminho dos tempos

for block in 2 4 8 16 32 ; do
	echo "blockbi 2 " $block
  	./../scripts/exec.sh blocksortingbi${block}_2.exe $input > $time/blocksortingbi${block}_2.out 
	./../scripts/exec.sh blocksortingsharedbi${block}_2.exe $input > $time/blocksortingsharedbi${block}_2.out 
done

for block in 4 8 16 32 ; do
 	echo "blockbi 4 " $block
	./../scripts/exec.sh blocksortingbi${block}_4.exe $input > $time/blocksortingbi${block}_4.out
	./../scripts/exec.sh blocksortingsharedbi${block}_4.exe $input > $time/blocksortingsharedbi${block}_4.out 
done

for block in 8 16 32 ; do
 	echo "block 8 " $block
	./../scripts/exec.sh blocksortingbi${block}_8.exe $input > $time/blocksortingbi${block}_8.out
	./../scripts/exec.sh blocksortingsharedbi${block}_8.exe $input > $time/blocksortingsharedbi${block}_8.out 
done

for block in 16 32 ; do
 	echo "blockbi 16 " $block
	./../scripts/exec.sh blocksortingbi${block}_16.exe $input > $time/blocksortingbi${block}_16.out
	./../scripts/exec.sh blocksortingsharedbi${block}_16.exe $input > $time/blocksortingsharedbi${block}_16.out
done

echo "blockbi 32" $block
./../scripts/exec.sh blocksortingbi32_32.exe $input > $time/blocksortingbi32_32.out 
./../scripts/exec.sh blocksortingsharedbi32_32.exe $input > $time/blocksortingsharedbi32_32.out 

