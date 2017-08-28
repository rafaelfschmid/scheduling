
input=$1 #arquivos de entrada
time=$2 #arquivos de tempo

./../scripts/exec.sh min.exe 		$input > $time/min.time
./../scripts/exec.sh minsort.exe 	$input > $time/minsort.time
./../scripts/exec.sh hybridmerge.exe 	$input > $time/hybridmerge.time
./../scripts/exec.sh hybridradix.exe 	$input > $time/hybridradix.time
./../scripts/exec.sh gpumerge.exe 	$input > $time/gpumerge.time
./../scripts/exec.sh gpuradix.exe 	$input > $time/gpuradix.time


