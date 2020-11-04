times=$1
#progs=(hybridmerge hybridradix gpumergeblocks gpuradixblocks gpumintaskcol minsort) #min)
progs=(hybridmerge hybridradix gpumergeblocks gpuradixblocks minsort) #min)

for p in `seq 0 4` ; do
	echo "" > ${times}/${progs[p]}.time
done

t=1024
while [ $t -le 65536 ]
do
	m=2
	((dim=$t*$m))
	while [[ $m -lt $t && dim -lt 536870912 ]] 
	do
		echo ${t}x${m}

		for p in `seq 0 4` ; do
			echo ${t}x${m} >> ${times}/${progs[p]}.time
		done

		for j in `seq 1 3` ; do
			./../scripts/generator ${t} ${m} 1 1 0 > gen.log

			for p in `seq 0 4` ; do
				for k in `seq 1 5` ; do
					./${progs[p]}.exe < B.u_c_hihi >> ${times}/${progs[p]}.time
				done
			done
		done

		for p in `seq 0 4` ; do
			echo " " >> ${times}/${progs[p]}.time
		done
		((m=$m*2))
		((dim=$t*$m))
	done
	((t=$t*2))
done
