times=$1
progs=(hybridmerge hybridradix gpumerge gpuradix gpumintaskcol minsort min)

for p in `seq 0 6` ; do
	echo "" > ${times}/${progs[p]}.time
done

for i in 1 2 4 8 16 32 64 128 ; do
	((t = $i * 1024))
	((m = $i * 32))

	for j in `seq 1 10` ; do
		./../scripts/generator ${t} ${m} 1 1 0 > gen.log

		echo ${t}x${m}_${j}
		for p in `seq 0 6` ; do
			echo ${t}x${m}_${j} >> ${times}/${progs[p]}.time

			for k in `seq 1 10` ; do
				./${progs[p]}.exe < B.u_c_hihi >> ${times}/${progs[p]}.time
			done
			echo " " >> ${times}/${progs[p]}.time
		done
	done
#	echo " "
done
