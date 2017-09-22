
for i in 16 32 ; do
	((t = $i * 1024))
	((m = $i * 32))
	echo ${t}x${m}          

	for j in `seq 4 20` ; do
		wget -P ~/cuda-workspace/inputs/schedule -c https://par-cga-sched.gforge.uni.lu/instances/etc/etc_c_${t}x${m}_hihi_${j}.dat.gz
	done
	echo " "
done



