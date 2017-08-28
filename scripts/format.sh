
output=$1
parsedir=$2

echo "" > ../times/$output.times

for filename in `ls -tr $parsedir`; do
	echo $filename >> ../times/$output.times
	cat ${parsedir}/${filename} >> ../times/$output.times
	echo " " >> ../times/$output.times
done

