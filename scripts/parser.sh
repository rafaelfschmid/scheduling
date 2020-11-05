
times=$1
outputdir=$2
#machine=$3

file=${times}".parsed"
echo $file

if test -s $outputdir/$file 
then
	mv $outputdir/$file  $outputdir/${file}".bkp"
fi

./parserstd.exe $outputdir/$file "$(ls -d  "$PWD"/$times/*)"


