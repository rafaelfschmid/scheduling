
times=$1
parsed=$2

for filename in `ls -tr $times`; do
	./parsermean.exe $times/$filename $parsed/$filename
done

