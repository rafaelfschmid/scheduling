
times=$1
parsed=$2

for filename in `ls -tr $times`; do
	./parser.exe $times/$filename $parsed/$filename
done

