
input=$1 #arquivos de entrada
time=$2 #arquivos de tempo

./../scripts/execblockuni.sh $input $time
./../scripts/execrowuni.sh $input $time
./../scripts/execblockbi.sh $input $time
./../scripts/execrowbi.sh $input $time
#./scripts/exectri.sh $input $time

