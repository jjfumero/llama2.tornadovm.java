logFolder="log_$(date +"%F_%H_%M_%S")"

mkdir -p $logFolder
## Java
bash butil.sh -v java stories15M.bin 2> $logFolder/log15_java
bash butil.sh -v java stories42M.bin 2> $logFolder/log42_java
bash butil.sh -v java stories110M.bin 2> $logFolder/log110_java

## Level Zero Device 0
bash butil.sh -v levelzero -d 0 stories15M.bin 2> $logFolder/log15_levelzero_0
bash butil.sh -v levelzero -d 0 stories42M.bin 2> $logFolder/log42_levelzero_0
bash butil.sh -v levelzero -d 0 stories110M.bin 2> $logFolder/log110_levelzero_0

## Level Zero Device 1
bash butil.sh -v levelzero -d 1 stories15M.bin 2> $logFolder/log15_levelzero_1
bash butil.sh -v levelzero -d 1 stories42M.bin 2> $logFolder/log42_levelzero_1
bash butil.sh -v levelzero -d 1 stories110M.bin 2> $logFolder/log110_levelzero_1


## TornadoVM Device 0
bash butil.sh -v tornadovm -d 0 stories15M.bin 2> $logFolder/log15_tornadovm_0
bash butil.sh -v tornadovm -d 0 stories42M.bin 2> $logFolder/log42_tornadovm_0
bash butil.sh -v tornadovm -d 0 stories110M.bin 2> $logFolder/log110_tornadovm_0

## Level Zero Device 1
bash butil.sh -v tornadovm -d 1 stories15M.bin 2> $logFolder/log15_tornadovm_1
bash butil.sh -v tornadovm -d 1 stories42M.bin 2> $logFolder/log42_tornadovm_1
bash butil.sh -v tornadovm -d 1 stories110M.bin 2> $logFolder/log110_tornadovm_1

