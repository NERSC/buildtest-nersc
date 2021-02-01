#!/bin/bash
#SBATCH -N 1
#SBATCH --reservation knl_fp_test
#SBATCH -C knl
#SBATCH -t 10:00
module swap craype-haswell craype-mic-knl
module load upcxx/2020.3.0
upcxx -g --network=aries src/hello_upcxx.cpp -o a.out
count=0
numthreads=272
while [ $count -le 2 ]; do
  upcxx-run -N 1 -n $numthreads ./a.out > outputfile
  linecount=`cat outputfile | wc -l`
  if [ $linecount -eq $numthreads ]; then
    echo "MATCH"
  else
    echo "FAIL"
  fi
  ((count++))
done
