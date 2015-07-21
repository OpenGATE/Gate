#!/bin/bash
echo "Hello Gater!"
pwd
echo $1
echo $2

echo $GATE_BINARY
echo $BENCHMARKS_DIRECTORY

cd $BENCHMARKS_DIRECTORY/$1/
pwd
ls
ls *

$GATE_BINARY mac/$2.mac

ls *

cd reference
tar xvzf /$1-reference.tgz
cd ..

diff reference output
