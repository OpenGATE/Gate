#!/bin/bash

if [ $# -ne 1 ]
then
  echo "Usage: `basename $0` FilePath"
  exit 1
fi

if test "$(uname)" = "Darwin"
then
  sha512 -q $1 >${1}.sha512
else
  echo $(sha512sum $1 | cut -f 1-1 -d ' ') >${1}.sha512
fi
git add ${1}.sha512

echo "/$(basename $1).sha512-stamp" >> $(dirname $1)/.gitignore
echo "/$(basename $1)"           >> $(dirname $1)/.gitignore
git add $(dirname $1)/.gitignore

rm $1

echo "GateAddBenchmarkData(\"DATA{${1#$(dirname $0)/}}\")" >> CMakeLists.txt

echo "Don't forget to upload data in https://data.kitware.com/#collection/5be2bffb8d777f21798e28bb/folder/5be2c0298d777f21798e28d3"
