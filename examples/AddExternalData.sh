#!/bin/bash

if [ $# -ne 1 ]
then
  echo "Usage: `basename $0` FilePath"
  exit 1
fi

if test "$(uname)" = "Darwin"
then
  md5 -q $1 >${1}.md5
else
  echo $(md5sum $1 | cut -f 1-1 -d ' ') >${1}.md5
fi
echo "/$(basename $1).md5-stamp" >> $(dirname $1)/.gitignore
echo "/$(basename $1)"           >> $(dirname $1)/.gitignore
rm $1

