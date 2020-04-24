
export ROOTPATH=$1
export ROOTSYS=$ROOTPATH/bin/root-config
export PATH=$ROOTSYS:$PATH
export THISROOT=$ROOTPATH/bin/thisroot.sh
echo $THISROOT
source $THISROOT
binFolder="$PWD"
testFolder="`dirname \"$0\"`"
cd $testFolder
export TERM="xterm"
$binFolder/Gate mac/carbon.mac
if [ $? -ne 0 ]; then
    exit 1
fi
cd $binFolder


