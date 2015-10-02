#!/bin/bash

# Ensures the output of the test will not be truncated.
echo CTEST_FULL_OUTPUT
echo

# 2 or 3 parameters should be provided, exit otherwise
if [ "$#" -le 1 ] || [ "$#" -ge 4 ]
then
    echo "Usage (example): `basename $0` benchRT gamma /tmp/dashboard_2015-10-02_11-37-08/opengate-creatis-dashboard-test"
    exit 1
fi

echo
echo "Current directory:"
pwd
echo

echo "Printing the three parameters for debugging:"
echo $1
echo $2
echo $3
echo
echo "Gate binary location:"
# GATE_BINARY is unset
if [ -z ${GATE_BINARY+x} ]; then
    which Gate
    # if Gate exists, the previous command returns 0 (0 is true)
    if [ `echo $?` -eq 0 ]; then
	GATE_BINARY=`which Gate`
	echo "GATE_BINARY is set to '$GATE_BINARY'"
    # if Gate does not exist, the previous command returns 1: exit the test
    else
	echo "Please provide a valid installation for Gate. Nothing was found by the command 'which Gate'."
	exit 1
    fi
# GATE_BINARY is set from dashboard CMakeLists
else
    echo "GATE_BINARY is set to '$GATE_BINARY'"
fi

echo

# $3 is not empty i.e ${Gate_SOURCE_DIR} is defined i.e. this test is launched from 'make test' or for the nightly dashboard
if [ ! -z ${3+x} ]; then
    echo "This test is launched from 'make test' or for the Nightly dashboard."
    BENCHMARKS_DIRECTORY=$3/benchmarks
# this script should be launched locally, from its containing folder
else
    # Note : ${BASH_SOURCE[0]} contains this script name
    # Tests -ge instead of -eq to include the case of there are backup versions of the script in the current folder.
    if [ `ls | grep ${BASH_SOURCE[0]} | wc -l` -ge 1 ]; then
	echo "This script is launched locally, from its containing folder."
	BENCHMARKS_DIRECTORY=`pwd`
    # exit otherwise
    else
	echo "You are launching gate_run_test.sh from an improper location. Please launch this script locally, from its containing folder."
	exit 1
    fi
fi

echo "Benchmarks directory:"
echo $BENCHMARKS_DIRECTORY
echo
cd $BENCHMARKS_DIRECTORY/$1/

echo "Working directory:"
pwd

echo
echo
echo "--------------------------------------------------------------------------------------------------"
echo "Launching Gate binary on mac/$2.mac."
$GATE_BINARY mac/$2.mac > gate_simulation_log.txt 2>&1

# Testing if there is a Geant4 installation problem
# Exit if any 'G4Exception' is found in gate_simulation_log.txt.
if [ `cat gate_simulation_log.txt | grep G4Exception | wc -l` -ge 1 ]; then
    echo "gate_simulation_log.txt contains 'G4Exception'."
    echo "Please install Geant4 properly by launching a command such as:"
    echo "source GEANT4_INSTALL_PATH/bin/geant4.sh"
    exit 1
fi

echo "Gate binary has finished."
echo "See at the end of this report for the generated output."
echo "--------------------------------------------------------------------------------------------------"
echo
echo

cd reference
echo "----------------------------------------------------"
echo "Reference archive content:"
tar xvzf $1-reference.tgz
# Testing if the archive exists, otherwise explain how to get it
if [ `echo $?` -ge 1 ]; then
    echo "The archive $1-reference.tgz is not present in the folder $BENCHMARKS_DIRECTORY/$1/reference."
    echo "To retrieve it:"
    echo "Go to Gate compilation folder ;"
    echo "Launch 'ccmake .' and set the options 'BUILD_TESTING' and 'GATE_DOWNLOAD_BENCHMARKS_DATA' to ON ;"
    echo "Configure and Generate ;"
    echo "Launch 'make'."
    exit 1
fi
echo "----------------------------------------------------"
cd ..

echo
echo "----------------------------------------------------"
echo "Folder in which diff will be performed contains:"
ls *
echo "----------------------------------------------------"
echo
echo "Performing detailed diff ('diff -s reference output') from folder:"
echo $BENCHMARKS_DIRECTORY/$1/
echo

exit_status_final=0

# Each line of the $2.txt file is splitted (separator is space) and stored in an array 'WORD'
while read -a WORD
do
    echo "${WORD[@]}"

    if [ ${WORD[0]} == "diff" ]; then
	echo "diff: Performing diff -s"
	diff -s reference/${WORD[1]} output/${WORD[1]}
	exit_status_partial=$?
    fi

    if [ ${WORD[0]} == "diff_stat" ]; then
	echo "diff_stat: Performing detailed diff on the 6th first lines of stat-gamma.txt:"
	diff -s <(head -n 6 reference/stat-gamma.txt) <(head -n 6 output/stat-gamma.txt)
	exit_status_partial=$?
    fi

    echo "exit_status_partial is:"
    echo $exit_status_partial
    echo
    exit_status_final=$(($exit_status_final+$exit_status_partial))
done <$BENCHMARKS_DIRECTORY/$1/reference/$2.txt

echo
echo "--------------------------"
echo
echo "Meaning of the exit_status_partial:"
echo "'0': no difference i.e. SUCCESSFUL TEST"
echo "'1': difference in a text file i.e. FAILING TEST"
echo "'2': difference on a binary file or missing file i.e. FAILING TEST"
echo
echo "--------------------------"
echo

echo "--------------------------"
echo "exit_status_final is:"
echo $exit_status_final
echo "--------------------------"

echo
echo
echo "--------------------------------------------------------------------------------------------------"
echo "For debugging information, here is the generated output of the simulation launched with Gate binary."
cat gate_simulation_log.txt
echo "--------------------------------------------------------------------------------------------------"


exit $exit_status_final
