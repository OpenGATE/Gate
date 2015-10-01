#!/bin/bash

# Ensures the output of the test will not be truncated.
echo CTEST_FULL_OUTPUT
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

# Testing if there is Geant4 installation problem
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
    echo "Launch make."
    exit 1
fi
echo "----------------------------------------------------"
cd ..

while read line
do
    echo "$ line \ n"
done <$BENCHMARKS_DIRECTORY/$1/reference/$2.txt

echo
echo "----------------------------------------------------"
echo "Folder in which diff will be performed contains:"
ls *
echo "----------------------------------------------------"
echo
echo "Performing detailed diff ('diff -s reference output') in folder:"
echo $BENCHMARKS_DIRECTORY/$1/
echo
diff -s reference output
exit_status_folder=$?
echo
echo "exit_status_folder is:"
echo $exit_status_folder
echo
echo "--------------------------"
echo
echo "Performing detailed diff on the 6th first lines of stat-gamma.txt:"
echo
diff -s <(head -n 6 excluded_from_test/stat-gamma_reference.txt) <(head -n 6 excluded_from_test/stat-gamma_output.txt)
exit_status_stat=$?
echo
echo "exit_status_stat is:"
echo $exit_status_stat
echo
echo "--------------------------"
echo
echo "Meaning of these partial exit_status:"
echo "'0': no difference i.e. SUCCESSFUL TEST"
echo "'1': missing file or difference in a text file i.e. FAILING TEST"
echo "'2': difference on a binary file i.e. FAILING TEST"
echo
echo "--------------------------"
echo
exit_status_final=$(($exit_status_folder+$exit_status_stat))
echo "exit_status_final = exit_status_folder + exit_status_stat is:"
echo $exit_status_final


echo
echo
echo "--------------------------------------------------------------------------------------------------"
echo "For debugging information, here is the generated output of the simulation launched with Gate binary."
cat gate_simulation_log.txt
echo "--------------------------------------------------------------------------------------------------"


exit $exit_status_final
