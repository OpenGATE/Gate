#!/bin/bash

# Ensures the output of the test will not be truncated.
echo CTEST_FULL_OUTPUT
echo
echo "Current directory:"
pwd
echo

echo "Printing the two parameters for debugging:"
echo $1
echo $2
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
echo "Gate binary has finished."
echo "See at the end of this report for the generated output."
echo "--------------------------------------------------------------------------------------------------"
echo
echo

cd reference
echo "----------------------------------------------------"
echo "Reference folder content:"
tar xvzf $1-reference.tgz
echo "----------------------------------------------------"
cd ..

mkdir excluded_from_test
mv output/BenchAnalyse.C excluded_from_test
mv output/output-gamma-Edep.mhd excluded_from_test
mv output/output-gamma-Edep.raw excluded_from_test
mv output/stat-gamma.txt excluded_from_test/stat-gamma_output.txt
mv output/README excluded_from_test
mv reference/stat-gamma.txt excluded_from_test/stat-gamma_reference.txt
mv reference/README excluded_from_test
mv reference/benchRT-reference.tgz excluded_from_test
mv reference/benchRT-reference.tgz.md5 excluded_from_test
mv reference/benchRT-reference.tgz.md5-stamp excluded_from_test

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
echo "If it ends by 'G4Exception', please install Geant4 properly by launching a command such as:"
echo "source GEANT4_INSTALL_PATH/bin/geant4.sh"
cat gate_simulation_log.txt
echo "--------------------------------------------------------------------------------------------------"


exit $exit_status_final
