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
echo $GATE_BINARY
echo
echo "Benchmarks directory:"
echo $BENCHMARKS_DIRECTORY
echo
cd $BENCHMARKS_DIRECTORY/$1/
echo "Working directory:"
pwd

echo
echo
echo --------------------------------------------------------------------------------------------------
echo "Launching Gate binary on mac/$2.mac."
$GATE_BINARY mac/$2.mac > gate_simulation_log.txt 2>&1
echo "Gate binary has finished."
echo "See at the end of this report for the generated output."
echo --------------------------------------------------------------------------------------------------
echo
echo

echo "Reference folder content:"
cd reference
echo "--------------------------------"
tar xvzf $1-reference.tgz
echo "--------------------------------"
cd ..

mkdir excluded_from_test
mv output/BenchAnalyse.C excluded_from_test
mv output/output-gamma-Edep.mhd excluded_from_test
mv output/output-gamma-Edep.raw excluded_from_test
mv output/stat-gamma.txt excluded_from_test
mv reference/benchRT-reference.tgz excluded_from_test
mv reference/benchRT-reference.tgz.md5 excluded_from_test
mv reference/benchRT-reference.tgz.md5-stamp excluded_from_test

echo
echo "Directory content before diff:"
echo "--------------------------------"
ls *
echo "--------------------------------"
echo
echo "Performing detailed diff ('diff -s reference output') in folder:"
echo $BENCHMARKS_DIRECTORY/$1/
echo
diff -s reference output
exit_status=$?
echoecho
echo "exit_status is 
echo $exit_status
echo
echo "Meaning:"
echo "'0': no difference i.e. SUCCESSFUL TEST"
echo "'1': missing file or difference in a text file i.e. FAILING TEST
echo "'2': difference on a binary file i.e. FAILING TEST"



echo
echo
echo --------------------------------------------------------------------------------------------------
echo "For debugging information, here is the generated output of the simulation launched with Gate binary:"
less gate_simulation_log.txt
echo --------------------------------------------------------------------------------------------------


exit $exit_status
