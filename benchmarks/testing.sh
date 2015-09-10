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
mv output/stat-gamma.txt excluded_from_test
mv output/README excluded_from_test
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
echo
echo "exit_status_folder is:"
echo $exit_status_folder
echo
echo
echo "Performing detailed diff on the 6th first lines of stat-gamma.txt:"
echo
diff -s <(head -n 6 reference/stat-gamma.txt) <(head -n 6 excluded_from_test/stat-gamma.txt)
exit_status_stat=$?
echo
echo
echo "exit_status_stat is:"
echo $exit_status_stat
echo
echo "Meaning of these exit_status:"
echo "'0': no difference i.e. SUCCESSFUL TEST"
echo "'1': missing file or difference in a text file i.e. FAILING TEST"
echo "'2': difference on a binary file i.e. FAILING TEST"



echo
echo
echo "--------------------------------------------------------------------------------------------------"
echo "For debugging information, here is the generated output of the simulation launched with Gate binary:"
less gate_simulation_log.txt
echo "--------------------------------------------------------------------------------------------------"

exit_status_final=$(($exit_status_folder+$exit_status_stat))

echo "exit_status_final=exit_status_folder + exit_status_stat is:"
echo $exit_status_final

exit $exit_status_final
