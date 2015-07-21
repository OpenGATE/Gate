#!/bin/bash
# Ensures the outpur of the test will not be truncated.
echo CTEST_FULL_OUTPUT
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

$GATE_BINARY mac/$2.mac > test_log.txt 2>&1

ls *

cd reference
tar xvzf $1-reference.tgz
cd ..

mkdir excluded_from_test
mv output/BenchAnalyse.C excluded_from_test
mv output/output-gamma-Edep.mhd excluded_from_test
mv output/output-gamma-Edep.raw excluded_from_test
mv output/stat-gamma.txt excluded_from_test
mv reference/benchRT-reference.tgz excluded_from_test
mv reference/benchRT-reference.tgz.md5 excluded_from_test
mv reference/benchRT-reference.tgz.md5-stamp excluded_from_test

echo "Performing diff in folder:"
echo $BENCHMARKS_DIRECTORY/$1/

diff --normal reference output
diff --normal /tmp/dashboard_2015-07-21_12-04-41/opengate-creatis-dashboard-test/benchmarks/benchRT/output/gamma-3d-Edep.mhd output/gamma-3d-Edep.mhd

echo "diff in brief (diff -q)"
diff -q reference output
diff -q /tmp/dashboard_2015-07-21_12-04-41/opengate-creatis-dashboard-test/benchmarks/benchRT/output/gamma-3d-Edep.mhd output/gamma-3d-Edep.mhd

echo "diff in details (diff -s)"
diff -s reference output
diff -s /tmp/dashboard_2015-07-21_12-04-41/opengate-creatis-dashboard-test/benchmarks/benchRT/output/gamma-3d-Edep.mhd output/gamma-3d-Edep.mhd

echo "diff on .raw"
diff /tmp/dashboard_2015-07-21_12-04-41/opengate-creatis-dashboard-test/benchmarks/benchRT/output/gamma-3d-Edep-Squared.raw /tmp/dashboard_2015-07-21_12-04-41/opengate-creatis-dashboard-test/benchmarks/benchRT/output/gamma-3d-Edep.raw
diff -q /tmp/dashboard_2015-07-21_12-04-41/opengate-creatis-dashboard-test/benchmarks/benchRT/output/gamma-3d-Edep-Squared.raw /tmp/dashboard_2015-07-21_12-04-41/opengate-creatis-dashboard-test/benchmarks/benchRT/output/gamma-3d-Edep.raw
diff -s /tmp/dashboard_2015-07-21_12-04-41/opengate-creatis-dashboard-test/benchmarks/benchRT/output/gamma-3d-Edep-Squared.raw /tmp/dashboard_2015-07-21_12-04-41/opengate-creatis-dashboard-test/benchmarks/benchRT/output/gamma-3d-Edep.raw 

echo "diff path"
which diff
