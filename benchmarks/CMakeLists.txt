
# to upload in http://midas3.kitware.com/midas/community/28

GateAddBenchmarkData("DATA{benchPET/benchPET.pdf}")
GateAddBenchmarkData("DATA{benchPET/benchmarkPET-GATE-Reference.gif}")
GateAddBenchmarkData("DATA{benchPET/benchmarkPET-fGATE.gif}")
GateAddBenchmarkData("DATA{benchPET/benchmarkPET.gif}")

# GateAddBenchmarkData("DATA{benchRT/output/Config-gamma-Edep.hdr}")
# GateAddBenchmarkData("DATA{benchRT/output/Config-gamma-Edep.img}")
# GateAddBenchmarkData("DATA{benchRT/output/Config-proton-Edep.hdr}")
# GateAddBenchmarkData("DATA{benchRT/output/Config-proton-Edep.img}")
# GateAddBenchmarkData("DATA{benchRT/output/benchmarkRT.gif}")

GateAddBenchmarkData("DATA{benchSPECT/SPECT-reference-results.gif}")
GateAddBenchmarkData("DATA{benchSPECT/resultsSPECTfgate.gif}")
GateAddBenchmarkData("DATA{benchSPECT/benchSPECTresult.pdf}")

GateAddBenchmarkData("DATA{benchVoxelRegular25x25x25/bench.gif}")
GateAddBenchmarkData("DATA{benchVoxelRegular25x25x25/maps/corps_cubique.i33}")

GateAddBenchmarkData("DATA{benchImageNavigators/pet/data/brain_phantom.i33}")
GateAddBenchmarkData("DATA{benchImageNavigators/pet/data/brain_phantom.raw}")
GateAddBenchmarkData("DATA{benchImageNavigators/rt/data/patient-2mm.raw}")

GateAddBenchmarkData("DATA{benchImaging/data/brain_phantom.i33}")

SET(GATE_BINARY ${Gate_BINARY_DIR}/Gate) 
if(BUILD_TESTING)
  GateAddBenchmarkData("DATA{benchRT/reference/benchRT_gamma-reference.tgz}")
  ADD_TEST(NAME benchRT_gamma
    COMMAND /bin/bash  ${Gate_SOURCE_DIR}/benchmarks/gate_run_test.sh benchRT gamma ${Gate_SOURCE_DIR} ${GATE_BINARY})
endif(BUILD_TESTING)

if(BUILD_TESTING)
  GateAddBenchmarkData("DATA{benchRT/reference/benchRT_proton-reference.tgz}")
  ADD_TEST(NAME benchRT_proton
    COMMAND /bin/bash  ${Gate_SOURCE_DIR}/benchmarks/gate_run_test.sh benchRT proton ${Gate_SOURCE_DIR} ${GATE_BINARY})
endif(BUILD_TESTING)

if(BUILD_TESTING)
  GateAddBenchmarkData("DATA{benchRT/reference/benchRT_carbon-reference.tgz}")
  ADD_TEST(NAME benchRT_carbon
    COMMAND /bin/bash  ${Gate_SOURCE_DIR}/benchmarks/gate_run_test.sh benchRT carbon ${Gate_SOURCE_DIR} ${GATE_BINARY})
endif(BUILD_TESTING)

if(BUILD_TESTING)
  GateAddBenchmarkData("DATA{benchImaging/reference/benchImaging_ct-reference.tgz}")
  ADD_TEST(NAME benchImaging_ct
    COMMAND /bin/bash  ${Gate_SOURCE_DIR}/benchmarks/gate_run_test.sh benchImaging ct ${Gate_SOURCE_DIR} ${GATE_BINARY})
endif(BUILD_TESTING)

if(BUILD_TESTING AND GATE_USE_OPTICAL)
  GateAddBenchmarkData("DATA{benchImaging/reference/benchImaging_optical-reference.tgz}")
  ADD_TEST(NAME benchImaging_optical
    COMMAND /bin/bash  ${Gate_SOURCE_DIR}/benchmarks/gate_run_test.sh benchImaging optical ${Gate_SOURCE_DIR} ${GATE_BINARY})
endif(BUILD_TESTING AND GATE_USE_OPTICAL)

if(BUILD_TESTING)
  GateAddBenchmarkData("DATA{benchImaging/reference/benchImaging_spect-reference.tgz}")
  ADD_TEST(NAME benchImaging_spect
    COMMAND /bin/bash  ${Gate_SOURCE_DIR}/benchmarks/gate_run_test.sh benchImaging spect ${Gate_SOURCE_DIR} ${GATE_BINARY})
endif(BUILD_TESTING)

if(BUILD_TESTING AND GATE_USE_ECAT7)
  GateAddBenchmarkData("DATA{benchImaging/reference/benchImaging_pet-reference.tgz}")
  ADD_TEST(NAME benchImaging_pet
    COMMAND /bin/bash  ${Gate_SOURCE_DIR}/benchmarks/gate_run_test.sh benchImaging pet ${Gate_SOURCE_DIR} ${GATE_BINARY})
endif(BUILD_TESTING AND GATE_USE_ECAT7)
