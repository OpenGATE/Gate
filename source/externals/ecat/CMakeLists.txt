project(ecat)
include_directories(${PROJECT_SOURCE_DIR}/src)
add_library(ecat 
        src/crash.c src/make3dfilter.c src/makefilter.c src/matpkg.c
	src/matrix_extra.c src/matrix.c src/matrix_xdr.c src/sino_dets.c
	src/machine_indep.c src/matrix_64.c src/convert_64.c src/convert_70.c src/rts_cmd.c src/rfa_xdr.c
	src/matrix_slice.c src/ecat_model.c src/plandefs.c src/interfile.c src/analyze.c src/num_sort.c
	src/save_volume7.c src/load_volume7.c
	)
set_property(TARGET ecat PROPERTY C_STANDARD 90)
target_link_libraries(ecat rpcsvc nsl socket -lm )
