#CC = gcc 				# use gcc for sunos4.1
#CC = cc -std1          # -std1 enforces ANSI C for Digital Unix
CC = cc 
DEBUG =  -O
SRCDIR = .
SRC_COMPRESS = $(SRCDIR)/lib_compress

CFLAGS = $(DEBUG) -c -I$(SRCDIR)
DEP_FLAGS = $(DEBUG) -I$(SRCDIR)

ECAT_OBJS = crash.o make3dfilter.o makefilter.o matpkg.o \
	matrix_extra.o matrix.o matrix_xdr.o sino_dets.o \
	machine_indep.o matrix_64.o convert_64.o convert_70.o rts_cmd.o rfa_xdr.o \
	matrix_slice.o ecat_model.o plandefs.o interfile.o analyze.o num_sort.o \
	save_volume7.o load_volume7.o

UNCOMPILED = lib3d.o rtsacs.o

LIB_OBJS = $(ECAT_OBJS)

LIB = libecat.a

$(LIB) : $(LIB_OBJS)
	ar r $(LIB) $(LIB_OBJS)

clean:
	/bin/rm -f *.o *.a *.bak


crash.o : $(SRCDIR)/crash.c
	$(CC) $(CFLAGS) $(SRCDIR)/crash.c
lib3d.o : $(SRCDIR)/lib3d.c
	$(CC) $(CFLAGS) $(SRCDIR)/lib3d.c
make3dfilter.o : $(SRCDIR)/make3dfilter.c
	$(CC) $(CFLAGS) $(SRCDIR)/make3dfilter.c
makefilter.o : $(SRCDIR)/makefilter.c
	$(CC) $(CFLAGS) $(SRCDIR)/makefilter.c
machine_indep.o : $(SRCDIR)/machine_indep.c
	$(CC) $(CFLAGS) $(SRCDIR)/machine_indep.c
matpkg.o : $(SRCDIR)/matpkg.c
	$(CC) $(CFLAGS) $(SRCDIR)/matpkg.c
matrix.o : $(SRCDIR)/matrix.c
	$(CC) $(CFLAGS) $(SRCDIR)/matrix.c
matrix_64.o : $(SRCDIR)/matrix_64.c
	$(CC) $(CFLAGS) $(SRCDIR)/matrix_64.c
matrix_xdr.o : $(SRCDIR)/matrix_xdr.c
	$(CC) $(CFLAGS) $(SRCDIR)/matrix_xdr.c
convert_64.o : $(SRCDIR)/convert_64.c
	$(CC) $(CFLAGS) $(SRCDIR)/convert_64.c
convert_70.o : $(SRCDIR)/convert_70.c
	$(CC) $(CFLAGS) $(SRCDIR)/convert_70.c
matrix_extra.o : $(SRCDIR)/matrix_extra.c
	$(CC) $(CFLAGS) $(SRCDIR)/matrix_extra.c
mat_read.o : $(SRCDIR)/mat_read.c
	$(CC) $(CFLAGS) $(SRCDIR)/mat_read.c
rtsacs.o : $(SRCDIR)/rtsacs.c
	$(CC) $(CFLAGS) $(SRCDIR)/rtsacs.c
rts_cmd.o : $(SRCDIR)/rts_cmd.c
	$(CC) $(CFLAGS) $(SRCDIR)/rts_cmd.c
rfa_xdr.o : $(SRCDIR)/rfa_xdr.c
	$(CC) $(CFLAGS) $(SRCDIR)/rfa_xdr.c
sino_dets.o : $(SRCDIR)/sino_dets.c
	$(CC) $(CFLAGS) $(SRCDIR)/sino_dets.c
matrix_slice.o : $(SRCDIR)/matrix_slice.c
	$(CC) $(CFLAGS) $(SRCDIR)/matrix_slice.c
ecat_model.o : $(SRCDIR)/ecat_model.c
	$(CC) $(CFLAGS) $(SRCDIR)/ecat_model.c
interfile.o : $(SRCDIR)/interfile.c
	$(CC) $(CFLAGS) $(SRCDIR)/interfile.c
analyze.o : $(SRCDIR)/analyze.c
	$(CC) $(CFLAGS) $(SRCDIR)/analyze.c
plandefs.o : $(SRCDIR)/plandefs.c
	$(CC) $(CFLAGS) $(SRCDIR)/plandefs.c
num_sort.o : $(SRCDIR)/num_sort.c
	$(CC) $(CFLAGS) $(SRCDIR)/num_sort.c

depend:
	makedepend -Dunix $(DEP_FLAGS) $(SRCDIR)/*.c $(SRC_COMPRESS)/*.c
# DO NOT DELETE THIS LINE -- make depend depends on it.
