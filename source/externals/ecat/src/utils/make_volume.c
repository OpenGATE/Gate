#include <stdio.h>
#include "load_volume.h"

static usage() {
	fprintf(stderr, "usage: make_volume -i in_matspec -o out_matspec [-c]\n");
	fprintf(stderr, "-c : make cubic volume using linear interpolation\n");
	exit(1);
}
main( argc, argv)
  int argc;
  char **argv;
{
	MatrixFile *mptr1, *mptr2;
	MatrixData *matrix;
	Main_header proto;
	char fname[256], *in_spec=NULL, *out_spec=NULL;
	int matnum1=0, matnum2=0;
	struct Matval matval;
	int c, cubic = 0, interpolate=0, verbose=0;
	float q_scale = 1.0;
	extern char *optarg;

   while ((c = getopt (argc, argv, "i:o:cq:v")) != EOF) {
        switch (c) {
        case 'i' :
            in_spec = optarg;
            break;
        case 'o' :
            out_spec    = optarg;
            break;
		case 'c' : cubic=1; interpolate = 1; break;
		case 'v' : verbose =1; break;
		case 'q' : sscanf(optarg,"%g",&q_scale);	/* undocumented option */
		}
	}


	if (in_spec==NULL || out_spec==NULL) usage();
	matspec( in_spec, fname, &matnum1);
	mptr1 = matrix_open( fname, MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
	if (!mptr1)
	  crash( "%s: can't open file '%s'\n", argv[0], fname);

	mat_numdoc( matnum1, &matval);
	matrix = load_volume( mptr1, matval.frame,cubic,interpolate);
	if (!matrix)
	  crash( "%s: specified matrix not found\n", argv[0]);
	matrix->scale_factor *= q_scale;
	if (verbose) fprintf(stderr,"scale_factor*%g = %g\n",q_scale,matrix->scale_factor);
	matspec( out_spec, fname, &matnum2);
	if (matnum2 == 0) matnum2 = matnum1; 	/* use same specifications */
	proto = *mptr1->mhptr;
	proto.sw_version = 70;
 	proto.file_type = PetVolume;		/* volume mode */	
	proto.plane_separation = matrix->z_size;
	proto.num_planes = matrix->zdim;
	mptr2 = matrix_create( fname, MAT_OPEN_EXISTING, &proto);
	if (!mptr2)
	  matrix_perror( argv[0]);
	if (matrix->data_type == VAX_Ix2) {
		matrix->data_type = SunShort;
		((Image_subheader*)matrix->shptr)->data_type = SunShort;
	}
	matrix_write( mptr2, matnum2, matrix);
	matrix_close( mptr1);
	matrix_close( mptr2);
}
