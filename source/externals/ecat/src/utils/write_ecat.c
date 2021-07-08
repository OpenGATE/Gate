/*  static char sccsid[] = "%W% UCL-TOPO %E%";
 *
 * Creation date : 29-Jan-1996
 * Author : Merence Sibomana <sibomana@topo.ucl.ac.be>
 *
 * Usage : write_ecat file_name [specs [new_file]]
 *      This program reads the standard input and writes to the specified
 *      ECAT data file.
 *		specs format is frame,plane,gate,data,bed (e.g. 1,1,1,0,0)
 *      It may be used instead of the CTI dwritedata program
 *      provided by CTI to interface IDL programs.
 *      if "new_file" argument is set to 1 a new file is created.
 */
#include "string.h"
#include "matrix.h"

static int debug = 1;
static int verbose = 1;

main( argc, argv)
  int argc;
  char **argv;
{
    int matnum=0, segments=0, nplanes=0, plane, npixels;
	caddr_t data_in;
	Main_header mh;
	Image_subheader *imh;
    short type=0;       /* 0 for main header */
	char matspec[80], *mk;
	int nmats=1, specs[5];
	int i=0, c=1;
	int new_file = 0;
    MatrixFile *mptr;
    MatrixData *matrix;
	struct Matval val;
	int el_size=2;
	

    if (argc<2) crash( "usage    : %s file_name [specs [new_file]]\n", argv[0]);
	if (argc>3) sscanf(argv[3],"%d",&new_file);

	if (fread(&mh, sizeof(Main_header), 1, stdin) != 1) 
		crash("%s : error reading main header\n",argv[0]);
	if (new_file) {
		if ((mptr = matrix_create(argv[1], MAT_CREATE_NEW_FILE,&mh)) == NULL)
			 crash("%s : can't create %s\n",argv[0],argv[1]);
	} else {
		if ((mptr = matrix_create(argv[1],MAT_OPEN_EXISTING,&mh)) == NULL)
			crash("%s : can't open %s\n",argv[0],argv[1]);
	}
	
	if (fread(&nmats, sizeof(int), 1, stdin) != 1)
		 crash("%s : error reading number of matrices\n",argv[0]);
if (debug) fprintf(stderr,"number of matrices : %d\n",nmats);
	matspec[i] = 0;
	if (fread(&matnum, sizeof(int), 1, stdin) != 1)
		 crash("%s : error reading matnum \n",argv[0]);
	mat_numdoc(matnum,&val);
	specs[0] = val.frame;
	specs[1] = val.plane;
	specs[2] = val.gate;
	specs[3] = val.data;
	specs[4] = val.bed;
if (debug) fprintf(stderr,"matnum : %d,%d,%d,%d,%d\n", specs[0],specs[1],specs[2],specs[3],specs[4]);
	if (argc > 2) {
		i = 0;
		mk = strtok(argv[2],",");
		while (mk != NULL) {
			sscanf(mk,"%d",&specs[i++]);
			mk = strtok(NULL,",");
		}
if (debug) fprintf(stderr,"specs : %d,%d,%d,%d,%d\n", specs[0],specs[1],specs[2],specs[3],specs[4]);
		matnum = mat_numcod(specs[0],specs[1],specs[2],specs[3],specs[4]);
	}
	matrix = (MatrixData*)calloc(1,sizeof(MatrixData));
	matrix->shptr = (caddr_t)calloc(2,MatBLKSIZE);
	switch(mh.file_type)
	{
	   case ByteVolume :
	   case PetImage :
	   case PetVolume :
			if (fread(matrix->shptr, sizeof(Image_subheader), 1, stdin) != 1)
				crash("%s: error reading Image subheader\n", argv[0]);
			imh = (Image_subheader*)matrix->shptr;
			matrix->xdim = imh->x_dimension;
			matrix->ydim = imh->y_dimension;
			matrix->zdim = imh->z_dimension;
			if (imh->data_type == ByteData) el_size = 1;
			matrix->data_size = matrix->xdim*matrix->ydim*matrix->zdim*el_size;
			break;
	   default :
		  crash("%s: type %d insupported\n", argv[0], mptr->mhptr->file_type);
	}
	if (fread(&segments, sizeof(int), 1, stdin) != 1)
		 crash("%s : error reading segments \n",argv[0]);
if (debug) fprintf(stderr,"segments : %d\n",segments);
	if (fread(&npixels, sizeof(int), 1, stdin) != 1)
		 crash("%s : error reading npixels \n",argv[0]);
if (debug) fprintf(stderr,"npixels : %d\n",npixels);
	if (fread(&nplanes, sizeof(int), 1, stdin) != 1)
		 crash("%s : error reading nplanes \n",argv[0]);
if (debug) fprintf(stderr,"nplanes : %d\n",nplanes);
	matrix->data_ptr = (caddr_t)calloc(matrix->data_size,1);
	fread(matrix->data_ptr,1,matrix->data_size,stdin);
	matrix_write(mptr, matnum, matrix);
	matrix_close(mptr);
	exit(0);
}
