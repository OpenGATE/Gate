/*  static char sccsid[] = "%W% UCL-TOPO %E%";
 *
 * Creation date : 29-Jan-1996
 * Author : Merence Sibomana < sibomana@topo.ucl.ac.be>
 *
 * Usage : read_ecat file_name specs [sort_order [data]]
 * 		This program reads the specified information from ECAT data
 *      file and writes to the standard output.
 *		specs format is frame,plane,gate,data,bed (e.g. 1,1,1,0,0)
 *		It may be used instead of the CTI dbreaddata program
 *		provided by CTI to interface IDL programs.
 *		if "data" argument is set to 1 the matrix data is written.
 */
#include <string.h>
#include "matrix.h"
static int debug = 0;
static int verbose = 0;
main(argc, argv)
  int argc;
  char **argv;
{
	struct Matval val;
	MatrixFile *mptr=NULL;
	MatrixData *matrix=NULL;
	MatDirNode *node=NULL;
	char *mk=NULL, buf[MatBLKSIZE];
	int sort_order=1, no_data = 1;
	int i, j, specs[5];
	short dsize[5];
	int *matnums=NULL, nmats=0;
	int mode = MAT_SUB_HEADER;
	int start, end, tmp_matnum;
	caddr_t data;

	if (argc<3)
		crash( "usage\t: %s filename specs [sort_order [data]]\n", argv[0]);
if (debug) fprintf(stderr,"filename = %s\n", argv[1]);
	mptr = matrix_open( argv[1], MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
	if (!mptr) matrix_perror(argv[1]);
if (verbose) fprintf(stderr,"opening %s succedeed\n", argv[1]);
	for (i=0; i<5; i++) specs[i] = 0;
if (debug) fprintf(stderr,"matrix specs %s \n", argv[2]);
	mk = strtok(argv[2],",");
	for (i=0; i<5; i++) {
		if (mk!=NULL) {
			if (*mk == '*') specs[i] = -1;
			else specs[i] = atoi(mk);
			mk = strtok(NULL,",");
		}
	}
	if (argc>3) sscanf(argv[3],"%d",&sort_order);
if (verbose) if (argc>3) fprintf(stderr,"sort_order=%s(%d)\n",argv[3],sort_order);
	if (argc>4) sscanf(argv[4],"%d",&no_data);
if (verbose) if (argc>4) fprintf(stderr,"no_data=%s(%d)\n",argv[4],no_data);
 	if (no_data==0) mode = UnknownMatDataType;
													/* build selected matnums */
	for (node=mptr->dirlist->first; node!=NULL; node = node->next) nmats++;
	matnums = (int*)calloc(sizeof(int),nmats);
	nmats = 0;
	for (node=mptr->dirlist->first; node!=NULL; node=node->next) {
		mat_numdoc( node->matnum, &val);
		if (specs[0]>=0 && specs[0] != val.frame) continue;
		if (specs[1]>=0 && specs[1] != val.plane) continue;
		if (specs[2]>=0 && specs[2] != val.gate) continue;
		if (specs[3]>=0 && specs[3] != val.data) continue;
		if (specs[4]>=0 && specs[4] != val.bed) continue;
		matnums[nmats++] = node->matnum;
	}
	if (nmats>1) {
		mat_numdoc(matnums[0],&val); start = val.plane;
		mat_numdoc(matnums[nmats-1],&val); end = val.plane;
		if (start>end) { 	/* reverse matnums */
			start = 0; end = nmats-1;
			while (start<end) {
				tmp_matnum = matnums[start];
				matnums[start] = matnums[end];
				matnums[end] = tmp_matnum;
				start++; end--;
			}
		}
	}
if (verbose) fprintf(stderr,"%d matrices selected\n",nmats);
mptr->mhptr->septa_state =nmats;
mptr->mhptr->align_3 = 21;
    fwrite(mptr->mhptr,sizeof(Main_header),1,stdout);	/* main header */
	if (fwrite(&nmats,sizeof(int),1,stdout) != 1)
		perror("stdout");			/* matrices number */

	for (i=0; i<nmats; i++) {
		matrix = matrix_read(mptr, matnums[i],MAT_SUB_HEADER);
		switch( mptr->mhptr->file_type) {
		case PetImage :
		case PetVolume :
		case ByteImage :
		case ByteVolume :
		case InterfileImage:
			fwrite(matrix->shptr,sizeof(Image_subheader),1,stdout);
			break;
		case AttenCor :
			fwrite(matrix->shptr,sizeof(Attn_subheader),1,stdout);
			break;
		case Normalization :
			fwrite(matrix->shptr,sizeof(Norm_subheader),1,stdout);
			break;
		case Norm3d :
			fwrite(matrix->shptr,sizeof(Norm3D_subheader),1,stdout);
			break;
		case Float3dSinogram :
		case Short3dSinogram :
			fwrite(matrix->shptr, sizeof(Scan3D_subheader),1,stdout);
			break;
		case Sinogram :
			fwrite(matrix->shptr,sizeof(Scan_subheader),1,stdout);
			break;
		default:
			crash("unknown file type %d",mptr->mhptr->file_type);
		}
		fwrite(&matnums[i],sizeof(int),1,stdout);
		dsize[0] = matrix->xdim;
		dsize[1] = matrix->ydim;
		dsize[2] = matrix->zdim;
		dsize[3] = 1;
		dsize[4] = matrix->data_type;
if (debug) fprintf(stderr,"dim: %d %d %d\n",
	matrix->xdim, matrix->ydim, matrix->zdim);
		free_matrix_data(matrix);
	}
if (debug) fprintf(stderr,"dim: %d %d %d\n",
	matrix->xdim, matrix->ydim, matrix->zdim);
	if (no_data == 0) {
		if (dsize[2] == 1) dsize[2] = nmats;
		fwrite(dsize,sizeof(short),5,stdout);
if (debug) fprintf(stderr,"dsize: %d %d %d %d %d\n",
	dsize[0],dsize[1],dsize[2],dsize[3],dsize[4]);
		for (i=0; i<nmats; i++) {
			matrix = matrix_read(mptr, matnums[i],UnknownMatDataType);
			data = matrix->data_ptr;
			for (j=0; j<matrix->zdim; j++) {
				switch(dsize[4]) {
				case ByteData:
					fwrite(data, 1, dsize[0]*dsize[1], stdout);
					data += dsize[0]*dsize[1];
					break;
				default:
				case VAX_Ix2:
				case SunShort:
					fwrite(data, 2, dsize[0]*dsize[1], stdout);
					data += dsize[0]*dsize[1]*2;
					break;
				case VAX_Rx4:
				case IeeeFloat:
				case SunLong:
					fwrite(data, 4, dsize[0]*dsize[1], stdout);
					data += dsize[0]*dsize[1]*4;
					break;
				}
				fprintf(stderr,".");
			}
		}
		fprintf(stderr,"\n");
	}
if (verbose) fprintf(stderr,"program read_ecat done\n");
	return(0);
}
