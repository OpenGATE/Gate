
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "matrix.h"	
#define FAIL 1
static char *version = "1.0";
static char *program_date = "10-may-2001";

main (argc, argv)
	int             argc;
	char           *argv[];
{

	MatrixFile     *matrix_file;			 /* file for ANALYZE hdr */
	FILE           *fd_if;			 /* output Interfile Format header */
	MatrixData     *matrix=NULL;
	MatDirNode *node;
	char *ECAT_file=NULL, *ECAT_base=NULL, *IF_hdr=NULL;
	char *p;

	if (argc < 2) {
		fprintf (stderr,
			"Usage: %s ecat_file [IF_hdr]\n", argv[0]);
		exit (FAIL);
	}
	ECAT_file = argv[1];
	if (argc > 2) IF_hdr = argv[2];

	ECAT_base = strdup(ECAT_file);
	if ( (p = strrchr(ECAT_base,'/')) == NULL)  p = ECAT_base;
	if ((p = strrchr(p,'.')) != NULL) *p = '\0';
	if (IF_hdr == NULL) {
		IF_hdr = (char*)malloc(strlen(ECAT_base)+6);
		sprintf(IF_hdr,"%s.h33",ECAT_base);
	}
	matrix_file = matrix_open (ECAT_file, MAT_READ_ONLY,MAT_UNKNOWN_FTYPE);
	if (matrix_file==NULL) 
		crash("%s : matrix_open error\n", ECAT_file);
	node = matrix_file->dirlist->first;
	if (node == NULL) 
		crash("%s: no matrix found\n",ECAT_file);
	if (node->next != NULL)
		fprintf(stderr,"Warning : using first of many %s matrices\n", ECAT_file);

	matrix = matrix_read(matrix_file, node->matnum, MAT_SUB_HEADER);
	if (matrix==NULL) crash ("%s : Error reading matrix header\n", ECAT_file);
	fprintf(stderr,"interfile header file : %s\n",IF_hdr);
	if ((fd_if = fopen (IF_hdr, "w")) == NULL) {
		perror (IF_hdr);
		exit (FAIL);
	}

	fprintf (fd_if, "INTERFILE :=\n");
	fprintf (fd_if, "version of keys    := 3.3\n");
	fprintf (fd_if, "conversion program := cti2ifh\n");
	fprintf (fd_if, "program version    := %s\n", version);
	fprintf (fd_if, "program date   := %s\n", program_date);
	fprintf (fd_if, "original institution   := %s\n",matrix_file->mhptr->facility_name);
	fprintf (fd_if, "subject ID	:= %s\n", matrix_file->mhptr->patient_name);
	fprintf (fd_if, "name of data file  := %s\n", ECAT_file);
	if (matrix_file->mhptr->file_type == Short3dSinogram &&
		matrix_file->mhptr->file_type == Float3dSinogram)
		fprintf (fd_if, "data offset in bytes := %d\n", (node->strtblk+1)*MatBLKSIZE);
	else fprintf (fd_if, "data offset in bytes := %d\n", node->strtblk*MatBLKSIZE);
    switch(matrix->data_type) {
        case ByteData :
            fprintf (fd_if, "number format  := unsigned integer\n");
            fprintf (fd_if, "number of bytes per pixel  := 1\n");
            break;
        case SunShort:
            fprintf (fd_if, "number format  := signed integer\n");
    		fprintf (fd_if, "imagedata byte order := bigendian\n");
            fprintf (fd_if, "number of bytes per pixel  := 2\n");
            break;
        case VAX_Ix2:
            fprintf (fd_if, "number format  := signed integer\n");
    		fprintf (fd_if, "imagedata byte order := littlendian\n");
            fprintf (fd_if, "number of bytes per pixel  := 2\n");
            break;
        case IeeeFloat:
            fprintf (fd_if, "number format  := short float\n");
    		fprintf (fd_if, "imagedata byte order := bigendian\n");
            fprintf (fd_if, "number of bytes per pixel  := 4\n");
            break;
    }
	fprintf (fd_if, "number of dimensions   := 3\n");
	fprintf (fd_if, "matrix size [1]    := %d\n", matrix->xdim);
	fprintf (fd_if, "matrix size [2]    := %d\n", matrix->ydim);
	fprintf (fd_if, "matrix size [3]    := %d\n", matrix->zdim);
	fprintf (fd_if, "scaling factor (mm/pixel) [1]  := %f\n",
		10*matrix->pixel_size);
	fprintf (fd_if, "scaling factor (mm/pixel) [2]  := %f\n",
		10*matrix->y_size);
	fprintf (fd_if, "scaling factor (mm/pixel) [3]  := %f\n",
		10*matrix->z_size);
	fprintf (fd_if, ";%%quantification units := %g\n", matrix->scale_factor);
	matrix_close(matrix_file);
	fclose (fd_if);
	exit (0);
}
