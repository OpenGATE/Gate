#include <stdio.h>
#ifndef FILENAME_MAX /* SunOs 4.1.3 */
#define FILENAME_MAX 256
#endif
#include "load_volume.h"

static char line1[256], line2[256], line3[256], line4[256];
matrix_info(mh, matrix)
Main_header *mh;
MatrixData *matrix;
{
  struct Matval mat;
  int i, nvoxels;
  char *units;
  u_char *b_data;
  short *s_data;
  float *f_data;
  double total=0, mean=0;
  int data_unit = mh->calibration_units_label;
  float ecf = mh->calibration_factor;
  
  mat_numdoc(matrix->matnum, &mat);
  printf("%d,%d,%d,%d,%d\n",
	 mat.frame, mat.plane, mat.gate, mat.data, mat.bed);
  nvoxels = matrix->xdim *matrix->ydim*matrix->zdim ;
  switch(matrix->data_type) {
  case ByteData:
    b_data = (u_char*)matrix->data_ptr;
    for (i=0; i<nvoxels; i++) total += *b_data++;
    break;
  case IeeeFloat :
    f_data = (float*)matrix->data_ptr;
    for (i=0; i<nvoxels; i++) total += *f_data++;
    break;
  default:		/* short */
    s_data = (short*)matrix->data_ptr;
    for (i=0; i<nvoxels; i++) total += *s_data++;
  }
  printf("Dimensions := %dx%dx%d\n",matrix->xdim,matrix->ydim,matrix->zdim);
  if (mh->calibration_units == Uncalibrated || ecf <= 1.0 ||
      data_unit>numDisplayUnits)  data_unit = 0;
  if (data_unit) units = customDisplayUnits[data_unit];
  else units = "";
  total *= matrix->scale_factor;
  mean = total/nvoxels;
  sprintf(line1, "Minimum := %g %s",matrix->data_min, units);
  sprintf(line2, "Maximum := %g %s", matrix->data_max, units);
  sprintf(line3, "Mean    := %g %s", mean, units);
  sprintf(line4, "Total   := %g %s", total, units);
  if (data_unit == 1) { /* convert Ecat counts to Bq/ml */
    units = customDisplayUnits[2];
    sprintf(line1+strlen(line1), ", %g Bq/ml", ecf*matrix->data_min);
    sprintf(line2+strlen(line2), ", %g Bq/ml", ecf*matrix->data_max);
    sprintf(line3+strlen(line3), ", %g Bq/ml", ecf*mean);
    sprintf(line4+strlen(line4), ", %g Bq/ml", ecf*total,units);
  }
  printf("%s\n%s\n%s\n%s\n",line1, line2, line3, line4);
}

main(argc, argv)
     int argc;
     char **argv;
{
  MatDirNode *node;
  MatrixFile *mptr;
  MatrixData *matrix;
  struct Matval mat;
  char fname[FILENAME_MAX];
  int i, ftype, frame = -1, matnum=0, cubic=0, interpolate=0;
  
  if (argc < 2) crash("usage : %s matspec\n",argv[0]);
  matspec( argv[1], fname, &matnum);
  mptr = matrix_open(fname, MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
  if (mptr == NULL) {
    matrix_perror(fname);
    return 0;
  }
  ftype = mptr->mhptr->file_type;
  if (ftype <0 || ftype >= NumDataSetTypes)
    crash("%s : unkown file type\n",fname);
  printf( "%s file type  : %s\n", fname, datasettype[ftype]);
  if (!mptr) matrix_perror(fname);
  if (matnum) {
    matrix = matrix_read(mptr,matnum, UnknownMatDataType);
    if (!matrix) crash("%d,%d,%d,%d,%d not found\n",
		       mat.frame, mat.plane, mat.gate, mat.data, mat.bed);
    matrix_info(mptr->mhptr,matrix);
  } else {
    node = mptr->dirlist->first;
    while (node)
      {
	mat_numdoc(node->matnum, &mat);
	if (ftype == PetImage) {
	  if (frame != mat.frame) {
	    frame = mat.frame;
	    matrix = load_volume(mptr,frame,cubic,interpolate);
	    matrix_info(mptr->mhptr,matrix);
	  }
	} else {
	  matrix = matrix_read(mptr,node->matnum,UnknownMatDataType);
	  if (!matrix) crash("%d,%d,%d,%d,%d not found\n",
			     mat.frame, mat.plane, mat.gate, mat.data, mat.bed);
	  matrix_info(mptr->mhptr,matrix);
	}
	node = node->next;
      }
  }
  matrix_close(mptr);
  return 1;
}
