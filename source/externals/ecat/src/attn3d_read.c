
/* static char sccsid[] = "@(#)attn3d_read.c	1.1 UCL-TOPO 95/11/29"; */

#include	<math.h>
#include	<stdlib.h>
#include	<fcntl.h>
#include	<string.h>
#include	"matrix.h"

extern float find_fmax();

MatrixData *attn3d_read(mptr, slice_matnum, dtype)
  MatrixFile *mptr ;
  int	slice_matnum, dtype;
{
  struct MatDir matdir;
  struct Matval val;
  int i, matnum, slice;
  int  group=0, slice_count=0, z_elements;
  int file_pos, nblks, data_size ;
  Attn_subheader *attnsub ;
  int y, line_size, skip_size, npixels;
  char *tmp_line, *line, *p;
  MatrixData *data = NULL;
  float *fdata, scale_f, fmax;
  short *sdata;

	if (!mptr || !mptr->mhptr || 
		mptr->mhptr->file_type != AttenCor ) return NULL;

	mat_numdoc( slice_matnum, &val);
	slice = val.plane-1;
	matnum = mat_numcod(val.frame,1,val.gate,val.data,val.bed);
	if (matrix_find(mptr,matnum,&matdir) < 0) return NULL;

	/* allocate space for MatrixData structure and initialize */
	data = (MatrixData *) calloc( 1, sizeof(MatrixData)) ;
	if (!data) return NULL;

	/* allocate space for subheader and initialize */
	data->shptr = (caddr_t) calloc(1, MatBLKSIZE) ;
	if (!data->shptr) return NULL;


	data->matnum = slice_matnum;
	data->matfile = mptr;
	data->mat_type = mptr->mhptr->file_type;
	attnsub = (Attn_subheader *) data->shptr ;
	mat_read_attn_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk, attnsub) ;
	data->data_type = attnsub->data_type ;
/* dimesnsions if storage_order = 0  i.e.
	 (((projs x z_elements)) x num_angles) x Ringdiffs
*/
	data->xdim = attnsub->num_r_elements ;
	data->ydim = attnsub->num_angles ;
	data->zdim = attnsub->z_elements[0];
	for (i=1; attnsub->z_elements[i]; i++) 
		data->zdim += attnsub->z_elements[i];
	data->scale_factor = attnsub->scale_factor ;
	data->pixel_size = attnsub->x_resolution;
	data->data_max = -1;
	if (dtype == MAT_SUB_HEADER) return  data;

	data_size = data->xdim*data->ydim*sizeof(float);
	file_pos = matdir.strtblk*MatBLKSIZE;
	while (attnsub->z_elements[group]) {
		slice_count += attnsub->z_elements[group];
		if (slice < slice_count) break;
		file_pos += attnsub->z_elements[group]*data_size;
		group++;
	}
	if (slice >= slice_count) {
		free_matrix_data(data);
		return NULL;
	}
	z_elements = attnsub->z_elements[group];

	nblks = (data_size+511)/512;
	data_size = data->data_size = nblks * 512;
	fdata = (float*) malloc(data_size) ;
	if (!fdata) {
		free_matrix_data(data);
		return NULL;
	}
	npixels = data->xdim*data->ydim;
	line_size = data->xdim*sizeof(float);
	skip_size = line_size*(z_elements-1);
	file_pos += slice*line_size;
	fseek(mptr->fptr,file_pos,0); /* jump to location of this slice*/
	line = malloc(line_size);
	tmp_line = malloc(line_size);
	data->zdim = 1;
	p = (char*)fdata;
	for (y=0; y<data->ydim; y++) {	/* for each planar view fixed theta */
		if (y > 0) 			/* skip to next planar view */
			fseek(mptr->fptr,skip_size,1);
		fread(line,sizeof(float),data->xdim,mptr->fptr);
		if (ntohs(1) != 1) {
			swab(line,tmp_line,line_size);
			swaw((short*)tmp_line,(short*)p,line_size/2);
		}
		else memcpy(p,line,line_size);
		p += line_size;
	}
	data->data_max = find_fmax(fdata,npixels);
	fmax = fabs(*fdata);
    for (i=0; i<nvals; i++)
      if (fabs(fdata[i]) > fmax) fmax = fabs(fdata[i]);
	scale_f = 32767.0/fmax;
	data->scale_factor = 1.0/scale_f;
	sdata = (short*)malloc(data->xdim*data->ydim*sizeof(short));
	data->data_ptr = (caddr_t)sdata;
	for (i=0; i<npixels; i++) sdata[i] = (short)(fdata[i]*scale_f + 0.5);
	free(fdata);
	free(line);
	return data;
}
