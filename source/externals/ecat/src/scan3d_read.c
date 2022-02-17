
/* static char sccsid[] = "@(#)scan3d_read.c	1.1 UCL-TOPO 95/11/29"; */
#include	"matrix.h"
#include	<stdlib.h>
#include	<fcntl.h>
#include	<string.h>


MatrixData *scan3d_read(mptr, slice_matnum, dtype)
  MatrixFile *mptr ;
  int	slice_matnum, dtype;
{
  struct MatDir matdir;
  struct Matval val;
  int i, matnum, slice;
  int  group=0, slice_count=0, z_elements;
  int file_pos, nblks, data_size ;
  Scan3D_subheader *scan3Dsub ;
  int y, line_size, skip_size;
  char *line, *p;
  MatrixData *data = NULL;

	if (!mptr || !mptr->mhptr || 
		mptr->mhptr->file_type != Short3dSinogram ) return NULL;

	mat_numdoc( slice_matnum, &val);
	slice = val.plane-1;
	matnum = mat_numcod(val.frame,1,val.gate,val.data,val.bed);
	if (matrix_find(mptr,matnum,&matdir) < 0) return NULL;

	/* allocate space for MatrixData structure and initialize */
	data = (MatrixData *) calloc( 1, sizeof(MatrixData)) ;
	if (!data) return NULL;

	/* allocate space for subheader and initialize */
	data->shptr = (caddr_t) calloc(2, MatBLKSIZE) ;
	if (!data->shptr) return NULL;


	data->matnum = slice_matnum;
	data->matfile = mptr;
	data->mat_type = mptr->mhptr->file_type;
	scan3Dsub = (Scan3D_subheader *) data->shptr ;
	mat_read_Scan3D_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk, scan3Dsub) ;
	data->data_type = scan3Dsub->data_type;
/* dimesnsions if storage_order = 0  i.e.
	 (((projs x z_elements)) x num_angles) x Ringdiffs
*/
	data->xdim = scan3Dsub->num_r_elements ;
	data->ydim = scan3Dsub->num_angles ;
	data->zdim = scan3Dsub->num_z_elements[0];
	for (i=1; scan3Dsub->num_z_elements[i]; i++) 
		data->zdim += scan3Dsub->num_z_elements[i];
	data->scale_factor = scan3Dsub->scale_factor ;
	data->pixel_size = scan3Dsub->x_resolution;
	data->data_max = scan3Dsub->scan_max * scan3Dsub->scale_factor ;
	if (dtype == MAT_SUB_HEADER) return  data;

	data_size = data->xdim*data->ydim*sizeof(short);
	file_pos = (matdir.strtblk+1)*MatBLKSIZE;
	while (scan3Dsub->num_z_elements[group]) {
		slice_count += scan3Dsub->num_z_elements[group];
		if (slice < slice_count) break;
		file_pos += scan3Dsub->num_z_elements[group]*data_size;
		group++;
	}
	if (slice >= slice_count) {
		free_matrix_data(data);
		return NULL;
	}
	z_elements = scan3Dsub->num_z_elements[group];

	nblks = (data_size+511)/512;
	data_size = data->data_size = nblks * 512;
	p = data->data_ptr = (caddr_t) calloc(1, data_size) ;
	if (!data->data_ptr) {
		free_matrix_data(data);
		return NULL;
	}

	line_size = data->xdim*sizeof(short);
	skip_size = line_size*(z_elements-1);
	file_pos += slice*line_size;
	fseek(mptr->fptr,file_pos,0); /* jump to location of this slice*/
	line = malloc(line_size);
	data->zdim = 1;
	for (y=0; y<data->ydim; y++) {	/* for each planar view fixed theta */
		if (y > 0) 			/* skip to next planar view */
			fseek(mptr->fptr,skip_size,1);
		fread(line,sizeof(short),data->xdim,mptr->fptr);
		if (ntohs(1) != 1) swab(line,p,line_size);
		else memcpy(p,line,line_size);
		p += line_size;
	}
	data->data_max = find_smax((short*)data->data_ptr,data->xdim*data->ydim);
	free(line);
	return data;
}
