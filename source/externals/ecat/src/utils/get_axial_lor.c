/*
 *  static char sccsid[] = "%W% UCL-TOPO %E%";
 *
 *  Author : Sibomana@topo.ucl.ac.be
 *	Creation date :		21-Jun-1995
 */

#include "plandefs.h"
#include "ecat_model.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static int verbose = 0;

static void usage() {
	 fprintf(stderr,"usage : \nget_axial_lor -i scan_spec -l r1,r2 -t r_elements -a angles [-o out_file] [-v]\n");
	fprintf(stderr,"Result :\n");
	fprintf (stderr,"writes on out_file, the sinogram located by [r1,r2] in the michelogram\n\n");
	fprintf (stderr,"-t option trims in radial direction to specified number of elements\n");
	fprintf (stderr,"-a option generates speficied number of views (angles), using linear interporation in angular direction\n");
	exit(1);
}

static matrix_trim(matrix,nprojs)
MatrixData *matrix;
int nprojs;
{
	Scan3D_subheader *scan_sub;
	Attn_subheader *attn_sub;
	int hw, nviews, i, j, nblks;

	if (nprojs > matrix->xdim)
		crash("matrix_trim : new size(%d) > current size(%d)\n",
 			nprojs, matrix->xdim);
	nviews = matrix->ydim;
	hw = (matrix->xdim - nprojs)/2;		/* half with */
	if (hw == 0) return;
	matrix->xdim = matrix->xdim - 2*hw;
	switch (matrix->data_type) {
	case IeeeFloat :
	{
		float *p0, *p1, *data;
		p0 = (float*)matrix->data_ptr;
		matrix->data_size = nprojs*nviews*sizeof(float);
		p1 = data = (float*)malloc(matrix->data_size);
		for (i=0; i<nviews; i++) {
			p0 += hw;
			for (j=0; j<nprojs; j++) *p1++ = *p0++;
			p0 += hw;
		}
		free(matrix->data_ptr);
		matrix->data_ptr = (caddr_t)data;
		break;
	}
	case VAX_Ix2:
    case SunShort:
	{
		short *p0, *p1, *data;
		p0 = (short*)matrix->data_ptr;
		matrix->data_size = nprojs*nviews*sizeof(short);
		p1 = data = (short*)malloc(matrix->data_size);
		for (i=0; i<nviews; i++) {
			p0 += hw;
			for (j=0; j<nprojs; j++) *p1++ = *p0++;
			p0 += hw;
		}
		free(matrix->data_ptr);
		matrix->data_ptr = (caddr_t)data;
		break;
	}
	default:
		crash("data type : %d not implemented\n", matrix->data_type);
		break;
	}
}

static matrix_flip(matrix) 
MatrixData *matrix;
{
	int i,j,nviews, nprojs;
	float *view, *data, *p;

	nprojs = matrix->xdim;
	nviews = matrix->ydim;
	view = (float*)calloc(nprojs,sizeof(float));
	data = (float*)matrix->data_ptr;
	for (i=0; i<nviews; i++) {
		p = data + (i+1)*nprojs;
		for (j=0; j<nprojs; j++) view[j] = *p--;
		memcpy(p+1,view,nprojs*sizeof(float));
	}
	free(view);
}
	
static matrix_rebin_angle(matrix,nviews)
MatrixData *matrix;
int nviews;
{
	Scan3D_subheader *scan_sub;
	Attn_subheader *attn_sub;
	int j, nprojs, view, v0, nblks;
	float w0, w1;

	float dv = (float)matrix->ydim/nviews;
	nprojs = matrix->xdim;
	switch (matrix->data_type) {
	case IeeeFloat :
	{
		float *old_data, *data, *p;
		float *view0, *view1;
		old_data = (float*)matrix->data_ptr;
		matrix->data_size = nprojs*nviews*sizeof(float);
		p = data = (float*)malloc(matrix->data_size);
		for (view=0; view<nviews; view++) {
			v0 = (int)(floor(view*dv));
			view0 = old_data + v0*nprojs;
			if (v0 < matrix->ydim-1) view1 = view0 + nprojs;
			else view1 = view0;
			w0 = view*dv - v0;
			w1 = 1.0 - w0;
			for (j=0; j<nprojs; j++) *p++ = w1*view0[j] + w0*view1[j];
		}
		free(matrix->data_ptr);
		matrix->data_ptr = (caddr_t)data;
		break;
	}
	case VAX_Ix2:
    case SunShort:
	{
		short *old_data, *view0, *view1;
		float  *data, *p;
		old_data = (short*)matrix->data_ptr;
		matrix->data_size = nprojs*nviews*sizeof(float);
		p = data = (float*)malloc(matrix->data_size);
		for (view=0; view<nviews; view++) {
			v0 = (int)(floor(view*dv));
			view0 = old_data + v0*nprojs;
			if (v0 < matrix->ydim-1) view1 = view0 + nprojs;
			else view1 = view0;
			w0 = view*dv - v0;
			w1 = 1.0 - w0;
			for (j=0; j<nprojs; j++) *p++ = w1*view0[j] + w0*view1[j];
		}
		free(matrix->data_ptr);
		matrix->data_ptr = (caddr_t)data;
		matrix->data_type = IeeeFloat;
		break;
	}
	default:
		crash("data type : %d not implemented\n", matrix->data_type);
		break;
	}
	matrix->ydim = nviews;
}


static MatrixData *get_axial_lor(mptr, volume, z_elements, slice)
MatrixFile *mptr;
MatrixData *volume;
short *z_elements;
int slice;
{
	int group, segment;
	MatrixData *matrix;
	
	for (group=0; z_elements[group] > 0; group++) {
		if ((slice-z_elements[group]) >= 0) {
			slice -= z_elements[group];
		} else break;
	}
	segment = group;
	if (group > 0 &&  slice > (z_elements[group]/2)) {
			segment = -segment;
			slice -= z_elements[group]/2;
	}
	if (verbose) fprintf(stderr,"reading slice %d in segment %d ...",slice,segment);
	if ((matrix = matrix_read_slice(mptr, volume,slice,segment)) == NULL)
   		crash( "segment %d, slice %d not found\n",slice,segment);
	if (verbose) fprintf(stderr,"%dx%dx%d done\n", matrix->xdim,
		matrix->ydim, matrix->zdim);
	return matrix;
}

int main (argc, argv)
int  argc;
char** argv;
{
    MatrixFile *mptr;
    MatrixData *volume;
	MatrixData *matrix_1, *matrix_2;		/* r1,r2 and r2,r1 matrices */
    char fname[256];
    int c, matnum;
	FILE *out_file;
	short type;
	caddr_t data;
	int slice, segment, group=0;
	EcatModel *model;
	int r1,r2, rmax, span, ring_difference, *vplanes;
	short *z_elements;
	int nprojs = 0, nviews =0;
	Scan3D_subheader *scan_sub=0;
	Attn_subheader *attn_sub=0;
	extern int optind, opterr;
	extern char *optarg;
	
	fname[0] = '\0';
	while ((c = getopt (argc, argv, "i:hl:o:t:a:v")) != EOF) {
		switch (c) {
		case 'i' :
			matspec( optarg, fname, &matnum);
			break;
		case 'o' :
			if ((out_file = fopen(optarg,"w")) == NULL) 
				crash("can't create %s\n",optarg);
			break;
		case 'l' :
			if (sscanf(optarg,"%d,%d",&r1,&r2) != 2) 
				crash("error decoding -l %s\n",optarg);
			break;
		case 't' :
			if (sscanf(optarg,"%d",&nprojs) != 1)
				crash("error decoding -t %s\n",optarg);
			break;
		case 'a' :
			if (sscanf(optarg,"%d",&nviews) != 1)
				crash("error decoding -a %s\n",optarg);
			break;
		case 'v':
			verbose = 1;
			break;
		default:
			usage();
		}
	}
	if (fname[0] == '\0')  usage();
    mptr = matrix_open( fname, MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
    if (!mptr)
      crash( "%s: can't open file '%s'\n", argv[0], fname);
	if ((model = ecat_model(mptr->mhptr->system_type)) == NULL)
		crash("unkown system type %d\n",mptr->mhptr->system_type);
	rmax = model->dirPlanes;
	vplanes = (int*)calloc(rmax*rmax, sizeof(int));
	if ( (volume = matrix_read(mptr, matnum, MAT_SUB_HEADER)) == NULL)
		crash("volume not found");
	switch(mptr->mhptr->file_type) {
	case Short3dSinogram :
		scan_sub = (Scan3D_subheader*)volume->shptr;
		span = scan_sub->axial_compression;
		ring_difference = scan_sub->ring_difference;
		z_elements = scan_sub->num_z_elements;
		break;
	case AttenCor:
		attn_sub = (Attn_subheader*)volume->shptr;
		span = attn_sub->span;
		z_elements = attn_sub->z_elements;
		ring_difference = attn_sub->ring_difference;		/* bad value */
		while (z_elements[group] > 0) group++;
        ring_difference = ((span-1)/2)+(span*group)-1;
		break;
	default :
    	crash("input is not an Sinogram or Attenuation data set\n");
		break;
	}
	plandefs(rmax,span,ring_difference, vplanes);

/* get r1, r2 sinogram */
	slice = vplanes[r2*rmax+r1];
	if (slice < 0) crash("location[%d,%d] not valid\n",r1,r2);
	matrix_1 = get_axial_lor(mptr, volume, z_elements, slice) ;

/* get r2, r1 sinogram */
	if ( (slice = vplanes[r1*rmax+r2]) < 0)
		crash("location[%d,%d] not valid\n",r1,r2);
	matrix_2 = get_axial_lor(mptr, volume, z_elements, slice) ;

	if (nprojs) {
		matrix_trim(matrix_1,nprojs);
		matrix_trim(matrix_2,nprojs);
	}
	if (nviews) {
		matrix_rebin_angle(matrix_1,nviews);
		matrix_rebin_angle(matrix_2,nviews);
	}
	matrix_flip(matrix_2);

    switch(matrix_1->data_type) {
    case ByteData:
        type = 0;
        break;
    default:
    case VAX_Ix2:
    case SunShort:
        type = 1;
        break;
    case SunLong :
        type = 2;
        break;
    case VAX_Rx4:
    case IeeeFloat:
        type = 3;
        break;
    }

	if (out_file == NULL) out_file = stdout;
	fwrite(&type,sizeof(short),1,out_file);
	matrix_1->ydim *= 2;
	fwrite(&matrix_1->xdim,sizeof(short),1,out_file);
    fwrite(&matrix_1->ydim,sizeof(short),1,out_file);
    fwrite(&matrix_1->zdim,sizeof(short),1,out_file);
	fwrite(matrix_1->data_ptr,1,matrix_2->data_size,out_file);
	fwrite(matrix_2->data_ptr,1,matrix_2->data_size,out_file);
	return 1;
}
