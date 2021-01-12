#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include "matrix.h"

typedef struct _Tslice
{
    int matnum;
    float zloc;
} Tslice;

#if defined(__STDC__)
typedef (*qsort_func)(const void *, const void *);
static int compare_zloc(const Tslice *a, const Tslice *b)
#else
static int compare_zloc(a,b)
Tslice *a, *b;
#endif
{
    if (a->zloc < b->zloc) return(-1);
    else if (a->zloc > b->zloc) return (1);
    else return (0);
}
#if defined(__STDC__)
static int slice_sort(Tslice *slices, int count)
{
    qsort(slices, count, sizeof(Tslice), (qsort_func)compare_zloc);
    return 1;
}
#else
static int slice_sort(slices,count)
Tslice *slices;
int count;
{
    qsort(slices, count, sizeof(Tslice), compare_zloc);
    return 1;
}
#endif

static Tslice slice[MAX_SLICES];
static  byte_scale(matrix, dest, f_min, f_max) 
MatrixData *matrix;
u_char *dest;
float f_min, f_max;
{
	int i, v, i_min=255, i_max = 0, size;
	float byte_scale;
	short *sdata=NULL;
	u_char *bdata=NULL;
	u_char *p1;
	int d_min, d_max, nvals = 256;

	d_min = (int)(0.5 + 0.01*f_min/matrix->scale_factor);
	d_max = (int)(0.5 + 0.01*f_max/matrix->scale_factor);
	if (matrix->data_type != ByteData)
		sdata = (short *)matrix->data_ptr;
	else bdata = (u_char *)matrix->data_ptr;
	size = matrix->xdim*matrix->ydim*matrix->zdim;
	p1 = dest;
	byte_scale = (float)(d_max-d_min)/(nvals-1);
	for (i=0; i<size; i++) {
		v = sdata? (*sdata++) : (*bdata++);
		if (v<=d_min) v=0;
		else v = (int)((v-d_min)/byte_scale);
		if (v >= nvals) v = nvals-1;
		if (v<i_min) i_min = v;
		if (v>i_max) i_max = v;
		*p1++ = v;
	}
}
	
static usage() {
	fprintf(stderr,
		"usage : byte_volume -i matspec -o matspec -r range_min,range_max\n");
		exit(1);
}

main (argc, argv)
	int	 argc;
	char	  **argv;

{

	int i, n, sz=0, frame=1;
	char i_fname[256], o_fname[256], selector[24];
	int i_matnum=0, o_matnum=0;
	int npixels, nslices=0;
	char *p;
	MatrixFile	 *file, *o_file;
	MatrixData *matrix=NULL, *b_matrix=NULL;
	Image_subheader *imh;
	MatDirNode *node;
	u_char* dest;
	Main_header mh;
	struct Matval mat;
	int verbose = 0, ret=0, err=0;
	float f_min = 0, f_max = -1;
	extern char *optarg;

	i_fname[0] = o_fname[0] = '\0';
	while ((i = getopt (argc, argv, "i:o:r:v")) != EOF) {
		switch(i) {
		case 'v' : verbose = 1;
			break;
		case 'i' :	  /* input file */
			matspec(optarg, i_fname, &i_matnum);
			break;
		case 'o' :
			matspec(optarg, o_fname, &o_matnum);
			break;
		case 'r' :
			if (sscanf(optarg,"%g,%g",&f_min, &f_max) != 2)
 				fprintf(stderr, "invalid dynamic range : %s\n",optarg, err++);
			break;
		case '?' : usage();
			break;
		}
	}
	if (i_fname[0] == '\0')
		fprintf(stderr, "input file not specified\n", err++);
	if (i_matnum == 0) i_matnum = mat_numcod(1,1,1,0,0);
	if (o_fname[0] == '\0')
		fprintf(stderr, "output file not specified\n", err++);
 	if (f_max <= f_min)
		fprintf(stderr, " no or invalid dynamic range\n", err++);
	if (err) usage();
	file =  matrix_open (i_fname, MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
	mh = *file->mhptr;
	mh.sw_version = 70;
	mh.num_frames = 1;
	mh.file_type = ByteVolume;
	if ((p = strrchr(file->fname,'/')) == NULL) p = file->fname;
	strcpy(mh.original_file_name,p);
	sprintf(mh.study_description,"byte volume : range = %g,%g", f_min,f_max);
	o_file = matrix_create (o_fname, MAT_CREATE, &mh);
	if (o_file == NULL) crash("can't create %s\n",o_fname);
	mat_numdoc(i_matnum, &mat);
	frame = mat.frame;
	if (mat.plane == 0) mat.plane = 1;
	if (mat.bed == 0) mat.bed = 1;
	node = file->dirlist->first;
	while (node) {
		mat_numdoc(node->matnum, &mat);
		if (mat.frame != frame) continue;
		slice[nslices].matnum = node->matnum;
		slice[nslices].zloc = mat.plane-1;
		nslices++;
		node = node->next;
	}
	if (nslices>1) slice_sort( slice, nslices);
	for (i=0; i<nslices; i++) {
		mat_numdoc(slice[i].matnum, &mat);
		matrix = matrix_read(file, slice[i].matnum, GENERIC);
		if (b_matrix == NULL) {
			b_matrix = (MatrixData*)malloc(sizeof(MatrixData));
			*b_matrix = *matrix;
			b_matrix->shptr = (caddr_t)malloc(sizeof(Image_subheader));
			memcpy(b_matrix->shptr,matrix->shptr,sizeof(Image_subheader));
			npixels = matrix->xdim*matrix->ydim*matrix->zdim;
			b_matrix->data_ptr = (caddr_t)malloc(npixels*nslices);
			b_matrix->data_size = npixels*nslices;
			dest = (u_char*)b_matrix->data_ptr;
		}
		byte_scale(matrix,dest,f_min,f_max);
		dest += npixels;
		free_matrix_data(matrix);
	}
	imh = (Image_subheader*)b_matrix->shptr;
	imh->data_type = b_matrix->data_type = ByteData;
	b_matrix->scale_factor =  imh->scale_factor = 1;
	b_matrix->data_min = imh->image_min = 0;
	b_matrix->data_max = imh->image_max = 255;
	imh->num_dimensions = 3;
	if (nslices > 1) imh->z_dimension = nslices;
	if (o_matnum > 0) b_matrix->matnum = o_matnum;
	matrix_write( o_file, b_matrix->matnum, b_matrix);
	free_matrix_data(b_matrix);
	matrix_close(file);
	matrix_close(o_file);
}
