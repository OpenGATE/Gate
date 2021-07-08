/*  19-sep-2002: 
 * Merge with bug fixes and support for CYGWIN provided by Kris Thielemans@csc.mrc.ac.uk
*/
#include <stdlib.h>
#include <malloc.h>
#include <string.h>

#include "matrix.h"

#ifndef ERROR
#define ERROR -1
#endif

#ifndef MIN
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif

#ifndef ABS
#define ABS(x) (((x) >= 0) ? (x) : -(x))
#endif

#ifndef SGN
#define SGN(x) ((x) > 0 ? 1 : (x) < 0 ? -1 : 0)
#endif

extern MatrixErrorCode matrix_errno;


#if defined(__STDC__) || defined(__cplusplus)
typedef (*qsort_func)(const void *, const void *);
#endif
#if defined(__cplusplus)
extern void display_message(const char*);
#else
#define display_message(s) 
#endif



#if  defined(__cplusplus)
int save_volume7( MatrixFile *mfile, Image_subheader *shptr, float *data_ptr, int frame, int gate, int data, int bed )
#else
int save_volume7( mfile, shptr, data_ptr, frame, gate, data, bed )
MatrixFile *mfile;
Image_subheader *shptr;
float *data_ptr;
int frame, gate, data, bed;
#endif
{
	float		pixel, scale_factor, calibration_factor;
	float		data_min, data_max;
	short		min, max;
	int		i, num_frames, plane, plane_size, data_size, nblks;
	MatDirNode	*node;
	char		frames[MAX_FRAMES];
	struct Matval	mat;
	short		*ibufptr;
	float		*fptr;
	MatrixData	*mdata;

	calibration_factor = 1.0;

	plane_size = shptr->x_dimension * shptr->y_dimension;
	data_size  = plane_size * shptr->z_dimension;

	fptr = data_ptr;
	data_min = *fptr;
	data_max = *fptr;
	fptr++;
	for( i = 1 ; i < data_size ; i++ ) {
		if( *fptr > data_max ) data_max = *fptr;
		else if( *fptr < data_min ) data_min = *fptr;
		fptr++;
	}

	if( ABS(data_max) >= ABS(data_min) ) {
		scale_factor = ABS(data_max) / calibration_factor / 32767.0;
		min = (short)( data_min / (scale_factor * calibration_factor) + .5 * SGN(data_min) );
		max = 32767;
	} else {
		scale_factor = ABS(data_min) / calibration_factor / 32768.0;
		min = -32768;
		max = (short)( data_max / (scale_factor * calibration_factor) + .5 * SGN(data_max) );
	}

	/******************************/
	/* setup MatrixData structure */
	/******************************/
	if( (mdata = (MatrixData *)calloc( 1, sizeof(MatrixData))) == NULL ) {
		return( ERROR ) ;
	}
	if( (mdata->shptr = (caddr_t)malloc(sizeof(Image_subheader))) == NULL ) {
		free( mdata );
		return( ERROR ) ;
	}
	nblks = (data_size*sizeof(short)+MatBLKSIZE-1)/MatBLKSIZE;
	if( (mdata->data_ptr = (caddr_t)calloc(nblks, MatBLKSIZE)) == NULL ) {
		free( mdata->shptr );
		free( mdata );
		return(ERROR);
	}
	*((Image_subheader *)mdata->shptr) = *shptr;

	mdata->matnum = mat_numcod( frame, 1, gate, data, bed );		/* matrix number */
	mdata->mat_type = PetVolume;						/* type of matrix? */
	mdata->data_type = SunShort;						/* type of data */
	mdata->data_size = data_size * sizeof( short );				/* size of data in bytes */
	mdata->xdim = ((Image_subheader *)mdata->shptr)->x_dimension;		/* dimensions of data */
	mdata->ydim = ((Image_subheader *)mdata->shptr)->y_dimension;		/* y dimension */
	mdata->zdim = ((Image_subheader *)mdata->shptr)->z_dimension;		/* for volumes */
	mdata->scale_factor = scale_factor;					/* valid if data is int? */
	mdata->pixel_size = ((Image_subheader *)mdata->shptr)->x_pixel_size;	/* xdim data spacing (cm) */
	mdata->y_size     = ((Image_subheader *)mdata->shptr)->y_pixel_size;	/* ydim data spacing (cm) */
	mdata->z_size     = ((Image_subheader *)mdata->shptr)->z_pixel_size;	/* zdim data spacing (cm) */
	mdata->data_min = data_min;						/* min value of data */
	mdata->data_max = data_max;						/* max value of data */

	((Image_subheader *)mdata->shptr)->image_min    = min;
	((Image_subheader *)mdata->shptr)->image_max    = max;
	((Image_subheader *)mdata->shptr)->scale_factor = scale_factor;
	((Image_subheader *)mdata->shptr)->num_dimensions = 3;
	((Image_subheader *)mdata->shptr)->data_type = SunShort;

	mfile->mhptr->sw_version = V7;
	mfile->mhptr->plane_separation = mdata->z_size;
	mfile->mhptr->calibration_factor = 1.0;
	mfile->mhptr->file_type = PetVolume;

	/***************/
	/* copy volume */
	/***************/	
	ibufptr = (short*)mdata->data_ptr;
	for( plane = 0 ; plane < mdata->zdim ; plane++ ) {
		fptr = &(data_ptr[(mdata->zdim - plane - 1) * plane_size]);
		for( i = 0 ; i < plane_size ; i++ ) {
			pixel = *fptr++;
			pixel = pixel / scale_factor;
			*ibufptr++ = (short)( pixel + .5*SGN(pixel) );
		}
	}

	/*************/
	/* save file */
	/*************/
	if( matrix_write( mfile, mdata->matnum, mdata ) == ERROR ) {
		free_matrix_data(mdata);
		return( ERROR );
	}

	free_matrix_data(mdata);

	/**********************/
	/* update main_header */
	/**********************/
	node = mfile->dirlist->first;
	memset( frames,0,MAX_FRAMES );
	while( node ) {
		mat_numdoc( node->matnum, &mat);
		frames[mat.frame] = 1;
		node = node->next;
	}
	num_frames = 0;
	for( i = 0 ; i < MAX_FRAMES ; i++ ) {
		if( frames[i] ) num_frames++;
	}
	mfile->mhptr->num_frames = num_frames;
	mfile->mhptr->num_bed_pos = 0;

	if( mat_write_main_header( mfile->fptr, mfile->mhptr ) == ERROR ) return( ERROR );

	return( 0 );
}
