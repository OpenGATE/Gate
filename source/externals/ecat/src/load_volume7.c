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
extern char	       matrix_errtxt[];


#if  defined(__cplusplus)
MatrixData *load_volume7( MatrixFile *matrix_file, int frame, int gate, int data, int bedstart, int bedend )
#else
MatrixData *load_volume7(matrix_file, frame, gate, data, bedstart, bedend)
MatrixFile *matrix_file;
int frame, gate, data, bedstart, bedend;
#endif
{
	int		i, j, k, rev, ret;
	MatrixData	*mat;
	int		matnum, dimension;
	float		zsep, maxval, minval;
	Main_header	*mh;
	Image_subheader	*imh = NULL;
	int		plane, planes;
	struct MatDir	entry;
	MatrixData	*volume;
	int		npixels, nvoxels, overlap[MAX_BED_POS], ovlp, ovl, sbed, ebed, nbed, actbed, splane, eplane, sframe, eframe;
	float		scale_factor, calibration_factor, w1, w2;
	short		*p1;
	float		*vp = NULL, *fvp;


	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';

	mh = matrix_file->mhptr;

	rev = mh->patient_orientation & 1;
	if( mh->num_bed_pos && (mh->bed_offset[0] < 0) ) rev = !rev;
	if( (mh->patient_orientation < 0) || (mh->patient_orientation >= UnknownOrientation) ) rev = !rev;

	if( bedstart < 0 ) {
		sbed = 0;
		ebed = mh->num_bed_pos;
	} else {
		if( bedstart > mh->num_bed_pos || bedend > mh->num_bed_pos || bedstart > bedend ) {
			if( bedstart > mh->num_bed_pos ) {
				sprintf( matrix_errtxt, "Bed %d", bedstart );
			} else if( bedend > mh->num_bed_pos ) {
				sprintf( matrix_errtxt, "Bed %d", bedend );
			} else if( bedstart > bedend ) {
				sprintf( matrix_errtxt, "Bedstart > Bedend" );
			}
			matrix_errno = MAT_MATRIX_NOT_FOUND;
			return( NULL );
		}
		sbed = bedstart;
		ebed = bedend;
	}

	/***************************************/
	/* malloc matrix_data structure volume */
	/***************************************/
	volume = (MatrixData*)calloc(1,sizeof(MatrixData));
	if( !volume ) return( NULL );
	imh = (Image_subheader*)calloc(1,sizeof(Image_subheader));
	if( !imh ) {
		free( volume );
		return( NULL );
	}

	/************************/
	/* load image subheader */
	/************************/
	matnum = mat_numcod( frame, 1, gate, data, sbed );
	mat = matrix_read( matrix_file, matnum, MAT_SUB_HEADER );
	if( !mat ) {
		free_matrix_data( volume );
		return( NULL );
	}
	*imh = *(Image_subheader*)mat->shptr;
	*volume = *mat;
	free_matrix_data( mat );

	volume->mat_type = (DataSetType)mh->file_type;
	volume->shptr = (caddr_t)imh;

	/*************************/
	/* calculate z_dimension */
	/*************************/
	zsep = mh->plane_separation;
	planes = volume->zdim;

	if( sbed == ebed ) {
		overlap[0] = 0;
	} else {
		for( i = 0 ; (i < mh->num_bed_pos) && (sbed < ebed) ; i++ ) {
			if( i == 0 ) {
				overlap[i] = planes - (int)(ABS(mh->bed_offset[0]) / zsep + .5);
			} else {
				overlap[i] = planes - (int)((ABS(mh->bed_offset[i]) - ABS(mh->bed_offset[i-1])) / zsep + .5);
			}
			if( overlap[i] < 0 || overlap[i] > planes ) {
				free_matrix_data( volume );
				matrix_errno = MAT_INVALID_MBED_POSITION;
				sprintf( matrix_errtxt, "overlap < 0 or overlap > planes" );
				return( NULL );
			}
		}
		for( nbed = sbed ; nbed < ebed ; nbed++ ) {
			if( rev ) {
				actbed = nbed;
			} else {
				actbed = ebed - ( nbed - sbed ) - 1;
			}
			volume->zdim += (planes - overlap[actbed]);
		}
	}
	npixels = volume->xdim * volume->ydim;
	nvoxels = npixels * volume->zdim;
	volume->data_ptr = (caddr_t)calloc( nvoxels, sizeof(float) );
	if( !volume->data_ptr ) {
		free_matrix_data( volume );
		return( NULL );
	}

	calibration_factor = mh->calibration_factor;
	if( calibration_factor == 0.0 ) calibration_factor = 1.0;

	/********************/
	/* read matrix      */
	/********************/
	for( nbed = sbed ; nbed <= ebed ; nbed++ ) {
		if( rev ) {
			actbed = nbed;
			if( nbed == sbed ) ovlp = 0;
			else ovlp = overlap[actbed-1];
		} else {
			actbed = ebed - ( nbed - sbed );
			if( nbed == sbed ) ovlp = 0;
			else ovlp = overlap[actbed];
		}

		matnum = mat_numcod( frame, 1, gate, data, actbed );

		if( !mat_lookup( matrix_file->fptr, mh, matnum, &entry ) ) {
			free_matrix_data( volume );
			sprintf( matrix_errtxt, "Frame %d, Gate %d, Data %d, Bed %d", frame, gate, data, actbed );
			matrix_errno = MAT_MATRIX_NOT_FOUND;
			return( NULL );
		}

		dimension = imh->num_dimensions;
		volume->data_type = imh->data_type;

		if( dimension == 2 ) {
/*			for( plane = 1 ; plane <= mh->num_planes ; plane++ ) {
				matnum = mat_numcod( frame, plane, gate, data, actbed );
			}
*/
			free_matrix_data(volume);
			matrix_errno = MAT_INVALID_DIMENSION;
			return( NULL );

		} else if( dimension == 3 ) {
			mat = matrix_read( matrix_file, matnum, UnknownMatDataType );
			if( !mat ) {
				free_matrix_data( volume );
				return( NULL );
			}

			scale_factor = mat->scale_factor * calibration_factor;
			p1 = (short*)mat->data_ptr;

			if( nbed == sbed ) {
				fvp = (float*)volume->data_ptr + (volume->zdim - planes) * npixels;
				ovl = 0;
			} else {
				fvp -= (planes - ovlp) * npixels;
				ovl = ovlp;
			}

			for( i = 0 ; i < planes ; i++ ) {
				vp = &fvp[(planes - i - 1) * npixels];
				if( i < ovl ) {
					w2 = (float)(i+1) / (float)(ovl + 1);
					w1 = 1.0 - w2;
					for( k = 0 ; k < npixels ; k++, vp++ ) *vp = *vp * w1 + (float)(*p1++) * scale_factor * w2;
				} else {
					for( k = 0 ; k < npixels ; k++ ) *vp++ = (float)(*p1++) * scale_factor;
				}
			}

			if( sbed == ebed ) {
				maxval = mat->data_max * calibration_factor;
				minval = mat->data_min * calibration_factor;
			}
			free_matrix_data(mat);

		} else {
			free_matrix_data(volume);
			matrix_errno = MAT_INVALID_DIMENSION;
			return( NULL );
		}
	}

	if( sbed != ebed ) {
		vp = (float*)volume->data_ptr;
		maxval = *vp;
		minval = *vp;
		for( j = 1 ; j < nvoxels ; j++ ) {
			vp++;
			if( *vp > maxval ) {
				maxval = *vp;
			} else if( *vp < minval ) {
				minval = *vp;
			}
		}
	}

	volume->z_size = zsep;
	volume->scale_factor = 1.0;
	volume->pixel_size = imh->x_pixel_size;
	volume->y_size     = imh->y_pixel_size;
	volume->data_size = nvoxels * sizeof(float);
	volume->matnum = mat_numcod( frame, 1, gate, data, sbed );
	volume->data_min = minval;
	volume->data_max = maxval;

	imh->num_dimensions = 3;
	imh->z_dimension = volume->zdim;

/*	if( bedstart < 0 ) mh->num_bed_pos = 0;*/

	return volume;
}
