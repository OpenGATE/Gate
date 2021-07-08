/* @(#)matpkg.h	1.2 3/15/91 */
#ifndef matpkg_h_defined
#define matpkg_h_defined
#include <math.h>

typedef struct matrix
	{
	  int ncols, nrows;
	  float *data;
	}
*Matrix;

typedef struct vol3d
	{
	  int xdim, ydim, zdim;
	  float voxel_size;
	  float *data;
	}
*Vol3d;

typedef struct stack3d
	{
	  int xdim, ydim, zdim;
	  float xy_size, z_size;
	  float *data;
	}
*Stack3d;

typedef struct view2d
	{
	  int xdim, ydim;
	  float x_pixel_size, y_pixel_size;
	  float *data;
	}
*View2d;

#if defined(__cplusplus)
extern "C" {
	matmpy(Matrix res, Matrix a, Matrix b);
	mat_print(Matrix);
	mat_unity(Matrix);
	Matrix mat_alloc(int ncols, int nrows);
	mat_copy(Matrix a, Matrix b);
	rotate(Matrix a,float rx, float ry, float rz);
	translate(Matrix a,float tx, float ty, float tz);
	scale(Matrix a,float sx, float sy, float sz);
	mat_free(Matrix);
}
#endif /* __cplusplus */

Matrix mat_alloc();
Vol3d make3d_volume();
Stack3d make3d_stack();
#endif /* matpkg_h_defined */
