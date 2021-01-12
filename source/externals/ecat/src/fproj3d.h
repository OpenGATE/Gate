/* @(#)fproj3d.h	1.2 7/10/91 */

#ifndef fproj3d_def

typedef struct vol3d_info
	{
	  unsigned int xdim, ydim, zdim;
	  float voxel_size;
	}
*Vol3d_info;

typedef struct view2d_info
	{
	  unsigned int xdim, ydim;
	  float x_pixel_size, y_pixel_size;
	}
*View2d_info;

typedef struct stack3d_info
	{
	  unsigned int xdim, ydim, zdim;
	  float xy_size, z_size;
	}
*Stack3d_info;

#define fproj3d_def
#endif
