/* @(#)lib3d.c	1.19 10/4/93 */

static char sccsid[]="@(#)lib3d.c	1.19 10/4/93 Copyright 1991-92 CTI Pet Systems, Inc.";

#include "matrix.h"
#include "matpkg.h"
#include "fproj3d.h"
#include <math.h>

#define max(a,b) ((a>b)?a:b)
#define min(a,b) ((a<b)?a:b)
#define abs(a) ((a<-a)?-a:a)

static int *mat_xlate=NULL;

int matnumfx( a, b, flag, frame, dflag)
  int a, b, flag, frame, dflag;
{
	int plane, i;

	switch(flag)
	{
	  case -1:
		return mat_numcod( frame, a+b+1, 1, dflag, 0);
	  case 0:	/* New ACS numbering (2048 planes 0-2047) */
		plane = ((a&0x10)<<5)+((a&0x08)<<4)+(a&7)+
			((b&0x10)<<4)+((b&15)<<3)+1;
		return mat_numcod( frame, plane, 1, dflag, 0);
	  case 1:	/* Pre-ACS sinogram numbering */
	  	return matnum_3d( a, b);
	  case 2:	/* RPT numbering */
	  	return matnum_3dfx( a, b, frame);
	  case 3:	/* Old 256 plane ACS sinogram numbering */
		plane = (1+(a<<3)+(b&0x07)+((b&0x08)<<4))%256;
		return mat_numcod( frame, plane, 1, dflag, 0);
	  case 4:	/* Actual 921 3-ring numbering */
		if (!mat_xlate)
		{
		  mat_xlate = (int*) malloc( 1024*sizeof(int));
		  for (plane=i=0; i<1024; i++)
		  {
			mat_xlate[i] = -1;
			if ((i >=   0 && i < 320) ||
			    (i >= 384 && i < 448) ||
			    (i >= 512 && i < 640) ||
			    (i >= 768 && i < 832)) mat_xlate[i] = plane++;
		  }
		}
		plane = ((a&0x10)<<5)+((a&0x08)<<4)+(a&7)+
			((b&0x10)<<4)+((b&15)<<3);
		return mat_numcod( frame, mat_xlate[plane]+1, 1, dflag, 0);
	  case 5:	/* Reversed ACS numbering (2048 planes 0-2047) */
		plane = ((b&0x10)<<5)+((b&0x08)<<4)+(b&7)+
			((a&0x10)<<4)+((a&15)<<3)+1;
		return mat_numcod( frame, plane, 1, dflag, 0);

	}
}

matnumx( a, b, flag)
  int a, b, flag;
{
	return matnumfx( a, b, flag, 1, 0);
}

int matnum_3d( ring_a, ring_b)
  int ring_a, ring_b;
{
	int frame, plane;

	frame = 1+((ring_b&0x8)>>2)+((ring_a&0x8)>>3);
	plane = 1+((ring_a&0x7)<<3)+(ring_b&0x7);
	return mat_numcod( frame, plane, 1, 0, 0);
}

int matnum_3dx( ring_a, ring_b)
  int ring_a, ring_b;
{
	int frame, plane;

	frame = 1+((ring_b&0x8)>>2)+((ring_a&0x8)>>3);
	plane = 1+((ring_a&0x7)<<3)+(ring_b&0x7);
	return mat_numcod( frame, plane, 0, 0, 0);
}

int matnum_3dfx( ring_a, ring_b, fr)
  int ring_a, ring_b, fr;
{
	int frame, plane;

	frame = 1+4*(fr-1)+((ring_b&0x8)>>2)+((ring_a&0x8)>>3);
	plane = 1+((ring_a&0x7)<<3)+(ring_b&0x7);
	return mat_numcod( frame, plane, 0, 0, 0);
}

struct tslice			/* temporary slice data */
	{ int matnum;
	  float scale_factor,
		max_pixel,
		zloc;
	};

load_stack3d( sinfo, sdata, file, matstr)
  Stack3d_info sinfo;
  float *sdata;
  char *file, *matstr;
{
	Main_header mhead;		/* holds file main header */
	Image_subheader h;		/* holds 1 image sub-header */
	struct matdir *dir, *mat_read_dir();	/* holds file directory */
	short int *b1;
	FILE *fptr;
	int i,j,k,plane,bed,dtype;
	int ranges[2][5];
	float max_voxel, zloc, plane_separation;
	int nvoxels, npixels, xdim, ydim, zdim, nslices;
	float *vp;
	struct Matval matval;
	float bed_pos[16];
	struct tslice slice[MAX_SLICES];

	fptr = mat_open( file, "r");	/* Open the file for reading */
	if (!fptr) return 0;
	decode_selector( matstr, ranges);
	dir = mat_read_dir( fptr, matstr);
	mat_read_main_header( fptr, &mhead);
	bed_pos[0] = 0.0;
	for (i=1; i<mhead.num_bed_pos; i++)
	  bed_pos[i] = mhead.bed_offset[i-1];
	max_voxel = 0.0;
	nslices = 0;
	plane_separation = mhead.plane_separation;
	if (plane_separation <= 1.0e-3) plane_separation = 0.675;
		/* old 931 image? */
	if (plane_separation > 10.0) plane_separation = 0.675;
	for (i=0; i<dir->nmats; i++)
	{ if (!matrix_selector(dir->entry[i].matnum, ranges)) continue;
	  mat_read_image_subheader( fptr, dir->entry[i].strtblk, &h);
	  dtype = h.data_type;
	  slice[nslices].scale_factor = h.quant_scale;
	  slice[nslices].max_pixel = h.quant_scale * h.image_max;
	  slice[nslices].matnum = dir->entry[i].matnum;
	  if (slice[nslices].max_pixel > max_voxel)
		max_voxel = slice[nslices].max_pixel;
	  sinfo->xy_size = h.pixel_size;
	  sinfo->z_size = plane_separation;
	  sinfo->xdim=sinfo->ydim=h.dimension_1;
	  mat_numdoc( slice[nslices].matnum, &matval);
	  plane = matval.plane;
	  bed = matval.bed;
	  slice[nslices].zloc = bed_pos[bed]+(plane-1)*plane_separation;
	  nslices++;
	}
	if (nslices==0)
	{ mat_close(fptr);
	  return 0;
	}
	sort_by_zloc( slice, nslices);
	sinfo->zdim = nslices;
	npixels = sinfo->xdim*sinfo->ydim;
	nvoxels = npixels * sinfo->zdim;
	b1 = (short int*) malloc( npixels*sizeof(short int));
	vp = sdata;
	for (i=0; i<nslices; i++)
	{
	  read_slice_data( fptr, &slice[i], b1, npixels, dir, dtype);
	  for (j=0; j<npixels; j++)
	    *vp++ = slice[i].scale_factor*b1[j];
	}
	for (j=0; j<npixels; j++)
	  *vp++ = 0.0;
	mat_close(fptr);
	free(b1);
	return 1;
}


load_vol3d( vinfo, vdata, file, matstr)
  Vol3d_info vinfo;
  float *vdata;
  char *file, *matstr;
{
	Main_header mhead;		/* holds file main header */
	Image_subheader h;		/* holds 1 image sub-header */
	struct matdir *dir, *mat_read_dir();	/* holds file directory */
	short int *b1, *b2, *p1, *p2;	/* image data pointers */
	FILE *fptr;
	int i,j,k,plane,bed,dtype;
	int ranges[2][5];
	float max_voxel, w1, w2, zloc, plane_separation;
	int nvoxels, npixels, xdim, ydim, zdim, nslices;
	float *vp;
	struct Matval matval;
	float bed_pos[16];
	struct tslice slice[MAX_SLICES];
	int iw1, iw2;
	char msg[256];

	fptr = mat_open( file, "r");	/* Open the file for reading */
	if (!fptr) return 0;
	decode_selector( matstr, ranges);
	dir = mat_read_dir( fptr, matstr);
	mat_read_main_header( fptr, &mhead);
	bed_pos[0] = 0.0;
	for (i=1; i<mhead.num_bed_pos; i++)
	  bed_pos[i] = mhead.bed_offset[i-1];
	max_voxel = 0.0;
	nslices = 0;
	plane_separation = mhead.plane_separation;
	if (plane_separation <= 1.0e-3) plane_separation = 0.675;
		/* old 931 image? */
	if (plane_separation > 10.0) plane_separation = 0.675;
	for (i=0; i<dir->nmats; i++)
	{ if (!matrix_selector(dir->entry[i].matnum, ranges)) continue;
	  mat_read_image_subheader( fptr, dir->entry[i].strtblk, &h);
	  dtype = h.data_type;
	  slice[nslices].scale_factor = h.quant_scale;
	  slice[nslices].max_pixel = h.quant_scale * h.image_max;
	  slice[nslices].matnum = dir->entry[i].matnum;
	  if (slice[nslices].max_pixel > max_voxel)
		max_voxel = slice[nslices].max_pixel;
	  vinfo->voxel_size = h.pixel_size;
	  vinfo->xdim=vinfo->ydim=h.dimension_1;
	  mat_numdoc( slice[nslices].matnum, &matval);
	  plane = matval.plane;
	  bed = matval.bed;
	  slice[nslices].zloc = bed_pos[bed]+(plane-1)*plane_separation;
	  nslices++;
	}
	if (nslices==0)
	{ mat_close(fptr);
	  return 0;
	}
	sort_by_zloc( slice, nslices);
	vinfo->zdim = 2 +
	  (int)(0.5+slice[nslices-1].zloc/vinfo->voxel_size);
	npixels = vinfo->xdim*vinfo->ydim;
	nvoxels = npixels * vinfo->zdim;
	b1 = (short int*) malloc( npixels*sizeof(short int));
	b2 = (short int*) malloc( npixels*sizeof(short int));
	i = 1;
	read_slice_data( fptr, &slice[0], b1, npixels, dir, dtype);
	read_slice_data( fptr, &slice[1], b2, npixels, dir, dtype);
	vp = vdata;
	for (j=0; j<vinfo->zdim;j++)
	{ zloc = j*vinfo->voxel_size;
	  if (zloc > slice[i].zloc)
	  { p1=b1;b1=b2;b2=p1;
	    if (i+1<nslices)
	    {
	      read_slice_data( fptr, &slice[i+1], b2, npixels, dir, dtype);
	      i++;
	    }
	  }
	  w2=(zloc-slice[i-1].zloc)/(slice[i].zloc-slice[i-1].zloc);
	  w1=1.0-w2;
	  w1=w1*slice[i-1].scale_factor;
	  w2=w2*slice[i].scale_factor;
	  p1 = b1; p2 = b2;
	  for (k=0; k<npixels; k++)
	    *vp++ = w1*(*p1++)+w2*(*p2++);

	} /* make next slice */
	mat_close(fptr);
	free(b1);
	free(b2);
	return 1;
}

compare_zloc( a, b)
  struct tslice *a, *b;
{	if (a->zloc < b->zloc) return (-1);
	else if (a->zloc > b->zloc) return (1);
	else return 0;
}

sort_by_zloc( slices, n)
  struct tslice slices[]; int n;
{
	qsort( slices, n, sizeof(struct tslice), compare_zloc);
}

read_slice_data( fptr, slice, bufr, npixels, dir, dtype)
  FILE *fptr;
  struct tslice *slice;
  short int *bufr;
  int npixels, dtype;
  struct matdir *dir;
{	int i, strtblk, endblk;

	for (i=0;i<dir->nmats;i++)
	  if (dir->entry[i].matnum == slice->matnum) break;
	strtblk = dir->entry[i].strtblk;
	endblk = dir->entry[i].endblk;
	mat_read_data( fptr, strtblk+1, 512*(endblk-strtblk), bufr, dtype);
}

read_slice_zdata( fptr, slice, bufr, npixels, dir, dtype)
  FILE *fptr;
  struct tslice *slice;
  short int *bufr;
  int npixels, dtype;
  struct matdir *dir;
{	int i, strtblk, endblk;

	for (i=0;i<dir->nmats;i++)
	  if (dir->entry[i].matnum == slice->matnum) break;
	strtblk = dir->entry[i].strtblk;
	endblk = dir->entry[i].endblk;
	mat_read_data( fptr, strtblk+1, 512*(endblk-strtblk), bufr, dtype);
	for (i=0;i<npixels;i++)
	  if (bufr[i] < 0) bufr[i] = 0;
}


typedef unsigned char byte;

int * compute_swap_lors( nprojs, nviews, nptr)
  int nprojs, nviews, *nptr;
{
	static byte fix951[8][3] = {
		{0,5,0}, {0,6,0}, {0,7,1}, {1,6,1},
		{10,15,0}, {9,15,0}, {8,15,2}, {9,14,2}};
	static byte fix953[8][3] = {
		{0,3,0}, {0,4,0}, {0,5,1}, {1,4,1},
		{8,11,0}, {7,11,0}, {7,10,2}, {6,11,2}};
	int deta, detb, v, e, a, b, *list, i, n, m;
	byte *fixer;

	if (nviews%2 == 0) fixer = (byte *)fix951;
	if (nviews%3 == 0) fixer = (byte *)fix953;
	n = 0;
	list = (int*) malloc( 32*32*8*sizeof(int));
	for (i=0; i<8; i++)
	{
	  a = fixer[3*i];
	  b = fixer[3*i+1];
	  m = fixer[3*i+2];
	  for (deta=0; deta<32; deta++)
	    for (detb=0; detb<32; detb++)
	    {
		dets_to_ve( a*32+deta, b*32+detb, &v, &e, nviews*2);
		e += nprojs/2;
		if (m==1 && v<nviews/2) continue;
		if (m==2 && v>nviews/2) continue;
		if ((e+1>0) && (e<nprojs)) list[n++] = v*nprojs+e;
	    }
	}
	*nptr = n;
	return (list);
}
#ifdef TEST

Matrix plane_matrix( vol, zoff, theta, phi, size)
  Vol3d vol;
  float zoff, theta, phi, size;
{
	Matrix m[4], result;
	int i,w,h,l;
	double sint, cost;
	float max_dim;

	result = matrix(4,4);
	for (i=0; i<4; i++)
	{
	  m[i] = matrix(4,4);
	  mat_unity(m[i]);
	}
	w = vol->xdim;
	h = vol->ydim;
	l = vol->zdim;
	max_dim = (float)max(w,max(h,l));
	m[0]->data[3] = -size/2.0;
	m[0]->data[7] = -size/2.0;
	m[0]->data[11] = zoff;
	m[0]->data[0] = size;
	m[0]->data[5] = size;
	sincos((double)phi*M_PI/180., &sint, &cost);
	m[1]->data[5] = cost;
	m[1]->data[10] = cost;
	m[1]->data[6] = sint;
	m[1]->data[9] = -sint;
	sincos((double)theta*M_PI/180., &sint, &cost);
	m[2]->data[0] = cost;
	m[2]->data[5] = cost;
	m[2]->data[1] = sint;
	m[2]->data[4] = -sint;
	m[3]->data[3] = (float)w/max_dim/2.0;
	m[3]->data[7] = (float)h/max_dim/2.0;
	m[3]->data[11] = (float)(l-2)/max_dim/2.0;
	matmpy(result, m[0], m[1]);
	matmpy(m[0], result, m[2]);
	matmpy(result, m[0], m[3]);
	for (i=0; i<4; i++)
	  mat_free(m[i]);
	return result;
}

View2d make_slice( vol, slice_mx)
  Vol3d vol;
  Matrix slice_mx;
{
/*
*	produce an oblique slice of the volume as specified by slice_mx.
*	1. transform corners of the slice [[0,1,0],[0,0,0],[1,0,0]]
*	   using the slice_mx transformation matrix.
*	2. compute the coordinates of points on this plane (s,t)=>(x,y,z)
*	   and interpolate the value of the volume at each point.
*/
	float x, y, z, v, *optr, *p;
	int i,j,k,s,t;
	int w,h,l;
	float wx, wy, wz;
	float max_dim;
	View2d slice;
	int slice_size;

	w = vol->xdim;
	h = vol->ydim;
	l = vol->zdim;
	slice_size = max(w,max(h,l));
	max_dim = (float) slice_size;
	slice = (View2d) malloc( sizeof(struct view2d));
	slice->xdim = slice->ydim = slice_size;
	slice->data = (float*) malloc( slice_size*slice_size*sizeof(float));
	optr = slice->data;
	for (t=0; t<slice_size; t++)
	{
	  x = slice_mx->data[3]*max_dim+slice_mx->data[1]*t;
	  y = slice_mx->data[7]*max_dim+slice_mx->data[5]*t;
	  z = slice_mx->data[11]*max_dim+slice_mx->data[9]*t;
	  for (s=0; s<slice_size; s++)
	  {
	    i=x, j=y, k=z, v=0.0;
	    if ((i+1>0)&&(i<w-1) &&
		(j+1>0)&&(j<h-1) &&
		(k+1>0)&&(k<l-1))
	    {
		wx=x-i, wy=y-j, wz=z-k;
		p=vol->data+i+j*w+k*w*h;
		v = (*(p))*(1.-wx)*(1.-wy)*(1.-wz) +
		    (*(p+1))*wx*(1.-wy)*(1.-wz) +
		    (*(p+w))*(1.-wx)*wy*(1.-wz) +
		    (*(p+w+1))*wx*wy*(1.-wz) +
		    (*(p+w*h))*(1.-wx)*(1.-wy)*wz +
		    (*(p+w*h+1))*wx*(1.-wy)*wz +
		    (*(p+w*h+w))*(1.-wx)*wy*wz +
		    (*(p+w*h+w+1))*wx*wy*wz;
	    }
	    *optr++ = v;
	    x += slice_mx->data[0];
	    y += slice_mx->data[4];
	    z += slice_mx->data[8];
	  }
	}
	return slice;
}

#endif TEST
