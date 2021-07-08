/* @(#)make3dfilter.c	1.8 10/13/93 */

#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
make3dfilter( nprojs, nslices, phi, phi0, filter, ftype_x, cutoff_x, ftype_y, cutoff_y)
  int nprojs, nslices, ftype_x, ftype_y;
  float phi, phi0, *filter, cutoff_x, cutoff_y;
{

	float u, v, delta_u, delta_v, rho, rho_sqr, fval, cosphi;
	float cond, sin2_phi, sin2_phi0, dx, dy, damping(), fmax;
	int x, y;


	cosphi = (float) cos((double)phi);
	sin2_phi = (float) sin((double) phi);
	sin2_phi *= sin2_phi;
	sin2_phi0 = (float) sin((double) phi0);
	sin2_phi0 *= sin2_phi0;
	delta_u = 1.0/(nprojs);
	delta_v = 1.0/(nslices);
	fmax = 0.0;
	for (y=0; y<=nslices/2; y++)
	{
	  v = y*delta_v;
	  dy = damping( v, ftype_y, cutoff_y);
	  for (x=0; x<=nprojs/2; x++)
	  {
	    u = x*delta_u;
	    dx = damping( u, ftype_x, cutoff_x);
	    if (x+y == 0)
	      filter[0] = 0.0;
	    else
	    {
		rho_sqr = u*u+v*v;
		rho = sqrt(rho_sqr);
		fval = 2.0*rho;
		cond = ((u*u+v*v*sin2_phi)/(rho_sqr*sin2_phi0))-1.0;
		if (cond > 0.0)
		  fval = M_PI*rho/atan2(1.0,sqrt(cond));
		if (fval > fmax) fmax = fval;
		filter[ x+y*nprojs ] = fval;
	    }
	  }
	}
	for (y=0; y<=nslices/2; y++)
	{
	  v = y*delta_v;
	  dy = damping( v, ftype_y, cutoff_y);
	  for (x=0; x<=nprojs/2; x++)
	  {
	    u = x*delta_u;
	    dx = damping( u, ftype_x, cutoff_x);
	    fval = filter[x+y*nprojs] * dx * dy / fmax;
	    filter[ x+y*nprojs ] = fval;
	    if (x>0) filter[nprojs-x+y*nprojs] = fval;
	    if (y>0) filter[x +(nslices-y)*nprojs] = fval;
	    if (x*y>0) filter[nprojs-x+(nslices-y)*nprojs] = fval;
	  }
	}
}

float damping( f, ftype, cutoff)
  float f, cutoff;
  int ftype;
{
	float d;

	switch( ftype)
	{
	  case 1:
		return (f<=cutoff) ? 1.0 : 0.0;
	  case 3:
		d = 0.5+0.5*cos(M_PI*f/cutoff);
		return (f<=cutoff) ? d : 0.0;
	  case 4:
		d = 0.54+0.46*cos(M_PI*f/cutoff);
		return (f<=cutoff) ? d : 0.0;
	  default:
		return (f<=cutoff) ? 1.0 : 0.0;
	}
}

#ifdef TEST

#include <stdio.h>

main( argc, argv)
  int argc;
  char **argv;
{
	float phi, phi0, *filter, cutoff_x, cutoff_y;
	int nprojs, nslices, x, y, ftype_x, ftype_y, i, fmax;
	FILE *optr;

	sscanf( argv[1], "%d,%d,%f,%f,%d,%f,%d,%f", &nprojs, &nslices,
	  &phi, &phi0, &ftype_x, &cutoff_x, &ftype_y, &cutoff_y);
	phi = phi*M_PI/180.;
	phi0 = phi0*M_PI/180.;
	filter = (float*) malloc( nprojs*nslices*sizeof(float));
	make3dfilter( nprojs, nslices, phi, phi0, filter, ftype_x, cutoff_x,
		ftype_y, cutoff_y);
	if (argc > 2)
	{
	  optr = fopen( argv[2], "wb");
	  fwrite( filter, nprojs*nslices, sizeof(float), optr);
	  close( optr);
	}
	fmax = 0;
	for (i=1; i<nprojs*nslices; i++)
	  if (filter[i] > filter[fmax]) fmax=i;
	printf("Filter max @ %d = %e\n", fmax, filter[fmax]);
	exit(0);
}

#endif
