/* @(#)makefilter.c	1.1 6/13/90 */

/* Make the frequency filter */

#include <math.h>
#include <stdlib.h>

#define	NO_FILTER	0
#define	RAMP		1
#define	BUTTERWORTH	2
#define	HANN		3
#define	HAMMING		4
#define	PARZEN		5
#define	SHEPP		6
#define EXP		7

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef	lint
static char sccsid[]="@(#)makefilter.c	1.1 6/13/90 Copyright 1990 CTI Pet Systems, Inc.";
#endif	/* lint */


static float *tramp=0;
static int tramp_size=0;

#define MAXFFT 2048

xramp( fx, fpars, nprojs)
  float *fx, fpars[];
  int nprojs;
{
	int i,n;
	float h[MAXFFT];
	float fft[MAXFFT], ra, temp[MAXFFT];
	float tau;

	if (tramp_size != nprojs)
	{ if (tramp) free(tramp);
	  tramp = (float*)malloc( nprojs*sizeof(float));
	  tau = fpars[1];
	  if (tau <= 0.0) tau = 1.0;
	  n=nprojs/2;

	  for (i=0; i<nprojs; i++)
	    h[i] = 0.0;			/* h[even] = 0.0 */
	  h[n] = 1.0/(4*tau*tau);		/* h[0] */
	  for (i=1; i<n; i+=2)
	    h[n-i] = h[n+i] = -1.0/(((float)i*M_PI*tau)*((float)i*M_PI*tau));
	  xfrf_( tramp, &ra, h, &n);
	  for (i=0; i<n; i++)
	  {
	    tramp[2*i] = 0.5*sqrt((double)(tramp[2*i]*tramp[2*i]+
			        tramp[2*i+1]*tramp[2*i+1]));
	    tramp[2*i+1] = 0.0;
	  }
	  tramp[1] = 0.5*ra;
	}
	for (i=0; i<nprojs; i++)
	  fx[i] = ((float)(i>>1)*0.5/(float)n > fpars[0]) ? 0.0 : tramp[i];
}

ramp(fx, fpars, nprojs)

float	*fx, fpars[] ;
int	nprojs ;

{
int	i ;
float	*ptr ;
float	cutoff, dc_component, ramp_slope, v ;

	cutoff = (fpars[0] == 0.0) ? 0.5 : fpars[0] ;
	dc_component = fpars[1] ;
	ramp_slope = (fpars[2] == 0.0) ? 0.5 : fpars[2] ;
	v = ramp_slope/(float)(nprojs/2) ;

	fx[0] = dc_component * v ;
	fx[1] = ( ramp_slope <= cutoff) ? ramp_slope : 0.0;
	ptr = fx + 2;
	for (i=1 ; i<nprojs/2 ; i++) {
	  *ptr++ = ((float)i*v <= cutoff) ? (float)i*v : 0.0;
	  *ptr++ = 0.0 ;
	}
}

hann( fx, fpars, nprojs)
  float *fx, fpars[];
  int nprojs;
{
	int i;

	fx[1] *= 0.5*(1.0+cos(M_PI*fx[1]/fpars[0])); 
	for (i=1; i<nprojs/2; i++)
	  fx[2*i] *= 0.5*(1.0+cos(M_PI*fx[2*i]/fpars[0]));
}

hamm( fx, fpars, nprojs)
  float *fx, fpars[];
  int nprojs;
{
	int i;

	fx[1] *= 0.54+0.46*cos(M_PI*fx[1]/fpars[0]);
	for (i=1; i<nprojs/2; i++)
	  fx[2*i] *= 0.54+0.46*cos(M_PI*fx[2*i]/fpars[0]);

}

parz( fx, fpars, nprojs)
  float *fx, fpars[];
  int nprojs;
{
	int i;
	float q;

	q = fx[1]/fpars[0];
	fx[1] *= (q<0.5)? 1.0-6.0*q*q*(1.0-q) :
			     2.0*(1.0-q)*(1.0-q)*(1.0-q);

	for (i=1; i<nprojs/2; i++)
	{ q = fx[2*i]/fpars[0];
	  fx[2*i] *= (q<0.5)? 1.0-6.0*q*q*(1.0-q) :
			     2.0*(1.0-q)*(1.0-q)*(1.0-q);
	}
}

float xpow( x, p)
  float x;  int p;
{
	float r=1.0; int i;

	for (i=0; i<p; i++) r=r*x;
	return r;
}

exp_filter( fx, fpars, nprojs)
  float *fx, fpars[];
  int nprojs;
{
	int i, pow;
	float x, cutoff, xp;

	cutoff = fpars[0];
	if (cutoff <= 0.0) cutoff = 0.5;
	pow = (int) fpars[1];
	if (pow < 1) pow = 1;
	for( i=0; i<nprojs/2; i++)
	{
	  x = (float) i/(float)(nprojs/2);
	  xp = xpow(2.0*x/cutoff, pow);
	  fx[2*i] = x*exp(-xp);
	  fx[2*i+1] = 0.0;
	}
	fx[1] = 0.5*exp(-xpow(1.0/cutoff));
}

makefilter(nprojs, fx, fcode, fpars)

int	fcode, nprojs ;
float	*fx, fpars[] ;

{
int	i, retcode ;
float	*ptr ;

	retcode = 1 ;
	switch (fcode) {
	  case NO_FILTER :
	     ptr = fx ;
	     for (i=0 ; i<nprojs/2 ; i++) {
		*ptr++ = 1.0 ;
		*ptr++ = 0.0 ;
	     } 
	     fx[1] = 1.0;
	     break ;
	  case RAMP :
	     ramp(fx, fpars, nprojs) ;
	     break ;
	  case BUTTERWORTH :
	     break ;
	  case HANN :
	     ramp(fx, fpars, nprojs) ;
	     hann(fx, fpars, nprojs) ;
	     break ;
	  case HAMMING:
	     ramp(fx, fpars, nprojs) ;
	     hamm(fx, fpars, nprojs) ;
	     break ;
	  case PARZEN :
	     ramp(fx, fpars, nprojs) ;
	     parz(fx, fpars, nprojs) ;
	     break ;
	  case SHEPP :
	     break ;
	  case EXP:
	     exp_filter(fx, fpars, nprojs);
	     break;
	  case -RAMP :
	     xramp(fx, fpars, nprojs) ;
	     break ;
	  case -BUTTERWORTH :
	     break ;
	  case -HANN :
	     xramp(fx, fpars, nprojs) ;
	     hann(fx, fpars, nprojs) ;
	     break ;
	  case -HAMMING:
	     xramp(fx, fpars, nprojs) ;
	     hamm(fx, fpars, nprojs) ;
	     break ;
	  case -PARZEN :
	     xramp(fx, fpars, nprojs) ;
	     parz(fx, fpars, nprojs) ;
	     break ;
	  case -SHEPP :
	     break ;
	  case -EXP:
	     exp_filter(fx, fpars, nprojs);
	     break;
	  default :
	     retcode = 0 ;
	     break ;

	}  /* end switch */
	return retcode ;
}
