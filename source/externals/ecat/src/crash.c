/* @(#)crash.c	1.1 6/7/90 */

#include <stdio.h>
#include <stdlib.h>

crash( fmt, a0,a1,a2,a3,a4,a5,a6,a7,a8,a9)
  char *fmt, *a0,*a1,*a2,*a3,*a4,*a5,*a6,*a7,*a8,*a9;
{
	fprintf( stderr, fmt, a0,a1,a2,a3,a4,a5,a6,a7,a8,a9);
	exit(1);
}
