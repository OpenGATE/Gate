#include <stdlib.h>
#include <stdio.h>
# define RMAX 32	/* Maximum number of crystal rings supported. */


void plandefs(rmax,span,amax,vplane)

	int	rmax;	/* Number of crystal rings. (May be virtual) */
	int	amax;	/* Maximum number of cross planes to accept
			 * for all projections. (Acceptance angle) */
	int span;	/* Axial compression */
	int	*vplane;/* An rmax by rmax array to receive the plane numbers */
{
	int	dmax;	/* Maximum number of cross planes to accept
			 * in a projection. */
	int	nmax;	/* Maximum angular projection index. */
	int	zindex[ RMAX ];	/* The index of the begining of the
				 * positive projection.	*/
	int	zsize[ RMAX ];	/* The z offset to the begining of the projection. */
	int	nvdp;	/* The number of virtual direct planes */
	int	i;
	int	r1,r2;	/* Ring numbers. */
	int	nabs;	/* Projection index. */
	int	plane;	/* The virtual plane number */
	
	/* Compute plane offsets at the begining of the projections and
	 * the size of the projections.	*/
	
	dmax = (span+1)/2;
	nmax = amax/(2*dmax -1);
	nvdp = 2*rmax-1;
	zsize[0] = 0;
	zindex[0] = 0;
	zsize[1] = dmax;
	zindex[1] = nvdp;
	for( i=2; i <= nmax; i++)
	{
		zindex[i] = zindex[i-1]+2*(nvdp-2*zsize[i-1]);
		zsize[i] = (2*i-1)*dmax-i+1;
	}
	
	/* Compute the plane number for each pair of crystals */
	for (r1=0; r1 < rmax; r1++)
		for(r2=0; r2 < rmax; r2++)
		{
			nabs = ( abs(r1-r2) + dmax - 1 ) / (2*dmax-1);
			if (nabs <= nmax)
			{
				plane = (r1 + r2) - zsize[nabs];
				plane += zindex[nabs];
				if ( ((r1 - r2) < 0) && (nabs > 0) )
					plane += nvdp-2*zsize[nabs];
			}
			else
				plane = -1;
			vplane[r1*rmax+r2] = plane;
		}
	return;
}
			
#ifdef TEST
static void usage() {
	fprintf(stderr,
		"plandefs rings span ring_difference\n");
	exit(1);
}

int main(argc, argv)
	int argc;
	void *argv[];
{
	int rmax,span,amax;
	int i,j;
	int *vplane;
	
	if (argc < 4) usage();
	if (sscanf(argv[1],"%d",&rmax) != 1) usage();
	if (sscanf(argv[2],"%d",&span) != 1) usage();
	if (sscanf(argv[3],"%d",&amax) != 1) usage();
	vplane = (int*)calloc(rmax*rmax,sizeof(int));
	
	plandefs(rmax,span,amax,vplane);
	for (i=0; i<rmax; i++) {
		for (j=0; j<rmax; j++) printf("%3d ",vplane[(rmax-i-1)*rmax+j]);
		printf("\n");
	}
}
#endif
