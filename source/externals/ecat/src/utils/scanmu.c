/* @(#)import_2dscan.c	1.6 8/5/97 

	Prepare 2D scan data for 2D reconstruction with  bkproj3d
	---------------------------------------------------------
	Due to memory limitation in Quad SC of BC1, a 2D view should fit in a 256*47 matrix (2d fft limitation), 
	trim should be applied to reduce the FOV from 336 to 256.
	When part of the head is outside the 42 cm FOV (.165*256), it should be recentered
	When the object is larger than the 42 cm FOV (cardiac studies), rebin should be applied (2 is enough)
	Rotation is introduced at reconstruction but the intrinsic tilt should be introduced
	to correct the x and y offsets.
	Correct scan scale factor with 1/rebin**2 for quantification
		please note that bkproj3d allows axial smoothing ...
	
	Version 1.4: correct decay_correction factor: exact formula rather than Taylor expansion....
	Version 1.5: handle multibed positions (February 1997)
	Version 1.6: fills processing code (May 1997)... 
*/

#include <stdio.h>
#include <math.h>
#include <matrix.h>
#include <string.h>
#include <sys/types.h>
#include <sys/file.h>

#ifndef lint
	static char sccsid[]="@(#)import_2dscan.c  version 1.6  8 May 97 - UCL- C. Michel";
#endif
static char *progname;
 
main(argc, argv)
  int 		argc;
  char 		*argv[];
{
   short int	*scand;
   int       	numProjs, numViews, numPlanes, numFrames, numBeds, trim=336, mash=1, d_rebin=1, nv, np, i, j, verbose=0;
   int       	deadtimeFlag=0, decayFlag=0, arcorFlag=0, normFlag=0, attnFlag=0, hhattnFlag=0;
   int       	frame=1, plane=1, bed=0;
   int       	arg, scan_matnum, nrm_matnum=0, attn_matnum=0, hhattn_matnum=0;
   int      	TotalNviews=0, input_mash=1;
   int		scanSize;
   int 		c;

   float       *fdata=NULL, *ftmp=NULL, *fmash=NULL;
   float       sf=1.0;
   double      ln2, exp(), log(), lamdaT;

   MatrixFile    *sinfp=NULL, *soutfp=NULL, *nfp=NULL, *hhafp=NULL, *afp=NULL;
   MatrixData    *scan , *norm, *hhattn, *attn; 

   Main_header     *mh;
   Scan_subheader  *ssh;
   char *scanin, *scanout;
   extern char *optarg;
   extern int optind, opterr;

	float *m1, *m2, *mult;

    char* in_file =0;
    char* out_file =0;
    char* mult_file =0;
    int im_num=1;
    FILE* mult_fp;



   progname = argv[0];
   if (argc < 3) {
      crash("usage: %s -i scanin -m multfile -o scanout -n num -v\n", progname);
   }
 

	while ((c = getopt (argc, argv, "i:o:m:n:v")) != EOF) {
		switch (c) {
			case 'i' :
				in_file = optarg;
				break;
			case 'o' :
				out_file = optarg;
				break;
			case 'm' :
				mult_file = optarg;
				break;
			case 'n' :
				sscanf(optarg,"%d",&im_num);
				break;
			case 'v' :
				verbose=1;
				break;
		}
	}

	if (in_file == 0 || out_file == 0 || mult_file == 0) usage();

   sinfp = matrix_open(in_file, MAT_READ_ONLY, MAT_SCAN_DATA);
   if (!sinfp) matrix_perror( progname);

/* fill-up main header for output file */

  mh = sinfp->mhptr;
  numPlanes = mh->num_planes;
  numFrames = mh->num_frames;
  numBeds = mh->num_bed_pos;
	mh->num_frames=1;
  soutfp = matrix_create( out_file, MAT_OPEN_EXISTING, mh);
  if (verbose) fprintf (stdout,"Study has %3d  planes and %3d frames and %3d beds\n", numPlanes, numFrames, numBeds);

	m1 = (float*)malloc(numFrames*sizeof(float));
    m2 = (float*)malloc(numFrames*sizeof(float));
/* Decode multiplicative factors */
    if ((mult_fp = fopen(mult_file,"r")) == NULL) {
        crash("scanmult: can't open %s\n",mult_file);
    }
    for (i=0;i<numFrames; i++)
        if (fscanf(mult_fp,"%g %g",&m1[i],&m2[i]) != 2)
            crash("error in %s\n",mult_file);
    fclose(mult_fp);
    switch (im_num) {
        case 1:
            mult=m1;
            break;
        case 2:
            mult=m2;
            break;
    }

    if (verbose)
        for (j = 0; j<numFrames; j++)
            fprintf(stderr,"multiplicative factors %d: %g\n",j,mult[j]);

	scan = 0;
  for (frame=1;frame<= numFrames; frame++) {
  	for (plane=1; plane<=numPlanes; plane++) {
     	if (verbose) fprintf (stdout,"\nProcessing plane %d from frame %d and from bed %d\n", plane, frame, bed);
     	scan_matnum = mat_numcod (frame, plane, 1, 0, bed);
     	if (scan) matrix_free( scan);
     	scan = matrix_read( sinfp, scan_matnum, GENERIC);
     	if (!scan) crash( "..can't read scan matrix for frame %d, plane %d, bed %d\n", frame, plane, bed);

     	scand = (short int*) scan->data_ptr;
     	ssh = (Scan_subheader *) scan->shptr;
     	numProjs = scan->xdim;
     	numViews = scan->ydim;
     	if (verbose) fprintf (stdout,"num_Projs = %3d  num_Views = %3d\n", numProjs, numViews);
     	scanSize = numProjs*numViews;
     	if (!fdata) fdata = (float*) calloc( scanSize*numPlanes , sizeof(float));
     	sf = scan->scale_factor;
     	if (verbose) fprintf (stdout, "Scan scale factor = %f\n", sf);
	

    	for (i=0; i<scanSize; i++) 
			fdata[i+(plane-1)*scanSize] += sf*scand[i]*mult[frame-1];


	}
  }
  /* Convert to short and write */
  	for (plane=1; plane<=numPlanes; plane++) {
     	scan_matnum = mat_numcod (1,plane, 1, 0, 0);
     	if (scan) matrix_free( scan);
     	scan = matrix_read( sinfp, scan_matnum, GENERIC);
   		convert_float_scan( scan, fdata+(plane-1)*scanSize);
     	matrix_write(soutfp, scan_matnum, scan);
	}


   mat_close(sinfp);
   mat_close(soutfp);
}
