/*	=============================================================================
 *	Module:			analyze2ifh.c
 *	Date:			07-Apr-94
 *	Author:			Tom Videen
 *	Description:	Transform PET VII or ECAT images into Analyze format.
 *		Input images may be any format recognizable by getrealimg.
 *		Output images will be displayed by Analyze with left brain on the left
 *		and with the lower slices first.  This allows the 3D volume rendered
 *		brains to appear upright.
 *	History:
 *		29-Nov-94 (TOV)	Create an Interfile Format header file.
 *		21-Jul-95 (TOV) Add "atlas name" and "atlas origin" to ifh output.
 *      07-Nov-95  modified by sibomana@topo.ucl.ac.be for ECAT 7 support
 *                 non standard keywords start with ";%"
 *                 module name changed to cti2analyze
 *  02-jan-00 : Add imagedata byte order := bigendian in Interfile header
 *	===============================================================================
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "analyze.h"			 /* dsr */

#define MAXSTR 256
#define TRANSVERSE '\000'
#define CORONAL    '\001'
#define SAGITTAL   '\002'
#define FAIL 1

static char     rcsid[] = "$Header: cti2analyze.c,v 1.8 1995/07/21 17:38:14 tom Exp $";

/*	----------------------------
 *	Function:			analyze2ifh
 *	----------------------------
 */
static char            *version = "1.0";
static char	*program_date = "1995:11:08";
main (argc, argv)
	int             argc;
	char           *argv[];
{

	struct dsr      hdr;			 /* header for ANALYZE */
	FILE           *fd_hdr;			 /* file for ANALYZE hdr */
	FILE           *fd_if;			 /* output Interfile Format header */
	FILE           *fd_img;			 /* output ANALYZE image  */

	char           *PET_img=NULL;		 /* input PET image filename */
	char           *ANALYZE_hdr=NULL;	/* output Analyze header filename  */
	char           *ANALYZE_img=NULL;	/* output Analyze image filename */
	char           *IF_hdr=NULL; 		/* output Interfile header filename */
	char			*p, *ANALYZE_base;
	short           xdim, ydim,zdim;		 /* pixel dimensions */
	char            tmp[80];
	char patient_id[12];
	short spm_origin[5];


/*
 *	Get command line arguments and initialize filenames:
 *	---------------------------------------------------
 */

	if (argc < 2) {
		fprintf (stderr,
			"Usage: analyze2ifh ANALYZE_img [IF_hdr]\n");
		exit (FAIL);
	}
	PET_img = argv[1];
	ANALYZE_img = argv[1];
	if (argc > 2) IF_hdr = argv[2];

	ANALYZE_base = strdup(ANALYZE_img);
	if ( (p = strrchr(ANALYZE_base,'/')) == NULL)  p = ANALYZE_base;
	if ((p = strrchr(p,'.')) != NULL) *p = '\0';
	
	ANALYZE_img = malloc(strlen(ANALYZE_base)+10);
	sprintf(ANALYZE_img,"%s.img",ANALYZE_base);

	if (ANALYZE_hdr == NULL) {
		ANALYZE_hdr = malloc(strlen(ANALYZE_base)+10);
		sprintf(ANALYZE_hdr,"%s.hdr",ANALYZE_base);
	}
	fprintf(stderr,"analyze header file : %s\n",ANALYZE_hdr);
	if ((fd_hdr = fopen (ANALYZE_hdr, "r")) == NULL) {
		perror(ANALYZE_hdr);
		exit (FAIL);
	}


	if (IF_hdr == NULL) {
		IF_hdr = malloc(strlen(ANALYZE_base)+10);
		sprintf(IF_hdr,"%s.h33",ANALYZE_base);
	}
	fprintf(stderr,"interfile header file : %s\n",IF_hdr);
	if ((fd_if = fopen (IF_hdr, "w")) == NULL) {
		perror (IF_hdr);
		exit (FAIL);
	}

/*
 * Read Analyze hdr file
 */

	if (fread(&hdr,sizeof(struct dsr),1,fd_hdr) < 0) {
		perror(ANALYZE_hdr);
		exit (FAIL);
	}
	if (ntohs(1) != 1) {
		hdr.hk.sizeof_hdr = ntohl(hdr.hk.sizeof_hdr);
		hdr.hk.extents = ntohl(hdr.hk.extents);
		swab((char*)hdr.dime.dim,tmp,8*sizeof(short));
		memcpy(hdr.dime.dim,tmp,8*sizeof(short));
		hdr.dime.datatype = ntohs(hdr.dime.datatype);
		hdr.dime.bitpix = ntohs(hdr.dime.bitpix);
		swab((char*)hdr.dime.pixdim,tmp,8*sizeof(float));
		swaw((short*)tmp,(short*)hdr.dime.pixdim,8*sizeof(float)/2);
		swab((char*)&hdr.dime.funused1,(char*)tmp,sizeof(float));
        swaw((short*)tmp,(short*)&hdr.dime.funused1,sizeof(float)/2);
		hdr.dime.glmax = ntohl(hdr.dime.glmax);
		hdr.dime.glmin = ntohl(hdr.dime.glmin);
	}
	strncpy(patient_id,hdr.hist.patient_id,10);
	patient_id[10] ='\0';
	xdim = hdr.dime.dim[1]; ydim = hdr.dime.dim[2]; zdim = hdr.dime.dim[3];

	fprintf (fd_if, "INTERFILE :=\n");
	fprintf (fd_if, "version of keys    := 3.3\n");
	fprintf (fd_if, "conversion program := analyze2ifh\n");
	fprintf (fd_if, "program version    := %s\n", version);
	fprintf (fd_if, "program date   := %s\n", program_date);
	fprintf (fd_if, "original institution   := %s\n",hdr.hist.originator);
	fprintf (fd_if, "name of data file  := %s\n", ANALYZE_img);
	fprintf (fd_if, "subject ID	:= %s\n", hdr.hist.patient_id);
    switch(hdr.dime.datatype) {
        case 2 :
            fprintf (fd_if, "number format  := unsigned integer\n");
            fprintf (fd_if, "number of bytes per pixel  := 1\n");
            fprintf (fd_if, "maximum pixel count := %d\n",hdr.dime.glmax);
            fprintf (fd_if, ";minimum pixel count := %d\n",hdr.dime.glmin);
            break;
        case 4:
            fprintf (fd_if, "number format  := signed integer\n");
    		fprintf (fd_if, "imagedata byte order := bigendian\n");
            fprintf (fd_if, "number of bytes per pixel  := 2\n");
            fprintf (fd_if, "maximum pixel count := %d\n",hdr.dime.glmax);
            fprintf (fd_if, ";minimum pixel count := %d\n",hdr.dime.glmin);
            break;
        case 16:
            fprintf (fd_if, "number format  := short float\n");
    		fprintf (fd_if, "imagedata byte order := bigendian\n");
            fprintf (fd_if, "number of bytes per pixel  := 4\n");
            fprintf (fd_if, "maximum pixel count := %d\n",hdr.dime.glmax);
            fprintf (fd_if, ";minimum pixel count := %d\n",hdr.dime.glmin);
            break;
    }
	fprintf (fd_if, "number of dimensions   := 3\n");
	fprintf (fd_if, "matrix size [1]    := %d\n", xdim);
	fprintf (fd_if, "matrix size [2]    := %d\n", ydim);
	fprintf (fd_if, "matrix size [3]    := %d\n", zdim);
	fprintf (fd_if, "scaling factor (mm/pixel) [1]  := %f\n",
		hdr.dime.pixdim[1]);
	fprintf (fd_if, "scaling factor (mm/pixel) [2]  := %f\n",
		hdr.dime.pixdim[2]);
	fprintf (fd_if, "scaling factor (mm/pixel) [3]  := %f\n",
		hdr.dime.pixdim[3]);
	fprintf (fd_if, ";%%matrix initial element [1] := right\n");
	fprintf (fd_if, ";%%matrix initial element [2] := posterior\n");
	fprintf (fd_if, ";%%matrix initial element [3] := inferior\n");
					/* check spm origin */
	if (ntohs(1) != 1) swab(hdr.hist.originator,(char*)spm_origin,10);
	else memcpy(spm_origin,hdr.hist.originator,10);
	if (spm_origin[0]>1 && spm_origin[1]>1 && spm_origin[2]>1 &&
		spm_origin[0]<xdim && spm_origin[1]<ydim && spm_origin[2]<zdim) {
		fprintf (fd_if, ";%%atlas origin [1] := %d\n", spm_origin[0]);
		fprintf (fd_if, ";%%atlas origin [2] := %d\n", spm_origin[1]);
		fprintf (fd_if, ";%%atlas origin [3] := %d\n", spm_origin[2]);
	}
	fprintf (fd_if, ";%%quantification units := %g\n", hdr.dime.funused1);

	fclose (fd_hdr);
	fclose (fd_if);
	exit (0);
}
