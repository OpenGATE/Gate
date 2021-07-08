/*	===============================================================================
 *	Module:			updareanh.c
 *	Date:			09-Jul-98
 *	Author:			Tom Videen
 *	Description:	add the analyze magic number to analyze header
 *	History:
 *		29-Nov-94 (TOV)	Create an Interfile Format header file.
 *		21-Jul-95 (TOV) Add "atlas name" and "atlas origin" to ifh output.
 *      07-Nov-95  modified by sibomana@topo.ucl.ac.be for ECAT 7 support
 *                 non standard keywords start with ";%"
 *                 module name changed to cti2analyze
 *		from analyze2ifh a.coppens
 *	===============================================================================
 */

#include <stdio.h>
#include <math.h>
#include <time.h>
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
static void swaw( from, to, length)
  short *from, *to;
  int length;
{
    short int temp;
    int i;

    for (i=0;i<length; i+=2)
    {  temp = from[i+1];
       to[i+1]=from[i];
       to[i] = temp;
    }
}

main (argc, argv)
	int             argc;
	char           *argv[];
{

	struct dsr      hdr;			 /* header for ANALYZE */
	FILE           *fd_hdr;			 /* file for ANALYZE hdr */
	FILE           *fd_img;          /* output ANALYZE image  */
	char           fname[256];  /* output Analyze image filename */
	
	char           *ANALYZE_hdr=NULL;	/* output Analyze header filename  */
	char           *ANALYZE_base=NULL;	/* output Analyze header filename  */
	char* p;
	int c;
	char            tmp[80];
	float   scale_factor = -1;
	float input_scale_factor = -1;
	int flip = 0;
	short         *image, *plane, *line;             /* input PET image */
	char            *buf_line;      /* ANALYZE swap buffer */
	short       *s_line=NULL,*S_line;
	u_char       *b_line=NULL,*B_line;
	float       *f_line=NULL,*F_line;
	int xdim,ydim,num_slices,data_type;
	int i,j,k;
	int elem_size,size;

	extern char *optarg;

/*
 *  Get command line arguments and initialize filenames:
 *  ---------------------------------------------------
 */
	while ((c = getopt (argc, argv, "s:i:f")) != EOF) {
		switch (c) {
			case 'i' :
				ANALYZE_hdr = optarg;
			break;
			case 's' :
				sscanf(optarg,"%g",&scale_factor);
			break;
			case 'f' :
				flip = 1;
			break;
		}
	}
	if (ANALYZE_hdr ==NULL) {
		fprintf (stderr,
			 "Usage: updateanh -i ANALYZE_hdr [-s scale_factor -f ]\n");
		fprintf (stderr,"	 -f for flip\n");
		exit (FAIL);
	}


	fprintf(stderr,"analyze header file : %s\n",ANALYZE_hdr);
	if ((fd_hdr = fopen (ANALYZE_hdr, "r")) == NULL) {
		perror(ANALYZE_hdr);
		exit (FAIL);
	}


/*
 * Read Analyze hdr file
 */

	if (fread(&hdr,sizeof(struct dsr),1,fd_hdr) < 0) {
		perror(ANALYZE_hdr);
		exit (FAIL);
	}
	fclose (fd_hdr);
/*
*
* update magic number
*
*/
	if (ntohs(1) != 1) {
		hdr.hk.sizeof_hdr = ntohl(hdr.hk.sizeof_hdr);
		if (hdr.hk.sizeof_hdr != sizeof(struct dsr)) 
			hdr.hk.sizeof_hdr =  sizeof(struct dsr);
		hdr.hk.sizeof_hdr = htonl(hdr.hk.sizeof_hdr);
	} else {
		if (hdr.hk.sizeof_hdr != sizeof(struct dsr))
			hdr.hk.sizeof_hdr =  sizeof(struct dsr);
	}
/*
*
* update scale factor
*
*/
	if (ntohs(1) != 1) {
		swab(&hdr.dime.funused1,tmp,sizeof(float));
		swaw(tmp,&hdr.dime.funused1,sizeof(float)/2);
	}
	input_scale_factor = hdr.dime.funused1;
	if (scale_factor <= 0) {
		if (input_scale_factor <= 0) {
			input_scale_factor = 0;
		}
	} else {
		input_scale_factor = scale_factor;
	}
	hdr.dime.funused1 = input_scale_factor;
	if (ntohs(1) != 1) {
		swab(&hdr.dime.funused1,tmp,sizeof(float));
		swaw(tmp,&hdr.dime.funused1,sizeof(float)/2);
	}
	
		
	if ((fd_hdr = fopen (ANALYZE_hdr, "w")) == NULL) {
		perror(ANALYZE_hdr);
		exit (FAIL);
	}
	if (fwrite(&hdr,sizeof(struct dsr),1,fd_hdr) < 0) {
        perror(ANALYZE_hdr);
        exit (FAIL);
    }
	fclose (fd_hdr);

/*
*
* flip images
*
*/
	
	ANALYZE_base = strdup(ANALYZE_hdr);

	if ((p = strrchr(ANALYZE_base,'.')) != NULL) *p = '\0';
	sprintf(fname,"%s.img",ANALYZE_base);
	if (flip) {
		if (ntohs(1) != 1) {
			hdr.dime.dim[1] = ntohs(hdr.dime.dim[1]);
			hdr.dime.dim[2] = ntohs(hdr.dime.dim[2]);
			hdr.dime.dim[3] = ntohs(hdr.dime.dim[3]);
			hdr.dime.datatype = ntohs(hdr.dime.datatype);
		}
		xdim = hdr.dime.dim[1];
		ydim = hdr.dime.dim[2];
		num_slices = hdr.dime.dim[3];
		data_type = hdr.dime.datatype;
		size = xdim*ydim*num_slices;
		buf_line = (char*)calloc(xdim,4);
		switch(data_type) {
			case 2:
				elem_size=1;
				b_line = (u_char*)calloc(xdim,1);

			break;
			case 4:
				elem_size=2;
				s_line = (short*)calloc(xdim,2);
			break;
			case 16:
				elem_size=4;
				f_line = (float*)calloc(xdim,4);
			break;
		}
		if ((fd_img = fopen (fname, "r")) == NULL) {
			perror (fname);
			exit (FAIL);
		}
fprintf(stderr, "data_type %d elem_size %d\n",data_type,elem_size);
		image = (short*)calloc(size,elem_size);

		switch(data_type) {
			case 2:
				if (fread((char*)image,elem_size,size,fd_img)< size) {
					fprintf(stderr, "cant read %s \n",fname);
					fclose (fd_img);
				}
			break;
			case 4:
				if (fread(image,elem_size,size,fd_img)< size) {
					fprintf(stderr, "cant read %s \n",fname);
					fclose (fd_img);
				}
			break;
			case 16:
				if (fread((float*)image,elem_size,size,fd_img)< size) {
					fprintf(stderr, "cant read %s \n",fname);
					fclose (fd_img);
				}
			break;
		}
		fclose (fd_img);
		if ((fd_img = fopen (fname, "w")) == NULL) {
			perror (fname);
			exit (FAIL);
		}

		for (i = 0; i <num_slices; i++) {
			for (j = 0; j <ydim ; j++) {
				switch(data_type) {
					case 2:
						B_line = (u_char*)image + i*xdim*ydim + j*xdim;
						for (k = 0; k < xdim; k++)
							b_line[xdim-k-1] = B_line[k];
						if (fwrite (b_line, 1, xdim, fd_img) != xdim) {
							perror (fname);
							exit (FAIL);
						}
					break;
					case 4:
						S_line = image + i*xdim*ydim + j*xdim;
						if (ntohs(1) != 1) {
							swab((char*)S_line, buf_line,xdim*2);
							memcpy(S_line,buf_line,xdim*2);
						}
						for (k = 0; k < xdim; k++)
							s_line[xdim-k-1] = S_line[k];
						if (ntohs(1) != 1) {
							swab((char*)s_line, buf_line,xdim*2);
							memcpy(s_line,buf_line,xdim*2);
						}
						if (fwrite (s_line, 2, xdim, fd_img) != xdim) {
							perror (fname);
							exit (FAIL);
						}
					break;
					case 16:
						F_line = (float*)image + i*xdim*ydim + j*xdim;
						if (ntohs(1) != 1) {
							swab((char*)F_line, buf_line,xdim*4);
							swaw((short*)buf_line,(short*)F_line,xdim*2);
						}
						for (k = 0; k < xdim; k++)
							f_line[	xdim-k-1] = F_line[k];
						if (ntohs(1) != 1) {
							swab((char*)f_line, buf_line,xdim*4);
							swaw((short*)buf_line,(short*)f_line,xdim*2);
						}
						if (fwrite (f_line, 4, xdim, fd_img) != xdim) {
							perror (fname);
							exit (FAIL);
						}
					break;
				}
			}
		}
		fclose(fd_img);
	}
	exit (0);
}
