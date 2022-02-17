/*	===============================================================================
 *	Module:			img2analyze.c
 *	Date:			07-Apr-94
 *	Author:			Tom Videen
 *	Description:	Transform PETT VI or ECAT images into Analyze format.
 *		Input images may be any format recognizable by getrealimg.
 *		Output images will be displayed by Analyze with left brain on the left
 *		and with the lower slices first.  This allows the 3D volume rendered
 *		brains to appear upright.
 *	History:
 *		29-Nov-94 (TOV)	Create an Interfile Format header file.
 *		21-Jul-95 (TOV) Add "atlas name" and "atlas origin" to ifh output.
 *	  07-Nov-95  modified by sibomana@topo.ucl.ac.be for ECAT 7 support
 *				 non standard keywords start with ";%"
 *				 module name changed to cti2analyze
 *    09-Jul-96 modified  by sibomana@topo.ucl.ac.be : zero voxels below
 *              a threshold argument
 *    02-jan-00 : Add imagedata byte order := bigendian in Interfile header
 *    02-apr-01 : Fix study date bug month := tm.tm_mon+1
 *	==========================================================================
 */

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "load_volume.h"
#include "analyze.h"			 /* dsr */

#define MAXSTR 64
#define TRANSVERSE '\000'
#define CORONAL	'\001'
#define SAGITTAL   '\002'
#define FAIL 1


/*	----------------------------
 *	Function:			cti2analyze
 *	----------------------------
 */
static char *version = "2.0";
static char *program_date = "2000:01:02";
static void usage() {
	fprintf (stderr,
		"cti2analyze version %s date %s\n", version, program_date);
	fprintf (stderr,
		"Usage:\tcti2analyze [-t data_type] [-T min[,max]]  [-c calibration_factor] [-p] -i PET_img -o ANALYZE_img [-h spm]\n");
	fprintf(stderr,"\tdata_type: 2=byte,4=short integer, 16=float (default is short integer)\n");
	fprintf(stderr,"\t-T min[,max] : min and max threshold expressed as percent of image max, Voxels where value<threshold are set to 0\n"); 
	fprintf(stderr,"\t-p: positive only when specified (same as -T 0)\n");
	fprintf(stderr,"\t-h spm: Use spm convention with image origin\n");
	fprintf(stderr,"\t\t stored in the History Originator text field\n");
	fprintf(stderr,"\tANALYZE header is generated with .hdr extension\n");
	fprintf(stderr,"\tInterfile header is generated with .h33 extension\n");
	exit (FAIL);
}

static threshold(volume, t1, t2) 
MatrixData *volume;
float t1, t2;
{
	int i, v, nvoxels;
	float f0, f1, fv;

	nvoxels = volume->xdim * volume->ydim * volume->zdim;
	f0 = volume->scale_factor;
	if (volume->data_type == ByteData) {
		u_char *bdata = (u_char*)volume->data_ptr;
		f1 = t2/255.0;
		for (i=0; i<nvoxels; i++) {
			fv = bdata[i]*f0;
			if (fv > t1) {
				v = (int) ( 0.5 + bdata[i]*f0/f1 );
				if (v<255) bdata[i] = (u_char)v;
				else bdata[i] = 255;
			} else bdata[i] = 0;
		}
	} else {
		short *sdata = (short*)volume->data_ptr;
		f1 = t2/32767.0;
		for (i=0; i<nvoxels; i++) {
			fv = sdata[i]*f0;
			if (fv > t1) {
				v = (int) ( 0.5 + sdata[i]*f0/f1 );
				if (v<32767) sdata[i] = (short)v;
				else sdata[i] = 32767;
			} else sdata[i] = 0;
		}
	}
	volume->scale_factor = f1;
	volume->data_min = t1;
	volume->data_max = t2;
}
		
main (argc, argv)
	int			 argc;
	char		   *argv[];
{

	struct dsr	  hdr;			 /* header for ANALYZE */
	FILE		   *fd_hdr=0;			 /* file for ANALYZE hdr */
	FILE		   *fd_if=0;			 /* output Interfile Format header */
	FILE		   *fd_img=0;			 /* output ANALYZE image  */

	struct Matval   matval;
	MatrixFile	 *file=0;
	MatrixData	 *matrix=0;
	char		   *PET_img=NULL;		 /* input PET image filename */
	char		   PET_fname[256];
	char		   fname[256];	/* output Analyze image filename */
	char		   *ANALYZE_hdr=NULL;	/* output Analyze header filename  */
	char		   *ANALYZE_img=NULL;	/* output Analyze header filename  */
	char		   *IF_hdr=NULL; 		/* output Interfile header filename */
	char			*p=0, *ANALYZE_dir=0, *ANALYZE_base=0;
	char		   *ptr=0;
	char			tmp[80];
	char			study_date[12];
	char			institution[MAXSTR];
	char			model[MAXSTR];
	char			subject_id[MAXSTR];
	char		   atlas_name[MAXSTR];

	int			 zorigin=0;		 /* zorigin read from input file */
	int			 scanner;		 /* 6 = PETT VI;  10 = 953B */
	int			 dimension;		 /* xdim * ydim * num_slices */
	int			 global_max;		 /* global maximum */
	int			 global_min=-32768;		 /* global minimum */
	int			 i, j, k, n;
	int				matnum;

	float		   pixel_size;
	float		   plane_separation;
	short		  *image=0, *plane=0, *line=0;			 /* input PET image */
	int 			cubic = 0;
	char			*buf_line=0; 		/* ANALYZE swap buffer */
	short		   xdim, ydim;		 /* pixel dimensions */
	short		   num_slices;		 /* number of slices */
	struct tm 		tm;
	int 			c, verbose=0, data_type = 4;
	int threshold1=0, threshold2=0;
	float threshold_val1 = 0, threshold_val2 = 0;
	int				bitpix=16;
	u_char		*b_line=NULL;
	short		*s_line=NULL;
	int			spm_header = 0;
	short		spm_origin[5];
	float	 b_scale=1.0, scale_factor=1.0, *f_line=NULL;
	float calibration_factor = 1.0;
	extern char *optarg;

	matval.frame = 1;			 /* default values for matval */
	matval.plane = 1;
	matval.gate = 1;
	matval.data = 0;
	matval.bed = 0;
	matnum = mat_numcod(1,1,1,0,0);

	strcpy (atlas_name, "Talairach 1988");

/*
 *	Get command line arguments and initialize filenames:
 *	---------------------------------------------------
 */


     while ((c = getopt (argc, argv, "i:o:t:T:vh:pc:")) != EOF) {
        switch (c) {
        case 't' :
			sscanf(optarg,"%d",&data_type);
            break;
        case 'i' :
			PET_img = optarg;
            break;
        case 'o' :
            ANALYZE_img = optarg;
			break;
        case 'p' :
			threshold1 = 1;
			if (threshold_val1 < 0) threshold_val1 = 0;
			break;
        case 'T' :
            if (sscanf(optarg,"%f",&threshold_val1) == 1) threshold1 = 1;
            if (sscanf(optarg,"%f,%f",&threshold_val1, &threshold_val2) == 2)
				threshold2 = 1;
            break;
        case 'c' :
            sscanf(optarg,"%f",&calibration_factor);
            break;
        case 'h' :
            if (tolower(*optarg) == 's') 	/* just first char is tested */
				spm_header = 1;
            break;
        case 'v' :
            verbose = 1;
            break;
        }
    }
	if (!PET_img || !ANALYZE_img) usage();
	matspec( PET_img, PET_fname, &matnum);

	ANALYZE_dir = strdup(ANALYZE_img);
	if ( (p = strrchr(ANALYZE_dir,'/')) != NULL)  {
		*p = '\0';
		ANALYZE_base = p+1;
	} else {
		ANALYZE_base = ANALYZE_dir;
		ANALYZE_dir = ".";
	}
	if ((p = strrchr(ANALYZE_base,'.')) != NULL) *p = '\0';
	
	sprintf(fname,"%s/%s.img",ANALYZE_dir, ANALYZE_base);
	if (verbose) fprintf(stderr,"analyze data file : %s\n",fname);
	if (strcmp(PET_fname,fname) == 0) {
		fprintf(stderr,
		"requested analyze data file is identical to CTI image file\n" );
		exit(FAIL);
	}
	if ((fd_img = fopen (fname, "w")) == NULL) {
		perror (fname);
		exit (FAIL);
	}

	if (ANALYZE_hdr == NULL) {
		ANALYZE_hdr = malloc(strlen(ANALYZE_base)+10);
		sprintf(fname,"%s/%s.hdr",ANALYZE_dir, ANALYZE_base);
	} else sprintf(fname,"%s",ANALYZE_hdr);
	if (verbose) fprintf(stderr,"analyze header file : %s\n",fname);
	if ((fd_hdr = fopen (fname, "w")) == NULL) {
		perror(fname);
		exit (FAIL);
	}

	if (IF_hdr == NULL) {
		IF_hdr = malloc(strlen(ANALYZE_base)+10);
		sprintf(fname,"%s/%s.h33",ANALYZE_dir, ANALYZE_base);
	} else sprintf(fname,"%s",IF_hdr);
	if (verbose) fprintf(stderr,"interfile header file : %s\n",fname);
	if ((fd_if = fopen (fname, "w")) == NULL) {
		perror (fname);
		exit (FAIL);
	}
/*
 *	First image, first time -- get scanner and dimensions
 *	------------------------------------------------------------------
 */
	file = matrix_open (PET_fname, MAT_READ_ONLY,MAT_UNKNOWN_FTYPE);
	if (file ==  NULL) 
		crash ("Error: cannot open %s as an ECAT image file\n", PET_img);
	mat_numdoc(matnum, &matval);
	matrix = load_volume(file,matval.frame,cubic);
	if (matrix == NULL) crash ("matrix %s not found\n",PET_img);

	if (calibration_factor!=1.0) matrix->scale_factor *= calibration_factor;
	xdim = matrix->xdim;
	ydim = matrix->ydim;
	num_slices = matrix->zdim;
	pixel_size = 0.001*((int)(matrix->pixel_size*1000+0.5));	/* keep 3 decimal */
	plane_separation = matrix->z_size;
	if ( matrix->data_type != SunShort && matrix->data_type != VAX_Ix2)  {
		fprintf (stderr, "only integer 2 images are currently supported\n");
		exit(1);
	}
	if (verbose) {
		fprintf(stderr,"image\trange : %g %g\n",matrix->data_min,matrix->data_max);
		if (calibration_factor != 1.0) fprintf(stderr,
			"\tscale_factor*calibration_factor(%g) : %g\n",matrix->scale_factor);
		else fprintf(stderr,"\tscale_factor : %g\n",matrix->scale_factor);
	}
	threshold_val1 = threshold_val1*matrix->data_max/100;
	if (threshold2) threshold_val2 = threshold_val2*matrix->data_max/100;
	else threshold_val2 = matrix->data_max;
	if (threshold1 || threshold2) 
		threshold(matrix,threshold_val1, threshold_val2);

	global_min = (int)(matrix->data_min/matrix->scale_factor);
	global_max = (int)(matrix->data_max/matrix->scale_factor);

	printf ("Converting %s\n", PET_img);
/*	Get system, institution, subject_id, study_date */

	sprintf (model, "Siemens/CTI ECAT %d", file->mhptr->system_type);
	strncpy (institution, file->mhptr->facility_name,MAXSTR-1);
	institution[MAXSTR-1] = '\0';
	strncpy (subject_id, file->mhptr->patient_id,MAXSTR-1);
	subject_id[MAXSTR-1] = '\0';
	memcpy(&tm,localtime((time_t*)&file->mhptr->scan_start_time), sizeof(tm));
	sprintf(study_date,"%-d:%-d:%-d",tm.tm_year+1900,tm.tm_mon+1,tm.tm_mday);

/*
	comment_info (file->mhptr->study_description, &comment_info_data);
	zorigin = comment_info_data.zorigin;
*/

/*
 * Flip X, Y & Z, and write to output file
 */

	image = (short*)matrix->data_ptr;
	buf_line = (char*)calloc(xdim,4);
	scale_factor = matrix->scale_factor;
	switch(data_type) {
		case 2 :
			b_line = (u_char*)calloc(xdim,1);
			bitpix = 8;
			if (global_max > 255) {
				b_scale = 255.0/global_max;
				global_max = 255;
			}
			scale_factor = scale_factor/b_scale;
			global_min = 0;
			break;
		case 4:
			global_min = -32768;
			s_line = (short*)calloc(xdim,2);
			break;
		case 16:
			f_line = (float*)calloc(xdim,4);
			bitpix = 32;
			scale_factor = 1.0;
			break;
	}

	for (i = num_slices - 1; i >= 0; i--) {
		plane = image + i*xdim*ydim;
		for (j = ydim - 1; j >= 0; j--) {
			line = plane + j*xdim;
			switch(data_type) {
				case 2:
					for (k = 0; k < xdim; k++)
						b_line[xdim-k-1] = (int)(b_scale*line[k]);
					if (fwrite (b_line, 1, xdim, fd_img) != xdim) {
						perror (fname);
						exit (FAIL);
					}
					break;
				case 4:
					for (k = 0; k < xdim; k++)
						s_line[xdim-k-1] = line[k];
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
					for (k = 0; k < xdim; k++)
						f_line[xdim-k-1] = scale_factor*line[k];
					if (ntohs(1) != 1) {
						swab((char*)f_line, buf_line,xdim*4);
						swaw((short*)buf_line,(short*)f_line,xdim*2);
					}
					if (fwrite (f_line, 4, xdim, fd_img) != xdim) {
						perror (fname);
						exit (FAIL);
					}
			}
		}
	}

/*
 * Create Analyze hdr file
 */

	memset(&hdr,0,sizeof(hdr));
	strncpy (hdr.hk.db_name, PET_fname, 17);
	hdr.hk.sizeof_hdr = sizeof (struct dsr); /* required by developers */
	strcpy(hdr.hk.data_type,"dsr");
	hdr.hk.extents = 16384;			 /* recommended by developers  */
	hdr.hk.regular = 'r';			 /* required by developers */
	hdr.dime.dim[0] = 4;			 /* typically only 4 dimensions  */
	hdr.dime.dim[1] = xdim;			 /* x dimension of atlas   */
	hdr.dime.dim[2] = ydim;			 /* y dimension of atlas   */
	hdr.dime.dim[3] = num_slices;		 /* number of slices in volume */
	hdr.dime.dim[4] = 1;			 /* only one volume typically  */
	strcpy(hdr.dime.vox_units,"mm");
	hdr.dime.bitpix = bitpix;			 /* number of bits/pixel */
	hdr.dime.pixdim[1] = 10. * pixel_size;	 /* should be input for scanner  */
	hdr.dime.pixdim[2] = 10. * pixel_size;	 /* should be input for scanner  */
	hdr.dime.pixdim[3] = 10. * plane_separation;	/* z dimension of atlas   */
	if (verbose) fprintf(stderr,"voxel size : [%g,%g,%g] mm\n",
		hdr.dime.pixdim[1],hdr.dime.pixdim[2],hdr.dime.pixdim[3]);
	hdr.dime.funused1 = scale_factor;
	hdr.dime.datatype = data_type;
	hdr.dime.glmax = global_max;
	hdr.dime.glmin = global_min;
	strncpy (hdr.hist.descrip, PET_fname, 79);
	if (!spm_header) 
		strcpy (hdr.hist.originator, "cti2analy");
	else {
		spm_origin[0] = (xdim+1)/2;
		spm_origin[1] = (xdim+1)/2;
		spm_origin[2] = (num_slices+1)/2;
		spm_origin[3] = spm_origin[4] = 0;
		if (ntohs(1) != 1) swab((char*)spm_origin,hdr.hist.originator,10);
		else  memcpy(hdr.hist.originator,spm_origin,10);
	}
	strncpy (hdr.hist.patient_id, subject_id, 10);
	hdr.hist.orient = TRANSVERSE;
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

	if ((fwrite (&hdr, sizeof (struct dsr), 1, fd_hdr)) != 1) {
		printf ("Error writing to: %s\n", ANALYZE_hdr);
		exit (FAIL);
	}
/*
 * Create Interfile Format header file
 */

	fprintf (fd_if, "INTERFILE :=\n");
	fprintf (fd_if, "version of keys	:= 3.3\n");
	fprintf (fd_if, "image modality := pet\n");
	fprintf (fd_if, "originating system := %s\n", model);
	fprintf (fd_if, "conversion program := cti2analyze\n");
	fprintf (fd_if, "program version	:= %s\n", version);
	fprintf (fd_if, "program date   := %s\n", program_date);
	fprintf (fd_if, "original institution   := %s\n", institution);
	fprintf (fd_if, "name of data file  := %s.img\n", ANALYZE_base);
	fprintf (fd_if, "subject ID	:= %s\n", subject_id);
	fprintf (fd_if, "study date := %s\n", study_date);

	switch(data_type) {
		case 2 :
			fprintf (fd_if, "number format  := unsigned integer\n");
			fprintf (fd_if, "number of bytes per pixel  := 1\n");
			break;
		case 4:
			fprintf (fd_if, "imagedata byte order := bigendian\n");
			fprintf (fd_if, "number format  := signed integer\n");
			fprintf (fd_if, "number of bytes per pixel  := 2\n");
			break;
		case 16:
			fprintf (fd_if, "imagedata byte order := bigendian\n");
			fprintf (fd_if, "number format  := short float\n");
			fprintf (fd_if, "number of bytes per pixel  := 4\n");
			fprintf (fd_if, "maximum pixel count := %g\n",matrix->data_max);
			if (verbose) fprintf(stderr,"new max : %g\n",matrix->data_max);
			break;	
	}
	if (data_type != 16) {
		fprintf (fd_if, "maximum pixel count := %d\n",global_max);
		fprintf (fd_if, ";minimum pixel count := %d\n",global_min);
		if (verbose) fprintf(stderr,"new max : %d\n",global_max);
		if (verbose) fprintf(stderr,"new min : %d\n",global_min);
	}
	fprintf (fd_if, "number of dimensions   := 3\n");
	fprintf (fd_if, "matrix size [1]	:= %d\n", xdim);
	fprintf (fd_if, "matrix size [2]	:= %d\n", ydim);
	fprintf (fd_if, "matrix size [3]	:= %d\n", num_slices);
	fprintf (fd_if, "scaling factor (mm/pixel) [1]  := %f\n", 10 * pixel_size);
	fprintf (fd_if, "scaling factor (mm/pixel) [2]  := %f\n", 10 * pixel_size);
	fprintf (fd_if, "scaling factor (mm/pixel) [3]  := %f\n", 10 * plane_separation);
	fprintf (fd_if, ";%%matrix initial element [1] := right\n");
	fprintf (fd_if, ";%%matrix initial element [2] := posterior\n");
	fprintf (fd_if, ";%%matrix initial element [3] := inferior\n");

	fprintf (fd_if, ";%%atlas name := %s\n", atlas_name);
	fprintf (fd_if, ";%%quantification units := %g\n", scale_factor);
/*
 *	Free & Exit
 *	-----------
 */

	matrix_close(file);
	fclose (fd_hdr);
	fclose (fd_img);
	fclose (fd_if);
	exit (0);
}
