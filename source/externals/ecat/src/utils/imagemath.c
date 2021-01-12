/* @(#)imagemath.c	1.1 4/26/91 */

#ifndef	lint
static char sccsid[]="@(#)imagemath.c	1.1 4/26/91 Copyright 1990 CTI Pet Systems, Inc.";
#endif

/*
 * Feb-1996 :Updated by Sibomana@topo.ucl.ac.be for ECAT 7.0 support
 * Feb 2000: Updated by Michel@topo.ucl.ac.be to add masking operations:
 *	     imagec=c2*imagea if (imageb {gt,lt} c3) when op1={gt,lt}
 * Dec 2000 V2.3 :
 *       Updated by Sibomana@topo.ucl.ac.be to add matnum as argument
 * 02-Nov-2001 V2.4 :
 *       add bin as op2 operation to binarize image
 */
   

#include "matrix.h"
#include <math.h>
#include <string.h>
static char *version = "imagemath V2.4 02-Nov-2001";
static usage(pgm) 
char *pgm;
{
	fprintf(stderr,"usage: %s imagea imageb imagec c1,c2,c3 op1,op2\n%s%s",pgm,
		 "\twhere op1={+,-,*,/,not,and,gt,lt} and op2={log,exp,sqr,bin}\n",
		 "\timagec=op2(c1+(c2*imagea)op1(c3*imageb)) when op1={+,-,*,/,not,and}\n");
	fprintf(stderr,"and \timagec=c2*imagea if (imageb {gt,lt} c3) when op1={gt,lt}\n");
	fprintf(stderr,"\timageb may be a matrix_spec or a constant value\n");
	fprintf(stderr,"\timageb may be a matrix_spec or a constant value\n");
	fprintf(stderr,"\t%s\n", version);
	exit(1);
}

main( argc, argv)
  int argc;
  char **argv;
{
	MatrixFile *file1=NULL, *file2=NULL, *file3=NULL;
	MatrixData *image1=NULL, *image2=NULL, *image3=NULL;
	MatrixData *slice1=NULL, *slice2=NULL;
	Main_header *mh3=NULL;
	Image_subheader *imh;
	float *imagea, *imageb, *imagec;
	float valb;
	short int *sdata;
	u_char *bdata;
	float maxval, minval, scalef;
	int i,j, matnuma, matnumb, matnumc, plane, nblks;
	char op1 = ' ', op2=' ',  *ptr, fname[256];
	float c1=0,c2=1,c3=1;
	FILE *fptr;
	int npixels, nvoxels;
	int segment = 0;

	if (argc<6) usage(argv[0]);
/* get imagea */
	if (!matspec( argv[1], fname, &matnuma)) matnuma = mat_numcod(1,1,1,0,0);
	file1 = matrix_open( fname, MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
	if (!file1)
	  crash( "%s: can't open file '%s'\n", argv[0], fname);
	image1 = matrix_read(file1,matnuma,MAT_SUB_HEADER);
	if (!image1) crash( "%s: image '%s' not found\n", argv[0], argv[1]);
	switch(file1->mhptr->file_type) {
	case InterfileImage :
	case PetImage :
	case PetVolume :
	case ByteImage :
	case ByteVolume :
		break;
	default :
		crash("input is not a Image nor Volume\n");
		break;
	}
	if (!matspec( argv[2], fname, &matnumb)) matnumb = mat_numcod(1,1,1,0,0);
	file2 = matrix_open( fname, MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
	if (!file2) {		/* check constant value argument */
	  if (sscanf(argv[2],"%g",&valb) != 1)
			crash("%s: can't open file '%s'\n", argv[0], fname);
	}
	if (file2) {
		image2 = matrix_read(file2,matnumb,MAT_SUB_HEADER);
		if (!image2) crash( "%s: image '%s' not found\n", argv[0], argv[2]);
		if (image1->xdim != image2->xdim || image1->ydim != image2->ydim ||
			image1->zdim != image2->zdim)
			crash("%s and %s are not compatible\n",argv[1], argv[2]);
	}
	npixels = image1->xdim*image1->ydim;
	nvoxels = npixels*image1->zdim;

/* get imagec specification and write header */
	if (!matspec( argv[3], fname, &matnumc)) matnumc = mat_numcod(1,1,1,0,0);
	mh3 = (Main_header*)calloc(1, sizeof(Main_header));
	memcpy(mh3, file1->mhptr, sizeof(Main_header));
	mh3->file_type = PetVolume;
	file3 = matrix_create( fname, MAT_OPEN_EXISTING, mh3);
	if (!file3) crash( "%s: can't open file '%s'\n", argv[0], fname);
	image3 = (MatrixData*)calloc(1, sizeof(MatrixData));
	memcpy(image3, image1, sizeof(MatrixData));
	imh = (Image_subheader*)calloc(1,sizeof(Image_subheader));
	memcpy(imh,image1->shptr,sizeof(Image_subheader));
	imagec = (float*)malloc(nvoxels*sizeof(float));
	imagea = (float*)malloc(npixels*sizeof(float));
	imageb = (float*)malloc(npixels*sizeof(float));

/* Decode Operators and Coefficients */
	sscanf( argv[4], "%f,%f,%f", &c1, &c2, &c3);
	ptr = strtok(argv[5],",");
	op1 = *ptr;
	if ((ptr=strtok(NULL,"'")) != NULL) {
		if (!strcmp( ptr, "log")) op2 = 'l';
		else if (!strcmp( ptr, "exp")) op2 = 'e';
			else if (!strcmp( ptr, "sqr")) op2 = 's';
				else if (!strcmp( ptr, "bin")) op2 = 'b';
	}

/* compute and write data */
	if (!file2) {
		for (i=0; i<npixels; i++) imageb[i] = valb;
	}
	for (plane=0; plane< image1->zdim; plane++) {
		slice1 = matrix_read_slice(file1,image1,plane,segment);
		if (slice1->data_type==SunShort || slice1->data_type==VAX_Ix2) {
			sdata = (short*)slice1->data_ptr;
			for (i=0; i<npixels; i++)
	  			imagea[i] = slice1->scale_factor*sdata[i];
		} else {	/* assume byte data */
			bdata = (u_char*)slice1->data_ptr;
			for (i=0; i<npixels; i++)
				imagea[i] = slice1->scale_factor*bdata[i];
		}
		free_matrix_data(slice1);

		if (file2) {
			slice2 = matrix_read_slice(file2,image2,plane,segment);
			if (slice2->data_type==SunShort || slice2->data_type==VAX_Ix2) {
				sdata = (short*)slice2->data_ptr;
				for (i=0; i<npixels; i++)
	  				imageb[i] = slice2->scale_factor*sdata[i];
			} else {    /* assume byte data */
				bdata = (u_char*)slice2->data_ptr;
				for (i=0; i<npixels; i++)
					imageb[i] = slice2->scale_factor*bdata[i];
			}
			free_matrix_data(slice2);
		}

		switch( op1)
		{
	  	case '+':
			for (i=0; i<npixels; i++)
			  imagea[i] = c1+c2*imagea[i]+c3*imageb[i];
			break;
	  	case '-':
			for (i=0; i<npixels; i++)
			  imagea[i] = c1+c2*imagea[i]-c3*imageb[i];
			break;
	  	case '*':
			for (i=0; i<npixels; i++)
			  imagea[i] = c1+c2*imagea[i]*c3*imageb[i];
			break;
	  	case '/':
			for (i=0; i<npixels; i++)
			  imagea[i] = c1+(imageb[i]==0.0) ? 0.0 :
				c2*imagea[i]/(c3*imageb[i]);
			break;
		case 'a' :		/* c2*imagea if (imageb!=0) */
			for (i=0; i<npixels; i++)
			imagea[i] = (imageb[i]==0.0)? 0.0 : c2*imagea[i];
			break;
		case 'n' :	/* c2*imagea if (imageb==0) */
			for (i=0; i<npixels; i++)
			imagea[i] = (imageb[i]==0.0)? c2*imagea[i] : 0.0;
			break;
		case 'g' :	/* c2*imagea if (imageb>c3) */
			for (i=0; i<npixels; i++)
			imagea[i] = (imageb[i]>c3)? c2*imagea[i] : 0.0;
			break;
		case 'l' :	/* c2*imagea if (imageb<c3) */
			for (i=0; i<npixels; i++)
			imagea[i] = (imageb[i]< c3)? c2*imagea[i] : 0.0;
			break;

	  	default:
			crash("%s: illegal operator \"%c\"...chose from {+,-,*,/,and,not,gt,lt}\n",
			argv[0], op1);
		}
		switch (op2) {
		case 'b' :
	  		for (i=0; i<npixels; i++)
	    		imagea[i] = (imagea[i] > 0.0) ? 1 : 0;
			break;
		case 'l' :
	  		for (i=0; i<npixels; i++)
	    		imagea[i] = (imagea[i] < 0.0) ? 0.0 : log(imagea[i]);
			break;
		case 'e' :
			for (i=0; i<npixels; i++)
				imagea[i] = exp(imagea[i]);
			break;
		case 's' :
	  		for (i=0; i<npixels; i++)
	    		imagea[i] = (imagea[i] < 0.0) ? 0.0 : sqrt(imagea[i]);
			break;
	/*	default	:  no operation */
		}
		memcpy(imagec+npixels*plane, imagea, npixels*sizeof(float));
	}
/* scale the output image */

	minval = maxval = imagec[0];
	for (i=0; i<nvoxels; i++)
	{
	  if (imagec[i]>maxval) maxval = imagec[i];
	  if (imagec[i]<minval) minval = imagec[i];
	}

	if (image1->data_type!=ByteData || image2->data_type!=ByteData || minval<0)
	{
		image3->shptr = (caddr_t)imh;
		nblks = (nvoxels*sizeof(short)+511)/512;
		image3->data_size = nblks*512;
		image3->data_type = imh->data_type = SunShort;
		if (fabs(minval) < fabs(maxval)) scalef = fabs(maxval)/32767.;
		else scalef = fabs(minval)/32767.;
		sdata = (short*)imagec;				/* reuse huge float array */
		for (i=0; i<nvoxels; i++)
	  		sdata[i] = (short)(0.5+imagec[i]/scalef);
	} else {
		image3->shptr = (caddr_t)imh;
		image3->data_type = imh->data_type = ByteData;
		nblks = (nvoxels*sizeof(u_char)+511)/512;
		image3->data_size = nblks*512;
		scalef = fabs(maxval)/255;
		bdata = (u_char*)imagec;				/* reuse huge float array */
		for (i=0; i<nvoxels; i++)
			bdata[i] = (u_char)(0.5+imagec[i]/scalef);
	}
	image3->data_ptr = (caddr_t)imagec;
	image3->scale_factor = imh->scale_factor = scalef;
	imh->image_max = (short)((0.5+maxval)/scalef);
	imh->image_min = (short)((0.5+minval)/scalef);
	strncpy( imh->annotation, version, 40);
	matrix_write(file3, matnumc, image3);
	matrix_close(file1);
	if (file2) matrix_close(file2);
	matrix_close(file3);
}
