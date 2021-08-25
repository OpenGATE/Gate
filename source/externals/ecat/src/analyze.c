/* 
// sccsid = "@(#)analyze.c	1.5  UCL-TOPO	98/04/28"
/*
 *  Copyright (C) 1995 University of Louvain, Louvain-la-Neuve, Belgium
 *
 *  Author : <Merence Sibomana> Sibomana@topo.ucl.ac.be
 *
 *	  Positron Emission Tomography Laboratory
 *	  Universite Catholique de Louvain
 *	  Ch. du Cyclotron, 2
 *	  B-1348 Louvain-la-Neuve
 *		  Belgium
 *
 *  This program may be used free of charge by the members
 *  of all academic and/or scientific institutions.
 *	   ANY OTHER USE IS PROHIBITED.
 *  It may also be included in any package
 *	  -  provided that this text is not removed from
 *	  the source code and that reference to this original
 *	  work is done ;
 *	  - provided that this package is itself available
 *	  for use, free of charge, to the members of all
 *	  academic and/or scientific institutions.
 *  Nor the author, nor the Universite Catholique de Louvain, in any
 *  way guarantee that this program will fullfill any particular
 *  requirement, nor even that its use will always be harmless.
 *
 *  19-sep-2002: 
 * Merge with bug fixes and support for CYGWIN provided by Kris Thielemans@csc.mrc.ac.uk
 */

#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "analyze.h"
#include "interfile.h"
#if defined(_WIN32) && !defined(__CYGWIN__)
#define stat _stat
#define itoa _itoa
#endif

#define ERROR   -1
#define OK 0

#define R_MODE "rb"

static char* magicNumber = "interfile";
static struct dsr hdr;

int analyze_orientation=0;	/* default is spm neurologist convention */
#ifdef __STDC__
static int analyze_read_hdr(const char *fname) 
#else
static int analyze_read_hdr(fname) 
char *fname;
#endif
{
	FILE *fp;
	char tmp[80];

	if ((fp = fopen(fname, R_MODE)) == NULL) return 0;
	if (fread(&hdr,sizeof(struct dsr),1,fp) < 0) return 0;
	fclose(fp);
    if (ntohs(1) != 1) {
        hdr.hk.sizeof_hdr = ntohl(hdr.hk.sizeof_hdr);
        hdr.hk.extents = ntohl(hdr.hk.extents);
        swab((char*)hdr.dime.dim,tmp,4*sizeof(short));
        memcpy(hdr.dime.dim,tmp,4*sizeof(short));
        hdr.dime.datatype = ntohs(hdr.dime.datatype);
        hdr.dime.bitpix = ntohs(hdr.dime.bitpix);
        swab((char*)hdr.dime.pixdim,tmp,4*sizeof(float));
        swaw((short*)tmp,(short*)hdr.dime.pixdim,4*sizeof(float)/2);
        swab((char*)&hdr.dime.funused1,tmp,sizeof(float));
        swaw((short*)tmp,(short*)(&hdr.dime.funused1),sizeof(float)/2);
        hdr.dime.glmax = ntohl(hdr.dime.glmax);
        hdr.dime.glmin = ntohl(hdr.dime.glmin);
    }
	return 1;
}
#ifdef __STDC__
static int _is_analyze(const char* fname)
#else
static int _is_analyze(fname) 
char *fname;
#endif
{
	FILE *fp;
	int sizeof_hdr;
	int magic_number = sizeof(struct dsr);
	
	if ( (fp = fopen(fname, R_MODE)) == NULL) return 0;
	if (fread(&sizeof_hdr,sizeof(int),1,fp) == 1)
		sizeof_hdr = ntohl(sizeof_hdr);
	else sizeof_hdr = 0;
	fclose(fp);
	if (sizeof_hdr == magic_number) return 1;
	return 0;
}

#ifdef __STDC__
char* is_analyze(const char* fname)
#else
char* is_analyze(fname) 
char *fname;
#endif
{
	char *p, *hdr_fname=NULL, *img_fname=NULL;
	struct stat st;
	int elem_size = 0, nvoxels;
	short *dim=NULL;

	if (_is_analyze(fname)) {	/* access by .hdr */
		hdr_fname = strdup(fname);
		img_fname = malloc(strlen(fname)+4);
		strcpy(img_fname,fname);
		if ( (p=strrchr(img_fname,'.')) != NULL) *p++ = '\0';
		strcat(img_fname,".img");
	} else {					/* access by .img */
		if ( (p=strstr(fname,".img")) == NULL)
			return NULL;			/* not a .img file */
		img_fname = strdup(fname);
		hdr_fname = malloc(strlen(fname)+4);
		strcpy(hdr_fname,fname);
		if ( (p=strrchr(hdr_fname,'.')) != NULL) *p++ = '\0';
		strcat(hdr_fname,".hdr");
	}

	if (stat(img_fname,&st)>=0 &&
		_is_analyze(hdr_fname) && analyze_read_hdr(hdr_fname)) {
		dim = hdr.dime.dim;
		nvoxels = dim[1]*dim[2]*dim[3];
		switch (hdr.dime.datatype) {
		case 2:		/* byte */
			elem_size = 1;
			break;
		case 4:		/* short */
			elem_size = 2;
			break;
		case 8:		/* int */
			elem_size = 4;
			break;
		case 16:	/* float */
			elem_size = 4;
			break;
		}
		if (elem_size == 0) fprintf(stderr,"unkown ANALYZE data type : %d\n",
			hdr.dime.datatype);
		else {
			if (nvoxels*elem_size == st.st_size) {
				if (strcmp(hdr_fname, fname) != 0)
					fprintf(stderr,"using %s header for %s data file\n",
						hdr_fname, img_fname);
				free(img_fname);
				return hdr_fname;
			} else fprintf(stderr,
				"%s size does not match %s info\n",img_fname,hdr_fname);
		}
	}
	free(img_fname);
	free(hdr_fname);
	return NULL;
}

#if defined(unix) || defined (__osf__)
static char *itoa(i, buf, radix)
int i;
char *buf;
int radix;
{
        sprintf(buf ,"%d",i);
        return buf;
}
#endif

static char *ftoa(f, buf)
float f;
char *buf;
{
	sprintf(buf,"%g",f);
	return buf;
}

int analyze_open(mptr)
MatrixFile *mptr;
{
	Main_header *mh;
	struct MatDir matdir;
	char *data_file, *extension, patient_id[20];
	char image_extrema[128];
	short  *dim, spm_origin[5];
	int elem_size, data_offset=0, data_size;
	char buf[40];

	matrix_errno = 0;
	if (!analyze_read_hdr(mptr->fname)) return 0;
    strncpy(patient_id,hdr.hist.patient_id,10);
    patient_id[10] ='\0';
	mh = mptr->mhptr;
	sprintf(mh->magic_number,"%d",sizeof(struct dsr));
	mh->sw_version = 70;
	mh->file_type = InterfileImage;
	mptr->interfile_header = (char**)calloc(END_OF_INTERFILE,sizeof(char*));
	data_file = malloc(strlen(mptr->fname)+4);
	strcpy(data_file, mptr->fname);
	if ((extension = strrchr(data_file,'/')) == NULL)
		extension = strrchr(data_file,'.');
	else extension = strrchr(extension,'.');
	if (extension != NULL) strcpy(extension,".img");
	else strcat(data_file,".img");
	mptr->interfile_header[NAME_OF_DATA_FILE] = data_file;
	if ( (mptr->fptr = fopen(data_file, R_MODE)) == NULL) {
		free(mptr->interfile_header);
		mptr->interfile_header = NULL;
		return 0;
	}
	switch(hdr.dime.datatype) {
		case 2 :
			elem_size = 1;
			mptr->interfile_header[NUMBER_FORMAT] = "unsigned integer";
			mptr->interfile_header[NUMBER_OF_BYTES_PER_PIXEL] = "1";
			break;
		case 4 :
			elem_size = 2;
        	if (hdr.dime.glmax > 32767)
				mptr->interfile_header[NUMBER_FORMAT] = "unsigned integer";
			else mptr->interfile_header[NUMBER_FORMAT] = "signed integer";
			sprintf(image_extrema,"%d %d",hdr.dime.glmin,hdr.dime.glmax);
			mptr->interfile_header[IMAGE_EXTREMA] = strdup(image_extrema);
			mptr->interfile_header[NUMBER_OF_BYTES_PER_PIXEL] = "2";
			break;
		case 8 :
			elem_size = 4;
			mptr->interfile_header[NUMBER_FORMAT] = "signed integer";
			mptr->interfile_header[NUMBER_OF_BYTES_PER_PIXEL] = "4";
			break;
		case 16 :
			elem_size = 4;
			mptr->interfile_header[NUMBER_FORMAT] = "short float";
			mptr->interfile_header[NUMBER_OF_BYTES_PER_PIXEL] = "4";
			break;
		default :
			free(mptr->interfile_header);
			mptr->interfile_header = NULL;
			matrix_errno = MAT_UNKNOWN_FILE_TYPE;
			return 0;
	}
	mh->num_planes = hdr.dime.dim[3];
	mh->num_frames = mh->num_gates = 1;
	strcpy(mh->patient_id,patient_id);
	mptr->interfile_header[PATIENT_ID] = strdup(patient_id);
	mptr->interfile_header[NUMBER_OF_DIMENSIONS] = "3";
							/* check spm origin */
	dim = hdr.dime.dim;
	if (ntohs(1) != 1) swab(hdr.hist.originator,(char*)spm_origin,10);
	else memcpy(spm_origin,hdr.hist.originator,10);
	if (spm_origin[0]>1 && spm_origin[1]>1 && spm_origin[2]>1 && 
		spm_origin[0]<dim[1] && spm_origin[1]<dim[2] && spm_origin[2]<dim[3]) {
		mptr->interfile_header[ATLAS_ORIGIN_1] = strdup(itoa(spm_origin[0],buf,10));
		mptr->interfile_header[ATLAS_ORIGIN_2] = strdup(itoa(spm_origin[1],buf,10));
		mptr->interfile_header[ATLAS_ORIGIN_3] = strdup(itoa(spm_origin[2],buf,10));
	}
	mptr->interfile_header[MATRIX_SIZE_1] = strdup(itoa(dim[1],buf,10));
	mptr->interfile_header[MATRIX_SIZE_2] = strdup(itoa(dim[2],buf,10));
	mptr->interfile_header[MATRIX_SIZE_3] = strdup(itoa(dim[3],buf,10));
	mptr->interfile_header[MAXIMUM_PIXEL_COUNT] = strdup(itoa(hdr.dime.glmax,buf,10));
    if (analyze_orientation == 0)       /* Neurology convention */
        mptr->interfile_header[MATRIX_INITIAL_ELEMENT_1] = "right";
    else                                /* Radiology convention */
        mptr->interfile_header[MATRIX_INITIAL_ELEMENT_1] = "left";
	mptr->interfile_header[MATRIX_INITIAL_ELEMENT_2] = "posterior";
	mptr->interfile_header[MATRIX_INITIAL_ELEMENT_3] = "inferior";
	mptr->interfile_header[SCALE_FACTOR_1] = strdup(ftoa(hdr.dime.pixdim[1],buf));
	mptr->interfile_header[SCALE_FACTOR_2] = strdup(ftoa(hdr.dime.pixdim[2],buf));
	mptr->interfile_header[SCALE_FACTOR_3] = strdup(ftoa(hdr.dime.pixdim[3],buf));
	if (hdr.dime.funused1 > 0.0)
		mptr->interfile_header[QUANTIFICATION_UNITS] = strdup(ftoa(hdr.dime.funused1,buf));
	mh->plane_separation = hdr.dime.pixdim[3]/10;	/* convert mm to cm */
	data_size = hdr.dime.dim[1] * hdr.dime.dim[2] * hdr.dime.dim[3] * elem_size;
	mptr->dirlist = (MatDirList *) calloc(1,sizeof(MatDirList)) ;
	matdir.matnum = mat_numcod(1,1,1,0,0);
	matdir.strtblk = data_offset/MatBLKSIZE;
	matdir.endblk = (data_offset + data_size)/MatBLKSIZE;
	insert_mdir(matdir, mptr->dirlist) ;
	return 1;
}
