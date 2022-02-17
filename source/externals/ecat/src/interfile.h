#ifndef interfile_h
#define interfile_h

#include "matrix.h"
#ifdef ultrix
extern char* strdup();
#endif

/* 
 *  sccsid = "@(#)interfile.h	1.3  UCL-TOPO	96/05/29"
 *
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
 *
 */

/* only one energy window, static tomographic data supported */

typedef enum {
	VERSION_OF_KEYS,
	IMAGE_MODALITY,
	ORIGINAL_INSTITUTION,
	ORIGINATING_SYSTEM,
	NAME_OF_DATA_FILE,
	DATA_STARTING_BLOCK,
	DATA_OFFSET_IN_BYTES,
	PATIENT_ID,
	PATIENT_DOB,		/* date format is YYYY:MM:DD */
	PATIENT_SEX,
	STUDY_ID,
	EXAM_TYPE,
	DATA_COMPRESSION,
	DATA_ENCODE,
	TYPE_OF_DATA,
	TOTAL_NUMBER_OF_IMAGES,
	STUDY_DATE,
	STUDY_TIME,		/* Time Format is hh:mm:ss */
	IMAGEDATA_BYTE_ORDER,
	NUMBER_OF_WINDOWS,	/* Number of energy windows */
	NUMBER_OF_IMAGES,	/* Number of images/energy window */
	PROCESS_STATUS,
	NUMBER_OF_DIMENSIONS,
	MATRIX_SIZE_1,
	MATRIX_SIZE_2,
	MATRIX_SIZE_3,
	NUMBER_FORMAT,
	NUMBER_OF_BYTES_PER_PIXEL,
	SCALE_FACTOR_1,
	SCALE_FACTOR_2,
	SCALE_FACTOR_3,
	IMAGE_DURATION,
	IMAGE_START_TIME,
	IMAGE_NUMBER,
	LABEL,
	MAXIMUM_PIXEL_COUNT,
	TOTAL_COUNTS,
/*
 My Extensions
*/
	QUANTIFICATION_UNITS,	/* scale_factor units; eg 10e-3 counts/seconds */
	COLORTAB,
	DISPLAY_RANGE,
	IMAGE_EXTREMA,
	REAL_EXTREMA,
	INTERPOLABILITY,
	MATRIX_INITIAL_ELEMENT_1,
	MATRIX_INITIAL_ELEMENT_2,
	MATRIX_INITIAL_ELEMENT_3,
	ATLAS_ORIGIN_1,
	ATLAS_ORIGIN_2,
	ATLAS_ORIGIN_3,
	TRANSFORMER,
	END_OF_INTERFILE
} InterfileKeys;

typedef enum {
	STATIC,
	DYNAMIC,
	GATED,
	TOMOGRAPHIC,
	CURVE,
	ROI,
	OTHER,
/* My Externsion */
	CLICHE			/* with a fixed colormap */
}	TypeOfData;

typedef enum {
	UNSIGNED_INTEGER,
	SIGNED_INTEGER,
	SHORT_FLOAT,
	LONG_FLOAT,
/* My Externsion */
	COLOR_PIXEL
} NumberFormat;

typedef struct _InterfileItem {
	int key;
	char* value;
} InterfileItem;

extern InterfileItem used_keys[];

#if defined(__STDC__) || defined(__cplusplus)
#if defined(__cplusplus)
extern "C" {
/*
 * high level user functions
 */
#endif
int interfile_write_volume(MatrixFile* mptr, char *image_name,char *header_name,
		u_char* data_matrix, int size);
char *is_interfile(const char*);
int interfile_open(MatrixFile*);
MatrixData *interfile_read_slice(FILE*, char** ifh, MatrixData*, int slice,
	int u_flag);
int interfile_read(MatrixFile *mptr,int matnum, 
	MatrixData  *data, int dtype);
#if defined(__cplusplus)
}
#endif
#else /* __cplusplus */
extern char *is_interfile();
extern int interfile_open();
extern MatrixData *interfile_read_slice();
extern int interfile_read();
extern int interfile_write_volume();
#endif
#endif
