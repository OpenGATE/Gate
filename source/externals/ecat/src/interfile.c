/*
// sccsid = "@(#)interfile.c	1.4  UCL-TOPO	97/08/05"
*/
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
 *
 */

#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "interfile.h"
#include "machine_indep.h"

#define ERROR   -1
#define OK 0
#define END_OF_KEYS END_OF_INTERFILE+1

#define R_MODE "rb"
#define RW_MODE "rb+"
#define W_MODE "wb+"

InterfileItem used_keys[] = {
	VERSION_OF_KEYS, "version of keys",
	IMAGE_MODALITY, "image modality",
/*
 Main Header
*/
	ORIGINAL_INSTITUTION, "original institution",
	ORIGINATING_SYSTEM, "originating system",
	NAME_OF_DATA_FILE, "name of data file",
	DATA_STARTING_BLOCK, "data starting block",
	DATA_OFFSET_IN_BYTES, "data offset in bytes",
	PATIENT_ID, "patient id",
	PATIENT_DOB, "patient dob",
	PATIENT_SEX, "patient sex",
	STUDY_ID, "study id",
	EXAM_TYPE, "exam type",
	DATA_COMPRESSION, "data compression",
	DATA_ENCODE, "data encode",
	DISPLAY_RANGE, "display range",
	IMAGE_EXTREMA, "image extrema",
	ATLAS_ORIGIN_1, "atlas origin [1]",
	ATLAS_ORIGIN_2, "atlas origin [2]",
	ATLAS_ORIGIN_3, "atlas origin [3]",
	TYPE_OF_DATA, "type of data",
	TOTAL_NUMBER_OF_IMAGES, "total number of images",
	STUDY_DATE, "study date",
	STUDY_TIME, "study time",
	IMAGEDATA_BYTE_ORDER, "imagedata byte order",
	NUMBER_OF_WINDOWS, "number of energy windows",

/* static tomographic images */
	NUMBER_OF_IMAGES, "number of images/energy window",
	PROCESS_STATUS, "process status",
	NUMBER_OF_DIMENSIONS, "number of dimensions",
	MATRIX_SIZE_1, "matrix size [1]",
	MATRIX_SIZE_2, "matrix size [2]",
	MATRIX_SIZE_3, "matrix size [3]",
	NUMBER_FORMAT, "number format",
	NUMBER_OF_BYTES_PER_PIXEL, "number of bytes per pixel",
	MAXIMUM_PIXEL_COUNT, "maximum pixel count",
	MATRIX_INITIAL_ELEMENT_1, "matrix initial element [1]",
	MATRIX_INITIAL_ELEMENT_2, "matrix initial element [2]",
	MATRIX_INITIAL_ELEMENT_3, "matrix initial element [3]",
	SCALE_FACTOR_1, "scaling factor (mm/pixel) [1]",
	SCALE_FACTOR_2, "scaling factor (mm/pixel) [2]",
	SCALE_FACTOR_3, "scaling factor (mm/pixel) [3]",
	IMAGE_DURATION, "image duration",
	IMAGE_START_TIME, "image start time",
	IMAGE_NUMBER, "image number",
	LABEL, "label",
/*
not standard keys added by Sibomana@topo.ucl.ac.be : it is expressed as scale units;
e.g 10e-6 counts/second
*/
	QUANTIFICATION_UNITS, "quantification units",
		/* scale_factor and units label */
	REAL_EXTREMA, "real extrema",
	INTERPOLABILITY, "interpolability",
	TRANSFORMER, "transformer",
	COLORTAB, "colortab",
	END_OF_INTERFILE,"end of interfile",
	END_OF_KEYS, 0
};

static char* magicNumber = "interfile";
static char  line[256];

#ifdef ultrix
#include <string.h>
char* strdup(char* s) {
	char* dup =  malloc(strlen(s)+1);
	strcpy(dup,s);
	return dup;
}
#endif /* ultrix */

static void clean_eol(line) 
char *line;
{
	int len = strlen(line);
	if (len > 250) {
		fprintf(stderr,"line too long :\n %s",line);
		exit(1);
	}
	line[len-1] = ' ';
}

static int _elem_size(data_type) 
int data_type;
{
	switch(data_type) {
		case ByteData :
		case ColorData :
			return 1;
		case SunShort:
			return 2;
		case SunLong:
		case IeeeFloat:
			return 4;
		default:
			fprintf(stderr, "unkown data type, assume short int\n");
			return 2;
	}
}

static void find_data_max(data) 
MatrixData *data;
{
	int npixels = data->xdim*data->ydim*data->zdim;
	switch(data->data_type) {
		case ByteData :
		case ColorData :
			data->data_min = find_bmin((u_char*)data->data_ptr,npixels);
			data->data_max = find_bmax((u_char*)data->data_ptr,npixels);
			break;
		default :
		case SunShort:
			data->data_min = find_smin((short*)data->data_ptr,npixels);
			data->data_max = find_smax((short*)data->data_ptr,npixels);
			break;
		case SunLong:
			data->data_min = find_imin((int*)data->data_ptr,npixels);
			data->data_max = find_imax((int*)data->data_ptr,npixels);
			break;
		case IeeeFloat:
			data->data_min = find_fmin((float*)data->data_ptr,npixels);
			data->data_max = find_fmax((float*)data->data_ptr,npixels);
	}
	data->data_min *=  data->scale_factor;
	data->data_max *=  data->scale_factor;
}

static void flip_x(line,data_type,xdim)
caddr_t line;
int data_type,xdim;
{
	static caddr_t _line=NULL;
	static int line_size = 0;
	int x=0;
	int elem_size = _elem_size(data_type);

	if (line_size == 0) {
		line_size = xdim*elem_size;
		_line = (caddr_t)malloc(line_size);
	} else if (xdim*elem_size > line_size) {
		line_size = xdim*elem_size;
		_line = (caddr_t)realloc(_line, line_size);
	}
	switch(data_type) {
		case ColorData :
		case ByteData :
		{
			u_char *b_p0, *b_p1;
			b_p0 = (u_char*)line;
			b_p1 = (u_char*)_line + xdim - 1;
			for (x=0; x<xdim; x++) *b_p1-- = *b_p0++;
			memcpy(line,_line,xdim);
			break;
		} 
		case SunShort:
		default :
		{
			short *s_p0, *s_p1;
			s_p0 = (short*)line;
			s_p1 = (short*)(_line + (xdim-1)*elem_size);
			for (x=0; x<xdim; x++) *s_p1-- = *s_p0++;
			memcpy(line,_line,xdim*elem_size);
			break;
		}
		case SunLong:
		{
			int *i_p0, *i_p1;
			i_p0 = (int*)line;
			i_p1 = (int*)(_line + (xdim-1)*elem_size);
			for (x=0; x<xdim; x++) *i_p1-- = *i_p0++;
			memcpy(line,_line,xdim*elem_size);
			break;
		}
		case IeeeFloat:
		{
			float *f_p0, *f_p1;
			f_p0 = (float*)line;
			f_p1 = (float*)(_line + (xdim-1)*elem_size);
			for (x=0; x<xdim; x++) *f_p1-- = *f_p0++;
			memcpy(line,_line,xdim*elem_size);
			break;
		}
	}
}
static void flip_y(plane,data_type,xdim,ydim)
caddr_t plane;
int data_type,xdim,ydim;
{
	static caddr_t _plane=NULL;
	static int plane_size = 0;
	caddr_t p0, p1;
	int elem_size = _elem_size(data_type);
	int y=0;

	if (plane_size == 0) {
		plane_size = xdim*ydim*elem_size;
		_plane = (caddr_t)malloc(plane_size);
	} else if (xdim*ydim*elem_size > plane_size) {
		plane_size = xdim*ydim*elem_size;
		_plane = (caddr_t)realloc(_plane, plane_size);
	}
	p0 = plane;
	p1 = _plane + (ydim-1)*xdim*elem_size;
	for (y=0; y<ydim; y++) {
		memcpy(p1,p0,xdim*elem_size);
		p0 += xdim*elem_size;
		p1 -= xdim*elem_size;
	}
	memcpy(plane,_plane,xdim*ydim*elem_size);
}

static char* get_list(fp, str)
FILE *fp;
char *str;
{
	int end_of_table = 0;
	char *p, *ret, **lines;
	float r,g,b;
	int i, line_count = 0, cc = 0;

	if ( (p = strchr(str,'<')) != NULL) p++;
	else return NULL;
	lines = (char**) calloc(256,sizeof(char*));
	if (sscanf(p,"%g %g %g",&r,&g,&b) == 3) { /*valid entry */
		cc += strlen(p);
		lines[line_count++] = strdup(p);
	}
	while (!end_of_table && fgets(line,256,fp) != NULL) {
		clean_eol(line);
		if ((p = strchr(line,';')) != NULL) *p = ' ';
		if ((p = strchr(line,'>')) != NULL) {		/* end of table */
			*p = '\0';
			end_of_table = 1;
		}
		if (sscanf(line,"%g %g %g",&r,&g,&b) == 3) { /*valid entry */
			cc += strlen(line);
			lines[line_count++] = strdup(line);
		}
	}
	ret = malloc(cc+1);
	strcpy(ret,lines[0]); free(lines[0]);
	for (i=1; i<line_count; i++) {
		strcat(ret, lines[i]); free(lines[i]);
	}
	free(lines);
	return ret;
}

static InterfileItem  *get_next_item(fp)
FILE *fp;
{
	char *key_str, *val_str, *end;
	InterfileItem* item;
	static InterfileItem ret;

	while (fgets(line,82,fp) != NULL) {
		clean_eol(line);
		key_str = line;
		while (*key_str && (isspace(*key_str) || *key_str == '!')) key_str++;
		if (*key_str == ';') {
			if (key_str[1] != '%') continue;  /* comment line */
			else key_str += 2;				 /* My extrenstion */
		}
		if ( (val_str = strchr(key_str,':')) == NULL )
			continue; /* invalid line; skip */
		*val_str++ = '\0';
		if (*val_str == '=') val_str++;

/* clean up key_str and val_str */
		end = key_str + strlen(key_str)-1;
		while (isspace(*end)) *end-- = '\0';
		while (isspace(*val_str)) val_str++;
		end = val_str + strlen(val_str)-1;
		while (isspace(*end)) *end-- = '\0';
		for (end=key_str; *end != '\0'; end++)
			*end = tolower(*end);		/* to lower case */
	
/* find key */
		for (item=used_keys; item->value!=NULL; item++)
			if (strcmp(key_str, item->value) == 0) {
				ret.key = item->key;
				if (ret.key == TRANSFORMER || ret.key == COLORTAB )
					ret.value = get_list(fp, val_str);
				else {
					if (strlen(val_str) > 0) ret.value = strdup(val_str);
					else ret.value = NULL;
				}
				return &ret;
			}
	}
	return NULL;
}

#ifdef __STDC__
char *_is_interfile(const char* fname)
#else
char *_is_interfile(fname) 
char *fname;
#endif
{
	int c;
	FILE *fp;
	char *p = magicNumber;
	
	if ( (fp = fopen(fname, R_MODE)) == NULL) return 0;

/* skip spaces */
	while ( (c = fgetc(fp)) != EOF)
		if (!isspace(c)) break;

/*  check magic */
	if (c != EOF) {
		if (c != '!' && *p++ != tolower(c)) {
			fclose(fp);
			return 0;
		}
		while ( (c = fgetc(fp)) != EOF) {
			if (*p++ != tolower(c)) break;
			if (*p == '\0') {				/* OK */
				fclose(fp);
				return strdup(fname);
			}
		}
	}
	fclose(fp);
	return 0;
}

#ifdef __STDC__
char* is_interfile(const char* fname)
#else
char* is_interfile(fname)
char *fname;
#endif
{
    char *p, *hdr_fname=NULL;
#ifdef __STDC__
	const char *img_fname=NULL;
#else
	char *img_fname=NULL;
#endif
	InterfileItem* item;
	FILE *fp;

    if (_is_interfile(fname))  return strdup(fname);
    /* assume data_fname and check header */
    if ( (img_fname=strrchr(fname,'/')) == NULL) img_fname = fname;
	else img_fname++;
    hdr_fname = malloc(strlen(fname)+3);
    strcpy(hdr_fname,fname);
    if ( (p=strrchr(hdr_fname,'.')) != NULL) *p = '\0';
    strcat(hdr_fname,".h33");
    if (_is_interfile(hdr_fname) && (fp = fopen(hdr_fname, R_MODE))!=NULL) {
		while ((item=get_next_item(fp)) != NULL) {
			if (item->value == NULL) continue;
			if (item->key==NAME_OF_DATA_FILE) {
										/* check short and full name */
				if (strcmp(item->value,img_fname)==0 ||		
					strcmp(item->value,fname)==0) {
					fclose(fp);
					fprintf(stderr,"using %s header for %s data file\n",
						hdr_fname, fname);
					free(item->value);
					return hdr_fname;
				}
			}
			free(item->value);
		}
		fclose(fp);
	}
    free(hdr_fname);
    return NULL;
}


int unmap_interfile_header(ifh, imagesub) 
Image_subheader* imagesub;
char **ifh;
{
	int  sx, sy, sz=1, dim = 2, elem_size = 2;
	int x_flip=0, y_flip=0, z_flip=0;
	int hour,min,sec;
	float f;
	char *p;

	if (ifh[NUMBER_OF_DIMENSIONS] != NULL) {
		sscanf(ifh[NUMBER_OF_DIMENSIONS],"%d",&dim);
		if (dim != 2 && dim != 3) {
			matrix_errno = MAT_INVALID_DIMENSION;
			return ERROR;
		}
	}
	if (ifh[NUMBER_OF_BYTES_PER_PIXEL] == NULL ||
		sscanf(ifh[NUMBER_OF_BYTES_PER_PIXEL],"%d",&elem_size) != 1) {
		matrix_errno = MAT_INVALID_DATA_TYPE;
		return ERROR;
	}
	if (ifh[NUMBER_FORMAT] && strstr(ifh[NUMBER_FORMAT], "float")) {
		if (elem_size != 4) {
			matrix_errno = MAT_INVALID_DATA_TYPE;
			return ERROR;
		}
		imagesub->data_type = IeeeFloat;
	} else {  /* integer data type */
		if (elem_size != 1 && elem_size != 2 & elem_size != 4) {
			matrix_errno = MAT_INVALID_DATA_TYPE;
			return ERROR;
		}
		if (elem_size == 1) imagesub->data_type = ByteData;
			if (elem_size == 2) imagesub->data_type = SunShort;
			if (elem_size == 4) imagesub->data_type = SunLong;
	}
	if (ifh[MATRIX_SIZE_1] == NULL ||
		sscanf(ifh[MATRIX_SIZE_1],"%d",&sx) != 1) 
		matrix_errno = MAT_INVALID_DIMENSION;
	else if (ifh[MATRIX_SIZE_2] == NULL ||
		sscanf(ifh[MATRIX_SIZE_2],"%d",&sy) != 1)
		matrix_errno = MAT_INVALID_DIMENSION;
	else  if (dim == 3)  {
		if (ifh[MATRIX_SIZE_3] == NULL ||
			sscanf(ifh[MATRIX_SIZE_3],"%d",&sz) != 1)
			matrix_errno = MAT_INVALID_DIMENSION;
	}
	if (ifh[NUMBER_OF_IMAGES] != NULL) {
		if (sscanf(ifh[NUMBER_OF_IMAGES],"%d",&sz) != 1)
			matrix_errno = MAT_INVALID_DIMENSION;
	}
	if (matrix_errno) return ERROR;
	imagesub->num_dimensions = 3;
	imagesub->x_dimension = sx;
	imagesub->y_dimension = sy;
	imagesub->z_dimension = sz;
	if (ifh[QUANTIFICATION_UNITS] != NULL)
		sscanf(ifh[QUANTIFICATION_UNITS], "%g", &imagesub->scale_factor);
	imagesub->image_min = imagesub->image_max = 0;
	if (ifh[SCALE_FACTOR_1] && sscanf(ifh[SCALE_FACTOR_1], "%g", &f) == 1)
		imagesub->x_pixel_size = f/10.0;		/* mm to cm */
	if (ifh[SCALE_FACTOR_2] && sscanf(ifh[SCALE_FACTOR_2], "%g", &f) == 1)
		imagesub->y_pixel_size = f/10.0;		/* mm to cm */
	if (ifh[SCALE_FACTOR_3] && sscanf(ifh[SCALE_FACTOR_3], "%g", &f) == 1)
		imagesub->z_pixel_size = f/10.0;		/* mm to cm */

	if (ifh[IMAGE_DURATION] && sscanf(ifh[IMAGE_DURATION], "%g", &f) == 1)
		imagesub->frame_duration = f;
	if (ifh[IMAGE_START_TIME] && 
		sscanf(ifh[IMAGE_START_TIME], "%d:%d:%d", &hour, &min, &sec) == 3)
		imagesub->frame_start_time= sec + 60 * (min + 60 * hour);

	if (ifh[MATRIX_INITIAL_ELEMENT_3] &&
		*ifh[MATRIX_INITIAL_ELEMENT_3]=='i') z_flip = 1;

	if (ifh[MATRIX_INITIAL_ELEMENT_2] && 
		*ifh[MATRIX_INITIAL_ELEMENT_2] == 'p') y_flip = 1;
	if (ifh[MATRIX_INITIAL_ELEMENT_1] && 
		*ifh[MATRIX_INITIAL_ELEMENT_1] == 'r') x_flip = 1;
	return 1;
}

int interfile_open(mptr)
MatrixFile *mptr;
{
	InterfileItem* item;
	FILE *fp;		
	Main_header *mh;
	Image_subheader imh;
	struct MatDir matdir;
	time_t now, t;
	struct tm tm;
	char *p, dup[256], data_dir[256], data_file[256];
	char *year, *month, *day, *hour, *minute, *second;
	int this_year;
	int i, elem_size, data_offset=0, data_size;
	float scale_factor;
	int image_number =1, end_of_interfile=0;

	now = time(0);
	this_year = 1900 + (localtime(&now))->tm_year;
	if ((fp = fopen(mptr->fname, R_MODE)) == NULL) return ERROR;
	mh = mptr->mhptr;
	strcpy(mh->data_units,"none");
	mh->calibration_units = 2; /* Processed */
	mh->calibration_units_label = 0;
	strcpy(mh->magic_number,magicNumber);
	mh->sw_version = 70;
	mh->file_type = InterfileImage;
	mptr->interfile_header = (char**)calloc(END_OF_KEYS,sizeof(char*));
	if (mptr->interfile_header == NULL) return ERROR;
	mh->num_frames = mh->num_gates = mh->num_bed_pos = 1;
	mh->plane_separation = 1;
	while (!end_of_interfile && (item=get_next_item(fp)) != NULL) {
		if (item->value == NULL) continue;
		mptr->interfile_header[item->key] = item->value;
		switch(item->key) {
		case ORIGINATING_SYSTEM:
			mh->system_type = atoi(item->value);
			break;
		case QUANTIFICATION_UNITS:
			mh->calibration_units_label = 0;	/* for multiple keys */
			if (sscanf(item->value,"%g %s",&scale_factor,mh->data_units)==2) {
			for (i=0; i<numDisplayUnits; i++)
				if (strcmp(mh->data_units,customDisplayUnits[i]) == 0)
					mh->calibration_units_label = i;
			}
			break;	
		case EXAM_TYPE:
			strncpy(mh->radiopharmaceutical,item->value,32);
			mh->radiopharmaceutical[31] = 0;
			break;
		case ORIGINAL_INSTITUTION:
			strncpy(mh->facility_name,item->value, 20);
			mh->facility_name[19] = '\0';
			break;
		case PATIENT_ID:
			strncpy(mh->patient_id,item->value,16);
			mh->patient_id[15] = '\0';
			break;
		case PATIENT_DOB:
			strcpy(dup,item->value);
			if ( (year = strtok(dup,":")) == NULL) break;
			mh->patient_age = this_year - atoi(year);
			if ( (month = strtok(NULL,":")) == NULL) break;
			if ( (day = strtok(NULL,":")) == NULL) break;
			memset(&tm,0,sizeof(tm));
			tm.tm_year = atoi(year) - 1900;
			tm.tm_mon = atoi(month);
			tm.tm_mday = atoi(day);
#if defined(sun) && !defined(__SVR4)
			mh->patient_birth_date = timelocal(&tm);
#else
			mh->patient_birth_date = mktime(&tm);
#endif
			break;
		case PATIENT_SEX:
			mh->patient_sex[0] = item->value[0];
			break;
		case STUDY_ID:
			strncpy(mh->study_name,item->value,12);
			mh->study_name[11] = '\0';
		case STUDY_DATE:
			strcpy(dup,item->value);
			if ( (year = strtok(dup,":")) == NULL) break;	
			if ( (month = strtok(NULL,":")) == NULL) break;
			if ( (day = strtok(NULL,":")) == NULL) break;
			memset(&tm,0, sizeof(tm));
			tm.tm_year = atoi(year) - 1900;
			tm.tm_mon = atoi(month);
			tm.tm_mday = atoi(day);
#if defined(sun) && !defined(__SVR4)
			mh->scan_start_time = timelocal(&tm);
#else 
			mh->scan_start_time = mktime(&tm);
#endif
			break;
			
		case STUDY_TIME:
			strcpy(dup,item->value);
			if ( (hour = strtok(dup,":")) == NULL) break;	
			if ( (minute = strtok(NULL,":")) == NULL) break;
			if ( (second = strtok(NULL,":")) == NULL) break;
			t = mh->scan_start_time;
			memcpy(&tm,localtime(&t), sizeof(tm));
			tm.tm_hour = atoi(hour);
			tm.tm_min = atoi(minute);
			tm.tm_sec = atoi(second);
#if defined(sun) && !defined(__SVR4) 
			mh->scan_start_time = timelocal(&tm);
#else 
			mh->scan_start_time = mktime(&tm);
#endif
			break;
		case NUMBER_OF_IMAGES :
		case MATRIX_SIZE_3 :
			mh->num_planes = atoi(item->value);
			break;
		case SCALE_FACTOR_3:
			mh->plane_separation = atof(item->value)/10.0; /* mm to cm */
			break;
		case IMAGE_NUMBER:
			image_number = atoi(item->value);
			break;
		case END_OF_INTERFILE:
			end_of_interfile = 1;
		}
	}
	fclose(fp);

	if (mptr->interfile_header[NAME_OF_DATA_FILE] != NULL) {
		strcpy(data_file,mptr->interfile_header[NAME_OF_DATA_FILE]);
		if ( (mptr->fptr = fopen(data_file, R_MODE)) != NULL)
			 mptr->interfile_header[NAME_OF_DATA_FILE] = strdup(data_file);
		else {
			strcpy(data_dir, mptr->fname);
			if ( (p = strrchr(data_dir,'/')) != NULL) *p = '\0';
			if ( (p = strrchr(data_file,'/')) != NULL) {
				strcpy(dup,p+1);
				sprintf(data_file,"%s/%s",data_dir, dup);
			} else {
				strcpy(dup,data_file);
				sprintf(data_file,"%s/%s",data_dir,dup);
			}
			if ( (mptr->fptr = fopen(data_file, R_MODE)) != NULL)
				mptr->interfile_header[NAME_OF_DATA_FILE] = strdup(data_file);
			else {
				free(mptr->interfile_header);
				return ERROR;
			}
		}
	}

	unmap_interfile_header(mptr->interfile_header, &imh);
	elem_size = _elem_size(imh.data_type);

	data_size = imh.x_dimension * imh.y_dimension * imh.z_dimension *elem_size;
	mptr->dirlist = (MatDirList *) calloc(1,sizeof(MatDirList)) ;
	matdir.matnum = mat_numcod(image_number,1,1,0,0);
	matdir.strtblk = data_offset/MatBLKSIZE;
	matdir.endblk = (data_offset + data_size)/MatBLKSIZE;
	insert_mdir(matdir, mptr->dirlist) ;
	return OK;
}

int interfile_read(mptr, matnum, data, dtype) 
MatrixFile	*mptr ;
MatrixData	*data ;
int   matnum, dtype;
{
	Image_subheader *imagesub ;
	int y, z, image_min, image_max, npixels, nvoxels;
	int i, tmp, nblks, elem_size=2, data_offset=0;
	caddr_t plane, line;
	u_short u_max, *up=NULL;
	short *sp=NULL;
	int z_flip=0, y_flip=0, x_flip=0;
	float f;
	char **ifh, *p;
	int swap_order = 0;

	ifh = mptr->interfile_header;
	imagesub = (Image_subheader*)data->shptr;
	memset(imagesub,0,sizeof(Image_subheader));
	imagesub->x_pixel_size=imagesub->y_pixel_size=imagesub->z_pixel_size=1.0;
	imagesub->scale_factor = 1.0;
	unmap_interfile_header(ifh, imagesub);
	elem_size = _elem_size(imagesub->data_type);

	data->matnum = mat_numcod(1,1,1,0,0);
	data->xdim = imagesub->x_dimension;
	data->ydim = imagesub->y_dimension;
	data->zdim = imagesub->z_dimension;
	data->pixel_size = imagesub->x_pixel_size;
	data->y_size = imagesub->y_pixel_size;
	data->z_size = imagesub->z_pixel_size;
	data->data_type = imagesub->data_type;
	data->scale_factor = imagesub->scale_factor ;
	if (ifh[MAXIMUM_PIXEL_COUNT] &&
		 sscanf(ifh[MAXIMUM_PIXEL_COUNT], "%d",&image_max) == 1) 
		imagesub->image_max = image_max;
	if (ifh[IMAGE_EXTREMA] &&
		 sscanf(ifh[IMAGE_EXTREMA],"%d %d",&image_min, &image_max) == 2) {
		 imagesub->image_min = image_min;
		 imagesub->image_max = image_max;
	}

	if (ifh[REAL_EXTREMA]) 
		sscanf(ifh[REAL_EXTREMA],"%g %g",&data->data_min, &data->data_max);
	if (elem_size == 1) {
		if (imagesub->image_max>0 && data->data_max>data->data_min) {
			imagesub->scale_factor = (data->data_max - data->data_min)/
				(imagesub->image_max - imagesub->image_min);
			data->scale_factor = imagesub->scale_factor;
		}
	}

	if (ifh[ATLAS_ORIGIN_1] &&
		sscanf(ifh[ATLAS_ORIGIN_1], "%g", &f) == 1) {
		data->x_origin = f * data->pixel_size;
	}
	if (ifh[ATLAS_ORIGIN_2] &&
		sscanf(ifh[ATLAS_ORIGIN_2], "%g", &f) == 1) {
		data->y_origin = f * data->y_size;
	}
	if (ifh[ATLAS_ORIGIN_3] &&
		sscanf(ifh[ATLAS_ORIGIN_3], "%g", &f) == 1) {
		data->z_origin = f  * data->z_size;
	}

	if (dtype == MAT_SUB_HEADER) {
		data->data_max = imagesub->image_max * data->scale_factor;
		return OK;
	}	/* else compute extrema */
	if (ifh[MATRIX_INITIAL_ELEMENT_3] &&
		*ifh[MATRIX_INITIAL_ELEMENT_3]=='i') {
		z_flip = 1;
		if (data->z_origin > 0)
			data->z_origin = (data->zdim -1)*data->z_size - data->z_origin;
		fprintf(stderr,
			"volume z direction is changed to superior->inferior\n");
	}
	if (ifh[MATRIX_INITIAL_ELEMENT_2] && 
		*ifh[MATRIX_INITIAL_ELEMENT_2] == 'p') {
		y_flip = 1;
		if (data->y_origin > 0)
			data->y_origin = (data->ydim-1)*data->y_size - data->y_origin;
		fprintf(stderr,
			"volume y direction is changed to anterior->posterior\n"); 
	}
	if (ifh[MATRIX_INITIAL_ELEMENT_1] && 
		*ifh[MATRIX_INITIAL_ELEMENT_1] == 'r') {
		x_flip = 1;
		if (data->x_origin> 0)
			 data->x_origin = (data->xdim-1)*data->pixel_size - data->x_origin;
		fprintf(stderr,
			"volume x direction is changed to left->right\n"); 
	}
	npixels = data->xdim * data->ydim;
	nvoxels = npixels * data->zdim;
	data->data_size = nvoxels * elem_size;
	nblks = (data->data_size+511)/512;
	data->data_ptr = (caddr_t) malloc(512*nblks);
	if (ifh[DATA_STARTING_BLOCK] &&
		sscanf(ifh[DATA_STARTING_BLOCK],"%d",&data_offset) ) {
		if (data_offset<0) data_offset = 0;
		else data_offset *= 512;
	}
	if (data_offset==0 && ifh[DATA_OFFSET_IN_BYTES] &&
		sscanf(ifh[DATA_OFFSET_IN_BYTES],"%d",&data_offset) ) {
		if (data_offset<0) data_offset = 0;
	}
	fseek(mptr->fptr,data_offset,0);
	for (z = 0; z < data->zdim; z++) {
		if (z_flip) 
			plane = data->data_ptr + (data->zdim-z-1)*elem_size*npixels;
		else plane = data->data_ptr + z*elem_size*npixels;
		if (fread(plane,elem_size,npixels,mptr->fptr) < npixels) {
			free_matrix_data(data);
			matrix_errno = MAT_READ_ERROR;
			return ERROR;
		}
		if (y_flip) 
			flip_y(plane,data->data_type,data->xdim,data->ydim);
		if (x_flip) {
			for (y = 0; y < data->ydim; y++) {
				line = plane + y*data->xdim*elem_size;
				flip_x(line,data->data_type,data->xdim);
			}
		}
	}

	if ( (p=ifh[IMAGEDATA_BYTE_ORDER]) != NULL && (*p=='b' || *p=='B'))
    { 
		if (ntohs(1) != 1) swap_order = 1;  /* big to little endian */
	}
	else
	{
		if (ntohs(1) == 1) swap_order = 1;  /* little to big endian */
	}
	if (swap_order)
	{
		int j=0;
		char *dptr, *tmp;
		tmp = malloc(512);
		dptr = data->data_ptr;
		if (elem_size == 2)
		{
			for (i=0, j=0; i<nblks; i++, j+=512)
			{
				swab(dptr+j, tmp, 512);
				memcpy(dptr+j, tmp, 512);
            }
		}
		if (elem_size == 4)
		{
			for (i=0, j=0; i<nblks; i++, j+=512)
			{
				swab(dptr+j, tmp, 512);
				swaw((short*)tmp, (short*)(dptr+j), 256);
			}
		}
		free(tmp);
	}
		
	if (elem_size == 2 &&
		ifh[NUMBER_FORMAT] && strstr(ifh[NUMBER_FORMAT], "unsigned") ) {
		up = (u_short*)data->data_ptr;
		u_max = *up++;
		for (i=1; i<nvoxels; i++, up++) 
			if (u_max< (*up)) u_max = *up;
		if (u_max > 32767) {
			fprintf(stderr,"converting unsigned to signed integer\n");
			sp = (short*)data->data_ptr;
			up = (u_short*)data->data_ptr;
			for (i=0; i<nvoxels; i++) {
				tmp = *up++;
				*sp++ = tmp/2;
			}
			data->scale_factor *= 2;
		}
	}
	find_data_max(data);		/*don't trust in header extrema*/
	return OK;
}

MatrixData *interfile_read_slice(fptr, ifh, volume, slice, u_flag)
FILE *fptr;
char **ifh;
MatrixData *volume;
int  slice;
int u_flag;
{
	caddr_t line;
	int i, npixels, file_pos, data_size, nblks, elem_size = 2;
	int  y, data_offset = 0;
	int z_flip=0,y_flip=0, x_flip=0;
	u_short *up=NULL;
	short *sp=NULL;
	MatrixData *data;

	if (ifh && ifh[MATRIX_INITIAL_ELEMENT_3] &&
		*ifh[MATRIX_INITIAL_ELEMENT_3]=='i') z_flip = 1;
	if (ifh && ifh[MATRIX_INITIAL_ELEMENT_2] && 
		*ifh[MATRIX_INITIAL_ELEMENT_2] == 'p') y_flip = 1;
	if (ifh && ifh[MATRIX_INITIAL_ELEMENT_1] && 
		*ifh[MATRIX_INITIAL_ELEMENT_1] == 'r') x_flip = 1;
	if (ifh && ifh[DATA_OFFSET_IN_BYTES])
        if (sscanf(ifh[DATA_OFFSET_IN_BYTES],"%d",&data_offset)!=1)
            data_offset=0;

	/* allocate space for MatrixData structure and initialize */
	data = (MatrixData *) calloc( 1, sizeof(MatrixData)) ;
	if (!data) return NULL;
	*data = *volume;
	data->zdim = 1;
	data->shptr = NULL;
	npixels = data->xdim*data->ydim;
	file_pos = data_offset;
	elem_size = _elem_size(data->data_type);
	data_size = data->data_size = npixels*elem_size;
	if (z_flip ==0) file_pos += slice*data_size;
	else file_pos += (volume->zdim-slice-1)*data_size;
	nblks = (data_size+(MatBLKSIZE-1))/MatBLKSIZE;
	if ((data->data_ptr = malloc(nblks * MatBLKSIZE)) == NULL) {
		free_matrix_data(data);
		return NULL;
	}
	fseek(fptr,file_pos,0); /* jump to location of this slice*/
	if (fread(data->data_ptr,elem_size,npixels,fptr) != npixels)
		perror("fread");
	file_data_to_host(data->data_ptr,nblks,data->data_type);
	if (y_flip) 
		flip_y(data->data_ptr,data->data_type,data->xdim,data->ydim);
	if (x_flip) {
		for (y = 0; y < data->ydim; y++) {
			line = data->data_ptr + y*data->xdim*elem_size;
			flip_x(line,data->data_type,data->xdim);
		}
	}
	data->shptr = NULL;
	if (ifh && ifh[NUMBER_FORMAT] && strstr(ifh[NUMBER_FORMAT], "unsigned"))
		u_flag = 1;
	if (u_flag && elem_size == 2) {
		short* sp = (short*)data->data_ptr;
		u_short* up = (u_short*)data->data_ptr;
		for (i=0; i<npixels; i++) *sp++ = (*up++)/2;
		data->scale_factor *= 2;
	}
	find_data_max(data);	/*  /*don't trust in header extrema*/
	return data;
}
int interfile_write_volume(mptr,image_name, header_name,data_matrix,size)
MatrixFile* mptr;
char *image_name,*header_name;
u_char* data_matrix;
int size;
{
	int count;
	FILE *fp_h, *fp_i;
	char** ifh;
	InterfileItem* item;
	if ((fp_i = fopen(image_name, W_MODE)) == NULL) return 0;
	count = fwrite(data_matrix,1,size,fp_i);
	fclose(fp_i);
	if (count != size) {
		return 0;
	}
	if ((fp_h = fopen(header_name, W_MODE)) == NULL) return 0;
	ifh = mptr->interfile_header;
	fprintf(fp_h,"!Interfile :=\n");
	fflush(fp_h);
	for (item=used_keys; item->value!=NULL; item++){
		if (ifh[item->key] != 0) 
			fprintf(fp_h,"%s := %s\n",item->value,ifh[item->key]);
		fflush(fp_h);
	}
	fclose(fp_h);
	
	return 1;
}
