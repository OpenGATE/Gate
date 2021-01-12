static char sccsid[]="@(#)matrix_extra.c	1.13 5/7/93 Copyright 1991,1992 CTI Pet Systems, Inc.";

/*
 * Modification history :
 * March-1996 :		Sibomana@topo.ucl.ac.be
 *      Add ECAT V70, Interfile and Analyze support
 *  
 * 5-Aug-1997:		Sibomana@topo.ucl.ac.be
 *      Add Error Handling facility provided by Helmut Lucht (MPI-Koeln)
 *      Error is set in errno global variable by system calls (malloc,calloc,
 *		fopen, fwrite, fread, ... )
 *		Error is set by matrix routines in matrix_errno for library specific
 *      	errors (MAT_INVALID_DIRBLK,MAT_ACS_FILE_NOT_FOUND, ...)
 * 25-Jan-1998:  Sibomana@topo.ucl.ac.be
 *		replace "#ifdef unix" by "#ifndef _WIN32" to support different unix
 *      	compilers.
 *  7-Dec-1999: Sibomana
 *		bug in mh_update corrected
 *  19-sep-2002: 
 *     Merge with bug fixes and support for CYGWIN provided by Kris Thielemans@csc.mrc.ac.uk
 *  1-may-2003: 
 *    Use RW_MODE if file exists and W_MODE for new file in matrix_open
 *  25-sep-2003:
 *    Remove '#include<varargs.h>' and functions using it (matrix_create_object, 
 *
 */


#include	<stdlib.h>
#include	<math.h>
#include	<fcntl.h>
#include	<string.h>
#include	<sys/types.h>
#include	<sys/stat.h>
#if defined(_WIN32) && !defined(__CYGWIN__)
#include <io.h>
#define stat _stat
#define access _access
#define F_OK 0
#else
#include	<unistd.h>
#endif
#include	"matrix.h"
#include	"interfile.h"
#include	"num_sort.h"
#include	"machine_indep.h"


#define ERROR   -1
#define OK 0
#define TRUE 1
#define FALSE 0

#define R_MODE "rb"
#define RW_MODE "rb+"
#define W_MODE "wb+"

#if !defined(OLD_C)
extern int analyze_open(MatrixFile*);
extern int write_z_data(MatrixFile* , int, MatrixData*);
extern int read_z_data(MatrixFile* , int, MatrixData*, int);
#endif 

MatrixErrorCode matrix_errno;
char		matrix_errtxt[132];

char* matrix_errors[] =
	{
		"No Error",
		"Read error",
		"Write error",
		"Invalid directory block",
		"ACS file not found",
		"Interfile open error",
		"File type not match",
		"Read from nil filepointer",
		"No mainheader file object",
		"Nil subheader pointer",
		"Nil data pointer",
		"Matrix not found",
		"Unknown filetype",
		"ACS create error",
		"Bad attribute",
		"Bad file access mode",
		"Invalid dimension",
		"No slices found",
		"Invalid data type",
		"Invalid multibed position"
	};

is_acs(fname)

  char	*fname ;

{
 	if (strstr(fname, "/sd") == fname)
		return(TRUE) ; 
	else
	   return(FALSE) ;
}

matrix_convert_data()
{
	return OK ;	/* dummy for now */
}

static void free_matrix_file(mptr)
  MatrixFile *mptr ;
{
	if (mptr == NULL) return;
	if (mptr->mhptr != NULL) free(mptr->mhptr) ;
	if (mptr->dirlist != NULL) matrix_freelist(mptr->dirlist) ;
	if (mptr->fptr) fclose(mptr->fptr);
	if (mptr->fname) free(mptr->fname);
	free(mptr);
}

matrix_freelist(matdirlist)

  MatDirList	*matdirlist ;

{
  MatDirNode	*node, *next ;

	if (matdirlist == NULL) return OK;
	if (matdirlist->first != NULL)
	{
	  node = matdirlist->first ;
	  do
	  {
		next = node->next ;
		free(node) ;
		node = next ;
	  }
	  while(next != NULL) ;
	}
	free(matdirlist) ;
	return OK;
}

int 
mh_update(file)
	MatrixFile     *file;
{
	MatDirNode     *node;
	struct Matval   val;
	int            *frames, *planes, *gates, *beds;
	int             num_frames = 0, num_planes = 0, num_gates = 0,
	                num_beds = 0;
	Main_header    *mh = file->mhptr;
	MatDirList     *dir_list = file->dirlist;
	int             mod = 0, nmats;					/* correction  7/12/99 MS */
	if (file->dirlist == NULL || file->dirlist->nmats == 0)
		return OK;
	frames = planes = gates = beds = NULL;
	nmats = file->dirlist->nmats;
	frames = (int *) calloc(nmats, sizeof(int));
	planes = (int *) calloc(nmats, sizeof(int));
	gates = (int *) calloc(nmats, sizeof(int));
	beds = (int *) calloc(nmats, sizeof(int));
	node = file->dirlist->first;
	while (node != NULL) {
		mat_numdoc(node->matnum, &val);
		if (bsearch(&val.frame, frames, num_frames, sizeof(int), compare_int) ==
		    NULL) {
			frames[num_frames++] = val.frame;
			sort_int(frames, num_frames);
		}
		if (bsearch(&val.plane, planes, num_planes, sizeof(int), compare_int) ==
		    NULL) {
			planes[num_planes++] = val.plane;
			sort_int(planes, num_planes);
		}
		if (bsearch(&val.gate, gates, num_gates, sizeof(int), compare_int) ==
		    NULL) {
			gates[num_gates++] = val.gate;
			sort_int(gates, num_gates);
		}
		if (bsearch(&val.bed, beds, num_beds, sizeof(int), compare_int) ==
		    NULL) {
			beds[num_beds++] = val.bed;
			sort_int(beds, num_beds);
		}
		node = node->next;
	}
	free(frames);
	free(planes);
	free(gates);
	free(beds);
	num_beds--;		/* ??? CTI convention ==> nombre d'offsets */
	if (mh->num_frames != num_frames) {
		mh->num_frames = num_frames;
		mod++;
	}
/*	if (mh->file_type == PetImage && mh->num_planes != num_planes) {
		mh->num_planes = num_planes;
		mod++;
	}
*/
	if (mh->num_gates != num_gates) {
		mh->num_gates = num_gates;
		mod++;
	}
	if (mh->num_bed_pos != num_beds) {
		mh->num_bed_pos = num_beds;
		mod++;
	}
	if (mod > 0)
		return mat_write_main_header(file->fptr, mh);
	return OK;
}

static int acs_read_scan_subheader( fname, mhptr, blknum, header)
  char *fname;
  Main_header *mhptr;
  int blknum;
  Scan_subheader *header;
{
  
#ifndef _WIN32
	char buf[MatBLKSIZE];
    rtsRblk(fname, blknum, buf);
    if (mhptr->sw_version < V7)
        return unmap64_scan_header(buf,header, mhptr);
    return unmap_scan_header(buf,header);
#else
	return ERROR;
#endif

}

static int acs_read_Scan3D_subheader( fname, mhptr, blknum, header)
  char *fname;
  Main_header *mhptr;
  int blknum;
  Scan3D_subheader *header;
{
#ifndef _WIN32
    char buf[2*MatBLKSIZE];
    rtsRblk(fname, blknum, buf);
    rtsRblk(fname, blknum+1, buf+MatBLKSIZE);
	return unmap_Scan3D_header(buf,header);
#else
	return ERROR;
#endif
}

MatDirBlk *mat_rdirblk(file, blknum)
	MatrixFile     *file;
	int             blknum;

{
	MatDirBlk      *matdirblk;
	int             i, j, err, ndirs;
	int             dirbufr[MatBLKSIZE / 4];
	FILE           *fptr = file->fptr;

	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';

	matdirblk = (MatDirBlk *) malloc(MatBLKSIZE);
	if (matdirblk == NULL)
		return (NULL);

	if (file->mhptr->sw_version < V7) {
		if (file->acs)
			err = read_raw_acs_data(file->fname, blknum, 1, (char *) dirbufr, VAX_Ix4);
		else
			err = read_matrix_data(fptr, blknum, 1, (char *) dirbufr, VAX_Ix4);
	} else {
		if (file->acs)
			err = read_raw_acs_data(file->fname, blknum, 1, (char *) dirbufr, SunLong);
		else
			err = read_matrix_data(fptr, blknum, 1, (char *) dirbufr, SunLong);
	}
	if (err == ERROR) {
		free(matdirblk);
		return (NULL);
	}
	matdirblk->nfree = dirbufr[0];
	matdirblk->nextblk = dirbufr[1];
	matdirblk->prvblk = dirbufr[2];
	matdirblk->nused = dirbufr[3];

	if (matdirblk->nused > 31) {
		matrix_errno = MAT_INVALID_DIRBLK;
		free(matdirblk);
		return (NULL);
	}
	ndirs = (MatBLKSIZE / 4 - 4) / 4;
	for (i = 0; i < ndirs; i++) {
		matdirblk->matdir[i].matnum = 0;
		matdirblk->matdir[i].strtblk = 0;
		matdirblk->matdir[i].endblk = 0;
		matdirblk->matdir[i].matstat = 0;
	}

	for (i = 0; i < matdirblk->nused; i++) {
		j = i + 1;
		matdirblk->matdir[i].matnum = dirbufr[j * 4 + 0];
		matdirblk->matdir[i].strtblk = dirbufr[j * 4 + 1];
		matdirblk->matdir[i].endblk = dirbufr[j * 4 + 2];
		matdirblk->matdir[i].matstat = dirbufr[j * 4 + 3];
	}
	return (matdirblk);
}

MatDirList *mat_read_directory(mptr)
	MatrixFile     *mptr;
{
	struct MatDir   matdir;
	MatDirList     *dirlist;
	MatDirBlk      *matdirblk;
	int             i, blknum;

	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';

	dirlist = (MatDirList *) calloc(1, sizeof(MatDirList));
	if (dirlist == NULL) return (NULL);

	blknum = MatFirstDirBlk;
	do {
		matdirblk = mat_rdirblk(mptr, blknum);
		if (matdirblk == NULL) {
			free(dirlist);
			return (NULL);
		}
		for (i = 0; i < matdirblk->nused; i++) {
			matdir.matnum = matdirblk->matdir[i].matnum;
			matdir.strtblk = matdirblk->matdir[i].strtblk;
			matdir.endblk = matdirblk->matdir[i].endblk;
			matdir.matstat = matdirblk->matdir[i].matstat;
			insert_mdir(matdir, dirlist);
		}
		blknum = matdirblk->nextblk;
		free(matdirblk);
	}
	while (blknum != MatFirstDirBlk);
	return (dirlist);
}

#ifdef __STDC__
extern char *is_analyze(const char*);
MatrixFile *matrix_open(const char* fname, int fmode, int mtype)
#else
extern char *is_analyze();
MatrixFile *matrix_open(fname, fmode, mtype)
  char	*fname ;
  int	fmode, mtype ;
#endif
{
#ifndef _WIN32
  int status;
#endif
  MatrixFile *mptr ;
  char *omode;

	if (fmode == MAT_READ_ONLY) omode = R_MODE;
	else {
		if (access(fname,F_OK) == 0) omode = RW_MODE;
		else omode = W_MODE;
	}

	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';

	/* allocate space for MatrixFile and main header data structures */
	if ( (mptr = (MatrixFile *)calloc(1,sizeof(MatrixFile))) == NULL) {
		 return(NULL) ;
	}

	if ( (mptr->mhptr = (Main_header *)calloc(1,sizeof(Main_header))) == NULL) {
		 free( mptr);
		 return(NULL) ;
	}
	mptr->acs = is_acs(fname) ;
	/* read the main header from the file */
	if (mptr->acs) {				/* located on the ACS filesystem */
#ifndef _WIN32 /* ACS access not implemented on WIN32 plateform */
		if ( (status = rts_rmhd(fname, mptr->mhptr)) == ERROR) {
#endif
		  matrix_errno = MAT_ACS_FILE_NOT_FOUND ;
		  free( mptr->mhptr);
		  free( mptr);
		  return (NULL);
#ifndef _WIN32
		}
		mptr->fname = strdup(fname);
#endif
	} else	{						/* located on nfs filesystem */
		if (fmode == MAT_READ_ONLY) { /* check if interfile or analyze format */
#ifndef _WIN32
			if ((mptr->fname=is_interfile(fname)) != NULL) {
				if (interfile_open(mptr) == ERROR) {
					/* matrix_errno set by interfile_open */
					free_matrix_file(mptr);
					return (NULL);
				}
				return mptr;
			}
			if ((mptr->fname=is_analyze(fname)) != NULL) {
				if (analyze_open(mptr) == ERROR) {
					/* matrix_errno is set by analyze_open */
					free_matrix_file(mptr);
					return (NULL);
				}
				return mptr;
			}
#endif
		}
									/* assume CTI/ECAT format */
		if ((mptr->fptr = fopen(fname, omode)) == NULL) {
	  		free_matrix_file(mptr);
	  		return (NULL);
		}
		mptr->fname = strdup(fname);
		if (mat_read_main_header(mptr->fptr, mptr->mhptr) == ERROR) {
			matrix_errno = MAT_NOMHD_FILE_OBJECT ;
			free_matrix_file(mptr);
			return(NULL);
		}
	}

	/*
	   if the matrix type doesn't match the requested type, that's
	   an error. Specify matrix type NoData to open any type.
	*/
	if (mtype != NoData && mtype != mptr->mhptr->file_type) {
		matrix_errno = MAT_FILE_TYPE_NOT_MATCH ;
		free_matrix_file(mptr);
		return (NULL);
	}

	/* read and store the matrix file directory.  */
	mptr->dirlist = mat_read_directory(mptr);
	if( (int)mptr->dirlist == 0 || (int)mptr->dirlist == -1 ) {
		free_matrix_file( mptr );
		return( NULL );
	}

	if( !strncmp(mptr->mhptr->magic_number, "MATRIX", strlen("MATRIX")) ) {
		mptr->file_format = ECAT7;
	} else {
		mptr->file_format = ECAT6;
	}

	if( matrix_errno == OK ) return( mptr ) ;
	free_matrix_file( mptr);
	return(NULL) ;
}

#ifdef __STDC__
MatrixFile     *matrix_create(const char *fname, int fmode, Main_header * proto_mhptr)
/**INDENT** Error@2: Unmatched #else */
#else
	MatrixFile     *matrix_create(fname, fmode, proto_mhptr)
	char           *fname;
	int             fmode;
	Main_header    *proto_mhptr;
/**INDENT** Error@7: Unmatched #endif */
#endif
{
	MatrixFile     *mptr = NULL;
	FILE           *fptr, *mat_create();

	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';
	switch (fmode) {
	case MAT_READ_WRITE:
	case MAT_OPEN_EXISTING:
		mptr = matrix_open(fname, MAT_READ_WRITE, proto_mhptr->file_type);
		if (mptr) break;
		/*
		 * if (matrix_errno != MAT_NFS_FILE_NOT_FOUND) break; if we
		 * got an NFS_FILE_NOT_FOUND error, then try to create the
		 * matrix file.
		 */
	case MAT_CREATE:
	case MAT_CREATE_NEW_FILE:
		matrix_errno = MAT_OK;
		matrix_errtxt[0] = '\0';
		if (is_acs(fname)) {
			matrix_errno = MAT_ACS_CREATE_ERR;
			return( NULL );
		}
		fptr = mat_create(fname, proto_mhptr);
		if (!fptr) return( NULL );

		mptr = (MatrixFile *) calloc(1, sizeof(MatrixFile));
		if (!mptr) {
			fclose( fptr );
			return( NULL );
		}

		mptr->fptr = fptr;

		mptr->fname = (char *) malloc(strlen(fname) + 1);
		if (!mptr->fname) {
			free( mptr );
			fclose( fptr );
			return( NULL );
		}

		strcpy(mptr->fname, fname);
		mptr->mhptr = (Main_header *) malloc(sizeof(Main_header));
		if (!mptr->mhptr) {
			free( mptr->fname );
			free( mptr );
			fclose( fptr );
			return( NULL );
		}

		memcpy(mptr->mhptr, proto_mhptr, sizeof(Main_header));
		mptr->dirlist = mat_read_directory(mptr);
		if (!mptr->dirlist) {
			free( mptr->fname );
			free( mptr->mhptr );
			free( mptr );
			fclose( fptr );
			return( NULL );
		}
		break;
	default:
		matrix_errno = MAT_BAD_FILE_ACCESS_MODE;
		return( NULL );
		break;
	}
	return mptr;
}

int matrix_close(mptr)
  MatrixFile *mptr ;
{
	int status = OK;
	matrix_errno = MAT_OK;
	if (mptr->fname) strcpy(matrix_errtxt,mptr->fname);
	else matrix_errtxt[0] = '\0';
	if (mptr == NULL) return status;
	if (mptr->mhptr != NULL) free(mptr->mhptr) ;
	if (mptr->dirlist != NULL) matrix_freelist(mptr->dirlist) ;
	if (mptr->fptr) status = fclose(mptr->fptr);
	if (mptr->fname) free(mptr->fname);
	free(mptr);
	return status;
}

MatrixData *matrix_read(mptr, matnum, dtype)
	MatrixFile     *mptr;
	int             matnum, dtype;
{
	MatrixData     *data;

	matrix_errno = OK;
	matrix_errtxt[0] = '\0';
	if (mptr == NULL)
		matrix_errno = MAT_READ_FROM_NILFPTR;
	else if (mptr->mhptr == NULL)
		matrix_errno = MAT_NOMHD_FILE_OBJECT;
	if (matrix_errno != OK)
		return (NULL);
	/* allocate space for MatrixData structure and initialize */
	data = (MatrixData *) calloc(1, sizeof(MatrixData));
	if (!data)
		return (NULL);

	/* allocate space for subheader and initialize */
	data->shptr = (caddr_t) calloc(2, MatBLKSIZE);
	if (!data->shptr) {
		free(data);
		return (NULL);
	}
/*	if (read_z_data(mptr, matnum, data, dtype) != OK) { */
	if (read_host_data(mptr, matnum, data, dtype) != OK) {
		free_matrix_data(data);
		data = NULL;
	} else if (dtype != NoData && data->data_type != dtype)
		matrix_convert_data(data, dtype);
	return (data);
}

static int matrix_write_slice(mptr,matnum,data,plane)
MatrixFile *mptr ;
MatrixData *data ;
int   matnum, plane;
{
	MatrixData *slice;
	Image_subheader *imh;
	struct Matval val;
	int i, npixels, nblks, s_matnum;
	short *sdata;
	u_char* bdata;
	int	ret;

	switch(mptr->mhptr->file_type) {
		case PetImage :
			if (data->data_type ==  ByteData) {
/*				fprintf(stderr,"Only short data type supported in V6\n");*/
				matrix_errno = MAT_INVALID_DATA_TYPE ;
				return ERROR;
			}
			mat_numdoc(matnum,&val);
			slice = (MatrixData*)malloc(sizeof(MatrixData));
			if( !slice ) return(ERROR) ;

			imh = (Image_subheader*)calloc(1,MatBLKSIZE);
			if( !imh ) {
				free( slice );
				return(ERROR) ;
			}
			memcpy(slice,data,sizeof(MatrixData));
			memcpy(imh,data->shptr,sizeof(Image_subheader));
			slice->shptr = (caddr_t)imh;
			slice->zdim = imh->z_dimension = 1;
			npixels = slice->xdim*slice->ydim;
			nblks = (npixels*2 + MatBLKSIZE-1)/MatBLKSIZE;
			slice->data_ptr = (caddr_t)calloc(nblks,MatBLKSIZE);
			if( !slice->data_ptr ) {
				free( slice );
				free( imh );
				return(ERROR) ;
			}
			slice->data_size = nblks*MatBLKSIZE;
			if (data->data_type ==  ByteData) {
				bdata = (u_char*)(data->data_ptr+(plane-1)*npixels);
				imh->image_min = find_bmin(bdata,npixels);
				imh->image_max = find_bmax(bdata,npixels);
				sdata = (short*)slice->data_ptr;
				for (i=0; i<npixels; i++)  sdata[i] = bdata[i];
			} else {
				sdata = (short*)(data->data_ptr+(plane-1)*npixels*2);
				imh->image_min = find_smin(sdata,npixels);
				imh->image_max = find_smax(sdata,npixels);
				memcpy(slice->data_ptr,sdata,npixels*2);
			}
			s_matnum = mat_numcod(val.frame,plane,val.gate,val.data,val.bed);
			ret = matrix_write(mptr,s_matnum,slice);
			free_matrix_data(slice);
			return( ret );
		case PetVolume :
		case ByteVolume :
		case InterfileImage:
/*			fprintf(stderr,
			"matrix_slice_write : Main_header file_type should be PetImage\n");
*/
			matrix_errno = MAT_FILE_TYPE_NOT_MATCH;
			return ERROR;
		default:
/*			fprintf(stderr,"V7 to V6 conversion only supported for images\n");*/
			matrix_errno = MAT_FILE_TYPE_NOT_MATCH;
			return ERROR;
		}
}

matrix_write(mptr, matnum, data)
  MatrixFile *mptr ;
  MatrixData *data ;
  int	matnum;
{
  int	slice ;

	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';
	if (mptr == NULL) matrix_errno = MAT_READ_FROM_NILFPTR ;
	else if (mptr->mhptr == NULL) matrix_errno = MAT_NOMHD_FILE_OBJECT ;
	else if (data->shptr == NULL) matrix_errno = MAT_NIL_SHPTR ;
	else if (data->data_ptr == NULL) matrix_errno = MAT_NIL_DATA_PTR ;
	if (matrix_errno != OK) return (ERROR) ;
	if ((mptr->mhptr->sw_version<V7 || mptr->mhptr->file_type==PetImage)
		 && data->zdim>1) {
		for (slice=1; slice<=data->zdim; slice++) {
			if (matrix_write_slice(mptr,matnum,data,slice) != OK)
				return ERROR;
		}
		return OK;
	}
/*	return  write_z_data(mptr, matnum, data); */
	return  write_host_data(mptr, matnum, data);
} 

#ifdef __STDC__
int find_bmax(const u_char *bdata, int nvals)
#else
int find_bmax( bdata, nvals)
  u_char *bdata;
  int nvals;
#endif
{
	int i;
	u_char bmax = bdata[0];
	for (i=1; i<nvals; i++)
	  if (bdata[i] > bmax) bmax = bdata[i];
	return bmax;
}
#ifdef __STDC__
int find_bmin(const u_char *bdata, int nvals)
#else
int find_bmin( bdata, nvals)
  u_char *bdata;
  int nvals;
#endif
{
	int i;
	u_char bmin = bdata[0];
	for (i=1; i<nvals; i++)
	  if (bdata[i] < bmin) bmin = bdata[i];
	return bmin;
}

#ifdef __STDC__
int find_smax( const short *sdata, int nvals)
#else
int find_smax( sdata, nvals)
  short *sdata;
  int nvals;
#endif
{
	int i;
	short smax = sdata[0];
	for (i=1; i<nvals; i++)
	  if (sdata[i] > smax) smax = sdata[i];
	return smax;
}
#ifdef __STDC__
int find_smin( const short *sdata, int nvals)
#else
int find_smin( sdata, nvals)
  short *sdata;
  int nvals;
#endif
{
	int i;
	short  smin = sdata[0];
	for (i=1; i<nvals; i++)
	  if (sdata[i] < smin) smin = sdata[i];
	return smin;
}

#ifdef __STDC__
int find_imax( const int *idata, int nvals)
#else
int find_imax( idata, nvals)
  int *idata;
  int nvals;
#endif
{
	int i, imax=idata[0];
	for (i=1; i<nvals; i++)
	  if (idata[i]>imax) imax = idata[i];
	return imax;
}
#ifdef __STDC__
int find_imin( const int *idata, int nvals)
#else
int find_imin( idata, nvals)
  int *idata;
  int nvals;
#endif
{
	int i, imin=idata[0];
	for (i=1; i<nvals; i++)
	  if (idata[i]<imin) imin = idata[i];
	return imin;
}

#ifdef __STDC__
float find_fmin( const float *fdata, int nvals)
#else
float find_fmin( fdata, nvals)
  float *fdata;
  int nvals;
#endif
{
	int i;
	float fmin = fdata[0];
	for (i=1; i<nvals; i++)
	  if (fdata[i]<fmin) fmin = fdata[i];
	return fmin;
}
#ifdef __STDC__
float find_fmax( const float *fdata, int nvals)
#else
float find_fmax( fdata, nvals)
  float *fdata;
  int nvals;
#endif
{
	int i;
	float fmax = fdata[0];
	for (i=1; i<nvals; i++)
	  if (fdata[i]>fmax) fmax = fdata[i];
	return fmax;
}


read_host_data(mptr, matnum, data, dtype) 

  MatrixFile	*mptr ;
  MatrixData	*data ;
  int	matnum , dtype;

 {
  struct MatDir matdir;
  int	 nblks, data_size ;
  Scan_subheader *scansub ;
  Scan3D_subheader *scan3Dsub ;
  Image_subheader *imagesub ;
  Attn_subheader *attnsub ;
  Norm_subheader *normsub ;
  Norm3D_subheader *norm3d;
  int sx,sy,sz;
  int elem_size= 2, datasize;

	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';

	if (mptr->interfile_header)		 /* read interfile */
		return interfile_read(mptr, matnum, data, dtype);
	if (matrix_find(mptr,matnum,&matdir) == ERROR)
	{
	   matrix_errno = MAT_MATRIX_NOT_FOUND ;
	   return(ERROR) ;
	}

	data->matnum = matnum;
	data->matfile = mptr;
	data->mat_type = mptr->mhptr->file_type;
	nblks = matdir.endblk - matdir.strtblk ;
	data_size = data->data_size = 512*nblks;
	if (dtype != MAT_SUB_HEADER)
	{
	  data->data_ptr = (caddr_t) calloc(1, data_size) ;
	  if (data->data_ptr == NULL)
	  {
		 return(ERROR) ;
	  }
	} 
	switch(mptr->mhptr->file_type)
	{
	   case Sinogram :
		scansub = (Scan_subheader *) data->shptr ;
		if (mptr->acs)
			acs_read_scan_subheader(mptr->fname, mptr->mhptr, matdir.strtblk,
			scansub) ;
		else mat_read_scan_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk,
			scansub) ;
		data->data_type = scansub->data_type ;
		data->xdim = scansub->num_r_elements ;
		data->ydim = scansub->num_angles ;
		data->zdim = scansub->num_z_elements ;
		data->scale_factor = scansub->scale_factor ;
		data->pixel_size = scansub->x_resolution;
		data->data_max = scansub->scan_max * scansub->scale_factor ;
		if (dtype == MAT_SUB_HEADER) break;
		if (mptr->acs) read_raw_acs_data(mptr->fname, matdir.strtblk+1,
			nblks, data->data_ptr, scansub->data_type);
		else read_matrix_data(mptr->fptr, matdir.strtblk+1, nblks,
		  data->data_ptr, scansub->data_type) ;
		break ;
	   case Short3dSinogram :
	   case Float3dSinogram :
		scan3Dsub = (Scan3D_subheader *) data->shptr ;
		if (mptr->acs)
			acs_read_Scan3D_subheader(mptr->fname, mptr->mhptr, matdir.strtblk,
			scan3Dsub) ;
		else mat_read_Scan3D_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk,
			scan3Dsub) ;
		data->data_type = scan3Dsub->data_type ;
		data->scale_factor = scan3Dsub->scale_factor ;
		data->pixel_size = scan3Dsub->x_resolution;
		data->y_size = data->pixel_size;
		data->z_size = scan3Dsub->z_resolution;
		data->data_max = scan3Dsub->scan_max * scan3Dsub->scale_factor ;
		sx = data->xdim = scan3Dsub->num_r_elements ;
		sz = data->ydim = scan3Dsub->num_angles ;
		sy = data->zdim = scan3Dsub->num_z_elements[0] ;
		if (dtype == MAT_SUB_HEADER) break;

		if (mptr->acs) 
			read_raw_acs_data(mptr->fname, matdir.strtblk+2, nblks-1,
		  		data->data_ptr, scan3Dsub->data_type) ;
		else read_matrix_data(mptr->fptr, matdir.strtblk+2, nblks-1,
		  		data->data_ptr, scan3Dsub->data_type) ;
		if (scan3Dsub->data_type == SunShort) {
			scan3Dsub->scan_max = find_smax((short*)data->data_ptr,sx*sy*sz);
			scan3Dsub->scan_min = find_smin((short*)data->data_ptr,sx*sy*sz);
			data->data_max = scan3Dsub->scan_max * scan3Dsub->scale_factor;
			data->data_min = scan3Dsub->scan_min * scan3Dsub->scale_factor;

		} else {
			data->data_max = find_fmax((float*)data->data_ptr,sx*sy*sz);
			data->data_min = find_fmin((float*)data->data_ptr,sx*sy*sz);
		}
		break ;
	   case ByteVolume :
	   case PetImage :
	   case PetVolume :
	   case InterfileImage:
		imagesub = (Image_subheader *) data->shptr ;
		mat_read_image_subheader(mptr->fptr,mptr->mhptr,matdir.strtblk,
			imagesub);
		data->data_type = imagesub->data_type ;
		sx = data->xdim = imagesub->x_dimension ;
		sy = data->ydim = imagesub->y_dimension ;
		data->scale_factor = imagesub->scale_factor ;
		if (data->data_type==ByteData || data->data_type==ColorData)
			elem_size =1 ;
		else elem_size = 2;
		sz = data->zdim = imagesub->z_dimension ;
		if( sx*sy*elem_size == 0 ) {
			matrix_errno = MAT_INVALID_DIMENSION;
			return( ERROR );
		}
/*
        if (sz != data_size/(sx*sy*elem_size)) {
			struct Matval mat;
			mat_numdoc(data->matnum, &mat);
			fprintf(stderr,"Warning : %d,%d,%d,%d,%d ",
        		mat.frame, mat.plane, mat.gate, mat.data, mat.bed);
		  fprintf(stderr,"header z_dimension (%d) doesn't agree with number of blocks (%d)\n",sz,nblks);
		}
*/
		if (sz > data_size/(sx*sy*elem_size))
	      sz = data->zdim = imagesub->z_dimension = data_size/(sx*sy*elem_size);
/* fix inconsistent file types */
		if (data->zdim > 1) {
			if (data->data_type == ByteData) data->mat_type = ByteVolume;
			else data->mat_type = PetVolume;
		} else data->mat_type = PetImage;

		data->pixel_size = imagesub->x_pixel_size;
		data->y_size = imagesub->y_pixel_size;
				/* if imagesub->y_pixel_size not filled assume square pixels */
		if (data->y_size <= 0) data->y_size = imagesub->x_pixel_size;
		data->z_size = imagesub->z_pixel_size;
				/* if imagesub->z_pixel_size not filled use palne separation */
		if (data->z_size <= 0) data->z_size = mptr->mhptr->plane_separation;
		data->data_max = imagesub->image_max * imagesub->scale_factor;
		/* KT added next 3 lines */
		data->z_origin = imagesub->z_offset;
		data->y_origin = imagesub->y_offset;
		data->x_origin = imagesub->x_offset;
		if (dtype == MAT_SUB_HEADER) break;
		read_matrix_data(mptr->fptr, matdir.strtblk+1, nblks,
		  data->data_ptr, imagesub->data_type) ;
		if (imagesub->data_type == ByteData) {
			imagesub->image_max = find_bmax((u_char*)data->data_ptr,sx*sy*sz);
			imagesub->image_min = find_bmin((u_char*)data->data_ptr,sx*sy*sz);
		} else {
			imagesub->image_max = find_smax((short*)data->data_ptr,sx*sy*sz);
			imagesub->image_min = find_smin((short*)data->data_ptr,sx*sy*sz);
		}
		data->data_max = imagesub->image_max * imagesub->scale_factor;
		data->data_min = imagesub->image_min * imagesub->scale_factor;
		break ;
	   case AttenCor :
		attnsub = (Attn_subheader *) data->shptr ;
		mat_read_attn_subheader(mptr->fptr,mptr->mhptr,matdir.strtblk,attnsub);
		data->data_type = attnsub->data_type ;
		sx = data->xdim = attnsub->num_r_elements ;
		sz = data->ydim = attnsub->num_angles ;
		sy = data->zdim = attnsub->z_elements[0] ;
		data->scale_factor = attnsub->scale_factor;
		data->pixel_size = attnsub->x_resolution;
		if (dtype == MAT_SUB_HEADER) break;
		read_matrix_data(mptr->fptr, matdir.strtblk+1, nblks,
			data->data_ptr, attnsub->data_type);
		data->data_max = find_fmax((float*)data->data_ptr,sx*sy*sz);
		break ;
	   case Normalization :
		normsub = (Norm_subheader *) data->shptr ;
		mat_read_norm_subheader(mptr->fptr,mptr->mhptr,matdir.strtblk,normsub);
		data->data_type = normsub->data_type ;
		data->xdim = normsub->num_r_elements ;
		data->ydim = normsub->num_angles ;
		data->zdim = normsub->num_z_elements ;
		data->scale_factor = normsub->scale_factor ;
		if (dtype == MAT_SUB_HEADER) break;
		read_matrix_data(mptr->fptr, matdir.strtblk+1, nblks,
		  data->data_ptr, normsub->data_type) ;
		data->data_max = data->scale_factor * 
		  find_fmax((float*)data->data_ptr, data->xdim * data->ydim);
		break ;
	   case Norm3d :
		norm3d = (Norm3D_subheader *) data->shptr ;
		mat_read_norm3d_subheader(mptr->fptr,mptr->mhptr,matdir.strtblk,norm3d);
		data->data_type = norm3d->data_type ;
		data->xdim = norm3d->num_r_elements;	/* 336 */
		data->ydim = norm3d->crystals_per_ring;	/* 784 */
		data->zdim = norm3d->num_crystal_rings;	/* 24 */
		data->scale_factor = 1.0;
		if (dtype == MAT_SUB_HEADER) break;
/*
		336*(1+7) + 24*784
		336*(1+7) =
		radial elements (plane geometry + Crystal Interference) Corrections
		24*784 = cristals efficiencies
*/
		read_matrix_data(mptr->fptr, matdir.strtblk+1, nblks,
		  data->data_ptr, norm3d->data_type) ;
		break ;
	   default :
		matrix_errno = MAT_UNKNOWN_FILE_TYPE ;
		return(ERROR) ;
		break ;
	}
	return(OK) ;
}


int write_host_data(mptr, matnum, data)

  MatrixFile *mptr ;
  MatrixData *data ;
  int	matnum ;
{
  struct MatDir matdir, dir_entry ;
  Scan3D_subheader *scan3Dsub = NULL;
  Scan_subheader *scansub ;
  Image_subheader *imagesub ;
  Attn_subheader *attnsub ;
  Norm_subheader *normsub ;
  int	status, blkno, nblks ;
	
	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';
	status = OK ;
	nblks = (data->data_size+511)/512;

	/* 3D sinograms subheader use one more block */
    if (mptr->mhptr->file_type == Short3dSinogram  ||
        mptr->mhptr->file_type == Float3dSinogram) nblks += 1;

	if (matrix_find(mptr, matnum, &matdir) == ERROR)
	{
	   blkno = mat_enter(mptr->fptr, mptr->mhptr, matnum, nblks) ;
	   if( blkno == ERROR ) return( ERROR );
	   dir_entry.matnum = matnum ;
	   dir_entry.strtblk = blkno ;
	   dir_entry.endblk = dir_entry.strtblk + nblks - 1 ;
	   dir_entry.matstat = 1 ;
	   insert_mdir(dir_entry, mptr->dirlist) ;
	   matdir = dir_entry ;
	}

	switch(mptr->mhptr->file_type)
	{
       case Float3dSinogram :
       case Short3dSinogram :
        scan3Dsub = (Scan3D_subheader*) data->shptr;
        if (mat_write_Scan3D_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk,
            scan3Dsub) == ERROR) return ERROR;
          if (write_matrix_data(mptr->fptr, matdir.strtblk+2,
            nblks, data->data_ptr, scan3Dsub->data_type) == ERROR) return ERROR;
          break;
	   case Sinogram :
		  scansub = (Scan_subheader *) data->shptr ;
		  if( mat_write_scan_subheader(mptr->fptr,mptr->mhptr, matdir.strtblk, scansub) == ERROR ) return( ERROR );
		  status = write_matrix_data(mptr->fptr, matdir.strtblk+1, nblks, data->data_ptr, scansub->data_type) ;
		  if( status == ERROR ) return( ERROR );
		  break ;
	   case ByteVolume :
	   case PetImage :
	   case PetVolume :
	   case InterfileImage:
		  imagesub = (Image_subheader *) data->shptr ;
	   	  if (imagesub == NULL) {
                imagesub = (Image_subheader *) calloc(1, MatBLKSIZE);
                data->shptr = (caddr_t)imagesub;
		  }							/* use MatrixData info */
		  imagesub->x_pixel_size = data->pixel_size;
		  imagesub->y_pixel_size = data->y_size;
		  imagesub->z_pixel_size = data->z_size;
		  imagesub->num_dimensions = 3;
		  imagesub->x_dimension = data->xdim;
		  imagesub->y_dimension = data->ydim;
		  imagesub->z_dimension = data->zdim;
		  imagesub->image_max = (int)(data->data_max/data->scale_factor);
		  imagesub->image_min = (int)(data->data_min/data->scale_factor);
		  imagesub->scale_factor = data->scale_factor;
		  imagesub->data_type = data->data_type;
		  if( mat_write_image_subheader(mptr->fptr,mptr->mhptr,matdir.strtblk, imagesub) == ERROR ) return( ERROR );
		  status = write_matrix_data(mptr->fptr, matdir.strtblk+1, nblks, data->data_ptr, imagesub->data_type) ;
		  if( status == ERROR ) return( ERROR );
		  break ;
	   case AttenCor :
		  attnsub = (Attn_subheader *) data->shptr ;
		  if( mat_write_attn_subheader(mptr->fptr,mptr->mhptr, matdir.strtblk, attnsub) == ERROR ) return( ERROR );
		  status = write_matrix_data(mptr->fptr, matdir.strtblk+1, nblks, data->data_ptr, attnsub->data_type);
		  if( status == ERROR ) return( ERROR );
		  break ;
	   case Normalization :
		  normsub = (Norm_subheader *) data->shptr ;
		  if( mat_write_norm_subheader(mptr->fptr,mptr->mhptr, matdir.strtblk, normsub) == ERROR ) return( ERROR );
		  status = write_matrix_data(mptr->fptr, matdir.strtblk+1, nblks, data->data_ptr, normsub->data_type) ;
		  if( status == ERROR ) return( ERROR );
		  break ;
	   default :	/* default treated as sinogram */
		  scansub = (Scan_subheader *) data->shptr ;
		  if( mat_write_scan_subheader(mptr->fptr,mptr->mhptr, matdir.strtblk, scansub) == ERROR ) return( ERROR );
		  status = write_matrix_data(mptr->fptr, matdir.strtblk+1, nblks, data->data_ptr, scansub->data_type) ;
		  if( status == ERROR ) return( ERROR );
		  break ;
	}
/* V6 num_frames is max frame number ==> update header only for V7 files */
/* Comment momentanne
	if (mptr->mhptr->sw_version == V7) return mh_update(mptr);
*/
	return(status) ;
}


int matrix_find( matfile, matnum, matdir)
  MatrixFile *matfile ;
  int	matnum ;
  struct MatDir *matdir;
{
  MatDirNode	*node ;

	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';
	if (matfile == NULL) return(ERROR) ;
	if (matfile->dirlist == NULL) return(ERROR) ;	
	node = matfile->dirlist->first ;
	while (node != NULL)
	{
	   if (node->matnum == matnum)
	   {
		matdir->matnum = node->matnum ;
		matdir->strtblk = node->strtblk ;
		matdir->endblk = node->endblk ;
		matdir->matstat = node->matstat ;
		break ;
	   }
	   node = node->next ;
	}
	if (node != NULL) return(OK) ;
	else return(ERROR) ;
}
	


int insert_mdir(matdir, dirlist)

  MatDirList	*dirlist ;
  struct MatDir	matdir ;

{
  MatDirNode	*node ;
 
	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';
	if (dirlist == NULL)
	{
		dirlist = (MatDirList *) malloc(sizeof(MatDirList)) ;
		if (dirlist == NULL) return(ERROR) ;
		dirlist->nmats = 0 ;
		dirlist->first = NULL ;
		dirlist->last = NULL ;
	}
	node = (MatDirNode *) malloc(sizeof(MatDirNode)) ;
	if (node == NULL) return(ERROR) ;

	node->matnum = matdir.matnum ;
	node->strtblk = matdir.strtblk ;
	node->endblk = matdir.endblk ;
	node->matstat = matdir.matstat;
	node->next = NULL ;

	if (dirlist->first == NULL)	/* if list was empty, add first node */
	{
	   dirlist->first = node ;
	   dirlist->last = node ;
	   dirlist->nmats = 1 ;
	}
	else
	{
	   (dirlist->last)->next = node ;
	   dirlist->last = node ;
	   ++(dirlist->nmats) ;
	}
	return OK;
}

void free_matrix_data(data)

  MatrixData	*data ;

{
	if (data != NULL)
	{
	   if (data->data_ptr != NULL) free(data->data_ptr) ;
	   if (data->shptr != NULL) free(data->shptr) ;
	   free(data) ;
	}

}



file_exists(filename)	/* subroutine to see if file exists or not */

  char *filename ;

{
#ifndef _WIN32
  Main_header mhead;
#endif
  struct stat stbuf;
	if (!is_acs(filename))
	{
		if (stat(filename, &stbuf) == -1) return(FALSE) ;
		return(TRUE) ;
	}
	else {
#ifndef _WIN32
		if (rts_rmhd(filename, &mhead) == OK) return (TRUE);
#endif
		return FALSE;
	}
}


/* subroutine to return starting position of substring within string */
/* return index of t in s, -1 if none */

strindex(s, t)

  char	s[], t[] ;

{
  int	i, j, k ;

	for (i=0 ; s[i] !='\0' ; i++)
	{
	   for (j=i, k=0 ; t[k] != '\0' && s[j]==t[k] ; j++, k++) ;
	   if (k > 0 && t[k] == '\0') return i ;
	}
	return -1 ;
}

#ifdef __STDC__
void matrix_perror( const char* s)
#else
void matrix_perror( s)
  char *s;
#endif
{
	if (matrix_errno)
		fprintf( stderr, "%s: %s\n", s, matrix_errors[matrix_errno]);
	else perror(s);
}

copy_proto_object( new, old)
  MatrixData *new, *old;
{
	int sh_size;

	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';
	if (new->mat_type == old->mat_type)
	{
	  switch (new->mat_type)
	  {
		case Sinogram:
			sh_size = sizeof(Scan_subheader);
			break;
		case PetVolume:
			sh_size = sizeof(Image_subheader);
			break;
		case AttenCor:
			sh_size = sizeof(Attn_subheader);
			break;
		case Normalization:
			sh_size = sizeof(Norm_subheader);
			break;
	  }
	  new->shptr = (caddr_t) malloc( sh_size);
	  if( !new->shptr ) return( ERROR );
	  memcpy(new->shptr, old->shptr, sh_size);
	}
	new->xdim = old->xdim;
	new->ydim = old->ydim;
	new->zdim = old->zdim;
	new->data_type = old->data_type;
	new->scale_factor = old->scale_factor;
	new->pixel_size = old->pixel_size;
	new->y_size = old->y_size;
	new->z_size = old->z_size;
	new->data_min = old->data_min;
	new->data_max = old->data_max;
	return( 0 );
}

Main_header *matrix_init_main_header( fname, ftype, mh_proto)
  char *fname;
  DataSetType ftype;
  Main_header *mh_proto;
{
	Main_header *mhptr;

	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';
	mhptr = (Main_header*) calloc( 1, sizeof(Main_header));
	if( !mhptr ) return( NULL );
	if (mh_proto)
	  memcpy(mhptr, mh_proto, sizeof(Main_header));
	mhptr->file_type = ftype;
	strncpy( mhptr->original_file_name, fname, 20);
	return mhptr;
}

void matrix_free( matrix)
  MatrixData *matrix;
{
	if (matrix->shptr) free( matrix->shptr);
	if (matrix->data_ptr) free( matrix->data_ptr);
	free( matrix);
}

convert_float_scan( scan, fdata)
  MatrixData *scan;
  float *fdata;
{
	int i, nvals, tot;
	float fmax, scale;
	short int *sdata;
	Scan_subheader *ssh;

	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';
	if (scan->data_ptr)
	  free(scan->data_ptr);
	nvals = scan->xdim*scan->ydim;
	sdata = (short int*) malloc( nvals*sizeof(short int));
	if (sdata == NULL) return ERROR;
	scan->data_ptr = (caddr_t) sdata;
	scan->data_size = nvals*sizeof(short int);
	fmax = fabs(*fdata);
	for (i=0; i<nvals; i++)
	  if (fabs(fdata[i]) > fmax) fmax = fabs(fdata[i]);
	scale = 1.0;
	if (fmax > 0.0) scale = 32767./fmax;
	tot = 0;
	for (i=0; i<nvals; i++)
	{
	  sdata[i] = (short)(scale*fdata[i] + 0.5);
	  tot += sdata[i];
	}
	scan->scale_factor = 1.0/scale;
	ssh = (Scan_subheader*) scan->shptr;
	ssh->scan_min = 0;
	ssh->scan_max = (short)(fmax*scale);
	ssh->num_r_elements = scan->xdim;
	ssh->num_angles = scan->ydim;
	ssh->net_trues = tot;
	ssh->scale_factor = 1.0/scale;
		ssh->x_resolution = scan->pixel_size;
	return OK;
}

convert_long_scan( scan, ldata)
  MatrixData *scan;
  int *ldata;
{
	int i, nvals, tot, lmax;
	float scale;
	short int *sdata;
	Scan_subheader *ssh;

	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';
	if (scan->data_ptr)
	  free(scan->data_ptr);
	nvals = scan->xdim*scan->ydim;
	sdata = (short int*) malloc( nvals*sizeof(short int));
	if (sdata==NULL) return ERROR;
	scan->data_ptr = (caddr_t) sdata;
	scan->data_size = nvals*sizeof(short int);
	lmax = abs(*ldata);
	for (i=0; i<nvals; i++)
	  if (abs(ldata[i]) > lmax) lmax = abs(ldata[i]);
	scale = 1.0;
	if (lmax > 0) scale = 32767./lmax;
	tot = 0;
	for (i=0; i<nvals; i++)
	{
	  sdata[i] = (short int) (scale*ldata[i]);
	  tot += sdata[i];
	}
	scan->scale_factor = 1.0/scale;
	ssh = (Scan_subheader*) scan->shptr;
	ssh->scan_min = 0;
	ssh->scan_max = (int)(lmax*scale);
	ssh->num_r_elements = scan->xdim;
	ssh->num_angles = scan->ydim;
	ssh->net_trues = tot;
	ssh->scale_factor = 1.0/scale;
	ssh->x_resolution = scan->pixel_size;
	return OK;
}


static void set_limits( ptr, n, s, e )
char	*ptr;
int	n, *s, *e;
{
	int	flag;

	flag = TRUE;
	switch( *ptr ) {
	case 0:
	case ',':
		break;
	case '-':
		sscanf( ptr+1, "%d", e );
		break;
	case '*':
		ptr++;
		if( *ptr == '-' ) {
			flag = FALSE;
		} else {
			*e = n;
			break;
		}
	default:
		if( flag ) sscanf( ptr, "%d", s );
		while( *ptr >= '0' && *ptr <= '9' ) ptr++;
		switch( *ptr++ ) {
		case '-':
			if( *ptr == '*' || *ptr == ',' || *ptr == 0 ) {
				*e = n;
			} else {
				sscanf( ptr, "%d", e );
			}
			break;
		default:
			*e = *s;
			break;
		}
		break;
	}	
}

int get_mat_spec( matspec, matnval, fname, matlim )
char	*matspec, *fname;
struct Matval		*matnval;
struct Matlimits	*matlim;
{
	int	 count;
	char	*ptr, s[1000];

	matlim->framestart = 1;
	matlim->frameend = 1;
	matlim->planestart = 1;
	matlim->planeend = matnval->plane;
	matlim->gatestart = 1;
	matlim->gateend = 1;
	matlim->datastart = 0;
	matlim->dataend = 0;
	matlim->bedstart = 0;
	matlim->bedend = 0;

	count = 0;

	strcpy( s, matspec );
	ptr = s;
	while( *ptr ) {
		if( *ptr == ' ' ) {
			strcpy( ptr, ptr+1 );
		} else {
			ptr++;
		}
	}
	ptr = strchr( s, ',' );
	if( ptr ) *ptr = 0;
	while( ptr++ ) {
		switch( count++ ) {
		case 0:	set_limits( ptr, matnval->frame, &matlim->framestart, &matlim->frameend );
			break;
		case 1:	set_limits( ptr, matnval->plane, &matlim->planestart, &matlim->planeend );
			break;
		case 2:	set_limits( ptr, matnval->gate, &matlim->gatestart, &matlim->gateend );
			break;
		case 3:	set_limits( ptr, matnval->data, &matlim->datastart, &matlim->dataend );
			break;
		case 4:	set_limits( ptr, matnval->bed, &matlim->bedstart, &matlim->bedend );
			break;
		}
		ptr = strchr(ptr, ',');
	}

	strcpy( fname, s );

	return( 0 );
}
MatrixData *matrix_read_scan(mptr, matnum, dtype, segment)
MatrixFile *mptr;
int matnum, dtype, segment;
{
	int i, nblks, plane_size;
	int sx, sy, sz;
	MatrixData *data;
	struct MatDir matdir;
	Scan3D_subheader *scan3Dsub;
	Attn_subheader *attnsub;
	int status, group;
	unsigned z_elements, file_pos=0;

/* Scan3D and Atten storage:
	storage_order = 0 : (((projs x z_elements)) x num_angles) x Ringdiffs
	storage_order != 0 : (((projs x num_angles)) x z_elements)) x Ringdiffs
    
*/

	matrix_errno = MAT_OK;
	matrix_errtxt[0] = '\0';

    if (matrix_find(mptr,matnum,&matdir) == ERROR)
		return NULL;

	if ((data = (MatrixData *) calloc( 1, sizeof(MatrixData))) != NULL) {
		if ( (data->shptr = (caddr_t) calloc(2, MatBLKSIZE)) == NULL) {
			free(data);
			return NULL;
		}
	} else return NULL;

    data->matnum = matnum;
    data->matfile = mptr;
    data->mat_type = mptr->mhptr->file_type;
    nblks = matdir.endblk - matdir.strtblk;
	group=abs(segment);
	switch(mptr->mhptr->file_type) {
	case Float3dSinogram :
	case Short3dSinogram :
		scan3Dsub = (Scan3D_subheader *)data->shptr;
		if (mptr->acs)
			status = acs_read_Scan3D_subheader(mptr->fname, mptr->mhptr,
				matdir.strtblk, scan3Dsub);
		else
			status = mat_read_Scan3D_subheader(mptr->fptr, mptr->mhptr,
				matdir.strtblk, scan3Dsub);
		if (status == ERROR) {
			free_matrix_data(data);	
			return NULL;
		}
		file_pos = (matdir.strtblk+1)*MatBLKSIZE;
		data->data_type = scan3Dsub->data_type;
		data->scale_factor = scan3Dsub->scale_factor;
		data->pixel_size = scan3Dsub->x_resolution;
		sx = data->xdim = scan3Dsub->num_r_elements;
		z_elements = scan3Dsub->num_z_elements[group];
		if (group > 0) z_elements /= 2;
		if (scan3Dsub->storage_order == 0) {
			data->z_size = data->pixel_size;
			data->y_size = mptr->mhptr->plane_separation;
			sy = data->ydim = z_elements;
			sz = data->zdim = scan3Dsub->num_angles;
			plane_size = sx*sz*sizeof(short);
		} else {
			data->y_size = data->pixel_size;
			data->z_size = mptr->mhptr->plane_separation;
			sy = data->ydim = scan3Dsub->num_angles;
			sz = data->zdim = z_elements;
			plane_size = sx*sy*sizeof(short);
		}
		if (mptr->mhptr->file_type == Float3dSinogram) plane_size *= 2;
		data->data_max = scan3Dsub->scan_max * scan3Dsub->scale_factor;
		if (dtype == MAT_SUB_HEADER) break;
		for (i=0; i<group; i++)
			file_pos += scan3Dsub->num_z_elements[i]*plane_size;
		if (segment < 0) file_pos += z_elements*plane_size;
		if (mptr->acs) {
			matrix_errno = MAT_BAD_FILE_ACCESS_MODE;
			free_matrix_data(data);
			return NULL;
		}
		data->data_size = z_elements*plane_size;
		nblks = (data->data_size+511)/512;
	  	if ((data->data_ptr = (caddr_t)calloc(nblks,512)) == NULL ||
			fseek(mptr->fptr,file_pos,0) == -1 ||
			fread(data->data_ptr, plane_size, z_elements, mptr->fptr)
				!= z_elements ||
			file_data_to_host(data->data_ptr,nblks,data->data_type) == ERROR) {
				free_matrix_data(data);
				return NULL;
		}
		if (mptr->mhptr->file_type == Short3dSinogram) {
			scan3Dsub->scan_max = find_smax((short*)data->data_ptr,sx*sy*sz);
			scan3Dsub->scan_min = find_smin((short*)data->data_ptr,sx*sy*sz);
			data->data_max = scan3Dsub->scan_max * scan3Dsub->scale_factor;
			data->data_min = scan3Dsub->scan_min * scan3Dsub->scale_factor;

		} else {
			data->data_max = find_fmax((float*)data->data_ptr,sx*sy*sz);
			data->data_min = find_fmin((float*)data->data_ptr,sx*sy*sz);
		}
		break;
	case AttenCor :
		attnsub = (Attn_subheader *) data->shptr;
		if (mat_read_attn_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk,
			attnsub) == ERROR) {
				free_matrix_data(data);
				return NULL;
		}
		file_pos = matdir.strtblk*MatBLKSIZE;
		data->data_type = attnsub->data_type;
        data->scale_factor = attnsub->scale_factor;
        data->pixel_size = attnsub->x_resolution;
        sx = data->xdim = attnsub->num_r_elements;
        z_elements = attnsub->z_elements[group];
        if (group > 0) z_elements /= 2;
        if (attnsub->storage_order == 0) {
            data->y_size = mptr->mhptr->plane_separation;
            data->z_size = data->pixel_size;
            sy = data->ydim = z_elements;
            sz = data->zdim = attnsub->num_angles;
        } else {
            data->z_size = mptr->mhptr->plane_separation;
            data->y_size = data->pixel_size;
            sy = data->ydim = attnsub->num_angles;
            sz = data->zdim = z_elements;
        }
        data->data_max = attnsub->attenuation_max;
        if (dtype == MAT_SUB_HEADER) break;
        plane_size = sx*attnsub->num_angles*sizeof(float);
        for (i=0; i<group; i++)
            file_pos += attnsub->z_elements[i]*plane_size;
        if (segment < 0) file_pos += z_elements*plane_size;
		if (dtype == MAT_SUB_HEADER) break;
		if (mptr->acs) {
			matrix_errno = MAT_BAD_FILE_ACCESS_MODE;
			free_matrix_data(data);
			return NULL;
		}
		data->data_size = z_elements*plane_size;
		nblks = (data->data_size+511)/512;
	  	if ((data->data_ptr = (caddr_t)calloc(nblks,512)) == NULL ||
			fseek(mptr->fptr,file_pos,0) == -1 ||
			fread(data->data_ptr,plane_size,z_elements,mptr->fptr) !=
				z_elements ||
			file_data_to_host(data->data_ptr,nblks,data->data_type) == ERROR) {
				free_matrix_data(data);
        		return NULL;
    	}
		data->data_max = find_fmax((float*)data->data_ptr,sx*sy*sz);
		break;
	default:
		matrix_errno = MAT_FILE_TYPE_NOT_MATCH;
		return NULL;
	}
	return data;
}
