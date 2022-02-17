/* @(#)matcopy.c	1.4 7/10/92 */
#ifndef lint
static char sccsid[]="(#)matcopy.c 1.4 7/10/92 Copyright 1990 CTI Pet Systems, Inc.";
#endif

/* 09-Nov-1995 : modified by sibomana@topo.ucl.ac.be */

#include <malloc.h>
#include <string.h>
#include "matrix.h"
extern MatrixData *matrix_read_scan();

static usage() {
	fprintf(stderr,
		"usage: matcopy -i matspec -o matspec [-V version -v] [-s storage_order]]\n");
	fprintf(stderr,"version is either 70 or 6 (default = 70)\n");
	fprintf(stderr,"-s storage_order (0 or 1); valid only for sinograms and attenuations\n");
	fprintf(stderr,"-p set number of plane to shift in scan (default is 0)\n");
	fprintf(stderr,"-v set verbose mode on ( default is off)\n");
	exit(1);
}

static int verbose=0;

static copy_scan(mptr1,matnum, mptr2,o_matnum,storage_order,pl_shift)
MatrixFile *mptr1, *mptr2;
int matnum, o_matnum, storage_order,pl_shift;
{
	MatrixData *matrix, *o_matrix;
	struct MatDir matdir, o_matdir;
	Scan3D_subheader *sh=NULL, *o_sh=NULL;
	Attn_subheader  *ah=NULL, *o_ah=NULL;
	caddr_t blk, sino, planar, dest;
	int keep_order = 1;
	int i, view, plane;
	int blkno, file_pos, view_pos;
	int nblks, line_size, num_views, num_planes;

	if (matrix_find(mptr1, matnum, &matdir) == -1) return 0;
	nblks = matdir.endblk-matdir.strtblk+1;
	if (matrix_find(mptr2, o_matnum, &o_matdir) == -1)  {
		blkno = mat_enter(mptr2->fptr, mptr2->mhptr, o_matnum, nblks) ;
		o_matdir.matnum = o_matnum;
		o_matdir.strtblk = blkno;
		o_matdir.endblk =  blkno + nblks - 1 ;
		insert_mdir(o_matdir, mptr2->dirlist) ;
	}
	matrix = matrix_read(mptr1,matnum,MAT_SUB_HEADER);
	blk = (caddr_t)malloc(MatBLKSIZE);
	switch (mptr1->mhptr->file_type) {
	case Float3dSinogram :
	case Short3dSinogram :
		sh = (Scan3D_subheader*)matrix->shptr;
		o_sh = (Scan3D_subheader*)calloc(2,MatBLKSIZE);
		memcpy(o_sh,sh,sizeof(Scan3D_subheader));
		if (storage_order>=0) o_sh->storage_order = storage_order;
		if (o_sh->storage_order != sh->storage_order) keep_order = 0;
		if (mptr1->mhptr->file_type == Float3dSinogram)
			line_size = sh->num_r_elements*sizeof(float);
		else line_size = sh->num_r_elements*sizeof(short);
		num_views = sh->num_angles;
		num_planes = sh->num_z_elements[0];
		mat_write_Scan3D_subheader(mptr2->fptr,mptr2->mhptr, o_matdir.strtblk,
			o_sh);
		nblks -= 2;
		file_pos = (o_matdir.strtblk+1)*MatBLKSIZE;
		break;
	case AttenCor :
		ah = (Attn_subheader*)matrix->shptr;
		o_ah = (Attn_subheader*)calloc(1,MatBLKSIZE);
		memcpy(o_ah,ah,sizeof(Attn_subheader));
		if (storage_order>=0) o_ah->storage_order = storage_order;
		if (o_ah->storage_order != ah->storage_order) keep_order = 0;
		line_size = ah->num_r_elements*sizeof(float);
		num_views = ah->num_angles;
		num_planes = ah->z_elements[0];
		mat_write_attn_subheader(mptr2->fptr,mptr2->mhptr, o_matdir.strtblk,
			o_ah);
		nblks -= 1;
		file_pos = o_matdir.strtblk*MatBLKSIZE;
		break;
	default:
		return 0;
	}
	if (verbose) fprintf(stderr,"view mode to sino mode\n");
	sino = (caddr_t)malloc(line_size*num_views);
	file_pos = ftell(mptr1->fptr);
	if (pl_shift <0) {
		for (plane=-pl_shift;plane<num_planes;plane++) {
			view_pos = file_pos + num_views*line_size*plane;
			if ((fseek(mptr1->fptr,view_pos,0) == -1) ||
				fread(sino,line_size*num_views,1,mptr1->fptr) != 1) {
					perror(mptr1->fname);
					exit(1);
			}

			if (fwrite(sino,line_size*num_views,1,mptr2->fptr) != 1) {
				perror(mptr2->fname);
					exit(1);
			}
		}
		for (plane=0; plane<-pl_shift;plane++) {
			view_pos = file_pos + num_views*line_size*(num_planes-1);
			if ((fseek(mptr1->fptr,view_pos,0) == -1) ||
				fread(sino,line_size*num_views,1,mptr1->fptr) != 1) {
					perror(mptr1->fname);
					exit(1);
			}

			if (fwrite(sino,line_size*num_views,1,mptr2->fptr) != 1) {
				perror(mptr2->fname);
					exit(1);
			}
		}
	} else {
		for (plane=0; plane<pl_shift;plane++) {
fprintf(stderr,"plane %d-> %d\n",plane,pl_shift);
			view_pos = file_pos + num_views*line_size*pl_shift ;
			if ((fseek(mptr1->fptr,view_pos,0) == -1) ||
				fread(sino,line_size*num_views,1,mptr1->fptr) != 1) {
				perror(mptr1->fname);
				exit(1);
			}
		if (fwrite(sino,line_size,num_views,mptr2->fptr) != num_views) {
			perror(mptr2->fname);
				exit(1);
		}
		}
		for (plane=0;plane<num_planes-pl_shift;plane++) {
fprintf(stderr,"plane %d-> %d\n",plane,plane+pl_shift);
			view_pos = file_pos + num_views*line_size *plane;
			if ((fseek(mptr1->fptr,view_pos,0) == -1) ||
				fread(sino,line_size*num_views,1,mptr1->fptr) != 1) {
				perror(mptr1->fname);
				exit(1);
			}
		if (fwrite(sino,line_size,num_views,mptr2->fptr) != num_views) {
			perror(mptr2->fname);
				exit(1);
		}
		}
	}
	if (mptr2->mhptr->sw_version == V7) mh_update(mptr2);
	return 1;
}
	
main( argc, argv)
  int argc;
  char **argv;
{
	MatrixFile *mptr1, *mptr2;
	MatrixData *matrix, *slice;
	Main_header proto;
	Image_subheader* imagesub;
	MatDirNode *node=NULL;
	char *mk, fname[256];
	int i, j, specs[5];
	char *in_spec=NULL, *out_spec=NULL;
	int c, version=V7, matnum=0, o_matnum=0;
	int plane,  npixels, slice_blks, slice_matnum;
	int elem_size=2, offset = 0;
	int storage_order = -1;
	int pl_shift = 0;
	short *sdata;
	u_char *bdata;
	int *matnums=NULL, nmats=0;
	struct Matval mat;
	extern char *optarg;

	while ((c = getopt (argc, argv, "i:o:V:s:p:v")) != EOF) {
		switch (c) {
		case 'i' :
			in_spec	= optarg;
            break;
		case 'o' :
			out_spec	= optarg;
            break;
		case 'V' :
			sscanf(optarg,"%d",&version);
            break;
		case 's' :
			if (sscanf(optarg,"%d",&storage_order) != 1 ||
			(storage_order!=0 && storage_order!=1)) usage();
            break;
		case 'p' :
			sscanf(optarg,"%d",&pl_shift);
            break;
		case 'v' :
			verbose = 1;
			break;
		}
	}
	
	if (in_spec == NULL || out_spec==NULL) usage();
	for (i=0; i<5; i++) specs[i] = 0;
	strcpy(fname,strtok(in_spec,","));
	mk = strtok(NULL,",");
	for (i=0; i<5; i++) {
		if (mk!=NULL) {
			if (*mk == '*') specs[i] = -1;
			else specs[i] = atoi(mk);
			mk = strtok(NULL,",");
		}
	}
	mptr1 = matrix_open( fname, MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
	if (!mptr1) crash( "%s: can't open file '%s'\n", argv[0], fname);
	if ( mptr1->dirlist->nmats == 0) crash("no matrix in %s\n",fname);
	matnums = (int*)calloc(sizeof(int),mptr1->dirlist->nmats);
	node = mptr1->dirlist->first;
	nmats = 0;
	if (specs[0] == 0) {	/* no matrix specified, use first */ 
		matnums[nmats++] = node->matnum;
	} else {				/* build specified matnums */
		while(node != NULL) {
			matnum = node->matnum;
			node = node->next;
        	mat_numdoc(matnum, &mat);
        	if (specs[0]>=0 && specs[0] != mat.frame) continue;
        	if (specs[1]>=0 && specs[1] != mat.plane) continue;
        	if (specs[2]>=0 && specs[2] != mat.gate) continue;
        	if (specs[3]>=0 && specs[3] != mat.data) continue;
        	if (specs[4]>=0 && specs[4] != mat.bed) continue;
        	matnums[nmats++] = matnum;
    	}
	}
	if (nmats == 0) crash( "%s: matrix not found\n", in_spec);
	matspec( out_spec, fname, &o_matnum);
	memcpy(&proto,mptr1->mhptr,sizeof(Main_header));
	proto.sw_version = version;
	if (version < V7) {
		if (proto.file_type != PetImage && proto.file_type != ByteVolume &&
		proto.file_type != PetVolume && proto.file_type != ByteImage &&
		proto.file_type != InterfileImage)
			crash ("version 6 : only images are supported \n");
		proto.file_type = PetImage;
	} else {
		if (proto.file_type == InterfileImage) {
			matrix = matrix_read( mptr1, matnums[0], MAT_SUB_HEADER);
			if (matrix->data_type == ByteData) proto.file_type = ByteVolume;
			else proto.file_type = PetVolume;
			free_matrix_data(matrix);
		}
	}
	if (proto.sw_version != mptr1->mhptr->sw_version) {
		fprintf(stderr,"converting version %d to version %d\n",
				mptr1->mhptr->sw_version, proto.sw_version); 
	} else {
		fprintf(stderr,"input/output version : %d\n",proto.sw_version);
	}
	mptr2 = matrix_create( fname, MAT_OPEN_EXISTING, &proto);
	if (!mptr2) crash( "%s: can't open file '%s'\n", argv[0], fname);
	
	for (i=0; i<nmats; i++) {
		if (nmats > 1 || o_matnum == 0) o_matnum = matnums[i];
		if (verbose) {
        	mat_numdoc(matnums[i], &mat);
			fprintf(stderr,"input matrix : %s,%d,%d,%d,%d,%d\n",
				mptr1->fname, mat.frame,mat.plane,mat.gate,mat.data,mat.bed);
        	mat_numdoc(o_matnum, &mat);
			fprintf(stderr,"output matrix : %s,%d,%d,%d,%d,%d\n",
				mptr2->fname, mat.frame,mat.plane,mat.gate,mat.data,mat.bed);
		}
		if (mptr1->mhptr->file_type==Short3dSinogram ||
			mptr1->mhptr->file_type==Float3dSinogram ||
			mptr1->mhptr->file_type==AttenCor)
			copy_scan( mptr1,matnums[i], mptr2,o_matnum,storage_order,pl_shift);
		else {
			matrix = matrix_read( mptr1,matnums[i], GENERIC);
			if (matrix != NULL) matrix_write( mptr2, o_matnum, matrix);
		}
	}
	matrix_close( mptr1);
	matrix_close( mptr2);
}
