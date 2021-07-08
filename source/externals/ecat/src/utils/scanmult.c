/* @(#)scanmath.c	1.1 4/26/91 */

/*
#ifndef	lint
static char sccsid[]="@(#)scanmult.c	1.1 4/26/91 Copyright 1990 CTI Pet Systems, Inc.";
#endif	lint
*/
/*
 * may1999 by coppens@topo.ucl.ac.be
 */
   

#include "matrix.h"
#include <math.h>

static usage() 
{
	fprintf(stderr,"usage: scanmult -i scanin -m multfile -o scanout -n num -v\n");
	exit(1);
}

main( argc, argv)
  int argc;
  char **argv;
{
	MatrixFile *file1=NULL, *file3=NULL;
	MatrixData *scan1=NULL, *scan3=NULL;
	MatrixData *slice1=NULL, *slice2=NULL;
	Scan3D_subheader *sh=NULL, *sh3=NULL;
	Main_header mh;
	struct MatDir matdir, dir_entry;
	float *scana, *scanc;
	float valb;
	short int *sdata;
	float *fdata;
	float maxval=0, minval=0, scalef=1;
	int i,j, matnum, blkno, nplblks, nblks;
	char *ptr, fname[256];
	int view, nviews, nprojs, nplanes, npixels;
	int offset;
	int num_fr;
	float *m1, *m2, *mult;

	extern char *optarg;
	char* in_file =0;
	char* out_file =0;
	char* mult_file =0;
	int verbose = 0;
	int im_num=1;
	FILE* mult_fp;
	int c;
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

	file1 = matrix_open( in_file, MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
	if (!file1)
	  crash( "scanmult: can't open file '%s'\n", in_file);

	matnum = mat_numcod(1,1,1,0,0);
	scan1 = matrix_read(file1,matnum,MAT_SUB_HEADER);
	if (!scan1) crash( "scanmults: scan %s,1,1,1,0,0 not found\n", in_file);
	if (file1->mhptr->file_type != Short3dSinogram && file1->mhptr->file_type != Float3dSinogram)
			crash("%s is not a Sinogram nor an Attenuation\n", in_file);
	matrix_find(file1, matnum, &matdir);

	sh = (Scan3D_subheader*)scan1->shptr;
	nviews = sh->num_angles;
	nprojs = sh->num_r_elements;
	nplanes = sh->num_z_elements[0];
	npixels = nprojs*nplanes;

/* get scanc specification and write header */
	memcpy(&mh, file1->mhptr, sizeof(Main_header));
	num_fr = mh.num_frames;
	mh.num_frames = 1;
	mh.num_planes = nplanes;
	mh.file_type = Float3dSinogram;
	file3 = matrix_create( out_file, MAT_OPEN_EXISTING, &mh);
	if (!file3) crash( "scanmult: can't open file '%s'\n", out_file);
	scan3 = (MatrixData*)calloc(1, sizeof(MatrixData));
	memcpy(scan3, scan1, sizeof(MatrixData));
	sh3 = (Scan3D_subheader*)calloc(2,512);
	memcpy(sh3,sh,sizeof(Scan3D_subheader));
	scan3->shptr= (caddr_t)sh3;
	scan3->data_type = sh3->data_type = IeeeFloat;
	nblks = (npixels*nviews*sizeof(float)+511)/512;
	nplblks = (npixels*sizeof(float)+511)/512;
	matnum = mat_numcod(1,1,1,0,0);
	if (matrix_find(file3, matnum, &matdir) == -1) {
        blkno = mat_enter(file3->fptr, file3->mhptr, matnum, nblks) ;
        dir_entry.matnum = matnum ;
        dir_entry.strtblk = blkno ;
        dir_entry.endblk = dir_entry.strtblk + nblks - 1 ;
        dir_entry.matstat = 1 ;
        insert_mdir(dir_entry, file3->dirlist) ;
        matdir = dir_entry ;
    } else {
        fprintf(stderr,"\7warning : existing matrix overwritten\n");
        blkno = matdir.strtblk;
    }
    mat_write_Scan3D_subheader(file3->fptr, file3->mhptr, matdir.strtblk, sh3);
	if (fseek(file3->fptr,0,SEEK_END) != 0)
			crash ("%s : error positioning eof\n",file3->fname);

	scan3->data_size = nblks*512;
	scan3->scale_factor = sh->scale_factor  = 1;
	scanc = (float*)calloc(nplblks,512);
	m1 = (float*)malloc(num_fr*sizeof(float));
	m2 = (float*)malloc(num_fr*sizeof(float));

/* Decode multiplicative factors */
	if ((mult_fp = fopen(mult_file,"r")) == NULL) {
		crash("scanmult: can't open %s\n",mult_file);
	}
	for (i=0;i<num_fr; i++)
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
		for (j = 0; j<num_fr; j++)
			fprintf(stderr,"multiplicative factors %d: %g\n",j,mult[j]);
/* compute and write data */
	for (view=0; view<nviews; view++) {
		if (verbose) 
			fprintf(stderr,"processing view %d\n",view+1);
		memset(scanc,0,nplblks*512);
		for (j = 0; j<num_fr; j++) {
			matnum = mat_numcod(j+1,1,1,0,0);
			scan1 = matrix_read(file1,matnum,MAT_SUB_HEADER);
			if (!scan1) 
				crash("scanmults: scan %s,%d,1,1 not found\n",in_file,j+1);
/*
			else fprintf(stderr," scan %s,%d,1,1  found\n",in_file,j+1);
*/

			slice1 = matrix_read_view(file1,scan1,view,0);
			if (slice1->data_type == IeeeFloat) {
				scana = (float*)slice1->data_ptr;
				for (i=0; i<npixels; i++)
					scanc[i] = scanc[i] + scana[i]*mult[j];
			} else {
				sdata = (short*)slice1->data_ptr;
				for (i=0; i<npixels; i++) {
					scanc[i] += sdata[i]*mult[j]*slice1->scale_factor;
				}
			}
			free_matrix_data(slice1);
	
		}
		file_data_to_host(scanc,nplblks,IeeeFloat);

		if (fwrite(scanc,sizeof(float),npixels,file3->fptr) != npixels)
			crash("%s view %d : error writing view\n",file3->fname,view+1);
	}

/* close all */
	matrix_close(file1);
	matrix_close(file3);
}
