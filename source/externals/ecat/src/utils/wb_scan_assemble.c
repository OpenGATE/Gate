/*
 Modification History
 8-sep-99 : Add more tolerance for bed shifts variation (plane separation)
            Use the most frequent bed shift.
 31-jan-00: Time normalization to 1 minute when bed duration is variable
 */
#include "matrix.h"
#include "ecat_model.h"
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#define MAX_BEDS 20
#define host_data_to_file file_data_to_host

typedef struct _BedPosition {
	int bed;
	time_t  duration;
	float position;
	MatrixData* matrix;
} BedPosition;

typedef struct _MergeFactor {
  float w1, w2;
} MergeFactor;

typedef struct _BedShift {
	double value; /* in mm */
	int count;
} BedShift;
static BedShift *bed_shifts;
static int num_bed_shifts = 0;

static int t_normalize=0;

static MatrixFile *file_in=NULL;
static MatrixFile *file_out=NULL;
static char in_name[128];
static char norm_name[128];
static char tmp_file[128];
static int verbose = 0;
static Main_header mhead;
static Scan3D_subheader *scan_sh;
static Scan_subheader *scan6_sh;
static Scan3D_subheader out_ssh;

static usage() {
	fprintf(stderr,"usage : wb_scan_assemble -i multi_bed_file -o wb_scan_file [-p bed_pos_file] [-m angular_mash]  [-r distance_rebin] [-v -x bi]\n");
	fprintf(stderr,"-m angular_mash (default = 1) \n");
	fprintf(stderr,"-r radial compression (default = 1) \n");
	fprintf(stderr,"The output file is overwritten if exists\n");
  	fprintf(stderr,"Version : 6 February 2003\n");
	exit(1);
}

/* add_bed_shift adds value and/or update repeat count,
   and sorts occurences vs repeat count.
*/

static void add_bed_shift(value)
double value;
{
	BedShift tmp;
	int i, found=0, sorted=0;
	for (i=0; i<num_bed_shifts && !found; i++)
		if (bed_shifts[i].value == value)
		{
    		bed_shifts[i].count++;
			found++;
		}
	if (!found)
	{
		bed_shifts[num_bed_shifts].value = value;
		bed_shifts[num_bed_shifts++].count = 1;
	}
	while (!sorted)	/* bubble sort */
	{
		sorted = 1;
		for (i=1; i<num_bed_shifts; i++)
		{
			if (bed_shifts[i].count > bed_shifts[i-1].count)
			{	/* swap and unset sorted flag */
				tmp = bed_shifts[i-1];
				bed_shifts[i-1] = bed_shifts[i];
				bed_shifts[i] = tmp;
				sorted = 0;
			}
		}
	}
}

static matrix_float(matrix)
MatrixData *matrix;
{
	float scalef, *fdata;
	short *sdata;
	int i, nblks, np = matrix->xdim*matrix->ydim;

	matrix->data_type = IeeeFloat;
	nblks = (np*sizeof(float)+511)/512;
	fdata = (float*)calloc(nblks,512);
	sdata = (short*)matrix->data_ptr;
	scalef = matrix->scale_factor;
	matrix->data_ptr = (caddr_t)fdata;
	for (i=0; i<np;i++) fdata[i] = scalef * sdata[i];
	matrix->scale_factor = 1.0;
	free(sdata);
}

static matrix_rebin(matrix, x_rebin, y_rebin)
MatrixData *matrix;
int x_rebin, y_rebin;
{
	int   i, j, k=0;
	float *scan_in, *scan_out;
	float *fp_in, *fp_out;
	float scf=1./(x_rebin*x_rebin*y_rebin);
	int nblks, nprojs = matrix->xdim, nviews = matrix->ydim;
	
	if (x_rebin<=1 && y_rebin<=1) return;
	nblks = (nprojs*nviews*sizeof(float)+511)/512;
	scan_out = (float*)calloc(nblks, 512);
/* integer x rebin */
	scan_in = (float*)matrix->data_ptr;
	fp_in = scan_in; fp_out = scan_out;
	if (x_rebin > 1 ) {
		for (i=0; i<nprojs*nviews; i += x_rebin, fp_out++) {
			for (j=0; j<x_rebin; j++) *fp_out += *fp_in++;
		}
		matrix->xdim /= x_rebin;
		nprojs = matrix->xdim;
	}
	
/* integer y rebin */
	if (y_rebin > 1 ) {
		if (x_rebin > 1 ) {						/* swap in and out only if x_rebin */
		    memcpy(scan_in,scan_out,nprojs*nviews*sizeof(float));
		    memset(scan_out,0,nprojs*nviews*sizeof(float));  		/* set to 0.0 */
		}
		fp_in = scan_in; fp_out = scan_out;
		for (i=0; i<nviews; i++) {
			fp_out = scan_out + nprojs*(i/y_rebin);
			for (j=0; j<nprojs; j++) fp_out[j] += *fp_in++;
		}
		matrix->ydim /= y_rebin;
		nviews = matrix->ydim;
	}
	free(scan_in);
	for (i=0; i<nprojs*nviews; i++) scan_out[i] *= scf;

	matrix->data_ptr = (caddr_t)scan_out;
	matrix->pixel_size *= x_rebin;
	matrix->y_size *= y_rebin;
}


#ifdef __STDC__
wb_scan_copy(int pl_out, BedPosition *bed, int pl_in, int pl_in_end,
int mash, int rebin)
#else
wb_scan_copy(pl_out,bed,pl_in, pl_in_end, mash,rebin)
int pl_out, pl_in, pl_in_end, mash, rebin;
BedPosition *bed;
#endif
{
	int i, j, count;
	int npixels, nblks;
	struct MatDir *matdir;
	MatrixData *slice;
	float *fdata;
	int num_projs, num_views;
/*	time_t duration;	bed duration in millisec */
	float tw;

/*	duration = ((Scan3D_subheader*)bed->matrix->shptr)->frame_duration; */
	tw = 60000.0/bed->duration;
	num_projs = bed->matrix->xdim/rebin;
	num_views = bed->matrix->ydim/mash;
	npixels = num_projs*num_views;
	nblks = (npixels*sizeof(float)+511)/512;
	for (i=pl_in; i<=pl_in_end; i++) {
		slice = matrix_read_slice(file_in,bed->matrix,i-1,0);
		if (slice->data_type != IeeeFloat) matrix_float(slice);
		matrix_rebin(slice,rebin,mash);
		fdata = (float*)slice->data_ptr;
		if (t_normalize)
			for (j=0; j<npixels; j++) fdata[j] *= tw;
		host_data_to_file(fdata,nblks,IeeeFloat);
		if ( (count=fwrite(fdata,sizeof(float),npixels,file_out->fptr)) !=
			npixels) crash("%s : fwrite error\n",file_out->fname);
		free_matrix_data(slice);
	}
	if (verbose) printf("copy %s,1,[%d-%d],1,0,%d	===> %s,1,[%d-%d],1 done\n",
		file_in->fname, pl_in, pl_in_end, bed->bed, file_out->fname,
		pl_out, pl_out+pl_in_end - pl_in);
}

#ifdef __osf__
wb_scan_merge(int pl_out, BedPosition *bed_1, int pl1, float w1,
BedPosition *bed_2, int pl2, float w2, int mash, int rebin)
#else
wb_scan_merge(pl_out,bed_1,pl1,w1,bed_2,pl2,w2,mash, rebin)
BedPosition *bed_1, *bed_2;
int pl_out, pl2,mash, rebin;
float w1, w2;
#endif
{
	int i, nblks, count;
	float  *fdata1, *fdata2, *fdata;
	int num_projs, num_views, npixels;
	MatrixData  *slice1, *slice2;
	float tw1=w1, tw2=w2;
/*	time_t duration1, duration2;

	duration1 = ((Scan3D_subheader*)bed_1->matrix->shptr)->frame_duration;
	duration2 = ((Scan3D_subheader*)bed_2->matrix->shptr)->frame_duration;
*/
	if (t_normalize) {
		tw1 *= 60000.0/bed_1->duration;
		tw2 *= 60000.0/bed_2->duration;
	}
	num_projs = bed_1->matrix->xdim/rebin;
	num_views = bed_1->matrix->ydim/mash;
	npixels = num_projs*num_views;
	nblks = (npixels*sizeof(float)+511)/512;
	fdata = (float*)calloc(nblks,512);
	slice1 = matrix_read_slice(file_in,bed_1->matrix,pl1-1,0);
	if (slice1->data_type != IeeeFloat) matrix_float(slice1);
	slice2 = matrix_read_slice(file_in,bed_2->matrix,pl2-1,0);
	if (slice2->data_type != IeeeFloat) matrix_float(slice2);
	matrix_rebin(slice1,rebin,mash);
	matrix_rebin(slice2,rebin,mash);
	fdata1 = (float*)slice1->data_ptr;
	fdata2 = (float*)slice2->data_ptr;
	for (i=0; i<npixels; i++)  {
		fdata[i] = tw1*fdata1[i]+tw2*fdata2[i];
	}
	host_data_to_file(fdata,nblks,IeeeFloat);
	if (fwrite(fdata, sizeof(float),npixels,file_out->fptr) != npixels) 
		crash("%s : fwrite error\n",file_in->fname);

	if (verbose) printf("merge %g * 1,%d,1,0,%d + %g * 1,%d,1,0,%d	===> %s,1,%d,1 done\n",
		w1,pl1,bed_1->bed,w2,pl2,bed_2->bed,file_out->fname,pl_out);
	free(fdata);
	free_matrix_data(slice1);
	free_matrix_data(slice2);

}

int main(argc,argv)
int argc;
char **argv;
{
	BedPosition tmp, *bed_positions = NULL, *bed_1, *bed_2;
	Main_header *mh;
	int i=0, j=0, count=0, sorted=0;
	int plane = 1, mash=1, rebin=1;
	float bed_pos0, *bed_offsets;
	char *x_beds;
	int matnum, blkno, nblks, data_size;
	char out_name[128], bed_pos_file[128];
	struct MatDir matdir, dir_entry;
	FILE *fp;
	int norm_flag = 0;
	int numplane, overplane, lastplane, span = 0, maxlors;
    	double  bed_shift, plane_sep;
	MergeFactor *mf;
    	EcatModel *model;
	time_t duration=0, all_duration=0;
 
	fprintf(stderr,"%s : Version 31-Jan-2000\n", argv[0]);
	in_name[0] = '\0'; out_name[0] = '\0'; norm_name[0] = '\0';
	bed_pos_file[0] = '\0';
	x_beds = calloc(MAX_BEDS,1);
	while ((i = getopt (argc, argv, "i:o:x:p:m:r:v")) != EOF) {
	switch(i) {
	case 'v' : verbose = 1;
		break;
	case 'i' :	/* input file */
			strncpy(in_name,optarg,127); in_name[127] = '\0';
			break;
	 case 'o' :		/* out file */
			strncpy(out_name,optarg,127); out_name[127] = '\0';
			break;
	case 'p' :		/* bed_pos file */
			strncpy(bed_pos_file,optarg,127); bed_pos_file[127] = '\0';
			break;
	case 'm':
		sscanf(optarg,"%d",&mash);
		break;
	case 'r':
		sscanf(optarg,"%d",&rebin);
		break;
	case 'x':
		j = atoi(optarg);
		if (j<MAX_BEDS) x_beds[j] = '1';
		break;
	case '?' : usage();
			break;
	 }
	}
	if (!strlen(in_name) || !strlen(out_name)) usage();
	if (strcmp(in_name,out_name) == 0) {
		printf("input and output should differ\n");
		exit(1);
	}
	if (verbose) printf("input file : %s\noutputfile : %s\n",in_name,out_name);
	file_in = matrix_open(in_name, MAT_READ_ONLY, GENERIC);
	if (file_in == NULL){
		matrix_perror(in_name);
		exit(1);
	}
	count =	file_in->mhptr->num_bed_pos+1;
	if (count < 2) 
		fprintf(stderr,"warning: %s is not a multi bed position file\n");
	bed_positions = (BedPosition*)calloc(count,sizeof(BedPosition));
	bed_positions = (BedPosition*)calloc(count,sizeof(BedPosition));
	bed_pos0 = file_in->mhptr->init_bed_position;
	bed_offsets = file_in->mhptr->bed_offset;
	numplane =  file_in->mhptr->num_planes;
    	plane_sep =  file_in->mhptr->plane_separation;
	if (bed_pos_file[0] && (fp=fopen(bed_pos_file,"r"))!= NULL) {
		for (i=0;i<count; i++)
			if (fscanf(fp,"%g",bed_offsets+i) != 1) break;
		fclose(fp);
	}
	for (i=0, j=0; i<count; i++) {
		if (x_beds[i]) continue;
		matnum = mat_numcod(1,1,1,0,i);
		bed_positions[j].matrix = matrix_read(file_in,matnum, MAT_SUB_HEADER);
		if (bed_positions[j].matrix == NULL) {
			fprintf(stderr,"matrix 1,1,1,0,%d not found\n",i);
			exit(1);
		}
		if (file_in->mhptr->file_type != Sinogram) {
			scan_sh = (Scan3D_subheader*)bed_positions[j].matrix->shptr;
			duration = scan_sh->frame_duration;
		}
		else {
			scan6_sh = (Scan_subheader*)bed_positions[j].matrix->shptr;
			duration = scan6_sh->frame_duration;
		}
		if (duration == 0) {
			fprintf(stderr,"matrix 1,1,1,0,%d : bed duration not set\n",i);
			exit(1);
		}
		if (j==0) all_duration = duration;
		if (all_duration != duration) t_normalize = 1;
		bed_positions[j].bed = i;
		bed_positions[j].duration = duration;
		if (i>0) bed_positions[j].position = bed_pos0+bed_offsets[i-1];
		else  bed_positions[j].position = bed_pos0;
		j++;
	 }
	count = j;

	/* buble sort */
	while (!sorted) {
		sorted = 1;
		for (i=1; i<count; i++) {
			if (bed_positions[i].position < bed_positions[i-1].position) {
				sorted = 0;
				tmp = bed_positions[i];
				bed_positions[i] = bed_positions[i-1];
				bed_positions[i-1] = tmp;
			}
		}
	}

    /* exit when bed shift is variable */
	bed_shifts = (BedShift*)calloc(count, sizeof(BedShift));
    bed_shift = bed_positions[1].position- bed_positions[0].position;
    bed_shift = rint(10*bed_shift); /* in mm */
	add_bed_shift(bed_shift);
    for (i=2; i<count; i++)
      {
        bed_shift = bed_positions[i].position-bed_positions[i-1].position;
        bed_shift = rint(10*bed_shift); /* in mm */
	    add_bed_shift(bed_shift);
      }
	bed_shift = bed_shifts[0].value;
	for (i=1; i<num_bed_shifts; i++)
      {
        if (fabs(bed_shift-bed_shifts[i].value) > 10*plane_sep)	/* 10*plane_sep mm */
          {
        fprintf(stderr,
            "%s : bed shift variation greater then plane separation (%g,%g) cm => aborted\n",
            argv[0],0.1*bed_shift, 0.1*bed_shifts[i].value);
        return 1;
          }
		else
          {
			if (bed_shift != bed_shifts[i].value)
              fprintf(stderr, "%s warning : bed shift variation (%g,%g) cm\n",
                argv[0],0.1*bed_shift, 0.1*bed_shifts[i].value);
          }
      }
    bed_shift /= 10;
	if (num_bed_shifts > 1)
		fprintf(stderr, "%s using %g cm bed shift\n", argv[0], bed_shift);

    if ((model = ecat_model(file_in->mhptr->system_type)) == NULL)
      return 1;
	/* don't trust main header num_planes */
	if( file_in->mhptr->file_type != Sinogram ) numplane = bed_positions[0].matrix->zdim;
    if (numplane != (model->dirPlanes*2-1))
	{
      fprintf(stderr,
		"MainHeader numplane %d disagree with scanner model %d \n",numplane, (model->dirPlanes*2-1));
	  return 1;
	}
    if (span == 0) span = model->def2DSpan;
    overplane = (int)(.5 +(numplane*plane_sep - bed_shift)/plane_sep);
	lastplane = numplane-overplane;
    mf = (MergeFactor*)calloc(overplane, sizeof(MergeFactor));
    maxlors = (span+1)/2;
    for (i=0; i<maxlors && i<overplane; i++)
      {
        mf[i].w2 = i+1;
        mf[overplane-i-1].w1 = i+1;
      }
    while (i<overplane)
      {
        mf[i].w2 = maxlors-1;
        mf[overplane-i-1].w1 = maxlors-1;
	i++;
        if (i<overplane)
          {
        mf[i].w2 = maxlors;
        mf[overplane-i-1].w1 = maxlors;
          }
	i++;
      }
    fprintf(stderr,"merge factors :");
    for (i=0; i<overplane; i++)
      fprintf(stderr," (%g,%g)", mf[i].w1, mf[i].w2);
    fprintf(stderr,"\n");
    for (i=0; i<overplane; i++)
    {
        mf[i].w1 =  mf[i].w1 / (mf[i].w1+ mf[i].w2);
        mf[i].w2 = 1.0 - mf[i].w1;
        fprintf(stderr," (%g,%g)", mf[i].w1, mf[i].w2);
      }
        fprintf(stderr,"\n");

	if (verbose) for (i=0; i<count; i++) 
		printf("%d : %g\n",bed_positions[i].bed, bed_positions[i].position);

	memcpy(&mhead,file_in->mhptr,sizeof(Main_header));
	if (mhead.sw_version < V7) mhead.sw_version = V7;
	mhead.num_planes = lastplane*count+overplane;
	mhead.file_type = Float3dSinogram;
	for (i=0;i<mhead.num_bed_pos;i++) mhead.bed_offset[i] = 0.0;
	mhead.num_bed_pos = 0;
	mhead.init_bed_position = bed_positions[0].position;
	file_out = matrix_create(out_name,MAT_CREATE_NEW_FILE, &mhead);
	if (!file_out) matrix_perror(out_name);

	bed_1 = bed_positions;
	if (file_in->mhptr->file_type != Sinogram) {
		scan_sh = (Scan3D_subheader*)bed_1->matrix->shptr;

		memcpy(&out_ssh,scan_sh,sizeof(Scan3D_subheader));
		out_ssh.num_angles = scan_sh->num_angles/mash;
		out_ssh.v_resolution = scan_sh->v_resolution*mash;
		out_ssh.num_r_elements = scan_sh->num_r_elements/rebin;
		out_ssh.x_resolution = scan_sh->x_resolution*rebin;
	} else {
		scan6_sh = (Scan_subheader*)bed_1->matrix->shptr;

		memset(&out_ssh,0,sizeof(Scan3D_subheader));
		out_ssh.num_angles = scan6_sh->num_angles/mash;
		out_ssh.v_resolution = scan6_sh->y_resolution*mash;
		out_ssh.num_r_elements = scan6_sh->num_r_elements/rebin;
		out_ssh.x_resolution = scan6_sh->x_resolution*rebin;
		/* complete subheader */
	 	out_ssh.corrections_applied = scan6_sh->corrections_applied;
	 	out_ssh.num_dimensions = scan6_sh->num_dimensions;
		out_ssh.axial_compression = model->def2DSpan;
		out_ssh.ring_difference = (model->def2DSpan + 1)/2;
		out_ssh.frame_start_time = scan6_sh->frame_start_time;
		out_ssh.frame_duration = scan6_sh->frame_duration;
		out_ssh.loss_correction_fctr = scan6_sh->loss_correction_fctr;
	}
		
	out_ssh.data_type = IeeeFloat; 
	out_ssh.storage_order = 1; 			/* force sino mode for easy handling */
	out_ssh.num_z_elements[0] = mhead.num_planes;
	out_ssh.scale_factor = 1.;
	if (t_normalize) {
		printf("\7Warning : variable bed duration is normalized to 1 min\n");
		out_ssh.frame_duration = 60000;
	}
	data_size = mhead.num_planes*out_ssh.num_angles*out_ssh.num_r_elements*
		sizeof(float);
	nblks = (data_size+511)/MatBLKSIZE;
	nblks += 1;                      /* sino3D subheader has one more block */
	matnum = mat_numcod(1,1,1,0,0);
	if (matrix_find(file_out, matnum, &matdir) == -1) {
		blkno = mat_enter(file_out->fptr,file_out->mhptr, matnum, nblks) ;
		dir_entry.matnum = matnum ;
		dir_entry.strtblk = blkno ;
		dir_entry.endblk = dir_entry.strtblk + nblks - 1 ;
		dir_entry.matstat = 1 ;
		insert_mdir(dir_entry, file_out->dirlist) ;
		matdir = dir_entry ;
	}
	mat_write_Scan3D_subheader(file_out->fptr,file_out->mhptr, matdir.strtblk,&out_ssh);
	wb_scan_copy(plane,bed_1,1,lastplane,mash,rebin);
	plane += lastplane; 

	for (i=1; i<count; i++)
	{
		bed_1 = &bed_positions[i-1]; bed_2 = &bed_positions[i];
/* merge bed_1 last overplane slices with bed_2 first overplane slices */
		for (j=0;j<overplane;j++)
			wb_scan_merge(plane++, bed_1, lastplane+j+1, mf[j].w1,
				bed_2, j+1,mf[j].w2, mash, rebin);
/*
 * copy the bed_2 middle slices
 */
		wb_scan_copy(plane,bed_2,overplane+1,lastplane, mash,rebin);
		plane += lastplane-overplane;
		free_matrix_data(bed_1->matrix);
		bed_1->matrix = 0;
	}

	bed_1 = &bed_positions[count-1];
	wb_scan_copy(plane,bed_1,lastplane+1,numplane,mash,rebin);
	plane += overplane;
	free_matrix_data(bed_1->matrix);
	matrix_close(file_in);
	if (norm_flag) unlink(tmp_file);
	matrix_close(file_out);
}
