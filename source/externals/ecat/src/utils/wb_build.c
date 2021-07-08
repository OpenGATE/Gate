/* sccsid = "%W%  UCL-TOPO  %E%" */
#include "matrix.h"
#include "ecat_model.h"
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
/*
#include "compute_decay.h"
suppose that images are decay corrected
*/
/* Modification History
 * version = "1.1 05-oct-2000";
 * 01-dec-2000 : v1.1 -> v1.2
 *			output_mhead.scan_start_time = ref_time
 *          imh->frame_start_time = 0;
 * 12-dec-2001 : add -f and -v flags
*/

static char *version = "1.21 12-dec-2001";

typedef struct _BedPosition
{
  int bed;
  float position;
  struct MatDir matdir;
  MatrixData* matrix;
} BedPosition;

typedef struct _MergeFactor
{
  float w1, w2;
} MergeFactor;


static time_t ref_time=0;		/* reference min scan start time */
static MatrixData *volume1, *volume2;
static int verbose = 0;
static int overwrite = 0;
static int debug = 0;
typedef struct _FilePosition {
	char *filename;
	float position;
} FilePosition;

static usage() {
	fprintf(stderr,"wb_build version %s\n", version);
	fprintf(stderr,"usage : wb_build [-f] [-v] file1 ... filen wb_file\n");
	fprintf(stderr,"\tfile1 ... filen must be image or scan stored as short integers\n");
	fprintf(stderr,"\teg : wb_build p01572fd*.img p01572fdg.img to assemble\n");
	fprintf(stderr,"\tp01572fd1.img ... p01572fd11.img into p01572fdg.img\n");
	fprintf(stderr,"\tinterframe decay correction is always applied except when\n");
	fprintf(stderr,"\t\tWB_BUILD_NODECAY environment variable is set to 1\n");
	fprintf(stderr,"\t-f sets overwrite flag\n");
	fprintf(stderr,"\t-v sets verbose flag\n");
	fprintf(stderr,"\tThe program does nothing if the output file exists and overwrite flag not set\n");
	exit(1);
}

#ifdef __STDC__
static float file_decay(MatrixFile* file)
#else
static float file_decay(file)
MatrixFile* file;
#endif
{
	float decay_factor=1.0, lamda, halflife, t;

	t = file->mhptr->scan_start_time - ref_time;
	halflife = file->mhptr->isotope_halflife;
	if (halflife > 0) {
		lamda = log(2.0)/halflife;
		decay_factor = exp(lamda*t);
	 }
	if (verbose) fprintf (stderr, "halflife, t times = %g,%g\n",halflife,t);
	if (verbose) fprintf (stderr, "Decay Correction = %g\n", decay_factor);
	return decay_factor;
}

static wb_copy(file_out,pl_out,file_in,pl_in,decay)
MatrixFile *file_in, *file_out;
int pl_out, pl_in;
float decay;
{
	Scan_subheader *sh;
	Image_subheader* imh;
	MatrixData *matrix = matrix_read_slice(file_in, volume1, pl_in-1, GENERIC);
	if (!matrix) matrix_perror(file_in->fname);
	matrix->scale_factor *= decay;
	matrix->data_max *= decay;
	if (file_out->mhptr->file_type == PetImage) {
		imh = (Image_subheader*)matrix->shptr;
		imh->decay_corr_fctr = decay;
		imh->scale_factor *= decay;
		imh->frame_start_time = 0;
	} else {
		sh = (Scan_subheader*)matrix->shptr;
		sh->frame_start_time = 0;
		sh->scale_factor *= decay;
	}
	matrix_write( file_out, mat_numcod(1,pl_out,1,0,0), matrix);
	if (debug) printf("copy %s,1,%d,1 (decay corrected : %g)  ===> %s,1,%d,1 done\n",
		file_in->fname, pl_in, decay, file_out->fname,pl_out);
	free_matrix_data(matrix);
}

static wb_merge( file_out, pl_out, file_in1,pl1,w1,file_in2,pl2,w2)
MatrixFile *file_in1, *file_in2, *file_out;
int pl_out, pl1,pl2;
float w1, w2;
{
	MatrixData *m1, *m2;
	Image_subheader *imh;
	Scan_subheader *sh;
	
	float *f_data, f_min, f_max, abs_max;
	float scale1,scale2, scale;
	short *p1, *p2;
	int i=0, size=0;

	m1 = matrix_read_slice( file_in1, volume1,pl1-1, GENERIC);
	if (!m1) matrix_perror(file_in1->fname);
	m2 = matrix_read_slice( file_in2, volume2, pl2-1, GENERIC);
	if (!m2) matrix_perror(file_in2->fname);
	size = m1->xdim*m1->ydim;
	f_data = (float*)calloc(size,sizeof(float));
	p1 = (short*)m1->data_ptr; p2 = (short*)m2->data_ptr;
	scale1 = m1->scale_factor; scale2 = m2->scale_factor;
	f_min = f_max = f_data[0] = (w1*scale1*p1[0])+(w2*scale2*p2[0]);
	for (i=1; i<size; i++) {
		f_data[i] = (w1*scale1*p1[i])+(w2*scale2*p2[i]);
		if (f_max < f_data[i]) f_max = f_data[i];
		if (f_min > f_data[i]) f_min = f_data[i];
	}
	abs_max = f_max;
	if (fabs(f_min) > f_max) abs_max = fabs(f_min); 
	scale =  32767.0/abs_max;
	if (debug)
		printf("merge %g * %s,1,%d,1 + %g * %s,1,%d,1  ===> %s,1,%d,1 done\n",
		w1,file_in1->fname,pl1,w2,file_in2->fname,pl2,file_out->fname,pl_out);
	if (debug) printf("data_max : %g,%g ===> %g\n",m1->data_max,m2->data_max,f_max);
	if (debug) printf("scale factors : %g,%g ===> %g\n",scale1,scale2,1/scale);
	m1->data_max = f_max;
	m1->data_min = f_min;
	for (i=0; i<size; i++) p1[i] = (short)(f_data[i]*scale);
	m1->scale_factor = abs_max/32767;
	if (file_out->mhptr->file_type == PetImage) {
		imh = (Image_subheader*)m1->shptr;
		imh->scale_factor = m1->scale_factor;
		imh->frame_start_time = 0;
		imh->image_max = m1->data_max/imh->scale_factor;
		imh->image_min = m1->data_min/imh->scale_factor;
	} else {
		sh = (Scan_subheader*)m1->shptr;
		sh->scale_factor = m1->scale_factor;
		sh->frame_start_time = 0;
	}
	matrix_write( file_out, mat_numcod(1,pl_out,1,0,0), m1);
	free_matrix_data(m1); free_matrix_data(m2);
	free(f_data);
}

int main(argc,argv)
int argc;
char **argv;
{
	FilePosition tmp, *file_positions = NULL;
	MatrixFile *file_in1, *file_in2, *file_out;
	Main_header mhead, *mh;
	int i=0, j=0, count=0, sorted=0;
	int plane = 1, numplanes, overplanes;
	int span = 0, maxlors;
	double   bed_shift, plane_sep;
	MergeFactor *mf;
	EcatModel *model;
	time_t file_time=0;		/* min scan start time */
	int apply_decay = 1;
	char* s;
	float decay_1=1.0, decay_2=1.0;
	struct tm *tm;
	
	if ((s=getenv("WB_BUILD_NODECAY")) != 0 && strcmp(s,"0")>0) apply_decay=0;
	if (verbose) {
		if (s!=NULL) printf("WB_BUILD_NODECAY=%s\n",s);
		if (apply_decay) fprintf(stderr,"interframe decay applied\n");
		else fprintf(stderr,"interframe decay IS NOT applied\n");
	}
	if (argc < 3) usage();
	file_positions = (FilePosition*)calloc(sizeof(FilePosition),count);
	for (i=1; i<(argc-1); i++) 
	{
		if (argv[i][0] != '-') file_positions[count++].filename = argv[i];
		else {
			if (strcmp(argv[i], "-o") == 0) overwrite = 1;
			else if (strcmp(argv[i], "-v") == 0) verbose = 0;
			else usage();
		}
	}
	for (i=0; i<count; i++) {
		file_in1 = matrix_open(file_positions[i].filename,
			MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
		if (file_in1 == NULL) matrix_perror(file_positions[i].filename);
		mh = file_in1->mhptr;
		file_time = mh->scan_start_time;
		if (i==0) ref_time = file_time;
		else if (ref_time > file_time) ref_time = file_time;
		file_positions[i].position = mh->init_bed_position;
		matrix_close(file_in1);
	}
	if (verbose) {
		tm = localtime(&ref_time);
		fprintf(stderr,"min scan start time = %d:%d:%d\n",
			tm->tm_hour, tm->tm_min, tm->tm_sec);
	}
/* buble sort */
	while (!sorted) {
		sorted = 1;
		for (i=1; i<count; i++) {
			if (file_positions[i].position < file_positions[i-1].position) {
				sorted = 0;
				tmp = file_positions[i];
				file_positions[i] = file_positions[i-1];
				file_positions[i-1] = tmp;
			}
		}
	}

	if (verbose) for (i=0; i<count; i++) 
		printf("%s : %g\n",file_positions[i].filename,
			file_positions[i].position);

	if (access(argv[argc-1],F_OK) == 0) {
		if (!overwrite) {
			fprintf(stderr, "%s already exists\7\n",argv[argc-1]);
			usage();
		} else unlink(argv[argc-1]);
	}
	file_in1 = matrix_open(file_positions[count-1].filename, MAT_READ_ONLY,
		MAT_UNKNOWN_FTYPE);
	numplanes = file_in1->mhptr->num_planes;
	plane_sep =  file_in1->mhptr->plane_separation;
	bed_shift = file_positions[1].position-file_positions[0].position;
	if ((model = ecat_model(file_in1->mhptr->system_type)) == NULL)
		return 1;
	if (numplanes != (model->dirPlanes*2-1))
	{
		fprintf(stderr,
			"MainHeader numplanes disagree with scanner model\n");
		return 1;
	}
	if (span == 0) span = model->def2DSpan;
	overplanes = (int)(.5 +(numplanes*plane_sep - bed_shift)/plane_sep);
	mf = (MergeFactor*)calloc(overplanes, sizeof(MergeFactor));
	maxlors = (span+1)/2;
	for (i=0; i<maxlors && i<overplanes; i++)
	{
		mf[i].w2 = i+1;
		mf[overplanes-i-1].w1 = i+1;
	}
	while (i<overplanes)
	{
		mf[i].w2 = maxlors-1;
		mf[overplanes-i-1].w1 = maxlors-1;
		i++;
		if (i < overplanes)
		{
			mf[i].w2 = maxlors;
			mf[overplanes-i-1].w1 = maxlors;
		}
		i++;
	}
	if (verbose) {
		fprintf(stderr,"merge factors :");
		for (i=0; i<overplanes; i++)
			fprintf(stderr," (%g,%g)", mf[i].w1, mf[i].w2);
		fprintf(stderr,"\n");
	}
	for (i=0; i<overplanes; i++)
	{
		mf[i].w1 =  mf[i].w1 / (mf[i].w1+ mf[i].w2);
		mf[i].w2 = 1.0 - mf[i].w1;
		if (verbose) fprintf(stderr," (%g,%g)", mf[i].w1, mf[i].w2);
	}
	if (verbose) fprintf(stderr,"\n");
	if (verbose) for (i=0; i<count; i++)
		printf("%s : %g\n",file_positions[i].filename, file_positions[i].position);
	memcpy(&mhead,file_in1->mhptr,sizeof(Main_header));
	matrix_close(file_in1);
	mhead.num_planes = (numplanes-overplanes)*count+overplanes;
	mhead.scan_start_time = ref_time;
	switch(mhead.file_type)
	{
	case PetImage:
	case PetVolume:
		mhead.file_type = PetImage;
		break;
	case Sinogram:
	case Short3dSinogram:
		mhead.file_type = Sinogram;
		break;
	default:
		crash("input must be Image or Sinogram stored as short integer\n");
	}
	file_out = matrix_create(argv[argc-1],MAT_OPEN_EXISTING, &mhead);
	if (!file_out) matrix_perror(argv[argc-1]);

	file_in1 = matrix_open(file_positions[0].filename,
		MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
	
	volume1 = matrix_read(file_in1, mat_numcod(1,1,1,0,0), MAT_SUB_HEADER);
	if (apply_decay) decay_1 = file_decay(file_in1);
	for (j=1; j<=(numplanes-overplanes); j++)
		wb_copy(file_out,plane++,file_in1,j,decay_1);
	free_matrix_data(volume1);
	matrix_close(file_in1);

	for (i=1; i<count; i++) {
		file_in1 = matrix_open(file_positions[i-1].filename,
			MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
		volume1 = matrix_read(file_in1, mat_numcod(1,1,1,0,0), MAT_SUB_HEADER);
		if (apply_decay) decay_1 = file_decay(file_in1);
		file_in2 = matrix_open(file_positions[i].filename,
			MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
		volume2 = matrix_read(file_in2, mat_numcod(1,1,1,0,0), MAT_SUB_HEADER);
		if (apply_decay) decay_2 = file_decay(file_in2);

		for (j=0; j<overplanes; j++)
		{
			wb_merge( file_out, plane++, file_in1,numplanes-overplanes+j+1,
				decay_1*mf[j].w1,file_in2,j+1, decay_2*mf[j].w2);
		}
		free_matrix_data(volume1); volume1 = volume2;
		for (j=overplanes+1; j<=(numplanes-overplanes); j++)
			wb_copy(file_out,plane++,file_in2,j,decay_2);

		free_matrix_data(volume1);
		matrix_close(file_in1); matrix_close(file_in2);
	}

	file_in1 = matrix_open(file_positions[count-1].filename,
		MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
	volume1 = matrix_read(file_in1, mat_numcod(1,1,1,0,0), MAT_SUB_HEADER);
	if (apply_decay) decay_1 = file_decay(file_in1);
	for (j=numplanes-overplanes+1; j<=numplanes; j++)
		wb_copy(file_out,plane++,file_in1,j,decay_1);
	free_matrix_data(volume1);
	matrix_close(file_in1);
}
