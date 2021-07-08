/*
 Modification History
 8-sep-99 :
 Add more tolerance for bed shifts variation (plane separation)
 Use the most frequent bed shift.
 20-mar-00 :
 Bug fix, bed_shift[0](line 348)  was not used 
 and assembling 2 positions was
 giving bad results . Use emacs indentation
 
 CJM :  
	adds Version 6 February 03 in usage
	do not exit if main header does not contain number of planes, use default
	corrects merging weights calculation 
	corrects number of planes in main header was made 
 */

#include "matrix.h"
#include "ecat_model.h"
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#define MAX_BEDS 20

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

typedef struct _BedShift
{
  double value; /* in mm */
  int count;
} BedShift;

static BedShift *bed_shifts;
static int num_bed_shifts = 0;

static MatrixFile *file_in=NULL;
static MatrixFile *file_out=NULL;
static int verbose = 0;

static usage()
{
  fprintf(stderr,"usage : wb_assemble -i multi_bed_file -o wb_file [-p bed_pos_file] [-v -x bi]\n");
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
  while (!sorted) /* bubble sort */
    {
      sorted = 1;
      for (i=1; i<num_bed_shifts; i++)
        {
	  if (bed_shifts[i].count > bed_shifts[i-1].count)
            {   /* swap and unset sorted flag */
	      tmp = bed_shifts[i-1];
	      bed_shifts[i-1] = bed_shifts[i];
	      bed_shifts[i] = tmp;
	      sorted = 0;
            }
        }
    }
}

static MatrixData *blank()
{
  static MatrixData *plane_out=NULL;
  MatrixData *template;
  Image_subheader *imh;
  if (plane_out == NULL)
    {
      plane_out = (MatrixData*)calloc(1,sizeof(MatrixData));
      imh = (Image_subheader*)calloc(1,sizeof(Image_subheader));
      plane_out->shptr = (caddr_t)imh;
      template = matrix_read(file_in,file_in->dirlist->first->matnum,
			     MAT_SUB_HEADER);
      memcpy(plane_out,template,sizeof(MatrixData));
      plane_out->data_type = imh->data_type = SunShort;
      plane_out->zdim = imh->z_dimension = 1;
      plane_out->data_size = plane_out->xdim*plane_out->ydim*sizeof(short);
      plane_out->data_ptr = (caddr_t)calloc(plane_out->data_size,1);
    } else memset(plane_out->data_ptr,plane_out->data_size,0);
  return plane_out;
}
	

#ifdef __STDC__
wb_copy(int pl_out, BedPosition *bed, int pl_in)
#else
wb_copy(pl_out,bed,pl_in)
int pl_out, pl_in;
BedPosition *bed;
#endif
{
  struct MatDir *matdir;
  MatrixData *slice;
  Image_subheader*  imh;
  int segment=0;
  
  matdir = &bed->matdir;
  if (!bed->matrix)
    bed->matrix = matrix_read(file_in, matdir->matnum, MAT_SUB_HEADER);
  if (!bed->matrix)
    {		/* assume 128x128 short int image */
      fprintf(stderr,"Warning :  Plane  %s,1,%d,1,0,%d missing\n",
	      file_in->fname, pl_in, bed->bed);
      matrix_write(file_out,mat_numcod(1,pl_out,1,0,0),blank());
    }
  else
    {
      slice = matrix_read_slice(file_in,bed->matrix,pl_in-1,segment);
      matrix_write(file_out,mat_numcod(1,pl_out,1,0,0),slice);
      if (verbose) printf("copy %s,1,%d,1,0,%d	===> %s,1,%d,1 done\n",
			  file_in->fname, pl_in, bed->bed, file_out->fname,pl_out);
      free_matrix_data(slice);
    }
}

#ifdef __osf__
wb_merge(int pl_out, BedPosition *bed_1, int pl1, float w1,
BedPosition *bed_2, int pl2, float w2)
#else
wb_merge(pl_out, bed_1,pl1,w1,bed_2,pl2,w2)
BedPosition *bed_1, *bed_2;
int pl_out, pl1,pl2;
float w1, w2;
#endif
{
  MatrixData *m1, *m2;
  Image_subheader*  imh;
  float *f_data, f_min, f_max;
  int i, size, segment=0;
  struct MatDir *matdir_1, *matdir_2;
  float scale1,scale2, scale;
  short *p1, *p2;
  
  matdir_1 = &bed_1->matdir;
  matdir_2 = &bed_2->matdir;
  if (!bed_1->matrix)
    {
      bed_1->matrix = matrix_read(file_in, matdir_1->matnum, MAT_SUB_HEADER);
      if (!bed_1->matrix)
	{
	  fprintf(stderr,"Warning :  Plane  %s,1,%d,1,0,%d missing\n",
		  file_in->fname, pl1, bed_1->bed);
	}
    }
  if (!bed_2->matrix)
    {
      bed_2->matrix = matrix_read(file_in, matdir_2->matnum, MAT_SUB_HEADER);
      if (!bed_2->matrix)
	{
	  fprintf(stderr,"Warning :  Plane  %s,1,%d,1,0,%d missing\n",
		  file_in->fname, pl2, bed_2->bed);
	}
    }
  
  if (bed_1->matrix==NULL && bed_2->matrix==NULL)
    {
      matrix_write( file_out,	mat_numcod(1,pl_out,1,0,0),blank());
      return;
    }
  if (bed_1->matrix==NULL)
    {		/* bed_2->matrix != NULL */
      wb_copy(pl_out,bed_2,pl2);
      return;
    }
  if (bed_2->matrix==NULL)
    {		/* bed_1->matrix != NULL */
      wb_copy(pl_out,bed_1,pl1);
      return;
    }
  /* bed_1->matrix!=NULL && bed_2->matrix!=NULL */
  m1 = matrix_read_slice(file_in,bed_1->matrix,pl1-1,segment);
  m2 = matrix_read_slice(file_in,bed_2->matrix,pl2-1,segment);
  p1 = (short*)m1->data_ptr; p2 = (short*)m2->data_ptr;
  scale1 = m1->scale_factor; scale2 = m2->scale_factor;
  size = m1->xdim*m1->ydim;
  f_data = (float*)calloc(size,sizeof(float));
  f_min = f_max = f_data[0] = (w1*scale1*p1[0])+(w2*scale2*p2[0]);
  for (i=1; i<size; i++)
    {
      f_data[i] = (w1*scale1*p1[i])+(w2*scale2*p2[i]);
      if (f_max < f_data[i]) f_max = f_data[i];
      if (f_min > f_data[i]) f_min = f_data[i];
    }
  scale = f_max/32767;
  imh = (Image_subheader*)m1->shptr;
  m1->data_max = f_max;
  m1->data_min = f_min;
  m1->scale_factor = imh->scale_factor = scale;
  m1->matnum = mat_numcod(1,pl_out,1,0,0);
  imh->scale_factor = scale;
  if (scale> 0) for (i=0; i<size; i++) p1[i] = (short)(f_data[i]/scale);
  imh->image_min = (short)(f_min/scale);
  imh->image_max = (short)(f_max/scale);
  matrix_write( file_out, m1->matnum,m1);
  free(f_data);
  free_matrix_data(m1);
  free_matrix_data(m2);
  if (verbose)
    printf("merge %g * 1,%d,1,0,%d + %g * 1,%d,1,0,%d  ===> %s,1,%d,1 done\n",
	   w1,pl1,bed_1->bed,w2,pl2,bed_2->bed,file_out->fname,pl_out);
  if (verbose)
    printf("scale factors : %g  %g  ===> %g \n", scale1, scale2, scale);
}

int main(argc,argv)
int argc;
char **argv;
{
  BedPosition tmp, *bed_positions = NULL, *bed_1, *bed_2;
  Main_header mhead, *mh;
  int i=0, j=0, count=0, sorted=0;
  int plane = 1, numplanes, overplanes;
  int span = 0, maxlors;
  float bed_pos0, *bed_offsets;
  double   bed_shift, plane_sep;
  char *x_beds;
  MergeFactor *mf;
  EcatModel *model;
  int matnum;
  char in_name[128], out_name[128], bed_pos_file[128];
  struct MatDir matdir;
  FILE *fp;
  extern int optind, opterr;
  extern char *optarg;
  
  in_name[0] = '\0'; out_name[0] = '\0'; bed_pos_file[0] = '\0';
  x_beds = calloc(MAX_BEDS,1);
  while ((i = getopt (argc, argv, "i:o:x:p:v")) != EOF)
    {
      switch(i)
	{
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
	case 'x':
	  j = atoi(optarg);
	  if (j<MAX_BEDS) x_beds[j] = '1';
	  break;
	case '?' : usage();
	  break;
	}
    }
  if (!strlen(in_name) || !strlen(out_name)) usage();
  if (strcmp(in_name,out_name) == 0)
    {
      printf("input and output should differ\n");
      exit(1);
    }
  if (verbose) printf("input file : %s\noutputfile : %s\n",in_name,out_name);
  file_in = matrix_open(in_name, MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
  if (file_in == NULL) matrix_perror(in_name);
  count =  file_in->mhptr->num_bed_pos+1;
  if (count < 2)
    {
      fprintf(stderr,"%s is not a multi bed position file\n");
      exit(1);
    }
  bed_positions = (BedPosition*)calloc(sizeof(BedPosition),count);
  bed_positions = (BedPosition*)calloc(sizeof(BedPosition),count);
  bed_pos0 = file_in->mhptr->init_bed_position;
  bed_offsets = file_in->mhptr->bed_offset;
  numplanes =  file_in->mhptr->num_planes;
  plane_sep =  file_in->mhptr->plane_separation;
  if (bed_pos_file[0] && (fp=fopen(bed_pos_file,"r"))!= NULL)
    {
      for (i=0;i<count; i++)
	if (fscanf(fp,"%g",bed_offsets+i) != 1) break;
      fclose(fp);
    }
  else
    {		  /* correct storage nonsense */
      /* problem fixed since Jun-1995 release 
	 for (i=count-1; i >= 2; i--) bed_offsets[i] = bed_offsets[i-2];
	 bed_offsets[1] = bed_offsets[2]/2;
	 bed_offsets[0] = 0;
      */
    }
  for (i=0, j=0; i<count; i++)
    {
      if (x_beds[i]) continue;
      matnum = mat_numcod(1,1,1,0,i);
      if (matrix_find(file_in,matnum,&matdir) < 0 )
	{
	  fprintf(stderr,"matrix 1,1,1,0,%d not found\n",i);
	}
      bed_positions[j].bed = i;
      bed_positions[j].matdir = matdir;
      if (i>0) bed_positions[j].position = bed_pos0+bed_offsets[i-1];
      else  bed_positions[j].position = bed_pos0;
      j++;
    }
  count = j;
  
  /* buble sort */
  while (!sorted)
    {
      sorted = 1;
      for (i=1; i<count; i++)
	{
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
      if (fabs(bed_shift-bed_shifts[i].value) > 10*plane_sep)
				/* plane_sep in mm */
	{
	  fprintf(stderr,
		  "%s : bed shift variation greater then plane separation (%g,%g) cm = > aborted\n",
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
  if (numplanes != (model->dirPlanes*2-1)) 
    {
      fprintf(stderr,"MainHeader number of planes %d disagree with scanner model %d \n",numplanes,model->dirPlanes*2-1); /* print values CJM */
      numplanes = model->dirPlanes*2-1;		/* set numplanes to its value  CJM */	
      /* return 1; 				CJM */
    }
  /* span is only accessible in axial_compression in Scan3D_subheader and is not available in image */
  if (span == 0) span = model->def2DSpan;
  fprintf(stderr,"Assumed span is %d \n",span); 	  
  overplanes = (int)(.5 +(numplanes*plane_sep - bed_shift)/plane_sep);
  fprintf(stderr,"Assumed number of overlapping planes is %d \n",overplanes); 
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
      i++;				/* CJM */
      if (i<overplanes)
	{
	  mf[i].w2 = maxlors;
	  mf[overplanes-i-1].w1 = maxlors;
	}
      i++;				/* CJM */
    }
  fprintf(stderr,"merge factors :");
  for (i=0; i<overplanes; i++)
    fprintf(stderr," (%g,%g)", mf[i].w1, mf[i].w2);
  fprintf(stderr,"\n");
  for (i=0; i<overplanes; i++)
    {
      mf[i].w1 =  mf[i].w1 / (mf[i].w1+ mf[i].w2);
      mf[i].w2 = 1.0 - mf[i].w1;
      fprintf(stderr," (%g,%g)", mf[i].w1, mf[i].w2);
    }
  fprintf(stderr,"\n");
  if (verbose) for (i=0; i<count; i++) 
    printf("%d : %g\n",bed_positions[i].bed, bed_positions[i].position);
  
  /*	output file overwrite is checked at executive level
	if (access(out_name,F_OK) == 0)
	{
	fprintf(stderr, "Warning: %s is overwritten\7\n",out_name);
	usage();
	}
  */
  memcpy(&mhead,file_in->mhptr,sizeof(Main_header));
  /* mhead.num_planes = 42+((5+37)*(count-1))+5; */
  mhead.num_planes = (numplanes-overplanes)*count+overplanes;
  fprintf(stderr,"Total number of planes is %d\n",mhead.num_planes);
  for (i=0;i<mhead.num_bed_pos;i++) mhead.bed_offset[i] = 0.0;
  mhead.num_bed_pos = 1;
  mhead.init_bed_position = bed_positions[0].position;
  file_out = matrix_create(out_name,MAT_OPEN_EXISTING, &mhead);
  if (file_out == NULL) matrix_perror(out_name);
  
  bed_1 = bed_positions;
  for (j=1; j<=(numplanes-overplanes); j++)
    wb_copy(plane++,bed_1,j);
  for (i=1; i<count; i++)
    {
      bed_1 = &bed_positions[i-1]; bed_2 = &bed_positions[i];
      for (j=0; j<overplanes; j++)
	{
	  wb_merge(plane++, bed_1,numplanes-overplanes+j+1,
		   mf[j].w1, bed_2, j+1, mf[j].w2);
	}
      for (j=overplanes+1; j<=(numplanes-overplanes); j++)
	wb_copy(plane++,bed_2,j);
      free_matrix_data(bed_1->matrix);
      bed_1->matrix = 0;
    }

  bed_1 = &bed_positions[count-1];
  for (j=numplanes-overplanes+1; j<=numplanes; j++)
    wb_copy(plane++,bed_1,j);
  free_matrix_data(bed_1->matrix);
  matrix_close(file_in);
  matrix_close(file_out);
}
