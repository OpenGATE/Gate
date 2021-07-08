/*
 *  static char sccsid[] = "%W% UCL-TOPO %E%";
 *
 *  Author : Sibomana@topo.ucl.ac.be
 *  Creation date :     30-Jun-1995
 *
 *  Uses find_center and compute_ellipse_attn  CTI code
static char sccsid[]="@(#)find_center.c	1.7 10/26/93 Copyright 1992 CTI Pet Systems, Inc.";
static char sccsid[]="@(#)compute_ellipse_attn.c	1.7 8/5/92 Copyright 1990 CTI Pet Systems, Inc.";
 */

#include <math.h>
#include <string.h>
#include "matrix.h"
#include "ecat_model.h"

struct line {
	float a, b, c;
	};

struct ellipse {
	float x0, y0, tilt, xr, yr;
	};


static int verbose = 0;

static find_center( image, thrsh, p_cx,p_cy)
MatrixData *image;
float thrsh, *p_cx, *p_cy;
{
	MatrixFile *mptr;
	Image_subheader *ishdr;
	short int *image_data;
	int i, x, y, xsize, ysize, dmax;
	float xmean, ymean, psize, intg, intg_pos;
	char fname[256];
	int matnum;
	int smin, cnt, left, right;

	ishdr = (Image_subheader*) image->shptr;
	xsize = image->xdim;
	ysize = image->ydim;
	psize = image->pixel_size;
	smin = ishdr->image_max * thrsh;
	xmean = 0.0;
	dmax = 0;
	cnt = 0;
	image_data = (short int*)image->data_ptr;
	for (y=0; y<ysize; y++)
	{
	  for (x=0; x<xsize; x++)
	    if (image_data[x+y*xsize] > smin) break;
	  left = x;
	  for (x=0; x<xsize; x++)
	    if (image_data[xsize-1-x+y*xsize] > smin) break;
	  right = xsize-1-x;
	  if (left < right)
	  {
	    xmean = cnt*xmean+(left+(float)(right-left)/2.0);
	    cnt++;
	    xmean = xmean/cnt;
	    if ((right-left)>dmax) dmax=right-left;
	  }
	}
	cnt = 0;
	ymean = 0.0;
	for (x=0; x<xsize; x++)
	{
	  for (y=0; y<ysize; y++)
	    if (image_data[x+y*xsize] > smin) break;
	  left = y;
	  for (y=0; y<ysize; y++)
	    if (image_data[x+(ysize-1-y)*xsize] > smin) break;
	  right = ysize-1-y;
	  if (left < right)
	  {
	    ymean = cnt*ymean+(left+(float)(right-left)/2.0);
	    cnt++;
	    ymean = ymean/cnt;
	    if ((right-left)>dmax) dmax=right-left;
	  }
	}
	intg = intg_pos = 0.0;
	for (i=0; i<xsize*ysize; i++)
	{
	  intg += image_data[i]*image->scale_factor;
	  if (image_data[i]>0) intg_pos += image_data[i]*image->scale_factor;
	}
	*p_cx = ishdr->x_offset + psize*(xmean-xsize/2);
	*p_cy = ishdr->y_offset - psize*(ymean-ysize/2);
	if (verbose) printf("image: center at %g,%g cm. diam = %g cm. int=%e,%e %e\n",
	  *p_cx, *p_cy, psize*dmax, intg, intg_pos, intg/intg_pos);
}


float* compute_ellipse_attn(nprojs,nviews,bin_size,ellipse,mu)
int nprojs,nviews;
float bin_size, mu;
struct ellipse *ellipse;
{
	char fname[256];
	int i, j, matnum;
	struct line line1, line2;
	double phi, theta, sintheta, costheta;
	float *a, *array, x, d, w1, w2, l1, l2;
	double length, ellipse_segment(), lmax;
	MatrixData *matrix;
	MatrixFile *mptr;
	Main_header mhead;
	Attn_subheader sub;

	array = a = (float*) malloc( nprojs*nviews*sizeof(float));
	theta = M_PI * ellipse->tilt / 180.;
	sintheta = sin(theta);
	costheta = cos(theta);
	d = bin_size;
	w1 = d;
	w2 = 2.0*d;
	lmax = 0.0;
	for (i=0; i<nviews; i++)
	{ phi =  M_PI * (float) i / (float) nviews;
	  line1.a = cos(phi);
	  line1.b = -sin(phi);
	  for (j=0; j<nprojs; j++)
	  {
	    x = bin_size * (float) (j-nprojs/2);
	    line1.c = x-w1/2;
	    translate_line( ellipse, &line1, &line2, sintheta, costheta);
	    l1 = ellipse_segment( &line2, ellipse);
	    line1.c = x+w1/2;
	    translate_line( ellipse, &line1, &line2, sintheta, costheta);
	    l2 = ellipse_segment( &line2, ellipse);
	    length = (l1+l2)/2.0;
	    *a++ = exp( (double) (mu*length));
	    if (length > lmax) lmax = length;
	  } /* next projection */
	} /* next view */
	return array;
}

translate_line( e, l1, l2, sint, cost)
  struct ellipse *e;
  struct line *l1, *l2;
  double sint, cost;
{
	l2->a = l1->a * cost + l1->b * sint;
	l2->b = l1->b * cost - l1->a * sint;
	l2->c = l1->c - l1->a * e->x0 - l1->b * e->y0;
}

#define pow2(x) (x*x)

double ellipse_segment( l, e)
  struct line *l;
  struct ellipse *e;
{
	double ax, bx, cx, ay, by, cy, dx2, dy2, sum;

	ax = pow2(l->b * e->yr) + pow2(l->a * e->xr);
	bx = 2.0 * l->a * l->c * pow2(e->xr);
	cx = pow2(e->xr) * (pow2(l->c) - pow2(l->b * e->yr));

	ay = ax;
	by = 2.0 * l->b * l->c * pow2(e->yr);
	cy = pow2(e->yr) * (pow2(l->c) - pow2(l->a * e->xr));

	dx2 = (pow2(bx)-4.0*ax*cx)/pow2(ax);
	dy2 = (pow2(by)-4.0*ay*cy)/pow2(ay);

	sum = dx2 + dy2;

	if (sum < 0.0) return (0.0);
	else return (sqrt(sum));
}

static void usage() {
	fprintf(stderr,"usage : phantom_attn -i image_file -o attn_file [-t thres -r radius -m mu -v] \n");
	fprintf(stderr,"-t threshold (image_max fraction, default = 0.15)\n");
	fprintf(stderr,"-r radius (in cm, default = 11.3 )\n");
	fprintf(stderr,"-m mu ( default = 0.096 )\n");
	fprintf(stderr,"-v set verbose mode\n");
	exit(1);
}

static char *image_file=NULL;
static char *attn_file=NULL;
static float threshold = 0.15;
static float radius = 11.3;
static float mu = 0.096;

main(argc, argv) 
int argc;
char *argv[];
{
	MatrixFile *mptr, *mptr1;
	Main_header mh;
	MatrixData *vol, *slice;
	Attn_subheader *ah;
	EcatModel *model;
	struct Matval m;
	struct MatDir matdir, dir_entry ;
	int matnum, slice_matnum, plane, mash=1;
	struct ellipse e;
	int nprojs, nviews, nplanes;
	int nblks, data_size, blkno, file_pos, skip_size, line_size;
	char *fname;
	int i,flag,y;
	float bin_size;
	float *attn_data, *all_1, *line, *scan_buf;
	float *xc, *yc, *z;
	float a,b,c,d,abdev,cddev;
	float *x_fit, *y_fit;
	extern char *optarg;

	while ((flag = getopt (argc, argv, "i:o:t:r:m:v")) != EOF) {
		switch (flag) {
		case 'i' : image_file = strdup(optarg);
			break;
		case 'o' : attn_file = strdup(optarg);
			break;
		case 't' : 
			if (sscanf(optarg,"%g",&threshold) != 1) {
				fprintf(stderr,"invalid threshold value : %s\n",optarg);
				usage();
			}
			break;
		case 'm' : 
			if (sscanf(optarg,"%g",&mu) != 1) {
				fprintf(stderr,"invalid mu value : %s\n",optarg);
				usage();
			}
			break;
		case 'r' : 
			if (sscanf(optarg,"%g",&radius) != 1) {
				fprintf(stderr,"invalid radius value : %s\n",optarg);
				usage();
			}
			break;
		case 'v' :
			verbose = 1;
			break;
		case '?' : default: usage();
		}
	}
	
	if (!image_file || !attn_file) usage();
	if ((mptr=matrix_open(image_file,MAT_READ_ONLY,MAT_UNKNOWN_FTYPE)) == NULL)
		crash("can't open %s\n",image_file);
	memcpy(&mh, mptr->mhptr, sizeof(Main_header));
	mh.file_type = AttenCor;
	mh.num_frames = 1;
	if ((fname=strrchr(attn_file,'/')) != NULL) fname = fname+1;
	else fname = attn_file;
	strncpy(mh.original_file_name,fname,32);
	mh.original_file_name[31] = '\0';
	mptr1 = matrix_create(attn_file, MAT_CREATE, &mh);
	if ( (model = ecat_model(mptr->mhptr->system_type)) == NULL)
		crash("unkown ecat_model %d\n",mptr->mhptr->system_type);
	if ( mptr->dirlist->nmats <= 0) crash("no image in %s\n",argv[1]);
	matnum = mptr->dirlist->first->matnum;
	mat_numdoc( matnum, &m);
	nprojs = model->defElements;
	for (i=0; i<mh.angular_compression; i++) mash *= 2;
	nviews = model->defAngles/mash;
	bin_size = model->binsize;
	e.tilt = model->intrTilt;
	e.xr = e.yr = radius;
	vol = matrix_read(mptr,matnum,MAT_SUB_HEADER);
	nplanes = vol->zdim;
	ah = (Attn_subheader*) calloc( 1, sizeof(Attn_subheader));
	ah->num_dimensions = 3;
	ah->num_r_elements = nprojs;
	ah->num_angles = nviews;
	ah->data_type = IeeeFloat;
	ah->x_resolution = bin_size;
	ah->span = model->def2DSpan;
	ah->ring_difference = model->def2DSpan/2;
	ah->scale_factor = 1.0;
	ah->z_elements[0] = ah->num_z_elements = vol->zdim;
	all_1 = (float*)malloc(MatBLKSIZE);
	scan_buf = (float*)malloc(nprojs*sizeof(float));
	for (i=0; i<128; i++) all_1[i] = 1.0;
	data_size =  nprojs*nviews*nplanes*sizeof(float);
	nblks = (data_size+511)/512;
	if (matrix_find(mptr1, matnum, &matdir) == -1) {
		blkno = mat_enter(mptr1->fptr, mptr1->mhptr, matnum, nblks) ;
		dir_entry.matnum = matnum ;
		dir_entry.strtblk = blkno ;
		dir_entry.endblk = dir_entry.strtblk + nblks - 1 ;
		dir_entry.matstat = 1 ;
		insert_mdir(dir_entry, mptr->dirlist) ;
		matdir = dir_entry ;
	} else {
		fprintf(stderr,"\7warning : existing matrix overwritten\n");
		blkno = matdir.strtblk;
	}
	mat_write_attn_subheader(mptr1->fptr, mptr1->mhptr, matdir.strtblk, ah);
	free(ah);
	file_pos = blkno*MatBLKSIZE;
	if (fseek( mptr1->fptr, blkno*MatBLKSIZE, 0) == EOF)
	crash("%s : fseek error\n", mptr1->fname);
	for (i=0; i<nblks; i++) {
		if (ntohs(1) != 1) {
			swab(all_1,scan_buf,MatBLKSIZE);
			swaw((short*)scan_buf,(short*)all_1,MatBLKSIZE/2);
		}
		if (fwrite( all_1, sizeof(float), 128, mptr1->fptr) != 128)
			crash("%s : fwrite error\n", mptr1->fname);
	}
	free(all_1);
	line_size = nprojs*sizeof(float);
	skip_size = line_size*(nplanes-1);
	xc = (float*)calloc(nplanes,sizeof(float));
	yc = (float*)calloc(nplanes,sizeof(float));
	z = (float*)calloc(nplanes,sizeof(float));
	x_fit = (float*)calloc(nplanes,sizeof(float));
	y_fit = (float*)calloc(nplanes,sizeof(float));
	for (plane=0; plane < nplanes; plane++) {
		slice = matrix_read_slice(mptr, vol, plane,0);
		find_center(slice,threshold, &xc[plane],&yc[plane]);
		if (verbose) fprintf(stderr,"plane %d : (x_center,y_center)=(%g,%g)\n",
			plane,xc[plane],yc[plane]);
		z[plane] = plane;
		free_matrix_data(slice);
	}
	
/* Numerical Recipes medfit works on unit offset vector */
	medfit(z-1,xc-1,nplanes,&a,&b,&abdev);
	medfit(z-1,yc-1,nplanes,&c,&d,&cddev);
	if (verbose) fprintf(stderr,"(a,b,abdev) = (%g,%g,%g)\n",a,b,abdev);
	if (verbose) fprintf(stderr,"(c,d,cddev) = (%g,%g,%g)\n",c,d,cddev);
	for (plane=0; plane < nplanes; plane++) {
		x_fit[plane] = a+b*plane;
		y_fit[plane] = c+d*plane;
	}

	for (plane=0; plane < nplanes; plane++) {
		e.x0 = x_fit[plane];
		e.y0 = y_fit[plane];
		if (verbose) fprintf(stderr,"plane %d : (x_fit,y_fit)=(%g,%g)\n",
			plane,e.x0, e.y0 );
		attn_data = compute_ellipse_attn(nprojs,nviews,bin_size,&e,mu);
		file_pos = blkno*MatBLKSIZE + plane*line_size;
		if (fseek(mptr1->fptr,file_pos,0) == -1)
			crash("%s : fseek error\n", mptr1->fname);
		if (ntohs(1) != 1) {
			swab(attn_data,scan_buf,nprojs*sizeof(float));
			swaw((short*)scan_buf,(short*)attn_data,nprojs*2);
		}
		fwrite(attn_data,sizeof(float),nprojs,mptr1->fptr);	/*write view 0*/
		line = attn_data+nprojs;;
		for (y=1; y<nviews; y++) {  /* for each planar view fixed theta */
			if (fseek(mptr1->fptr,skip_size,1) == -1)
				crash("%s : fseek error\n", mptr1->fname);
			if (ntohs(1) != 1) {
				swab(line,scan_buf,nprojs*sizeof(float));
				swaw((short*)scan_buf,(short*)line,nprojs*2);
			}
			fwrite(line,sizeof(float),nprojs,mptr1->fptr);
			line += nprojs;
		}
		free(attn_data);
		free_matrix_data(slice);
	}
	matrix_close(mptr);
	matrix_close(mptr1);
}
