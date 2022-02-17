/*
 *  static char sccsid[] = "%W% UCL-TOPO %E%";
 *
 *  Author : Sibomana@topo.ucl.ac.be
 *	Creation date :  24-Jan-1996
 */

#include "matrix.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
/* #define mod(i, j) ( (i < j) ? i : i-j); */

static int verbose = 0;
static int positive_only = 0;

static void usage() {
	fprintf(stderr,"usage : \nscan2if -i scan_spec - o IF_data [-h IF_header] [-r r_elements] [-a angles] [-t tilt] [-c x_offset,y_offset] [-z zoom] [-s scale_factor] [-p] [-v]\n");
	fprintf(stderr,"Result :\n");
	fprintf (stderr,"writes on (IF_data, IF_header) Interfile 3.3 sinogram\n");
	fprintf (stderr,"if IF_header not specified, the filename is made of IF_data with h33 extension\n");
	fprintf (stderr,"-r interpolates in radial direction to specified number of elements\n");
	fprintf (stderr,"-a interpolates in angular direction to specified number of angles\n");
	fprintf (stderr,"-t removes intrinsic tilt \n");
	fprintf (stderr,"-c recenter sinogram on X and Y\n");
	fprintf (stderr,"-z zoom sinogram in radial direction\n");
	fprintf (stderr,"-s scales sinogram with a given factor\n");
	fprintf (stderr,"-p positive only (negative values are set to 0)\n");
	fprintf (stderr,"-v verbose\n");
	exit(1);
}

static MatrixData* matrix_interp(matrix, sx1, sx2)
MatrixData *matrix;
int sx1, sx2;
{
	int i1, i2, j, k;
	int x1a, x1b, x2a, x2b;
	int sx1a, sx2a;
	float x1, x2; 
	float y1,y2,y3,y4;
	float t, u;
	float *ya, *y;
	float dx1 = ((float)matrix->xdim)/sx1;
	float dx2 = ((float)matrix->ydim)/sx2;

/*	Numerical Recipes ....
	for (i1=0; i1<sx1; i1++) {
			x1 = dx1 * i1;
			j = (int) x1;
		for (i2=0; i2<sx2; i2++) {
			x2 = dx2 * i2;
			k = (int)x2;
			y1 = ya[j][k];
			y2 = ya[j+1][k];
			y3 = ya[j+1][k+1];
			y4 = ya[j][k+1];
			t = (x1 -j);
			u = (x2 - k);
			y = (1-t)*(1-u)*y1 + t*(1-u)*y2 + t*u*y3 + (1-t)*u*y4;
		}
	}
*/
	sx1a = matrix->xdim; sx2a = matrix->ydim;
	y = (float*)calloc(sx1*sx2, sizeof(float));
	ya = (float*)matrix->data_ptr;
	for (i1=0; i1<sx1; i1++) {
		x1 = dx1 * i1;
		j = (int) x1;
		for (i2=0; i2<sx2; i2++) {
			x2 = dx2 * i2;
			k = (int)x2;
			y1 = ya[k*sx1a + j];
			y2 = ya[k*sx1a + (j+1)];
			y3 = ya[(k+1)*sx1a + (j+1)];
			y4 = ya[(k+1)*sx1a + j];
			t = (x1 -j);
			u = (x2 - k);
			y[i2*sx1+i1] = (1-t)*(1-u)*y1 + t*(1-u)*y2 + t*u*y3 + (1-t)*u*y4;
		}
	}
	matrix->xdim = sx1; matrix->ydim = sx2;
	matrix->pixel_size *= dx1;
	matrix->y_size *= dx2;
	matrix->data_ptr = (caddr_t)y;
	free(ya);
	return matrix;
}

static matrix_shift(matrix, x0, y0)
MatrixData *matrix;
float x0,y0;
{
   double	theta;
   int		i, j, k;
   int		np, nv;
   float	shift, x, w, *fdata_in, *fdata_out;

   fdata_out = (float*)calloc(matrix->xdim*matrix->ydim,sizeof(float));
   fdata_in = (float*)matrix->data_ptr;
   np=matrix->xdim;
   nv=matrix->ydim;   

   for (i=0; i<nv; i++)
   {	theta = M_PI*i/nv;
	
	shift = (y0*sin(theta)-x0*cos(theta))/matrix->pixel_size;
	for (j=0; j<np; j++)
	{   x = shift + j;
	    if (x<0.0 || x>(float)(np-2)) continue;
	    k = (int) x;
	    w = x-k;
	    fdata_out[k+i*np] += (1.0-w)*fdata_in[i*np+j];
	    fdata_out[k+1+i*np] += w*fdata_in[i*np+j];
	 }
    }
    matrix->data_ptr = (caddr_t)fdata_out;
    free(fdata_in);
}

static remove_tilt(matrix, tilt)
MatrixData *matrix;
float tilt;
{
   int		i, j, np,nv, shift;
   float	*fdata_in, *fdata_out, *line_in, *line_out;
   fdata_out = (float*)calloc(matrix->xdim*matrix->ydim,sizeof(float));
   fdata_in = (float*)matrix->data_ptr;
   np=matrix->xdim;
   nv=matrix->ydim; 
   shift = (int)(tilt * nv/180.);
   if (shift > 0) 
   { memcpy(fdata_out+(np*shift),fdata_in,(np*(nv-shift))*sizeof(float));
     for (i=0 ; i<shift ; i++)
     {	line_in = fdata_in + ((nv-shift)+i)*np;
	line_out = fdata_out + i*np;
	for (j=0; j<np; j++) line_out[np-j-1] = line_in[j];
     }
   }
   else
   { shift = -shift; 
     memcpy(fdata_out,fdata_in+(np*shift),(np*(nv-shift))*sizeof(float));
     for (i=0 ; i<shift ; i++)
     {	line_in = fdata_in + i*np;
	line_out = fdata_out + ((nv-shift)+i) *np;
	for (j=0; j<np; j++) line_out[np-j-1] = line_in[j];
     }
   }
   matrix->data_ptr = (caddr_t)fdata_out;
   free(fdata_in);
}

static  matrix_trim(matrix,trim)
MatrixData *matrix;
int trim;
{
   int		i, j, np, nv;
   float	*fdata_in, *fdata_out;

   fdata_in = (float*)matrix->data_ptr;
   np=matrix->xdim;
   nv=matrix->ydim; 
   if (verbose) fprintf(stderr, "projs %d views %d\n", np,nv);
   if (trim >= np) return;
   fdata_out = (float*)calloc(trim*matrix->ydim,sizeof(float));
   for (i=0; i<nv; i++)
     for (j=0; j<trim; j++)
	fdata_out[j+i*trim] = fdata_in[j+i*np+np/2-trim/2];

   matrix->data_ptr = (caddr_t)fdata_out;
   matrix->xdim=trim;
   free(fdata_in);
}			

static  matrix_float(matrix)
MatrixData *matrix;
{
	float scalef, *fdata;
	short *sdata;
	int i, np = matrix->xdim*matrix->ydim;

	matrix->data_type = IeeeFloat;
	fdata = (float*)calloc(np,sizeof(float));
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
	int nprojs = matrix->xdim, nviews = matrix->ydim;
	

	if (x_rebin<=1 && y_rebin<=1) return;
	scan_out = (float*)calloc(sizeof(float),nprojs*nviews);
/* integer x rebin */
	scan_in = (float*)matrix->data_ptr;
	fp_in = scan_in; fp_out = scan_out;
	if (x_rebin > 1 ) {
		matrix->xdim /= x_rebin;
		for (i=0; i<nprojs*nviews; i += x_rebin, fp_out++) {
			for (j=0; j<x_rebin; j++) *fp_out += *fp_in++;
		}
		memcpy(scan_in,scan_out,nprojs*nviews*sizeof(float));
		memset(scan_out,0,nprojs*nviews*sizeof(float));  /* set to 0.0 */
		nprojs = matrix->xdim;
	}
	
/* integer y rebin */
	fp_in = scan_in; fp_out = scan_out;
	if (y_rebin > 1 ) {
		matrix->ydim /= y_rebin;
		for (i=0; i<nviews; i++) {
			fp_out = scan_out + nprojs*(i/y_rebin);
			for (j=0; j<nprojs; j++) fp_out[j] += *fp_in++;
		}
	}
	free(scan_in);
	matrix->data_ptr = (caddr_t)scan_out;
}

int main(argc, argv)
int  argc;
char** argv;
{
    MatrixFile *mptr1=NULL;
	FILE *fp, *data_fp;
        MatrixData *volume, *scan, *slice;
        char fname[256], *p, *in=NULL, *out=NULL, *header=NULL;
	float *fdata, user_scale = 1.0, d_rescale=1.,a_rescale=1.,xoff=0., yoff=0., tilt=0., zoom=1., nzoom=1., quant =1.;
	float input_mash=1.,fmax =0., sum_in =0.,sum_out1=0.,sum_out2=0.;
	short *sdata, smax=0;
	int c, matnum=0, frame=0;
	int  segment=0;		/* ACS/2 segment 0 for 2D mode Acquisition*/
	struct Matval matval;
	MatDirNode *node;
	int i, plane, trim, nprojs = 0, nviews = 0;
	Scan_subheader *i_scan_sub=0;
	Scan3D_subheader *i_scan3d_sub=0;
	Attn_subheader *attn_sub=0;
	extern int optind, opterr;
	extern char *optarg;
	
	fname[0] = '\0';
	while ((c = getopt (argc, argv, "i:o:r:a:t:c:z:s:h:pv")) != EOF) {
		switch (c) {
		case 'i' :
			in = optarg;
			break;
		case 'o':
			out = optarg;
			break;
		case 'h':
			header = optarg;
			break;
		case 'r' :
			if (sscanf(optarg,"%d",&nprojs) != 1)
				crash("error decoding -t %s\n",optarg);
			break;
		case 'a' :
			if (sscanf(optarg,"%d",&nviews) != 1)
				crash("error decoding -a %s\n",optarg);
			break;
		case 't' :
			if (sscanf(optarg,"%f", &tilt) != 1)
				crash("error decoding -t %s\n",optarg);
			break;
		case 'c' :
			if (sscanf(optarg,"%f,%f", &xoff, &yoff) != 2)
				crash("error decoding -c %s\n",optarg);
			break;
		case 'z' :
			if (sscanf(optarg,"%f", &zoom) != 1)
				crash("error decoding -z %s\n",optarg);
			break;
		case 's' :
			if (sscanf(optarg,"%g",&user_scale) != 1)
				crash("error decoding -s %s\n",optarg);
			break;
		case 'v':
			verbose = 1;
			break;
		case 'p':
			positive_only = 1;
			break;
		default:
			usage();
		}
	}

	if (in==NULL || out==NULL) usage();
	matspec( in, fname, &matnum);
   	mptr1 = matrix_open( fname, MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
   	if (!mptr1)
   		crash( "%s: can't open file '%s'\n", argv[0], fname);
	if (matnum==0 && mptr1->dirlist->first)
		matnum = mptr1->dirlist->first->matnum;
	mat_numdoc(matnum, &matval);
	frame = matval.frame;
	if ( (volume = matrix_read(mptr1, matnum, MAT_SUB_HEADER)) == NULL)
		crash("input matrix not found");
	if (!nprojs) nprojs = volume->xdim;
	if (!nviews) nviews = volume->ydim;

								/* since fproj_volume does not fill mh properly */
	if (mptr1->mhptr->plane_separation == 0) {
		fprintf(stderr,"\7Unknown plane separation, assume ECAT 961HR\n");
		mptr1->mhptr->plane_separation = 0.3125;
	}
								/* count planes for slices mode Sinograms */
	if (mptr1->mhptr->file_type == Sinogram) {
		volume->zdim=0;
		node = mptr1->dirlist->first;
		while (node) {
			mat_numdoc( node->matnum, &matval);
			if (matval.frame == frame) volume->zdim++;
			node = node->next;
		}
	}

	if (!header) {
		matspec(out, fname, &matnum);
		if ((p = strrchr(fname,'/')) == NULL) p = fname;
    	if ((p = strrchr(p,'.')) != NULL) *p = '\0';
		strcat(fname,".h33");
	} else strcpy(fname, header);
	/* write Interfile header */
	if ((fp = fopen(fname, "w")) == NULL)
    	crash( "%s: can't create file '%s'\n", argv[0], fname);
	fprintf(fp,"!Interfile :=\n");
	fprintf(fp,"version of keys := 3.3\n");
	fprintf(fp,"name of data file := %s\n",out);
	fprintf(fp,"number of dimensions := 3\n");
	fprintf(fp,"matrix size [1] := %d\n",nprojs);
	fprintf(fp,"matrix size [2] := %d\n",nviews);
	fprintf(fp,"matrix size [3] := %d\n",volume->zdim);
	fprintf(fp,"number format := signed integer\n");
	fprintf(fp,"number of bytes per pixel := 2\n");
	fprintf(fp,"imagedata byte order := BIGENDIAN\n");
	
	/* create Interfile data file */
	if ((data_fp = fopen(out, "w")) == NULL)
		crash( "%s: can't create file '%s'\n", argv[0], out);
	scan = (MatrixData*)calloc(sizeof(MatrixData),1);
	scan->shptr = (caddr_t)calloc(sizeof(Scan_subheader),1);
	
	switch(mptr1->mhptr->file_type) {
	case Sinogram :
		i_scan_sub= (Scan_subheader*)volume->shptr;
		break;
	case Float3dSinogram :
	case Short3dSinogram :
		i_scan3d_sub = (Scan3D_subheader*)volume->shptr;
		break;
	case AttenCor:
		attn_sub = (Attn_subheader*)volume->shptr;
		break;
	default :
    	crash("input is not a Sinogram nor an Attenuation\n");
		break;
	}
	scan->xdim = nprojs;
	scan->ydim = nviews;
	scan->data_ptr = (caddr_t)calloc(nviews*nprojs, sizeof(short));
	sdata = (short*)scan->data_ptr;
	for (plane=1; plane<=volume->zdim; plane++) {
		if (mptr1->mhptr->file_type == Sinogram) 
			slice = matrix_read(mptr1,mat_numcod(frame,plane,1,0,0),Sinogram);
		else slice = matrix_read_slice(mptr1,volume,plane-1,segment);
		if (slice->data_type != IeeeFloat) 					/* matrix should be float */
			matrix_float(slice);

		fdata = (float*)slice->data_ptr;
		for (i=0; i<nprojs*nviews; i++) 
		{
			sum_in +=fdata[i];
		}

		if (tilt != 0.) 							/* remove intrinsic tilt */
		{	if (verbose) fprintf(stderr, "Intrinsic tilt corrected: %g degrees\n", tilt);
			remove_tilt(slice,tilt);
		}

  		if (xoff != 0. || yoff !=0.) 						/* recenter scan	*/
			matrix_shift(slice, xoff, yoff);

  		if (zoom > 1. && zoom < ((float)slice->xdim)/nprojs)  			/* zoom scan with a trim from xdim to min nprojs */
		{	trim = 2*(int) ((float)slice->xdim/(2.*zoom)+0.5);		/* should be even	 */
			nzoom = (float)slice->xdim/trim;				/* nearest zoom */
        		if (verbose) fprintf(stderr, "Trim at %d projections, nearest zoom is %g\n", trim, nzoom);
			matrix_trim(slice, trim);
			slice->xdim = trim;
   		}

		if (slice->xdim/nprojs>1 || slice->ydim/nviews>1) 			/* integer rebin	*/
		{	if (verbose) fprintf(stderr, "Integer rebin d at %d ,a at %d\n", slice->xdim/nprojs,slice->ydim/nviews);
			matrix_rebin(slice,slice->xdim/nprojs,slice->ydim/nviews);
		}

		if (nprojs!=slice->xdim || nviews!=slice->ydim)	
		{	if (verbose) fprintf(stderr, "Interpolation with nprojs %d ,nviews %d\n", nprojs,nviews);
			matrix_interp(slice,nprojs,nviews);				/* interpolate		*/
		}
		fdata = (float*)slice->data_ptr;
		fmax = 0.;
		for (i=0; i<nprojs*nviews; i++) 
		{
			sum_out1 +=fdata[i];
			if (fdata[i] > fmax) fmax = fdata[i];
		}
		if (fmax*user_scale > 32767.) crash (" The maximum scale factor is %g\n",32767./fmax);
		fdata = (float*)slice->data_ptr;
		for (i=0; i<nprojs*nviews; i++) {
			c = (int)(fdata[i]*user_scale + 0.5);
			c = c<32768? c : 32767; 					/* truncate to short integer*/
			if (positive_only && c<0) c = 0;
			if (c < -32768) c = -32768;
			if (smax<c)  smax = c;
			sdata[i] = c;
			sum_out2 += c;
		}
		if (ntohs(1) != 1) {
			swab(sdata, slice->data_ptr,nprojs*nviews*sizeof(short));
			fwrite(slice->data_ptr,nprojs*nviews,sizeof(short),data_fp);
		} else fwrite(sdata,nprojs*nviews,sizeof(short),data_fp);
		free_matrix_data(slice);
	}
	if (smax < 1.0)
		fprintf(stderr,"\7maximum is too low ; use -s option\n");
	fprintf(fp,"maximum pixel count := %d\n",smax);
	d_rescale = ((float)volume->xdim)/(nprojs*nzoom);
	input_mash = 392./volume->ydim;				/* valid for 961HR only */
	a_rescale = ((float) volume->ydim)*input_mash/nviews; 
        fprintf(stderr, "Rescaling factors: distance %g angle: %g\n", d_rescale, a_rescale);   
	fprintf(fp,"scaling factor (mm/pixel) [1] := %g\n",
		10.*d_rescale*volume->pixel_size);
	fprintf(fp,"scaling factor (mm/pixel) [2] := %g\n",
		10.*d_rescale*volume->pixel_size);
	fprintf(fp,"scaling factor (mm/pixel) [3] := %g\n",
		10.*mptr1->mhptr->plane_separation);
 /*     quant= 1./(user_scale*a_rescale*d_rescale*d_rescale);   keep quantification */
        quant= 1./(user_scale*a_rescale); 			/* keep quantification */
	fprintf(fp,";%%quantification units := %g\n", quant);
/*	fprintf(stdout,"%g\n", quant);				   pass quantification to calling script */
	sum_out1 /= nzoom;
	sum_out2 /= (nzoom*user_scale);
	fprintf(stderr,"Input Integral %g Output Integral %g => Ratio %g\n",sum_in,sum_out1,sum_in/sum_out1);
	fprintf(stderr,"Output Integral after integer conversion %g => Ratio %g\n",sum_out2,sum_in/sum_out2);

	free_matrix_data(scan);
	matrix_close(mptr1);
	fclose(fp);
	fclose(data_fp);
}
