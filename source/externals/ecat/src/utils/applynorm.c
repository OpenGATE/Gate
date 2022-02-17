/* 
	Implemented in C by Sibomana@topo.ucl.ac.be
	from IDL program applynorm.pro (Copyright (c) 1994 CTI PET Systems, Inc.)
*/
 
#include "matrix.h"
#include "ecat_model.h"
#include <math.h>

#define RMAX 32	/* Number of crystal rings supported */
#define AXIAL_CRYSTAL_RINGS 32
#define TRANSAXIAL_CRYSTALS 8

#define BEG_CRYSTAL	36
#define END_CRYSTAL	52
#define ROTATION_OFFSET 44
#define CRYS_A_OFFSET 	ROTATION_OFFSET
#define CRYS_B_OFFSET	192 + ROTATION_OFFSET

/*
 * These macros compute the crystal pair associated with a particular
 * view and element. For oblique projections det1 is always associated
 * with ring_a while det2 is associated with ring_b.
 */
#define ve_to_det1(e,v,nv) ( (v + (e >> 1) + 2*nv) % (2*nv) )
#define ve_to_det2(e,v,nv) ( (v - ((e+1) >> 1) +3*nv) % (2*nv) )


#define TIMES_FILENAME "times.dat"

int  verbose = 0 ;
int number_of_elements;		/* Number of radial elements in a projection */
int number_of_views;		/* Number of unmashed polar angles in a sinogram */
int mash;					/* Number of angles added together. 1, 2, 4, ... */
int transaxial_crystals;	/* Number of transaxial crystals in a block */
int axial_crystals;			/* Number of axial crystals in a block */
int block_rings;			/* Number of rings of blocks */
int crystal_rings;			/* Number of rings of crystals */
int crystals_per_ring;		/* Number of rings of crystals */
int span;			 		/* ring differences to accept in a projection */
int max_acceptance;			/* Maximum ring difference to accept */
int blocks_per_bucket;		/* Number of blocks in a bucket */
int crystals_per_bucket;	/* Number of transaxial crystals in a bucket */
int buckets_per_ring;		/* Number of buckets in a ring */
int septa_in;				/* One if septa is in field of view, 0 if out */
 
int *ringnums;
/* Points to an array of number_of_planes by span+2 containing
 * the number of ring pairs and the ring pairs for each plane. */

float *dtime_efficiency;
/* Points to a list of deadtime corrected crystal efficiencies.
 * One entry per crystal. This gets loaded for each frame. */
					 
float *crys_efficiency;
/* Points to a list of crystal efficiencies. One entry per
 * crystal. Gets loaded at initalization. */
					 
float *crystal_interference;
/* Points to a transaxial_crystals by number_of_elements array that
 * contains the crystal interference efficiencies. */
					 
float *geometric;
/* Points to a number_of_elements by number_of_planes array that 
 * contains the geometric efficiency for each plane. If septa is in
 * the field of view, this arrray will have a row for each plane.
 * If septa is out, this array will have only one row. */
					 
float *axial_t1;
/* Contains the paralyzing deadtimes for each axial crystal */
float *axial_t2;
/* Contains the non-paralyzing deadtimes for each axial crystal */

float *trans_t1;
/* Contains the non-paralyzing deadtimes for each transaxial
 * crystal in a block. */

 
/*
 * make_ring_numbers returns a pointer to an array containing the ring pairs for each
 * plane. The array is ringnums[nplanes][span+2]. The first entry is number of pairs followed
 * by the ring_a, ring_b pairs for each lor in the plane. If no memory is available, NULL is returned.
 */
#include <malloc.h>
#include <stdio.h>


int *make_ring_numbers(rmax,span,amax)

	int	rmax;		/* Number of crystal rings. (May be virtual) */
	int	span;		/* Sum of the number of cross planes in odd and even
			 	 * planes. */
	int	amax;		/* Maximum number of cross planes to accept
			 	 * for all projections. (Acceptance angle) */
{
	int	dmax;		/* Maximum number of cross planes in a projection */
	int	nmax;		/* Maximum angular projection index. */
	int	zindex[ RMAX ];	/* The index of the begining of the
				 * positive projection.	*/
	int	zsize[ RMAX ];	/* The z offset to the begining of the projection. */
	int	nvdp;		/* The number of virtual direct planes */
	int	i;
	int	r1,r2;		/* Ring numbers. */
	int	nabs;		/* Projection index. */
	int	plane;		/* The virtual plane number */
	int	*ringnums;	/* The array to hold the ring numbers */
	int	*plnptr;	/* points into a row */
	int	nplanes;	/* The number of planes */
	
	/* Compute plane offsets at the begining of the projections and
	 * the size of the projections.	*/
	
	span = (span < 3) ? 3 : span;
	dmax = span/2+1;
	amax = (amax < rmax) ? amax : rmax-1;
	nmax = (amax+1)/span;
	nvdp = 2*rmax-1;
	
	/* Get the number of planes and make the array */
	
	nplanes = number_of_planes(rmax,span,amax);
	ringnums = (int *) malloc( sizeof(int)*nplanes*(span+2) );
	if( ringnums == NULL )
	{
		printf(" ERROR: make_ring_nums: Not enough memory for ringnums array.\n");
		return 0;
	}
	for( i=0; i< nplanes*(span+2); i++)
		ringnums[i]=0;
	
	/* Compute plane offsets at the begining of the projections and
	 * the size of the projections.	*/

	zsize[0] = 0;
	zindex[0] = 0;
	zsize[1] = dmax;
	zindex[1] = nvdp;
	for( i=2; i <= nmax; i++)
	{
		zindex[i] = zindex[i-1]+2*(nvdp-2*zsize[i-1]);
		zsize[i] = (2*i-1)*dmax-i+1;
	}

	/* Compute the plane number for each pair of crystals 
	 * and stuff in the ringnums array. */

	for (r1=0; r1 < rmax; r1++)
		for(r2=0; r2 < rmax; r2++)
		{
			nabs = ( abs(r1-r2) + dmax - 1 ) / span;
			if (nabs <= nmax)
			{
				plane = (r1 + r2) - zsize[nabs];
				plane += zindex[nabs];
				if ( ((r1 - r2) < 0) && (nabs > 0) )
					plane += nvdp-2*zsize[nabs];
				if( plane >= nplanes)
				{
					printf(" Plane %d is larger than %d\n", plane,nplanes);
					return 0;
				}
				plnptr = &ringnums[plane*(span+2)];
				i=plnptr[0];
				plnptr[2*i+1]=r1;
				plnptr[2*i+2]=r2;
				plnptr[0]++;
			}
		}
	return ringnums;
}

int number_of_planes(rmax,span,amax)
	int	rmax;	/* Number of crystal rings. (May be virtual) */
	int	span;	/* Sum of the number of cross planes in odd and even
			 * planes. */
	int	amax;	/* Maximum number of cross planes to accept */
{
	int	nvdp;	/* The number of virtual direct planes */
	int	nplanes;/* Collects the number of planes */  
	int	i;
	
	span = (span < 3) ? 3 : span;
	amax = (amax < rmax) ? amax : rmax-1;
	nvdp = (2*rmax-1);
	nplanes = nvdp;
	for (i=1; i <= (amax+1)/span; i++)
		nplanes = nplanes + 2*( nvdp -( (2*i-1)*span + 1 ));
	return nplanes;
}

correct_for_dtime(projection,zoffset,zsize,mashed_view)
	float *projection;
	int zoffset;
	int zsize;
	int mashed_view;
	
{
	int crys_a;
	int crys_b;
	int elem;
	int view;
	int *zptr;	
	int z;
	int npairs;
	int pair;
	float lor_efficiency;
	float view_efficiency;
	float *geoptr;
	float *interferptr;
	int start_view;
	int end_view;
	
	start_view = mashed_view * mash;
	end_view = start_view + mash;
	
	zptr = &ringnums[zoffset*(span+2)];

	for( z=zoffset; z < zoffset+zsize; z++)
	{
		npairs = zptr[0];
		if (septa_in)
			geoptr = &geometric[z*number_of_elements];
		else
			geoptr = geometric;

		interferptr = crystal_interference;
		for( elem=-number_of_elements/2; elem < number_of_elements/2; elem++)
		{
			view_efficiency = 0.;
			for( view = start_view; view < end_view; view++)
			{
				lor_efficiency = 0.;
				crys_a=ve_to_det1(elem,view,number_of_views);
				crys_b=ve_to_det2(elem,view,number_of_views);
				for( pair = 1; pair <= npairs*2; pair += 2)
					lor_efficiency += dtime_efficiency[crys_a+crystals_per_ring*zptr[pair]]
						*dtime_efficiency[crys_b+crystals_per_ring*zptr[pair+1]];
				view_efficiency += lor_efficiency*interferptr[view % transaxial_crystals];
			}
			*projection++ /= (view_efficiency*( *geoptr++));
			interferptr += transaxial_crystals;
		}
		zptr += (span+2);
	}
	return 1;
}
								 
/* Called to initalize the efficiencies for each crystal. Allocates memory if none is allocated.
 * Routine needs the crystal deadtimes initalized by init_deadtime(). Returns a 1 for success,
 * a zero if memory could not be allocated.
 */

load_deadtime_array(singles)
	float *singles;
{
	int ring;
	int crystal;
	int bucket;
	float rate;
	int gantry_crystal;
	
	if (dtime_efficiency == NULL)
	{
		dtime_efficiency = (float *) malloc( sizeof( float) * crystal_rings * crystals_per_ring);
		if (dtime_efficiency == NULL)
			return 0;
	}
	for (ring = 0; ring < crystal_rings; ring++)
		for (crystal=0; crystal < crystals_per_ring; crystal++)
		{
			bucket = crystal/crystals_per_bucket + buckets_per_ring*(ring/axial_crystals);
			rate = singles[bucket] / (float) blocks_per_bucket;
			gantry_crystal = ring*crystals_per_ring+crystal;
			dtime_efficiency[gantry_crystal] = crys_efficiency[gantry_crystal]
				* exp( -axial_t1[ring]*rate/(1.+axial_t2[ring]*rate) )
				/( (1.+axial_t2[ring]*rate)*(1.+trans_t1[crystal % transaxial_crystals]*rate) );
		}
	return 1;
}

		
init_deadtime()
{
	int i;
	
	if (verbose) {
		printf("number_of_elements:\t%d\n",number_of_elements);
		printf("number_of_views:\t%d\n",number_of_views);
		printf("mash:\t%d\n",mash);
		printf("transaxial_crystals:\t%d\n",transaxial_crystals);
		printf("axial_crystals:\t%d\n",axial_crystals);
		printf("crystal_rings:\t%d\n",crystal_rings);
		printf("block_rings:\t%d\n",block_rings);
		printf("crystals_per_ring:\t%d\n",crystals_per_ring);
		printf("span:\t%d\n",span);	 
		printf("max_acceptance:\t%d\n",max_acceptance);
		printf("blocks_per_bucket:\t%d\n",blocks_per_bucket);
		printf("crystals_per_bucket:\t%d\n",crystals_per_bucket);
		printf("buckets_per_ring:\t%d\n",buckets_per_ring);
	}
	
	ringnums = make_ring_numbers( crystal_rings, span, max_acceptance );
	if (ringnums == NULL) return 0;
	return 1;
}

void prtringnum()
{
	int pair;
	int z;
	int *zptr;
	int npairs;
	for (z=0; z<(2*crystal_rings-1); z++)
	{
		zptr = &ringnums[z*(span+2)];
		npairs = zptr[0];
		printf("%d  %d  ",z,npairs);
		for(pair=1; pair<(npairs*2); pair += 2)
			printf("(%3d,%3d) ",zptr[pair],zptr[pair+1]);
		printf("\n");
	}
}

main(argc, argv)
int argc;
char **argv;
{
	int i, size, nelems, nviews, nplns, ncrystals;
	int iseg=0, seg, nsegs=0, view;
	int c, matnum, offset, zoffset, model;
	int attn_offset, blkno, nblks;
	char file_name[256], *scan_spec=NULL, *norm_spec=NULL, *attn_spec=NULL;
	MatrixFile *scan_mf=NULL, *norm_mf=NULL, *attn_mf=NULL;
	Main_header *scan_mh=NULL, *norm_mh=NULL, *attn_mh=NULL;
	Scan3D_subheader *scan_sh=NULL;
	Norm3D_subheader *norm_sh=NULL;
	Attn_subheader ah;
	Main_header mh;
	struct MatDir matdir, dir_entry;
	EcatModel *model_info;
	MatrixData *norm_matrix;
	float *proj, *norm, *all_1, fval;
	short *scan_data;
	caddr_t scan_buf;
	extern char *optarg;

	 while ((c = getopt (argc, argv, "i:o:n:v")) != EOF) {
		switch (c) {
		case 'i':
			scan_spec = optarg;
			break;
		case 'o':
			attn_spec = optarg;
			break;
		case 'n' :
			norm_spec = optarg;
			break;
		case 'v' :
			verbose = 1;
			break;
		}
	}
			
	if (!scan_spec || !norm_spec)
		crash("usage: applynorm -i scan_matrix -n norm_file -o atten_file\n");
	matspec( norm_spec, file_name, &matnum);
	norm_mf = matrix_open(file_name,MAT_READ_ONLY,Norm3d);
	if (!norm_mf)
		crash("can't open normalization file %s\n", file_name);
	norm_mh = norm_mf->mhptr;
	if (!matspec( scan_spec, file_name, &matnum))
		matnum = mat_numcod(1,1,1,0,0);
	scan_mf = matrix_open(file_name,MAT_READ_ONLY,Short3dSinogram);
	if (!scan_mf) 
		crash("can't open scan file %s\n", file_name); 
	scan_mh = scan_mf->mhptr;
	
	scan_sh = (Scan3D_subheader*)calloc(1,sizeof(Scan3D_subheader));
	if (matrix_find(scan_mf,matnum,&matdir) < 0)
		crash("scan matrix %s not found\n",argv[1]);
	mat_read_Scan3D_subheader(scan_mf->fptr, scan_mf->mhptr,
		matdir.strtblk, scan_sh) ;
	offset = (matdir.strtblk+1)*MatBLKSIZE;
	if (scan_sh->storage_order != 0) 
		crash("unsupported storage order\n");
	septa_in = (scan_mh->septa_state == 0) ? 1 : 0;
	nelems = scan_sh->num_r_elements;
	nviews = scan_sh->num_angles;
	nplns = scan_sh->num_z_elements[iseg];
	for (iseg=0; scan_sh->num_z_elements[iseg] != 0; iseg++) nsegs++;
	printf("Input sinogram size is %d x %d x %d x %d\n", nelems, nviews,
		nsegs*2-1,nplns);
	printf("sinogram acquisition mode is %s\n", septa_in? "2D" : "3D");
	model = scan_mh->system_type;
	if ( (model_info = ecat_model(model)) == NULL)
		crash("unkown model type %d\n",model);

	norm_matrix =  matrix_read(norm_mf,mat_numcod(1,1,1,0,0),GENERIC);
	if (!norm_matrix) 
		crash("norm matrix %s,1,1,1,0,0 not found\n",norm_mf->fname);
	norm_sh = (Norm3D_subheader *)norm_matrix->shptr;

	mash = 1;
	for (i=1; i<=scan_mh->angular_compression; i++) mash *= 2;
	number_of_elements = nelems;
	number_of_views = nviews*mash;
	transaxial_crystals = model_info->angularCrystalsPerBlock;
	axial_crystals = model_info->axialCrystalsPerBlock;
	block_rings = model_info->rings*model_info->axialBlocksPerBucket;
	crystal_rings = block_rings * axial_crystals;
	crystals_per_ring =  2*number_of_views;  /* Does not support wobble */
	span = scan_sh->axial_compression;
	max_acceptance = scan_sh->ring_difference;
	blocks_per_bucket = model_info->transBlocksPerBucket;
	crystals_per_bucket = blocks_per_bucket*transaxial_crystals;
	buckets_per_ring = crystals_per_ring / crystals_per_bucket;
	init_deadtime();
	geometric = (float*)norm_matrix->data_ptr;
	crystal_interference = geometric + norm_sh->num_geo_corr_planes*nelems;
	crys_efficiency = crystal_interference + 7*nelems;
	ncrystals = norm_sh->num_crystal_rings*norm_sh->crystals_per_ring;
	axial_t1 = norm_sh->ring_dtcor1;
	axial_t2  =  norm_sh->ring_dtcor2;
	trans_t1 = norm_sh->crystal_dtcor;
	if ((load_deadtime_array(scan_sh->uncor_singles)) != 1)
		crash("Error...in ldarray");
	zoffset = 0;
	offset = (matdir.strtblk+1)*MatBLKSIZE;
	size = nelems*nplns;
	proj = (float*)malloc(size*sizeof(float));
	norm = (float*)malloc(size*sizeof(float));
	scan_data = (short*)malloc(size*sizeof(short));
	scan_buf = malloc(size*sizeof(float));

/* create atten file */
	memcpy(&mh, scan_mf->mhptr, sizeof(Main_header));
	mh.file_type = AttenCor;
    mh.num_frames = mh.num_planes = mh.num_gates = 1;
	matspec( attn_spec, file_name, &matnum);
    attn_mf = matrix_create(file_name, MAT_OPEN_EXISTING, &mh);
	memset(&ah, 0, sizeof(Attn_subheader));
    ah.num_dimensions = 3;
    ah.num_r_elements = nelems;
    ah.num_angles = nviews;
    ah.data_type = IeeeFloat;
    ah.x_resolution = scan_sh->x_resolution;
    ah.span = model_info->def2DSpan;
    ah.ring_difference = model_info->def2DSpan/2;
    ah.scale_factor = 1.0;
	nblks = (nelems*nviews*nplns*sizeof(float)+511)/512;
	memcpy(ah.z_elements,scan_sh->num_z_elements,64*sizeof(short));
    ah.num_z_elements = ah.z_elements[0];
    if (matrix_find(attn_mf, matnum, &matdir) == -1) {
        blkno = mat_enter(attn_mf->fptr, attn_mf->mhptr, matnum, nblks) ;
        dir_entry.matnum = matnum ;
        dir_entry.strtblk = blkno ;
        dir_entry.endblk = dir_entry.strtblk + nblks - 1 ;
        dir_entry.matstat = 1 ;
        insert_mdir(dir_entry, attn_mf->dirlist) ;
        matdir = dir_entry ;
    } else {
        fprintf(stderr,"\7warning : existing matrix overwritten\n");
        blkno = matdir.strtblk;
    }
	attn_offset = matdir.strtblk*MatBLKSIZE;
    mat_write_attn_subheader(attn_mf->fptr,attn_mf->mhptr,matdir.strtblk,&ah);
	attn_offset = blkno*MatBLKSIZE;
    all_1 = (float*)malloc(MatBLKSIZE);
    for (i=0; i<128; i++) all_1[i] = 1.0;
	if (ntohs(1) != 1) {
		swab((char*)all_1,(char*)scan_buf,MatBLKSIZE);
		swaw((short*)scan_buf,(short*)all_1,MatBLKSIZE/2);
	}
	if (fseek( attn_mf->fptr, attn_offset, 0) == EOF)
		crash("%s : fwrite error\n", attn_mf->fname);
	for (i=0; i<nblks; i++) 
		if (fwrite( all_1, sizeof(float), 128, attn_mf->fptr) != 128)
			crash("%s : fwrite error\n", attn_mf->fname);
	free(all_1);

	for (iseg=1; iseg < 2*nsegs; iseg++) {
		seg = iseg/2;
		if( (iseg % 2) != 0 ) seg = -seg;
		printf("Normalizing segment %d z offset = %d file offset = %d\n",
			seg,zoffset, offset);
/*
WHY DIVISION BY 2: USE NEXT LINE
		nplns = scan_sh->num_z_elements[abs(seg)]/2;
*/
		nplns = scan_sh->num_z_elements[abs(seg)];
		for (i=0; i<size; i++) norm[i] = 1;

/*
 * norm = norm/((fltarr(nelems)+1)#[.92,.96,fltarr(nplns-4)+1,.96,.92])
 IDL intruction translated as next 6 lines
 */
		for (i=0; i<nelems; i++) {
			norm[i] /= .92;			/* plane 0 */
			norm[nelems+i] /= .96;	/* plane 1 */
			norm[(nplns-2)*nelems+i] /= .96; 	/* plane nplns - 1 */
			norm[(nplns-1)*nelems+i] /= .92; 	/* plane nplns */
		}
		for (view=0; view<nviews; view++) {
			if (fseek(scan_mf->fptr,offset,0) != 0) 
				crash ("view %d : read positioning error\n", view+1);
			if (fread(scan_buf,sizeof(short),size,scan_mf->fptr) != size)
				crash("view %d : error reading view\n",view+1);
			if (ntohs(1) != 1) swab(scan_buf,scan_data,size*sizeof(short));
			else memcpy(scan_data,scan_buf,size*sizeof(short));
			for (i=0; i<size; i++) proj[i] = norm[i]*scan_data[i];
			correct_for_dtime(proj,zoffset,nplns,view);

			if (ntohs(1) != 1) {
				swab((char*)proj,scan_buf,size*sizeof(float));
				swaw((short*)scan_buf,(short*)proj,size*2);
			}
			if (fseek(attn_mf->fptr,attn_offset,0) != 0) 
				crash ("view %d : write positioning error\n", view+1);
			if (fwrite(proj,sizeof(float),size,attn_mf->fptr) != size)
				crash("view %d : error writing view\n",view+1);
			offset = offset+size*sizeof(short);
			attn_offset = attn_offset+size*sizeof(float);
		}
		zoffset = zoffset + nplns;
	}
	matrix_close(scan_mf);
	matrix_close(norm_mf);
	matrix_close(attn_mf);
}
