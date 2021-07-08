/* @(#)show_header.c	1.2 7/27/92 */
/*
 * Jun-1995 :Updated by Sibomana@topo.ucl.ac.be for ECAT 7.0 support
 * March-1999: add dosage and dose_start_time
 * July 7 1999: sex printout modified
 *
 */

#include <math.h>
#include "matrix.h"

main( argc, argv)
  int argc;
  char **argv;
{
	int matnum=0;
	char fname[256];
	MatrixFile *mptr;
	MatrixData *matrix;
FILE *fp;

	if (argc<2)
	  crash( "usage    : %s matspec\n", argv[0]);
	matspec( argv[1], fname, &matnum);
	mptr = matrix_open( fname, MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
	if (!mptr) {
		matrix_perror(fname);
		exit(1);
	}
	if (matnum==0)
	{
	  show_main_header( mptr->mhptr);
	  exit(0);
	}
	matrix = matrix_read( mptr, matnum, MAT_SUB_HEADER);
	if (!matrix)
	  crash( "%s    : matrix not found\n", argv[0]);
	switch( mptr->mhptr->file_type)
	{
	  case Sinogram    :
		show_scan_subheader( matrix->shptr);
		break;
      case Short3dSinogram :
      case Float3dSinogram :
		show_Scan3D_subheader( matrix->shptr);
		break;
	  case PetImage  :
	  case PetVolume :
	  case ByteImage :
	  case ByteVolume :
	  case InterfileImage:
		show_image_subheader( matrix);
		break;
	  case AttenCor    :
		show_attn_subheader( matrix->shptr);
		break;
	  case Normalization :
		show_norm_subheader( matrix->shptr);
		break;
	  case Norm3d :
		show_norm3d_subheader( matrix->shptr);
		break;
	  default    :
		crash("%s    : unknown matrix file type (%d)\n", argv[0], mptr->mhptr->file_type);
	}
	exit(0);
}

static char *ftypes[]= {
	"Scan", "Image", "Atten", "Norm", "PolarMap",
	"ByteVolume", "PetVolume", "ByteProjection", "PetProjection", "ByteImage",
	"Short3dSinogram", "Byte3dSinogram", "Norm3d", "Float3dSinogram","Interfile" };

static char *dtypes[] = {
	"UnknownMatDataType", "ByteData", "VAX_Ix2", "VAX_Ix4",
    "VAX_Rx4", "IeeeFloat", "SunShort", "SunLong" };

static char *applied_proc[] = {
	"Norm", "Atten_Meas", "Atten_Calc",
    "Smooth_X", "Smooth_Y", "Smooth_Z", "Scat2d", "Scat3d",
    "ArcPrc", "DecayPrc", "OnCompPrc", "Random","","","",""};

static char* storage_order(idx)
int idx;
{
	switch (idx)
	{
	case 0: return "View mode";
	case 1: return "Sinogram mode";
	}
	return "Unknown mode";
}
	
show_main_header( mh)
  Main_header *mh;
{
	int ft, i;
	char *ftstr, tod[256];
	time_t t = mh->scan_start_time;

	printf("Matrix Main Header\n");
	printf("Original File Name          : %-32s\n", mh->original_file_name);
	printf("Software Version            : %d\n", mh->sw_version);
/*		moved in individual matrices
	printf("Data type                   : %d\n", mh->data_type);
*/
	printf("System type                 : %d\n", mh->system_type);
	ft = mh->file_type;
	ftstr = "UNKNOWN";
	if (ft>0 && ft<=NumDataSetTypes) ftstr = ftypes[ft-1];
	printf("File type                   : %d (%s)\n", ft, ftstr);
	printf("Node Id                     : %-10s\n", mh->serial_number);
	printf("Scan TOD                    : %s\n",ctime(&t));
/*
	day_time( tod, mh->scan_start_day, mh->scan_start_month, 
	  mh->scan_start_year, mh->scan_start_hour, mh->scan_start_minute,
	  mh->scan_start_second);
*/
	strncpy( tod, mh->isotope_code, 8);
	tod[8]='\0';
	printf("Isotope                     : %-8s\n", tod);
	printf("Isotope Half-life           : %0.5g sec.\n", mh->isotope_halflife);
	printf("Radiopharmaceutical         : %-32s\n", mh->radiopharmaceutical);
	printf("Gantry Tilt Angle           : %0.1f deg.\n", mh->gantry_tilt);
	printf("Gantry Rotation Angle       : %0.1f deg.\n", mh->gantry_rotation);
	printf("Bed Elevation               : %0.1f cm\n", mh->bed_elevation);
/*  ELIMINATED
	printf("Rotating Source Speed       : %d rpm\n", mh->rot_source_speed);
*/
	printf("Intrinsic tilt              : %4.2f degrees\n", mh->intrinsic_tilt);
	printf("Wobble Speed                : %d rpm\n", mh->wobble_speed);
	printf("Transmission Source Type    : %d\n", mh->transm_source_type);
	printf("Axial FOV Width             : %0.2f cm\n", mh->distance_scanned);
	printf("Transaxial FOV Width        : %0.2f cm\n", mh->transaxial_fov);
/*  ELIMINATED
	printf("Transaxial Sampling Mode    : %d\n", mh->transaxial_samp_mode);
*/
	printf("Coincidence Mode            : %d\n", mh->coin_samp_mode);
	printf("Axial Sampling Mode         : %d\n", mh->axial_samp_mode);
	printf("Calibration Factor          : %0.5e\n", mh->calibration_factor);
	printf("Calibration Units           : %d", mh->calibration_units);
	if (mh->calibration_units>0 && mh->calibration_units<3)
		printf(" (%s)\n",calstatus[mh->calibration_units]);
	else printf("\n");
	printf("Calibration Units Label     : %d", mh->calibration_units_label);
	if (mh->calibration_units_label>0 && mh->calibration_units_label<14)
		printf(" (%s)\n",customDisplayUnits[mh->calibration_units]);
	else printf("\n");
	printf("Compression Code            : %d\n", mh->compression_code);
	printf("Study Name                  : %-12s\n", mh->study_name);
	printf("Patient ID                  : %-16s\n", mh->patient_id);
	printf("        Name                : %-32s\n", mh->patient_name);
	if (mh->patient_sex[0]==0) 
		printf("        Sex                 : M\n");
	else if (mh->patient_sex[0]==1) 
		printf("        Sex                 : F\n");
	else  printf("        Sex                 : U\n");
	printf("        Age                 : %g\n", mh->patient_age);
	printf("        Height              : %g\n", mh->patient_height);
	printf("        Weight              : %g\n", mh->patient_weight);
	printf("        Dexterity           : %c\n", mh->patient_dexterity[0]);
	printf("Physician Name              : %-32s\n", mh->physician_name);
	printf("Operator Name               : %-32s\n", mh->operator_name);
	printf("Study Description           : %-32s\n", mh->study_description);
	printf("Acquisition Type            : %d\n", mh->acquisition_type);
/*	ELIMINATED
	printf("Bed Type                    : %d\n", mh->bed_type);
	printf("Septa Type                  : %d\n", mh->septa_type);
*/
	printf("Facility Name               : %-20s\n", mh->facility_name);
	printf("Number of Planes            : %d\n", mh->num_planes);
	printf("Number of Frames            : %d\n", mh->num_frames);
	printf("Number of Gates             : %d\n", mh->num_gates);
	printf("Number of Bed Positions     : %d\n", mh->num_bed_pos);
	printf("Initial Bed Position        : %.1f cm\n", mh->init_bed_position);
	for (i=0; i<mh->num_bed_pos; i++)
	  printf("  Offset %2d       %.1f cm\n", i+1, mh->bed_offset[i]);
	printf("Plane Separation            : %0.4f cm\n", mh->plane_separation);
	printf("Bin Size                    : %0.4f cm\n",mh->bin_size); 
	printf("Scatter Lower Threshold     : %d Kev\n", mh->lwr_sctr_thres);
	printf("True Lower Threshold        : %d Kev\n", mh->lwr_true_thres);
	printf("True Upper Threshold        : %d Kev\n", mh->upr_true_thres);
	printf("Acquisition Mode            : %d \n", mh->acquisition_mode);
	printf("User Processing Code        : %-10s\n", mh->user_process_code);
	printf("Branching Fraction          : %0.4f\n", mh->branching_fraction);
	if (dosage>1000.0f)
	{ /* Assume Bq */
		printf("Injected dose               : %g Bq\n", mh->dosage);
	} else
		printf("Injected dose               : %g mCi\n", mh->dosage);
	t = mh->dose_start_time;
	if (t > 0) {
	printf("Dose start time             : %s",ctime(&t));
	printf("Delay since injection       : %d\n",mh->scan_start_time - t);
	}
	else  {
	printf("Dose start time             : 0\n");
	printf("Delay since injection       : 0\n");
	}
}

show_scan_subheader( sh)
  Scan_subheader *sh;
{
	int i,j;

	printf("Scan Matrix Sub-header\n");
	printf("Data type                   : %d(%s)\n", sh->data_type,
		dtypes[sh->data_type] );
	printf("Dimension_1 (ring elements) : %d\n", sh->num_r_elements);
	printf("Dimension_2 (angles)        : %d\n", sh->num_angles);
	printf("corrections_applied         : %d", sh->corrections_applied);
	if ((j=sh->corrections_applied) > 0) {
		printf(" ( ");
		for (i=0;i<16; i++) 
			if ((j & (1<<i)) != 0) printf("%s ", applied_proc[i]);
		printf(")");
	}
	printf("\n");
	printf("num_z_elements              : %d\n", sh->num_z_elements);
	printf("x resolution (sample distance)     : %g\n", sh->x_resolution);
	printf("y resolution                : %g\n", sh->y_resolution);
	printf("z resolution                : %g\n", sh->z_resolution);
	printf("w resolution                : %g\n", sh->w_resolution);
/*	ELIMINATED
	printf("Processing code             : %d\n", sh->processing_code);
*/
/*		moved to main header
	printf("Isotope_halflife            : %0.4g sec.\n", sh->isotope_halflife);
*/
/*	ELIMINATED
	printf("Frame duration              : %d sec.\n", sh->frame_duration_sec);
*/
	printf("Gate duration               : %d msec.\n", sh->gate_duration);
	printf("R-wave Offset               : %d msec.\n", sh->r_wave_offset);
	printf("Scale factor                : %g\n", sh->scale_factor);
	printf("Scan_min                    : %d\n", sh->scan_min);
	printf("Scan_max                    : %d\n", sh->scan_max);
	printf("Prompt Events               : %d\n", sh->prompts);
	printf("Delayed Events              : %d\n", sh->delayed);
	printf("Multiple Events             : %d\n", sh->multiples);
	printf("Net True Events             : %d\n", sh->net_trues);
	for (i=0; i<16; i++)
	  printf("Avg Singles Bucket %2d       : %0.4E / %0.4E\n", i,
		sh->cor_singles[i], sh->uncor_singles[i]);
	printf("Average Singles (C/U)       : %0.4E / %0.4E\n", sh->tot_avg_cor,
						sh->tot_avg_uncor);
	printf("Total Coincidence Rate      : %d\n", sh->total_coin_rate);
	printf("Frame Start Time            : %d msec.\n", sh->frame_start_time);
	printf("Frame Duration              : %d msec.\n", sh->frame_duration);
	printf("Loss Correction Factor      : %0.5f\n", sh->loss_correction_fctr);
}

show_Scan3D_subheader( sh)
  Scan3D_subheader *sh;
{
	int i,j;

	printf("Scan Matrix Sub-header\n");
	printf("Data type                   : %d(%s)\n",
		sh->data_type, dtypes[sh->data_type]);
	printf("Num_dimensions              : %d\n", sh->num_dimensions);
	printf("Dimension_1(num_r_elements) : %d\n", sh->num_r_elements);
	printf("Dimension_2 (num_angles)    : %d\n", sh->num_angles);
	printf("corrections_applied         : %d\n", sh->corrections_applied);
	if ((j=sh->corrections_applied) > 0) {
		printf(" ( ");
		for (i=0;i<16; i++) 
			if ((j & (1<<i)) != 0) printf("%s ", applied_proc[i]);
		printf(")");
	}
	printf("\n");
	printf("num_z_elements              : %d", sh->num_z_elements[0]);
	for (i=1; sh->num_z_elements[i] != 0; i++)
		printf ("+%d", sh->num_z_elements[i]);
	printf("\n");
	printf("ring difference             : %d\n", sh->ring_difference);
	printf("storage order               : %d ( %s ) \n", sh->storage_order,
		storage_order(sh->storage_order));
	printf("axial compression           : %d\n", sh->axial_compression);
	printf("x resolution (sample distance)     : %g\n", sh->x_resolution);
	printf("v resolution                : %g\n", sh->v_resolution);
	printf("z resolution                : %g\n", sh->z_resolution);
	printf("w resolution                : %g\n", sh->w_resolution);
	printf("Gate duration               : %d msec.\n", sh->gate_duration);
	printf("R-wave Offset               : %d msec.\n", sh->r_wave_offset);
	printf("Scale factor                : %g\n", sh->scale_factor);
	printf("Scan_min                    : %d\n", sh->scan_min);
	printf("Scan_max                    : %d\n", sh->scan_max);
	printf("Prompt Events               : %d\n", sh->prompts);
	printf("Delayed Events              : %d\n", sh->delayed);
	printf("Multiple Events             : %d\n", sh->multiples);
	printf("Net True Events             : %d\n", sh->net_trues);
	printf("Average Singles (C/U)       : %0.4E / %0.4E\n", sh->tot_avg_cor,
						sh->tot_avg_uncor);
	printf("Total Coincidence Rate      : %d\n", sh->total_coin_rate);
	printf("Frame Start Time            : %d msec.\n", sh->frame_start_time);
	printf("Frame Duration              : %d msec.\n", sh->frame_duration);
	printf("Loss Correction Factor      : %0.5f\n", sh->loss_correction_fctr);
 	printf("Uncorrected Singles        :\n");
	for (i=0; i<16; i++) 	{
		for (j=0; j<8; j++)
 			printf("\t%g",sh->uncor_singles[i*8+j]);
		printf("\n");
	}
}

show_image_subheader(matrix)
  MatrixData *matrix;
{
	char tod[256], *fcode;
  	Image_subheader *ih;
	static char *fcodes[]={
		"None", "Ramp", "Butterworth", "Hanning", "Hamming", "Parzen",
		"Shepp", "Exponential"};
	int i,j,f, x0, y0, z0;

	ih = (Image_subheader*)matrix->shptr;
	x0 = (int)(0.5+matrix->x_origin/matrix->pixel_size);
	y0 = (int)(0.5+matrix->y_origin/matrix->y_size);
	z0 = (int)(0.5+matrix->z_origin/matrix->z_size);
	printf("Image Matrix Sub-header\n");
	printf("Data type                   : %d(%s)\n",
		ih->data_type, dtypes[ih->data_type]);
	printf("Num_dimensions              : %d\n", ih->num_dimensions);
	printf("Dimension_1                 : %d\n", ih->x_dimension);
	printf("Dimension_2                 : %d\n", ih->y_dimension);
	printf("Dimension_3                 : %d\n", ih->z_dimension);
	if (x0 > 1 || y0>1 || z0>1) {
		printf("X Origin                    : %d\n", x0);
		printf("Y Origin                    : %d\n", y0);
		printf("Z Origin                    : %d\n", z0);
	}
	printf("X Offset                    : %0.3f cm\n", ih->x_offset);
	printf("Y Offset                    : %0.3f cm\n", ih->y_offset);
	printf("Zoom (recon_scale)          : %0.3f\n", ih->recon_zoom);
	printf("Quant_scale                 : %g\n", ih->scale_factor);
/*	ELIMINATED
	printf("Quant_units                 : %d\n", ih->quant_units);
*/
	printf("Processing Code             : %d", ih->processing_code);
	if ((j=ih->processing_code) > 0) {
		printf(" ( ");
		for (i=0;i<16; i++) 
			if ((j & (1<<i)) != 0) printf("%s ", applied_proc[i]);
		printf(")");
	}
	printf("\n");
	printf("Image_min                   : %d\n", ih->image_min);
	printf("Image_max                   : %d\n", ih->image_max);
	printf("X Pixel Size                : %0.4f mm\n",10.0*ih->x_pixel_size);
	printf("Y Pixel Size                : %0.4f mm\n",10.0*ih->y_pixel_size);
	printf("Z Pixel Size                : %0.4f mm\n",10.0*ih->z_pixel_size);
/*	moved to main header
	printf("Slice width                 : %0.4f cm\n", ih->slice_width);
*/
	printf("Frame Duration              : %d msec\n", ih->frame_duration);
	printf("Frame Start Time            : %d msec\n", ih->frame_start_time);
/*	ELIMINATED
	printf("Slice Location              : %d\n", ih->slice_location);
	day_time( tod, ih->recon_start_day, ih->recon_start_month,
		ih->recon_start_year, ih->recon_start_hour,
		ih->recon_start_minute, ih->recon_start_sec);
	printf("Reconstruction TOD          : %s\n", tod);
*/
	fcode = "UNKNOWN";
	f=ih->filter_code;
	if (f<0) f=-f;
	if (f<8) fcode=fcodes[f];
	printf("Filter Type                 : %d (%s)\n", ih->filter_code, fcode);
	printf("Filter Parameters           : %f,0.0,%f,%f,%f,%f\n",
	  ih->filter_cutoff_frequency, ih->filter_ramp_slope, ih->filter_order,
	  ih->filter_scatter_fraction, ih->filter_scatter_fraction);
	printf("Image Rotation Angle        : %0.2f degrees\n", ih->z_rotation_angle);
/*	ELIMINATED
	printf("Plane Efficiency Factor     : %0.5f\n", ih->plane_eff_corr_fctr);
*/
	printf("Decay Correction Factor     : %0.5f\n", ih->decay_corr_fctr);
/*	MOVED TO MAIN HEADER
	printf("Loss Correction Factor      : %0.5f\n", ih->loss_corr_fctr);
	printf("ECAT Calibration Factor     : %0.5e\n", ih->ecat_calibration_fctr);
	printf("Well Counter Calib Fctr     : %0.5e\n", ih->well_counter_cal_fctr);
*/
	printf("Annotation                  : %-40s\n", ih->annotation);
}

day_time( str, day, month, year, hour, minute, sec)
  char *str;
  int day, month, year, hour, minute, sec;
{
	static char *months="JanFebMarAprMayJunJulAugSepOctNovDec";
	char mstr[4];

	if (day<1) day=1;
	if (day>31) day=31;
	if (month<1) month=1;
	if (month>12) month=12;
	if (year<1900) year+=1900;
	if (year<1970) year=1970;
	if (year>9999) year=9999;
	if (hour<0) hour=0;
	if (hour>23) hour=23;
	if (minute<0) minute=0;
	if (minute>59) minute=59;
	if (sec<0) sec=0;
	if (sec>59) sec=59;
	strncpy( mstr, months+(month-1)*3, 3);
	sprintf( str, "%02d-%s-%4d %02d:%02d:%02d", day, mstr, year, 
		hour, minute, sec);
}
show_attn_subheader( ah)
  Attn_subheader *ah;
{
	int i=0;
	printf("Attenuation Matrix Sub-header\n");
	printf("Data type                   : %d(%s)\n",
		ah->data_type, dtypes[ah->data_type]);
	printf("Attenuation_type            : %d\n", ah->attenuation_type);
	printf("storage order               : %d ( %s )\n", ah->storage_order,
		storage_order(ah->storage_order));
	printf("Dimension_1                 : %d\n", ah->num_r_elements);
	printf("Dimension_2                 : %d\n", ah->num_angles);
	printf("ring_difference             : %d\n",ah->ring_difference);
	printf("span                        : %d\n",ah->span);
	printf("z_elements                  : %d", ah->z_elements[0]);
	for (i=1; ah->z_elements[i] != 0; i++)
		printf ("+%d", ah->z_elements[i]);
	printf("\n");
	printf("Scale Factor                : %g\n", ah->scale_factor);
	printf("x resolution                : %g\n", ah->x_resolution);
	printf("y resolution                : %g\n", ah->y_resolution);
	printf("z resolution                : %g\n", ah->z_resolution);
	printf("w resolution                : %g\n", ah->w_resolution);
	printf("X origin                    : %0.4f cm\n", ah->x_offset);
	printf("Y origin                    : %0.4f cm\n", ah->y_offset);
	printf("X radius                    : %0.4f cm\n", ah->x_radius);
	printf("Y radius                    : %0.4f cm\n", ah->y_radius);
	printf("Tilt Angle                  : %0.2f deg\n", ah->tilt_angle);
	printf("Attenuation Coefficient     : %0.3f 1/cm\n", ah->attenuation_coeff);
/*	ELIMINATED
	printf("Sample Distance             : %0.5f cm\n", ah->sample_distance);
*/
}
show_norm_subheader( nh)
  Norm_subheader *nh;
{
	char tod[256];

	printf("Normalization Matrix Sub-header\n");
	printf("Data type                   : %d(%s)\n",
		nh->data_type, dtypes[nh->data_type]);
	printf("Num_dimensions              : %d\n", nh->num_dimensions);
	printf("Dimension_1(num_r_elements) : %d\n", nh->num_r_elements);
	printf("Dimension_2 (num_angles)    : %d\n", nh->num_angles);
	printf("num_z_elements              : %d\n", nh->num_z_elements);
	printf("ring difference             : %d\n", nh->ring_difference);
	printf("Scale Factor                : %g\n", nh->scale_factor);
/*	MOVED TO MAIN HEADER
	day_time( tod, nh->norm_day, nh->norm_month, nh->norm_year,
		nh->norm_hour, nh->norm_minute, nh->norm_second);
	printf("Normalization TOD           : %s\n", tod);
*/
	printf("FOV Source Width            : %0.4f cm\n", nh->fov_source_width);
	printf("storage order               : %d ( %s )\n", nh->storage_order,
		storage_order(nh->storage_order));
	printf("Norm_min                    : %d\n", nh->norm_min);
	printf("Norm_max                    : %d\n", nh->norm_max);
}

show_norm3d_subheader( nh)
  Norm3D_subheader *nh;
{
	int i;
	printf("Normalization Matrix Sub-header\n");
	printf("Data type                   : %d(%s)\n",
		nh->data_type, dtypes[nh->data_type]);
	printf("number of radial elements   : %d\n", nh->num_r_elements);
	printf("number of transaxial crystals  : %d\n",nh->num_transaxial_crystals);
	printf("number of crystal rings        : %d\n", nh->num_crystal_rings);
	printf("number of crystals per ring    : %d\n", nh->crystals_per_ring);
	printf("number of plane geometric corrections : %d\n",
		nh->num_geo_corr_planes);
	printf("Crystal deadtime correction factors\n");
	for (i=0; i<8 && fabs(nh->crystal_dtcor[i]) > 0.0; i++)
			printf("%g\n",nh->crystal_dtcor[i]);
	printf("Ring deadtime correction factors\n");
	for (i=0; i<32 && fabs(nh->ring_dtcor1[i]) > 0.0; i++)
			printf("%g\t%g\n",nh->ring_dtcor1[i],nh->ring_dtcor2[i]);
	
}

