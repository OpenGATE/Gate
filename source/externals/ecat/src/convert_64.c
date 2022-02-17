/*
 Modification history:
	02-oct-02: Added modification from Dr. Harald Fricke <HFricke@hdz-nrw.de>
*/
#include "matrix.h"
#include "matrix_64.h"
#include <math.h>
#include <string.h>
#include <time.h>
#define ELIMINATED  0
#define DEFMATNUM 0x1010001 /* added by H.F. */


static int mh64_data_type = SunShort;
static char *olddisplayunits[NumOldUnits] =
    {"Total counts", "Unknown Units", "ECAT counts/sec", "uCi/cc", "LMRGlu",
    "LMRGlu umol/min/100g", "LMRGlu mg/min/100g", "nCi/ml", "Well counts",
    "Bq/cc", "ml/min/100g", "ml/min/g"};

static int r_elements(system_type)
int system_type;
{
	switch (system_type) {
	case 953 :
	default:
		return 160;
	case 921 :
	case 922 : /* ACS II Exact added by H.F. */
	case 925 :
	case 951 :
		return 192;
	case 961:
		return 336;
	case 962:
		return 288;
	}
}

static int num_angles(system_type)
int system_type;
{
	switch (system_type) {
	case 921 :
	case 922 : /* ACS II Exact added by H.F. */
	case 925 :
	case 953 :
	default:
		return 192;
	case 951 :
		return 256;
	case 962:
		return 288;
	case 961:
		return 392;
	}
}

static float bin_size(system_type)
int system_type;
{
	switch (system_type) {
	case 921 :
	case 922 : /* ACS II Exact added by H.F. */
	case 925 :
	default:
		return 0.3375;
	case 961:
		return 0.165;
	case 962:
		return 0.225;
	case 951 :
	case 953 :
		return 0.312932;
	}
}
static float intrinsic_tilt(system_type)
int system_type;
{
    switch (system_type) {
    case 921 :
    case 922 : /* ACS II Exact added by H.F. */
    case 925 :
        return 15.0;
    case 961:
        return 13.0;
    case 962:
    case 951 :
    case 953 :
    default:
        return 0.0;
    }
}

void mh64_convert(h_64, h)
Main_header_64 *h_64;
Main_header * h;
{
	char	ustr1[33], ustr2[33], *ptr;
	int	i;
	struct tm strtTime;
	time_t t = h->scan_start_time;
	memset(h_64,0,sizeof(Main_header_64));  /* clear memory added by H.F. */
	strncpy(h_64->original_file_name, h->original_file_name,19);
	h_64->original_file_name[19] = '\0';	/* truncated to 19 cc */
	h_64->sw_version = 6;
	h_64->data_type = mh64_data_type;	/* use matrix data_type */
    h_64->system_type = h->system_type;
	switch(h->file_type) {
		case PetVolume:
		case ByteVolume:
    		h_64->file_type = PetImage;
			break;
		case Short3dSinogram:
			h_64->file_type = Sinogram;
			break;
		default:
			 h_64->file_type = h->file_type;
	}
    strcpy(h_64->node_id, h->serial_number);
	strtTime = *localtime(&t);
	h_64->scan_start_day = strtTime.tm_mday;
	h_64->scan_start_month = strtTime.tm_mon+1;
	h_64->scan_start_year = 1900+strtTime.tm_year;
	h_64->scan_start_hour = strtTime.tm_hour;
	h_64->scan_start_minute = strtTime.tm_min;
	h_64->scan_start_second = strtTime.tm_sec;
    strcpy(h_64->isotope_code, h->isotope_code);
    h_64->isotope_halflife = h->isotope_halflife;
    strcpy(h_64->radiopharmaceutical, h->radiopharmaceutical);
    h_64->gantry_tilt = h->gantry_tilt;
    h_64->gantry_rotation = h->gantry_rotation;
    h_64->bed_elevation = h->bed_elevation;
    h_64->rot_source_speed = ELIMINATED;
    h_64->wobble_speed = h->wobble_speed;
    h_64->transm_source_type = h->transm_source_type;
    h_64->axial_fov = h->distance_scanned;
    h_64->transaxial_fov = h->transaxial_fov;
	h_64->transaxial_samp_mode = ELIMINATED;
    h_64->coin_samp_mode = h->coin_samp_mode;
    h_64->axial_samp_mode = h->axial_samp_mode;
    h_64->compression_code = 0;
/*    h_64->calibration_units = h->calibration_units;*/
    switch( h->calibration_units ) {
	case Uncalibrated:
		h_64->calibration_units = EcatCountsPerSec;
		break;
	case Calibrated:
		h_64->calibration_units = BecquerelsPerCC;
		break;
	case Processed:
		h_64->calibration_units = UnknownEcfUnits;
		strcpy( ustr1, h->data_units );
		if( ptr = strstr( ustr1, "/ml" ) ) {
			*(ptr+1) = 'c';
			*(ptr+2) = 'c';
		}
		for( i = 0 ; i < NumOldUnits ; i++ ) {
			strcpy( ustr2, olddisplayunits[i] );
			if( ptr = strstr( ustr2, "/ml" ) ) {
				*(ptr+1) = 'c';
				*(ptr+2) = 'c';
			}
			if( !strcmp( ustr1, ustr2 ) ) {
				h_64->calibration_units = i;
				break;
			}
		}
		break;
	default:
		h_64->calibration_units = UnknownEcfUnits;
		break;
    }
    h_64->calibration_factor = h->calibration_factor;
    strcpy(h_64->study_name, h->study_name);
    strcpy(h_64->patient_id, h->patient_id);
    strcpy(h_64->patient_name, h->patient_name);
	h_64->patient_sex = h->patient_sex[0];
    h_64->patient_dexterity = h->patient_dexterity[0];
    if (h->patient_age) sprintf(h_64->patient_age,"%g",h->patient_age);
    if (h->patient_height) sprintf(h_64->patient_height,"%g",h->patient_height);
    if (h->patient_weight) sprintf(h_64->patient_weight,"%g",h->patient_weight);
    strcpy(h_64->physician_name, h->physician_name);
    strcpy(h_64->operator_name, h->operator_name);
    strcpy(h_64->study_description, h->study_description);
    h_64->acquisition_type = h->acquisition_type;
	h_64->bed_type = h_64->septa_type = ELIMINATED;
    strcpy(h_64->facility_name, h->facility_name);
    h_64->num_planes = h->num_planes;
    h_64->num_frames = h->num_frames;
    h_64->num_gates = h->num_gates;
    h_64->num_bed_pos = h->num_bed_pos;
    h_64->init_bed_position = h->init_bed_position;
	memcpy(h_64->bed_offset,h->bed_offset,15*sizeof(float));
    h_64->plane_separation = h->plane_separation;
    h_64->lwr_sctr_thres = h->lwr_sctr_thres;
    h_64->lwr_true_thres = h->lwr_true_thres;
    h_64->upr_true_thres = h->upr_true_thres;
	h_64->collimator = ELIMINATED;
    strcpy(h_64->user_process_code,h->user_process_code);
    h_64->acquisition_mode = h->acquisition_mode;
}

void sh64_convert(h_64, h, mh)
Scan_subheader_64* h_64;
Scan_subheader* h;
Main_header* mh;
{
	int i=0;

	mh64_data_type = h->data_type;	/* set main header data type */
    h_64->data_type = h->data_type;
    h_64->dimension_1 = h->num_r_elements ;
    h_64->dimension_2 = h->num_angles;
    h_64->processing_code = h->corrections_applied & 0xffff;	/* mod May 97 */
    h_64->smoothing = (h->corrections_applied >> 16) & 0xffff;	/* mod May 97 */
    h_64->sample_distance = h->x_resolution;
    h_64->isotope_halflife = mh->isotope_halflife;
	h_64->frame_duration_sec = ELIMINATED;
    h_64->gate_duration = h->gate_duration;
    h_64->r_wave_offset = h->r_wave_offset;
    h_64->scale_factor = h->scale_factor;
    h_64->scan_min = h->scan_min;
    h_64->scan_max = h->scan_max;
    h_64->prompts = h->prompts;
    h_64->delayed = h->delayed;
    h_64->multiples = h->multiples;
    h_64->net_trues = h->net_trues;
    memcpy(h_64->cor_singles, h->cor_singles, 16*sizeof(float));
    memcpy(h_64->uncor_singles, h->uncor_singles, 16*sizeof(float));
    h_64->tot_avg_cor = h->tot_avg_cor;
    h_64->tot_avg_uncor = h->tot_avg_uncor;
    h_64->total_coin_rate = h->total_coin_rate;
    h_64->frame_start_time = h->frame_start_time;
    h_64->frame_duration = h->frame_duration;
    h_64->loss_correction_fctr = h->loss_correction_fctr;
	for (i=0; i<8; i++) h_64->phy_planes[i] = h->phy_planes[i];
}

void ih64_convert(h_64, h, mh)
Image_subheader_64* h_64;
Image_subheader* h;
Main_header* mh;
{
	mh64_data_type = h->data_type;	/* set main header data type */
	memset(h_64,0,sizeof(Image_subheader_64));	/* clear memory */
    h_64->data_type = h->data_type;
    h_64->num_dimensions = 2;
    h_64->dimension_1 = h->x_dimension;
    h_64->dimension_2 = h->y_dimension;
    h->x_offset = h_64->x_origin = h->x_offset;
    h->y_offset = h_64->y_origin = h->y_offset;
    h_64->recon_scale = h->recon_zoom;
	h_64->ecat_calibration_fctr = mh->calibration_factor;
	h_64->well_counter_cal_fctr = 1.0;
	h_64->quant_scale = h->scale_factor;
	h_64->quant_units = 2;
    h_64->image_min = h->image_min;
    h_64->image_max = h->image_max;
    h_64->pixel_size = h->x_pixel_size;
   if (h->z_pixel_size>0) {
	h_64->slice_width = h->z_pixel_size;
   } else {
	h_64->slice_width = mh->plane_separation;
   }
    h_64->frame_duration = h->frame_duration;
    h_64->frame_start_time = h->frame_start_time;
    h_64->filter_code = h->filter_code;

    h_64->scan_matrix_num = DEFMATNUM; /* added by H.F. */
    h_64->norm_matrix_num = DEFMATNUM; /* added by H.F. */
    h_64->atten_cor_matrix_num = DEFMATNUM; /* added by H.F. */

    h_64->image_rotation = h->z_rotation_angle;
    h_64->decay_corr_fctr = h->decay_corr_fctr;
    h_64->processing_code = h->processing_code;
    h_64->filter_params[0] = h->filter_cutoff_frequency;
    h_64->filter_params[2] = h->filter_ramp_slope;
    h_64->filter_params[3] = h->filter_order;
    h_64->filter_params[4] = h->filter_scatter_fraction;
    h_64->filter_params[5] = h->filter_scatter_slope;
	h_64->intrinsic_tilt = mh->intrinsic_tilt;
    strcpy(h_64->annotation, h->annotation);
}

void nh64_convert(h_64, h, mh)
Norm_subheader_64* h_64;
Norm_subheader* h;
Main_header* mh;
{
	struct tm *normTime;
	time_t t = mh->scan_start_time;
	mh64_data_type = h->data_type;	/* set main header data type */
	memset(h_64,0,sizeof(Norm_subheader_64));		/* clear memory */
    h_64->data_type = h->data_type;
    h_64->dimension_1 = h->num_r_elements;
    h_64->dimension_2 = h->num_angles;
	h_64->scale_factor = h->scale_factor;
    h_64->fov_source_width = h->fov_source_width;
	h_64->ecat_calib_factor = mh->calibration_factor;
	normTime = localtime(&t);
	h_64->norm_day = normTime->tm_mday;
	h_64->norm_month = normTime->tm_mon+1; /* +1 by H.F.: struct tm counts months after January */
	h_64->norm_year = 1900+normTime->tm_year;
	h_64->norm_hour = normTime->tm_hour;
	h_64->norm_minute = normTime->tm_min;
	h_64->norm_second = normTime->tm_sec;
}

void ah64_convert(h_64, h, mh)
Attn_subheader_64* h_64;
Attn_subheader* h;
Main_header* mh;
{
	mh64_data_type = h->data_type;	/* set main header data type */
	memset(h_64,0,sizeof(Attn_subheader_64));		/* clear memory */
    h_64->data_type = h->data_type;
    h_64->attenuation_type = h->attenuation_type;
    h_64->dimension_1 = h->num_r_elements;
    h_64->dimension_2 = h->num_angles;
    h_64->sample_distance = h->x_resolution;
    h_64->scale_factor = h->scale_factor;
    h_64->x_origin = h->x_offset;
    h_64->y_origin = h->y_offset;
    h_64->x_radius = h->x_radius;
    h_64->y_radius = h->y_radius;
    h_64->tilt_angle = h->tilt_angle;
    h_64->attenuation_coeff = h->attenuation_coeff;
}
