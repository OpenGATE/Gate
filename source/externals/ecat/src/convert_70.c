/*
 Modification history:
        02-oct-02: Added modification from Dr. Harald Fricke <HFricke@hdz-nrw.de>
*/

#include <math.h>
#include "matrix.h"
#include "matrix_64.h"
#include "isotope_info.h"
#include <string.h>
#include <time.h>

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
	case 922 : /* added by H.F. */
	case 925 :
	case 951 :
		return 192;
	case 961:
		return 336;
	case 962:
	case 966:
		return 288;
	}
}

static int num_angles(system_type)
int system_type;
{
	switch (system_type) {
	case 921 :
	case 922 : /* added by H.F. */
	case 925 :
	case 953 :
	default:
		return 192;
	case 951 :
		return 256;
	case 962:
	case 966:
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
	case 922 : /* added by H.F. */
	case 925 :
	default:
		return 0.3375;
	case 961:
		return 0.165;
	case 962:
	case 966:
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
    case 922 : /* added by H.F. */
    case 925 :
    case 953 :
        return 15;
    case 961:
        return 13;
    case 951 :
    case 962:
    case 966:
    default:
        return 0;
    }
}

mh_convert(h, h_64)
Main_header * h;
Main_header_64 *h_64;
{
	int j=0;
	float branchFrac;
	struct tm strtTime;
    short units = h_64->calibration_units;
	if(units < 0 || units >= NumOldUnits)  units = UnknownEcfUnits;
	strcpy(h->data_units, olddisplayunits[units]);
    h->calibration_units_label = units;
    h->calibration_factor = (h_64->calibration_factor==0.0?1.0:h_64->calibration_factor)*ecfconverter[units];
    switch( units ) {
	case EcatCountsPerSec:
		h->calibration_units = Uncalibrated;
		break;
	case UCiPerCC:
	case NCiPerCC:
                /* Units have been converted to Bq/cc by scaling the ECF */
                /* Change data_units string to reflect the change.       */
                strcpy(h->data_units, olddisplayunits[BecquerelsPerCC]);
                /* fall through... I love C... H.F. */

	case BecquerelsPerCC:
		h->calibration_units = Calibrated;
		break;
	default:
		h->calibration_units = Processed;
		break;
    }

    strcpy(h->original_file_name, h_64->original_file_name);
	h->sw_version = h_64->sw_version;
    h->system_type = h_64->system_type;
    h->file_type = h_64->file_type;
    strcpy(h->serial_number, h_64->node_id);
	memset(&strtTime,0,sizeof(strtTime));
	strtTime.tm_mday = h_64->scan_start_day;
	strtTime.tm_mon = h_64->scan_start_month-1;
	strtTime.tm_year = h_64->scan_start_year - 1900;
	strtTime.tm_hour = h_64->scan_start_hour;
	strtTime.tm_min = h_64->scan_start_minute;
	strtTime.tm_sec = h_64->scan_start_second;
	strtTime.tm_isdst = -1;
	if( strtTime.tm_mday < 1 ) strtTime.tm_mday = 1;
	if( strtTime.tm_mon  < 0 ) strtTime.tm_mon = 0;
	h->scan_start_time = mktime(&strtTime);
    strcpy(h->isotope_code, h_64->isotope_code);
    h->isotope_halflife = h_64->isotope_halflife;
    strcpy(h->radiopharmaceutical, h_64->radiopharmaceutical);
    h->gantry_tilt = h_64->gantry_tilt;
    h->gantry_rotation = h_64->gantry_rotation;
    h->bed_elevation = h_64->bed_elevation;
    h->intrinsic_tilt = intrinsic_tilt(h->system_type);
    h->wobble_speed = h_64->wobble_speed;
    h->transm_source_type = h_64->transm_source_type;
    h->distance_scanned = h_64->axial_fov;
    h->transaxial_fov = h_64->transaxial_fov;
    h->angular_compression = h_64->compression_code%256;
    h->coin_samp_mode = h_64->coin_samp_mode;
    h->axial_samp_mode = h_64->axial_samp_mode;
    h->compression_code = h_64->compression_code/256;
    strcpy(h->study_name, h_64->study_name);
    strcpy(h->patient_id, h_64->patient_id);
    strcpy(h->patient_name, h_64->patient_name);
	h->patient_sex[0] = h_64->patient_sex;
    h->patient_dexterity[0] = h_64->patient_dexterity;
    h->patient_age = atof(h_64->patient_age);
    h->patient_height = atof(h_64->patient_height);
    h->patient_weight = atof(h_64->patient_weight);
    h->patient_birth_date = 0;
    strcpy(h->physician_name, h_64->physician_name);
    strcpy(h->operator_name, h_64->operator_name);
    strcpy(h->study_description, h_64->study_description);
    h->acquisition_type = h_64->acquisition_type;
    h->patient_orientation = UnknownOrientation;
    strcpy(h->facility_name, h_64->facility_name);
    h->num_planes = h_64->num_planes;
    h->num_frames = h_64->num_frames;
    h->num_gates = h_64->num_gates;
    h->num_bed_pos = h_64->num_bed_pos;
    h->init_bed_position = h_64->init_bed_position;
	memcpy(h->bed_offset,h_64->bed_offset,15*sizeof(float));
    h->plane_separation = h_64->plane_separation;
    h->lwr_sctr_thres = h_64->lwr_sctr_thres;
    h->lwr_true_thres = h_64->lwr_true_thres;
    h->upr_true_thres = h_64->upr_true_thres;
    strcpy(h->user_process_code,h_64->user_process_code);
    h->acquisition_mode = h_64->acquisition_mode;
    h->bin_size = bin_size(h_64->system_type);
    while (j < NumberOfIsotopes &&
		strncmp(isotope_info[j].name, h_64->isotope_code,strlen(isotope_info[j].name)) != 0) j++;
    if (j < NumberOfIsotopes) {
	sscanf(isotope_info[j].branch_ratio, "%f", &branchFrac);
 	h->branching_fraction = branchFrac;
/* fprintf (stdout, "Warning: branching fraction %f now added to main header\n", branchFrac); */
    }
    else fprintf (stdout, "Warning: improper or missing value for isotope(%s) in the main header\n", h_64->isotope_code);
    h->dose_start_time = 0.0;
    h->dosage = 0.0;
    h->well_counter_factor = 1.0;
    h->septa_state = SeptaExtended;
}

void sh_convert(h, h_64, mh)
Scan_subheader* h;
Scan_subheader_64* h_64;
Main_header* mh;
{
	int i=0;

    h->data_type = h_64->data_type;
    h->num_dimensions = 2;
    h->num_r_elements = h_64->dimension_1;
    h->num_angles = h_64->dimension_2;
    h->corrections_applied = h_64->processing_code;
    h->corrections_applied = h_64->processing_code | (h_64->smoothing << 16);
    h->num_z_elements = 1;
    h->x_resolution = h_64->sample_distance;
    h->y_resolution = h_64->sample_distance;
    h->z_resolution = mh->plane_separation;
    h->w_resolution = 0;
    h->gate_duration = h_64->gate_duration;
    h->r_wave_offset = h_64->r_wave_offset;
    h->num_accepted_beats = 0;
    h->scale_factor = h_64->scale_factor;
    h->scan_min = h_64->scan_min;
    h->scan_max = h_64->scan_max;
    h->prompts = h_64->prompts;
    h->delayed = h_64->delayed;
    h->multiples = h_64->multiples;
    h->net_trues = h_64->net_trues;
    memcpy(h->cor_singles, h_64->cor_singles, 16*sizeof(float));
    memcpy(h->uncor_singles, h_64->uncor_singles, 16*sizeof(float));
    h->tot_avg_cor = h_64->tot_avg_cor;
    h->tot_avg_uncor = h_64->tot_avg_uncor;
    h->total_coin_rate = h_64->total_coin_rate;
    h->frame_start_time = h_64->frame_start_time;
    h->frame_duration = h_64->frame_duration;
    h->loss_correction_fctr = h_64->loss_correction_fctr;
	for (i=0; i<8; i++) h->phy_planes[i] = h_64->phy_planes[i];
	h->ring_difference = 0;
	for (i=0; i<8; i++)
		if (h->phy_planes[i] > -1) h->ring_difference++;
}

void ih_convert(h, h_64, mh)
Image_subheader* h;
Image_subheader_64* h_64;
Main_header* mh;
{
	memset(h,0,sizeof(Image_subheader));			/* clear memory */
    h->data_type = h_64->data_type;
    h->num_dimensions = h_64->num_dimensions;
    h->x_dimension = h_64->dimension_1;
    h->y_dimension = h_64->dimension_2;
    h->z_dimension = 1;
    h->x_offset = h_64->x_origin;
    h->y_offset = h_64->y_origin;
    h->recon_zoom = h_64->recon_scale;
	h->scale_factor = h_64->quant_scale;
    h->image_min = h_64->image_min;
    h->image_max = h_64->image_max;
    h->x_pixel_size = h_64->pixel_size;
    h->y_pixel_size = h_64->pixel_size;
    h->z_pixel_size = h_64->slice_width; /* not mh->plane_separation  H.F. */
    h->frame_duration = h_64->frame_duration;
    h->frame_start_time = h_64->frame_start_time;
    h->filter_code = h_64->filter_code;
    h->num_r_elements = r_elements(mh->system_type);
    h->num_angles = num_angles(mh->system_type);
    h->z_rotation_angle = h_64->image_rotation;
    h->decay_corr_fctr = h_64->decay_corr_fctr;
    h->processing_code = h_64->processing_code;
    h->filter_cutoff_frequency = h_64->filter_params[0];
    h->filter_ramp_slope = h_64->filter_params[2];
    h->filter_order = h_64->filter_params[3];
    h->filter_scatter_fraction = h_64->filter_params[4];
    h->filter_scatter_slope = h_64->filter_params[5];
    strcpy(h->annotation, h_64->annotation);
}

void nh_convert(h, h_64, mh)
Norm_subheader* h;
Norm_subheader_64* h_64;
Main_header* mh;
{
	memset(h,0,sizeof(Norm_subheader));			/* clear memory */
    h->data_type = h_64->data_type;
    h->num_dimensions = 2;
    h->num_r_elements = h_64->dimension_1;
    h->num_angles = h_64->dimension_2;
    h->num_z_elements =  1;
	h->scale_factor = h_64->scale_factor;
/*
    h->norm_min;                     = recompute min(data);
    h->norm_max;                     = recompute max(data)
*/
    h->fov_source_width = h_64->fov_source_width;
    h->norm_quality_factor =  1.0;
    h->storage_order = ElVwAxRd;
    h->z_elements[0] = 1;
}

void ah_convert(h, h_64, mh)
Attn_subheader* h;
Attn_subheader_64* h_64;
Main_header* mh;
{
	memset(h,0,sizeof(Attn_subheader));			/* clear memory */
    h->data_type = h_64->data_type;
    h->num_dimensions = 2;
    h->attenuation_type = h_64->attenuation_type;
    h->num_r_elements = h_64->dimension_1;
    h->num_angles = h_64->dimension_2;
    h->num_z_elements = 1;
    h->x_resolution = h_64->sample_distance;
    h->y_resolution = h_64->sample_distance;
    h->z_resolution = mh->plane_separation;
    h->scale_factor = h_64->scale_factor;
    h->x_offset = h_64->x_origin;
    h->y_offset = h_64->y_origin;
    h->x_radius = h_64->x_radius;
    h->y_radius = h_64->y_radius;
    h->tilt_angle = h_64->tilt_angle;
    h->attenuation_coeff = h_64->attenuation_coeff;
/*
    h->attenuation_min = recompute min(data);
    h->attenuation_max = recompute max(data);
*/
    h->skull_thickness = 0.45;
    h->edge_finding_threshold = 0.1;
    h->storage_order = ElVwAxRd;
    h->z_elements[0] = 1;
}

/*
void analyze_convert(h, ah, mh)
Attn_subheader* h;
Analyze_header* ah;
Main_header* mh;
{
	memset(h,0,sizeof(Norm_subheader));
    h->data_type = ah->data_type;
    h->num_dimensions = 2;
.....
}
*/
