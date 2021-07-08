/* @(#)matrix.h	1.5 2/8/93 Copyright 1989 CTI, Inc.";*/

/* prevent recursive definition */

/*
 * modification by Sibomana@topo.ucl.ac.be		19-sep-1994
 * used to convert 6.4 image files in 7.0 format.
 *
*/

#ifndef		matrix_64_h
#define		matrix_64_h

#include "matrix.h"
typedef short int word;

typedef struct mat_main_header {
	char		original_file_name[20];
	word		sw_version;
	word		data_type;
	word		system_type;
	word		file_type;
	char		node_id[10];
	word		scan_start_day,
			scan_start_month,
			scan_start_year,
			scan_start_hour,
			scan_start_minute,
			scan_start_second;
	char		isotope_code[8];
	float		isotope_halflife;
	char		radiopharmaceutical[32];
	float		gantry_tilt,
			gantry_rotation,
			bed_elevation;
	word		rot_source_speed,
			wobble_speed,
			transm_source_type;
	float		axial_fov,
			transaxial_fov;
	word		transaxial_samp_mode,
			coin_samp_mode,
			axial_samp_mode;
	float		calibration_factor;
	word		calibration_units,
			compression_code;
	char		study_name[12],
			patient_id[16],
			patient_name[32],
			patient_sex,
			patient_age[10],
			patient_height[10],
			patient_weight[10],
			patient_dexterity,
			physician_name[32],
			operator_name[32],
			study_description[32];
	word		acquisition_type,
			bed_type,
			septa_type;
	char		facility_name[20];
	word		num_planes,
			num_frames,
			num_gates,
			num_bed_pos;
	float		init_bed_position,
			bed_offset[15],
			plane_separation;
	word		lwr_sctr_thres,
			lwr_true_thres,
			upr_true_thres;
	float		collimator;
	char		user_process_code[10];
	word		acquisition_mode;
	}
Main_header_64;

typedef struct mat_scan_subheader{
	word		data_type,
			dimension_1,
			dimension_2,
			smoothing,
			processing_code;
	float		sample_distance,
			isotope_halflife;
	word		frame_duration_sec;
	int		gate_duration,
			r_wave_offset;
	float		scale_factor;
	word		scan_min,
			scan_max;
	int		prompts,
			delayed,
			multiples,
			net_trues;
	float		cor_singles[16],
			uncor_singles[16],
			tot_avg_cor,
			tot_avg_uncor;
	int		total_coin_rate,
			frame_start_time,
			frame_duration;
	float		loss_correction_fctr;
	int		phy_planes[8];
	}
Scan_subheader_64;

typedef struct mat_image_subheader{
	word		data_type,
			num_dimensions,
			dimension_1,
			dimension_2;
	float		x_origin,
			y_origin,
			recon_scale,			/* Image ZOOM from reconstruction */
			quant_scale;			/* Scale Factor */
	word		image_min,
			image_max;
	float		pixel_size,
			slice_width;
	int		frame_duration,
			frame_start_time;
	word		slice_location,
			recon_start_hour,
			recon_start_minute,
			recon_start_sec;
	int		recon_duration;
	word		filter_code;
	int		scan_matrix_num,
			norm_matrix_num,
			atten_cor_matrix_num;
	float		image_rotation,
			plane_eff_corr_fctr,
			decay_corr_fctr,
			loss_corr_fctr,
			intrinsic_tilt ;
	word		processing_code,
			quant_units,
			recon_start_day,
			recon_start_month,
			recon_start_year;
	float		ecat_calibration_fctr,
			well_counter_cal_fctr,
			filter_params[6];
	char		annotation[40];
	}
Image_subheader_64;

typedef struct mat_norm_subheader{
	word		data_type,
			dimension_1,
			dimension_2;
	float		scale_factor;
	word		norm_hour,
			norm_minute,
			norm_second,
			norm_day,
			norm_month,
			norm_year;
	float		fov_source_width;
	float		ecat_calib_factor;
	}
Norm_subheader_64;

typedef struct mat_attn_subheader{
	word		data_type,
			attenuation_type,
			dimension_1,
			dimension_2;
	float		scale_factor,
			x_origin,
			y_origin,
			x_radius,
			y_radius,
			tilt_angle,
			attenuation_coeff,
			sample_distance;
	}
Attn_subheader_64;

#endif	/* 	matrix_64_h */
