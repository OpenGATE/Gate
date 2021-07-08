/* @(#)ecatRts.h	1.1 4/13/91 */

#ifndef ecatRts_defined
#define ecatRts_defined

#define RTS_SERVER 600000032
#define RTS_SERVER_VERSION 1
#define RTS_INFO_SERVER 600000036
#define RTS_INFO_VERSION 1

enum RTS_FUNCTIONS {
	CONF = 1,
	DPLN = 1 + 1,
	DFRM = 1 + 2,
	SINM = 1 + 3,
	RESO = 1 + 4,
	STRT = 1 + 5,
	STOP = 1 + 6,
	FCLR = 1 + 7,
	INFO = 1 + 8,
	STOR = 1 + 9,
	GANT = 1 + 10,
	RGAN = 1 + 11,
	SNGL = 1 + 12,
	DBUG = 1 + 13,
	CREF = 1 + 14,
	DCRF = 1 + 15,
	REMS = 1 + 16,
	MODL = 1 + 17,
	GIMI = 1 + 18,
	THRT = 1 + 19,
	CONT = 1 + 20,
	RMHD = 1 + 21,
	WMHD = 1 + 22,
	RSHD = 1 + 23,
	WSHD = 1 + 24,
	RDAT = 1 + 25,
	WDAT = 1 + 26,
	FDEL = 1 + 27,
	ASTR = 1 + 28,
	DFOV = 1 + 29,
	MASH = 1 + 30,
	RBLK = 1 + 31,
	WBLK = 1 + 32,
	RECT = 1 + 33,
	GLOG = 1 + 34,
};
typedef enum RTS_FUNCTIONS RTS_FUNCTIONS;
bool_t xdr_RTS_FUNCTIONS();


enum RTS_INFO_FUNCTIONS {
	INFO_STRT = 1,
	INFO_STOP = 1 + 1,
};
typedef enum RTS_INFO_FUNCTIONS RTS_INFO_FUNCTIONS;
bool_t xdr_RTS_INFO_FUNCTIONS();


struct XMAIN_HEAD {
	char original_file_name[20];
	short sw_version;
	short data_type;
	short system_type;
	short file_type;
	char node_id[10];
	short scan_start_day;
	short scan_start_month;
	short scan_start_year;
	short scan_start_hour;
	short scan_start_minute;
	short scan_start_second;
	char isotope_code[8];
	float isotope_halflife;
	char radiopharmaceutical[32];
	float gantry_tilt;
	float gantry_rotation;
	float bed_elevation;
	short rot_source_speed;
	short wobble_speed;
	short transm_source_type;
	float axial_fov;
	float transaxial_fov;
	short transaxial_samp_mode;
	short coin_samp_mode;
	short axial_samp_mode;
	float calibration_factor;
	short calibration_units;
	short compression_code;
	char study_name[12];
	char patient_id[16];
	char patient_name[32];
	char patient_sex[1];
	char patient_age[10];
	char patient_height[10];
	char patient_weight[10];
	char patient_dexterity[1];
	char physician_name[32];
	char operator_name[32];
	char study_description[32];
	short acquisition_type;
	short bed_type;
	short septa_type;
	char facility_name[20];
	short num_planes;
	short num_frames;
	short num_gates;
	short num_bed_pos;
	float init_bed_position;
	float bed_offset[15];
	float plane_separation;
	short lwr_sctr_thres;
	short lwr_true_thres;
	short upr_true_thres;
	float collimator;
	char user_process_code[10];
};
typedef struct XMAIN_HEAD XMAIN_HEAD;
bool_t xdr_XMAIN_HEAD();


struct XSCAN_SUB {
	short data_type;
	short dimension_1;
	short dimension_2;
	short smoothing;
	short processing_code;
	float sample_distance;
	float isotope_halflife;
	short frame_duration_sec;
	int gate_duration;
	int r_wave_offset;
	float scale_factor;
	short scan_min;
	short scan_max;
	int prompts;
	int delayed;
	int multiples;
	int net_trues;
	float cor_singles[16];
	float uncor_singles[16];
	float tot_avg_cor;
	float tot_avg_uncor;
	int total_coin_rate;
	int frame_start_time;
	int frame_duration;
	float loss_correction_fctr;
};
typedef struct XSCAN_SUB XSCAN_SUB;
bool_t xdr_XSCAN_SUB();


struct XIMAGE_SUB {
	short data_type;
	short num_dimensions;
	short dimension_1;
	short dimension_2;
	float x_origin;
	float y_origin;
	float recon_scale;
	float quant_scale;
	short image_min;
	short intimage_max;
	float pixel_size;
	float slice_width;
	int frame_duration;
	int frame_start_time;
	short slice_location;
	short recon_start_hour;
	short intrecon_start_minute;
	short recon_start_sec;
	int recon_duration;
	short filter_code;
	int scan_matrix_num;
	int norm_matrix_num;
	int atten_cor_matrix_num;
	float image_rotation;
	float plane_eff_corr_fctr;
	float decay_corr_fctr;
	float loss_corr_fctr;
	short processing_code;
	short quant_units;
	short recon_start_day;
	short recon_start_month;
	short recon_start_year;
	float ecat_calibration_fctr;
	float well_counter_cal_fctr;
	float filter_params[6];
	char annotation[40];
};
typedef struct XIMAGE_SUB XIMAGE_SUB;
bool_t xdr_XIMAGE_SUB();


struct XNORM_SUB {
	short data_type;
	short dimension_1;
	short dimension_2;
	float scale_factor;
	short norm_hour;
	short norm_minute;
	short norm_second;
	short norm_day;
	short norm_month;
	short norm_year;
	float fov_source_width;
};
typedef struct XNORM_SUB XNORM_SUB;
bool_t xdr_XNORM_SUB();


struct XATTEN_SUB {
	short data_type;
	short attenuation_type;
	short dimension_1;
	short dimension_2;
	float scale_factor;
	float x_origin;
	float y_origin;
	float x_radius;
	float y_radius;
	float tilt_angle;
	float attenuation_coeff;
	float sample_distance;
};
typedef struct XATTEN_SUB XATTEN_SUB;
bool_t xdr_XATTEN_SUB();


struct CREF_args {
	char *file_name;
	XMAIN_HEAD mhead;
	int data_size;
};
typedef struct CREF_args CREF_args;
bool_t xdr_CREF_args();


struct DPLN_args {
	int log_plane;
	int phy_plane[4];
};
typedef struct DPLN_args DPLN_args;
bool_t xdr_DPLN_args();


struct DFRM_args {
	int nframes;
	int delay;
	int duration;
	int nsegs;
};
typedef struct DFRM_args DFRM_args;
bool_t xdr_DFRM_args();


struct MODL_args {
	int model;
	int number_of_rings;
};
typedef struct MODL_args MODL_args;
bool_t xdr_MODL_args();


struct CFRM_args {
	int frame;
	int delay;
	int duration;
};
typedef struct CFRM_args CFRM_args;
bool_t xdr_CFRM_args();


struct CONF_args {
	int config;
	float pile_up_factor;
	float plane_factor;
};
typedef struct CONF_args CONF_args;
bool_t xdr_CONF_args();


struct STRT_args {
	int numberFrames;
	int acqType;
};
typedef struct STRT_args STRT_args;
bool_t xdr_STRT_args();


struct RECT_args {
	int frame;
	int plane;
	int segment;
	int data;
	int startView;
	int endView;
};
typedef struct RECT_args RECT_args;
bool_t xdr_RECT_args();


struct STOR_args {
	char *filename;
	int frame;
	int matframe;
	int bed;
};
typedef struct STOR_args STOR_args;
bool_t xdr_STOR_args();


struct ASTR_args {
	char *file_name;
	int auto_store;
};
typedef struct ASTR_args ASTR_args;
bool_t xdr_ASTR_args();


struct GLOG_args {
	char *filename;
	int frame;
	int plane;
	int segment;
	int data;
	int bed;
};
typedef struct GLOG_args GLOG_args;
bool_t xdr_GLOG_args();


struct GIMI_args {
	int frame;
	int plane;
	int segment;
	int data;
	int bed;
};
typedef struct GIMI_args GIMI_args;
bool_t xdr_GIMI_args();


struct SNGL_args {
	int start_bucket;
	int end_bucket;
	int frequency;
	int timeout;
	int num_ipcs;
};
typedef struct SNGL_args SNGL_args;
bool_t xdr_SNGL_args();


struct RBLK_args {
	char *filename;
	int block_number;
};
typedef struct RBLK_args RBLK_args;
bool_t xdr_RBLK_args();


struct WBLK_args {
	char *filename;
	int block_number;
	char wblk[512];
};
typedef struct WBLK_args WBLK_args;
bool_t xdr_WBLK_args();


struct GIMI_resp {
	int time;
	int start_time;
	int nproj;
	int nview;
	int minval;
	int maxval;
	int prompts;
	int delayeds;
	int multiples;
	int corrected_sing[32];
	int uncorrected_sing[32];
	int total_ipc_prompts;
	int total_ipc_delayeds;
	int total_ipc_multiples;
	int status;
};
typedef struct GIMI_resp GIMI_resp;
bool_t xdr_GIMI_resp();


struct GLOG_resp {
	int time;
	int start_time;
	int minval;
	int maxval;
	int prompts;
	int delayeds;
	int multiples;
	int net_trues;
	float dtcor;
	int status;
};
typedef struct GLOG_resp GLOG_resp;
bool_t xdr_GLOG_resp();


struct INFO_resp {
	int acq_status;
	int total_time;
	int prompt_events;
	int delayed_events;
	int multiple_events;
	int current_frame;
	int total_frames;
	int frame_time;
	int frame_end_time;
	int scan_end_time;
	int singles_Kcps_corrected;
	int singles_Kcps_uncorrected;
	int total_ipc_prompt_rate;
	int total_ipc_delayed_rate;
	int total_ipc_multiple_rate;
	int contig_disk_space;
};
typedef struct INFO_resp INFO_resp;
bool_t xdr_INFO_resp();


struct RBLK_resp {
	int status;
	char rblk[512];
};
typedef struct RBLK_resp RBLK_resp;
bool_t xdr_RBLK_resp();


struct RMHD_resp {
	int status;
	XMAIN_HEAD xmain_head;
};
typedef struct RMHD_resp RMHD_resp;
bool_t xdr_RMHD_resp();


struct WMHD_args {
	char *file_name;
	XMAIN_HEAD xmain_head;
};
typedef struct WMHD_args WMHD_args;
bool_t xdr_WMHD_args();


struct RSHD_args {
	char *file_name;
	int matnum;
};
typedef struct RSHD_args RSHD_args;
bool_t xdr_RSHD_args();


struct RSHD_resp {
	int status;
	char rhdat[512];
};
typedef struct RSHD_resp RSHD_resp;
bool_t xdr_RSHD_resp();


struct WSHD_args {
	char *file_name;
	int matnum;
	char whdat[512];
};
typedef struct WSHD_args WSHD_args;
bool_t xdr_WSHD_args();


struct RDAT_args {
	char *file_name;
	int matnum;
};
typedef struct RDAT_args RDAT_args;
bool_t xdr_RDAT_args();


struct RDAT_resp {
	int status;
	struct {
		u_int rdat_len;
		char *rdat_val;
	} rdat;
};
typedef struct RDAT_resp RDAT_resp;
bool_t xdr_RDAT_resp();


struct WDAT_args {
	char *file_name;
	int matnum;
	int data_type;
	struct {
		u_int wdat_len;
		char *wdat_val;
	} wdat;
};
typedef struct WDAT_args WDAT_args;
bool_t xdr_WDAT_args();

#endif ecatRts_defined
