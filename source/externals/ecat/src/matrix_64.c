
static char sccsid[]="@(#)matrix.c	1.11 6/7/93 Copyright 1989 CTI, Inc.";

/*
 * modification by Sibomana@topo.ucl.ac.be      19-sep-1994
 * used to convert 6.4 image files in 7.0 format.
 * 02-oct-02: Added modification from Dr. Harald Fricke <HFricke@hdz-nrw.de>
*/

#include	"matrix_64.h"
#include	"matrix.h"
#include        "machine_indep.h"
#include	<string.h>
#include	<stdlib.h>

#define  ERROR -1
#define OK 0

extern MatrixErrorCode matrix_errno;
extern char matrix_errtxt[];

float get_vax_float( bufr, off)
  unsigned short bufr[];
  int off;
{
	unsigned short t1, t2;
	union {unsigned int t3; float t4;} test;

	if (bufr[off]==0 && bufr[off+1]==0) return(0.0);
	t1 = bufr[off] & 0x80ff;
	t2=(((bufr[off])&0x7f00)+0xff00)&0x7f00;
	test.t3 = (t1+t2)<<16;
	test.t3 =test.t3+bufr[off+1];
	return(test.t4);
}

int get_vax_long( bufr, off)
  unsigned short bufr[];
  int off;
{
	return ((bufr[off+1]<<16)+bufr[off]);
}
  
int mat_lookup_64( fptr, matnum, entry)
  FILE *fptr;
  int matnum;
  struct MatDir *entry;
{
	
	int blk, i;
	int nfree, nxtblk, prvblk, nused, matnbr, strtblk, endblk, matstat;
	int dirbufr[MatBLKSIZE/4];
	char bytebufr[MatBLKSIZE];

	matrix_errno = 0;
	matrix_errtxt[0] = '\0';
	blk = MatFirstDirBlk;
	while(1) {
	if (mat_rblk( fptr, blk, bytebufr,1) == ERROR) return ERROR;
	if (ntohs(1) == 1) {
		swab( bytebufr, (char*)dirbufr, MatBLKSIZE);
		swaw( (short*)dirbufr, (short*)bytebufr, MatBLKSIZE/2);
	}
	memcpy(dirbufr, bytebufr, MatBLKSIZE);
	nfree  = dirbufr[0];
	nxtblk = dirbufr[1];
	prvblk = dirbufr[2];
	nused  = dirbufr[3];
	for (i=4; i<MatBLKSIZE/4; i+=4)
	{  matnbr  = dirbufr[i];
	   strtblk = dirbufr[i+1];
	   endblk  = dirbufr[i+2];
	   matstat = dirbufr[i+3];
	   if (matnum == matnbr) {
	      entry->matnum  = matnbr;
	      entry->strtblk = strtblk;
	      entry->endblk  = endblk;
	      entry->matstat = matstat;
	      return (1); }
 	}
	blk = nxtblk;
	if (blk == MatFirstDirBlk) break;
	}
	return (0);
}

int unmap64_main_header(buf, header)
  char *buf;
  Main_header *header;
{
	int i;
	Main_header_64 h_64;
	short b[MatBLKSIZE/2];
	Main_header_64 *h = &h_64;
	
	memset(h,0,sizeof(h_64));
	strncpy( h->original_file_name, buf+28, 19);
	strncpy( h->node_id, buf+56, 9);
	strncpy( h->isotope_code, buf+78, 7);
	strncpy( h->radiopharmaceutical, buf+90, 31);
	strncpy( h->study_name, buf+162, 11);
	strncpy( h->patient_id, buf+174, 15);
	strncpy( h->patient_name, buf+190, 31);
	h->patient_sex = buf[222];
	strncpy( h->patient_age, buf+223, 9);
	strncpy( h->patient_height, buf+233, 9);
	strncpy( h->patient_weight, buf+243, 9);
	h->patient_dexterity = buf[253];
	strncpy( h->physician_name, buf+254, 31);
	strncpy( h->operator_name, buf+286, 31);
	strncpy( h->study_description, buf+318, 31);
	strncpy( h->facility_name, buf+356, 19);
	strncpy( h->user_process_code, buf+462, 9);
	if (ntohs(1) == 1) swab( buf, (char*)b, MatBLKSIZE);
	else memcpy(b,buf, MatBLKSIZE);
	h->sw_version = b[24];
	h->data_type = b[25];
	h->system_type = b[26];
	h->file_type = b[27];
	h->scan_start_day = b[33];
	h->scan_start_month = b[34];
	h->scan_start_year = b[35];
	h->scan_start_hour = b[36];
	h->scan_start_minute = b[37];
	h->scan_start_second = b[38];
	h->isotope_halflife = get_vax_float(b, 43);
	h->gantry_tilt = get_vax_float(b, 61);
	h->gantry_rotation = get_vax_float(b, 63);
	h->bed_elevation = get_vax_float(b, 65);
	h->rot_source_speed = b[67];
	h->wobble_speed = b[68];
	h->transm_source_type = b[69];
	h->axial_fov = get_vax_float(b, 70);
	h->transaxial_fov = get_vax_float(b, 72);
	h->transaxial_samp_mode = b[74];
	h->coin_samp_mode = b[75];
	h->axial_samp_mode = b[76];
	h->calibration_factor = get_vax_float( b, 77);
	h->calibration_units = b[79];
	h->compression_code = b[80];
	h->acquisition_type = b[175];
	h->bed_type = b[176];
	h->septa_type = b[177];
	h->num_planes = b[188];
	h->num_frames = b[189];
	h->num_gates = b[190];
	h->num_bed_pos = b[191];
	h->init_bed_position = get_vax_float( b, 192);
	for (i=0; i<15; i++)
	  h->bed_offset[i] = get_vax_float( b, 194+2*i);
	h->plane_separation = get_vax_float( b, 224);
	h->lwr_sctr_thres = b[226];
	h->lwr_true_thres = b[227];
	h->upr_true_thres = b[228];
	h->collimator = get_vax_float( b, 229);
	h->acquisition_mode = b[236];
	mh_convert(header,h);
	return (0);
}

int unmap64_scan_header(buf, header, mh)
  char *buf;
  Scan_subheader *header;
  Main_header *mh;
{
	int i;
	Scan_subheader_64 h_64;
	short b[MatBLKSIZE/2];
	Scan_subheader_64 *h = &h_64;

	if (ntohs(1) == 1) swab( buf, (char*)b, MatBLKSIZE);
	else memcpy(b,buf, MatBLKSIZE);
	h->data_type = b[63];
	h->dimension_1 = b[66];
	h->dimension_2 = b[67];
	h->smoothing = b[68];
	h->processing_code = b[69];
	h->sample_distance = get_vax_float( b, 73);
	h->isotope_halflife = get_vax_float( b, 83);
	h->frame_duration_sec = b[85];
	h->gate_duration = get_vax_long( b, 86);
	h->r_wave_offset = get_vax_long( b, 88);
	h->scale_factor = get_vax_float( b, 91);
	h->scan_min = b[96];
	h->scan_max = b[97];
	h->prompts = get_vax_long( b, 98);
	h->delayed = get_vax_long( b, 100);
	h->multiples = get_vax_long( b, 102);
	h->net_trues = get_vax_long( b, 104);
	for (i=0; i<16; i++)
	{ h->cor_singles[i] = get_vax_float( b, 158+2*i);
	  h->uncor_singles[i] = get_vax_float( b, 190+2*i);}
	h->tot_avg_cor = get_vax_float( b, 222);
	h->tot_avg_uncor = get_vax_float( b, 224);
	h->total_coin_rate = get_vax_long( b, 226);
	h->frame_start_time = get_vax_long( b, 228);
	h->frame_duration = get_vax_long( b, 230);
	h->loss_correction_fctr = get_vax_float( b, 232);
	for (i=0; i<8; i++)
	  h->phy_planes[i] = get_vax_long( b, 234+(2*i));
	sh_convert(header,h,mh);
	return (0);
}

int unmap64_image_header(buf, header, mh)
  char* buf;
  Image_subheader *header;
  Main_header *mh;
{
	int i;
	Image_subheader_64 h_64;
	short b[MatBLKSIZE/2];
	Image_subheader_64 *h = &h_64;

	memset(h,0,sizeof(h_64));
	strncpy( h->annotation, buf+420, 39);
	if (ntohs(1) == 1) swab( buf, (char*)b, MatBLKSIZE);
	else memcpy(b,buf,MatBLKSIZE);
	h->data_type = b[63];
	h->num_dimensions = b[64];
	h->dimension_1 = b[66];
	h->dimension_2 = b[67];
	h->x_origin = get_vax_float( b, 80);
	h->y_origin = get_vax_float( b, 82);
	h->recon_scale = get_vax_float( b, 84);
	h->quant_scale = get_vax_float( b, 86);
	h->image_min = b[88];
	h->image_max = b[89];
	h->pixel_size = get_vax_float( b, 92);
	h->slice_width = get_vax_float( b, 94);
	h->frame_duration = get_vax_long( b, 96);
	h->frame_start_time = get_vax_long( b, 98);
	h->slice_location = b[100];
	h->recon_start_hour = b[101];
	h->recon_start_minute = b[102];
	h->recon_start_sec = b[103];
	h->recon_duration = get_vax_long( b, 104);
	h->filter_code = b[118];
	h->scan_matrix_num = get_vax_long( b, 119);
	h->norm_matrix_num = get_vax_long( b, 121);
	h->atten_cor_matrix_num = get_vax_long( b, 123);
	h->image_rotation = get_vax_float( b, 148);
	h->plane_eff_corr_fctr = get_vax_float( b, 150);
	h->decay_corr_fctr = get_vax_float( b, 152);
	h->loss_corr_fctr = get_vax_float( b, 154);
	h->intrinsic_tilt = get_vax_float( b, 156);
	h->processing_code = b[188];
	h->quant_units = b[190];
	h->recon_start_day = b[191];
	h->recon_start_month = b[192];
	h->recon_start_year = b[193];
	h->ecat_calibration_fctr = get_vax_float( b, 194);
	h->well_counter_cal_fctr = get_vax_float( b, 196);
	for (i=0; i<6; i++)
	  h->filter_params[i] = get_vax_float( b, 198+2*i);
	ih_convert(header,h,mh);
	return (0);
}

int unmap64_attn_header(buf,header,mh)
  char *buf;
  Attn_subheader *header;
  Main_header *mh;
{
	int i;
	Attn_subheader_64 h_64;
	short bufr[MatBLKSIZE/2];
	Attn_subheader_64 *h = &h_64;

	if (ntohs(1) == 1) swab( buf, (char*)bufr, MatBLKSIZE);
	else memcpy(bufr,buf,MatBLKSIZE);
	h->data_type = bufr[63];
	h->attenuation_type = bufr[64];
	h->dimension_1 = bufr[66];
	h->dimension_2 = bufr[67];
	h->scale_factor = get_vax_float(bufr, 91);
	h->x_origin = get_vax_float(bufr, 93);
	h->y_origin = get_vax_float(bufr, 95);
	h->x_radius = get_vax_float(bufr, 97);
	h->y_radius = get_vax_float(bufr, 99);
	h->tilt_angle = get_vax_float(bufr, 101);
	h->attenuation_coeff = get_vax_float(bufr, 103);
	h->sample_distance = get_vax_float(bufr, 105);
	ah_convert(header,h,mh);
	return 0;
}

int unmap64_norm_header(buf, header, mh)
  char *buf;
  Norm_subheader *header;
  Main_header *mh;
{
	int i;
	Norm_subheader_64 h_64;
	Norm_subheader_64 *h = &h_64;
	short bufr[MatBLKSIZE/2];

	if (ntohs(1) == 1) swab( buf, (char*)bufr, MatBLKSIZE);
	else memcpy(bufr,buf,MatBLKSIZE);
	h->data_type = bufr[63];
	h->dimension_1 = bufr[66];
	h->dimension_2 = bufr[67];
	h->scale_factor = get_vax_float(bufr, 91);
	h->norm_hour = bufr[93];
	h->norm_minute = bufr[94];
	h->norm_second = bufr[95];
	h->norm_day = bufr[96];
	h->norm_month = bufr[97];
	h->norm_year = bufr[98];
	h->fov_source_width = get_vax_float(bufr, 99);
	h->ecat_calib_factor = get_vax_float(bufr, 101);
	nh_convert(header,h,mh);
	return (0);
}

int map64_main_header( bbufr, mh70)
  char *bbufr;
  Main_header *mh70;
{
      short bufr[MatBLKSIZE/2];
      int err,i, loc;
	Main_header_64 header;
    
      memset(bufr,0,MatBLKSIZE);
   	  mh64_convert(&header,mh70);
      bufr[24] = header.sw_version;
      bufr[25] = header.data_type;
      bufr[26] = header.system_type;
      bufr[27] = header.file_type;
      bufr[33] = header.scan_start_day;
      bufr[34] = header.scan_start_month;
      bufr[35] = header.scan_start_year;
      bufr[36] = header.scan_start_hour;
      bufr[37] = header.scan_start_minute;
      bufr[38] = header.scan_start_second;
      ftovaxf (header.isotope_halflife, &bufr[43]);
      ftovaxf (header.gantry_tilt, &bufr[61]);
      ftovaxf (header.gantry_rotation, &bufr[63]);
      ftovaxf (header.bed_elevation, &bufr[65]);
      bufr[67] = header.rot_source_speed;
      bufr[68] = header.wobble_speed;
      bufr[69] = header.transm_source_type;
      ftovaxf (header.axial_fov, &bufr[70]);
      ftovaxf (header.transaxial_fov, &bufr[72]);
      bufr[74] = header.transaxial_samp_mode;
      bufr[75] = header.coin_samp_mode;
      bufr[76] = header.axial_samp_mode;
      ftovaxf (header.calibration_factor, &bufr[77]);
      bufr[79] = header.calibration_units;
      bufr[80] = header.compression_code;
      bufr[175] = header.acquisition_type;
      bufr[176] = header.bed_type;
      bufr[177] = header.septa_type;
      bufr[188] = header.num_planes;
      bufr[189] = header.num_frames;
      bufr[190] = header.num_gates;
      bufr[191] = header.num_bed_pos;
      ftovaxf (header.init_bed_position, &bufr[192]);
      for (i=0; i<15; i++)
      {
	ftovaxf (header.bed_offset[i], &bufr[194+2*i]);
      }
      ftovaxf (header.plane_separation, &bufr[224]);
      bufr[226] = header.lwr_sctr_thres;
      bufr[227] = header.lwr_true_thres;
      bufr[228] = header.upr_true_thres;
      ftovaxf (header.collimator, &bufr[229]);
      bufr[236] = header.acquisition_mode;
    
      if (ntohs(1) == 1) swab( (char*)bufr, bbufr, MatBLKSIZE);
	  else memcpy(bbufr,bufr, MatBLKSIZE);
    
      /* facility_name */ 
      memcpy(bbufr+356, header.facility_name, 20);
      memcpy(bbufr+28, header.original_file_name, 20);
      /* write the node_id - character string */
      memcpy(bbufr+56, header.node_id, 10);
      /* write the isotope code - char string */
      memcpy(bbufr+78, header.isotope_code, 8);
      /* write the radiopharmaceutical  - char string */
      memcpy(bbufr+90, header.radiopharmaceutical, 32);
      /* study_name - char string */
      memcpy(bbufr+162, header.study_name, 12);
      /* patient_id - char string */
      memcpy(bbufr+174, header.patient_id, 16);
      /* patient_name - char string */
      memcpy(bbufr+190, header.patient_name, 32);
      /* patient_sex - char */
      bbufr[222] = header.patient_sex;
      /* patient_age - char string */
      memcpy(bbufr+223, header.patient_age, 10);
      /* patient_height  - char string */
      memcpy(bbufr+233, header.patient_height, 10);
      /* patient_weight - char string */
      memcpy(bbufr+243, header.patient_weight, 10);
      /* patient_dexterity - char */
      bbufr[253] = header.patient_dexterity;
      /* physician_name - char string */
      memcpy(bbufr+254, header.physician_name, 32);
      /* operator_name - char string */
      memcpy(bbufr+286, header.operator_name, 32);
      /* study_description - char string */
      memcpy(bbufr+318, header.study_description, 32);
      /* user_process_code  - char string */
      memcpy(bbufr+462, header.user_process_code, 10);
    
      return 1;
}

static sunltovaxl( in, out)
  int in;
  unsigned short int out[2];
{
    out[0]=(in&0x0000FFFF);
    out[1]=(in&0xFFFF0000)>>16;
}

int map64_image_header( bbufr, h70, mh)
  char *bbufr;
  Image_subheader *h70;
  Main_header *mh;
{
	short bufr[MatBLKSIZE/2];
	int i;
	Image_subheader_64 header;

	ih64_convert(&header,h70,mh);
    memset(bufr,0,MatBLKSIZE);
	/* transfer subheader information */
	bufr[63] = header.data_type;
	bufr[64] = header.num_dimensions;
	bufr[66] = header.dimension_1;
	bufr[67] = header.dimension_2;
	ftovaxf(header.x_origin, &bufr[80]);
	ftovaxf(header.y_origin, &bufr[82]);
	ftovaxf(header.recon_scale, &bufr[84]);
	ftovaxf(header.quant_scale, &bufr[86]);
	bufr[88] = header.image_min;
	bufr[89] = header.image_max;
	ftovaxf(header.pixel_size, &bufr[92]);
	ftovaxf(header.slice_width, &bufr[94]);
	sunltovaxl(header.frame_duration, &bufr[96]);
	sunltovaxl(header.frame_start_time, &bufr[98]);
	bufr[100] = header.slice_location;
	bufr[101] = header.recon_start_hour;
	bufr[102] = header.recon_start_minute;
	bufr[103] = header.recon_start_sec;
	sunltovaxl(header.recon_duration, &bufr[104]);
	bufr[118] = header.filter_code;
	sunltovaxl(header.scan_matrix_num, &bufr[119]);
	sunltovaxl(header.norm_matrix_num, &bufr[121]);
	sunltovaxl(header.atten_cor_matrix_num, &bufr[123]);
	ftovaxf(header.image_rotation, &bufr[148]);
	ftovaxf(header.plane_eff_corr_fctr, &bufr[150]);
	ftovaxf(header.decay_corr_fctr, &bufr[152]);
	ftovaxf(header.loss_corr_fctr, &bufr[154]);
	ftovaxf(header.intrinsic_tilt, &bufr[156]);
	bufr[188] = header.processing_code;
	bufr[190] = header.quant_units;
	bufr[191] = header.recon_start_day;
	bufr[192] = header.recon_start_month;
	bufr[193] = header.recon_start_year;
	ftovaxf(header.ecat_calibration_fctr, &bufr[194]);
	ftovaxf(header.well_counter_cal_fctr, &bufr[196]);
	
	for (i=0; i<6; i++)
	ftovaxf(header.filter_params[i], &bufr[198+2*i]);
	
	/* swap the bytes */
	if (ntohs(1) == 1) swab( (char*)bufr, bbufr, MatBLKSIZE);
	else memcpy(bbufr,bufr, MatBLKSIZE);
	strcpy (bbufr+420, header.annotation);
	return 1;
}

int map64_scan_header( bbufr, h70, mh)
  char *bbufr;
  Scan_subheader *h70;
  Main_header *mh;
{
  Scan_subheader_64 header;
  	int i, err;
  	short bufr[MatBLKSIZE/2];
  
	sh64_convert(&header,h70,mh);
    memset(bufr,0,MatBLKSIZE);
	bufr[0] = 256;
	bufr[1] = 1;
	bufr[2] = 22;
	bufr[3] = -1;
	bufr[4] = 25;
	bufr[5] = 62;
	bufr[6] = 79;
	bufr[7] = 106;
	bufr[24] = 37;
	bufr[25] = -1;
	bufr[61] = 17;
	bufr[62] = -1;
 	bufr[78] = 27;
	bufr[79] = -1;
	bufr[105] = 52;
	bufr[106] = -1;
 	bufr[63] = header.data_type;
  	bufr[66] = header.dimension_1;			/* x dimension */
  	bufr[67] = header.dimension_2;			/* y_dimension */
  	bufr[68] = header.smoothing;			
  	bufr[69] = header.processing_code;			
  	ftovaxf(header.sample_distance, &bufr[73]);
  	ftovaxf(header.isotope_halflife, &bufr[83]);
  	bufr[85] = header.frame_duration_sec;
  	sunltovaxl(header.gate_duration, &bufr[86]);
  	sunltovaxl(header.r_wave_offset, &bufr[88]);
  	ftovaxf(header.scale_factor, &bufr[91]);
  	bufr[96] = header.scan_min;
  	bufr[97] = header.scan_max;
  	sunltovaxl(header.prompts, &bufr[98]);
  	sunltovaxl(header.delayed, &bufr[100]);
  	sunltovaxl(header.multiples, &bufr[102]);
  	sunltovaxl(header.net_trues, &bufr[104]);
  	for (i=0; i<16; i++)
  	{
    	  ftovaxf(header.cor_singles[i], &bufr[158+2*i]);
  	  ftovaxf(header.uncor_singles[i], &bufr[190+2*i]);
 	};
  	ftovaxf(header.tot_avg_cor, &bufr[222]);
  	ftovaxf(header.tot_avg_uncor, &bufr[224]);
  	sunltovaxl(header.total_coin_rate, &bufr[226]);		/* total coin rate */
  	sunltovaxl(header.frame_start_time, &bufr[228]);
  	sunltovaxl(header.frame_duration, &bufr[230]);
  	ftovaxf(header.loss_correction_fctr, &bufr[232]);
  	for (i=0; i<8; i++)
    	  sunltovaxl(header.phy_planes[i], &bufr[234+2*i]);

  	if (ntohs(1) == 1) swab( (char*)bufr, bbufr, MatBLKSIZE);
	else memcpy(bbufr,bufr, MatBLKSIZE);
	return 1;
}

int map64_attn_header( bbufr, h70, mh)
  char *bbufr;
  Attn_subheader *h70;
  Main_header *mh;
{
	Attn_subheader_64 header;
	int i,err;
	short bufr[MatBLKSIZE/2];
	
	ah64_convert(&header,h70,mh);
    memset(bufr,0,MatBLKSIZE);
	bufr[0] = 256;
	bufr[1] = 1;
	bufr[2] = 22;
	bufr[3] = -1;
	bufr[4] = 25;
	bufr[5] = 62;
	bufr[6] = 79;
	bufr[7] = 106;
	bufr[24] = 37;
	bufr[25] = -1;
	bufr[61] = 17;
	bufr[62] = -1;
 	bufr[78] = 27;
	bufr[79] = -1;
	bufr[105] = 52;
	bufr[106] = -1;
	bufr[63] = header.data_type;
	bufr[64] = header.attenuation_type;
	bufr[66] = header.dimension_1;
	bufr[67] = header.dimension_2;
	ftovaxf( header.scale_factor, &bufr[91]);
	ftovaxf( header.x_origin, &bufr[93]);
	ftovaxf( header.y_origin, &bufr[95]);
	ftovaxf( header.x_radius, &bufr[97]);
	ftovaxf( header.y_radius, &bufr[99]);
	ftovaxf( header.tilt_angle, &bufr[101]);
	ftovaxf( header.attenuation_coeff, &bufr[103]);
	ftovaxf( header.sample_distance, &bufr[105]);
	if (ntohs(1) == 1) swab( (char*)bufr, bbufr, 512);
	else memcpy(bbufr,bufr, MatBLKSIZE);
	return 1;
}

int map64_norm_header( bbufr,h70,mh)
  char *bbufr;
  Norm_subheader *h70;
  Main_header *mh;
{
	int i,err;
  	Norm_subheader_64 header;
	short bufr[MatBLKSIZE/2];

	nh64_convert(&header,h70,mh);
    memset(bufr,0,MatBLKSIZE);
	bufr[0] = 256;
	bufr[1] = 1;
	bufr[2] = 22;
	bufr[3] = -1;
	bufr[4] = 25;
	bufr[5] = 62;
	bufr[6] = 79;
	bufr[7] = 106;
	bufr[24] = 37;
	bufr[25] = -1;
	bufr[61] = 17;
	bufr[62] = -1;
 	bufr[78] = 27;
	bufr[79] = -1;
	bufr[105] = 52;
	bufr[106] = -1;
	bufr[63] = header.data_type;
	bufr[66] = header.dimension_1;
	bufr[67] = header.dimension_2;
	ftovaxf( header.scale_factor, &bufr[91]);
	bufr[93] = header.norm_hour;
	bufr[94] = header.norm_minute;
	bufr[95] = header.norm_second;
	bufr[96] = header.norm_day;
	bufr[97] = header.norm_month;
	bufr[98] = header.norm_year;
	ftovaxf( header.fov_source_width, &bufr[99]);
	ftovaxf( header.ecat_calib_factor, &bufr[101]);
	if (ntohs(1) == 1) swab( (char*)bufr, bbufr, MatBLKSIZE);
	else memcpy(bbufr,bufr,MatBLKSIZE);
	return 1;
}

