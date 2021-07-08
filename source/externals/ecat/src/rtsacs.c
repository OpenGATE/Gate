/* @(#)rtsacs.c	1.2 2/19/93 */

#ifndef	lint
static char sccsid[]="@(#)rtsacs.c	1.2 2/19/93 Copyright 1991 CTI Pet Systems, Inc.";
#endif	lint

#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <rpc/rpc.h>
#include <errno.h>
#include "matrix.h"
#include "ecatRts.h"
#include "ecatAcs.h"

extern	int	errno ;

static CLIENT *acqClient;
static struct timeval acqTimeout;

bool_t doAcsAcqCommand (command, inProc, inArg, outProc, outArg)
int command;
xdrproc_t inProc, outProc;
caddr_t *inArg, *outArg;
{
    int status;

    if (!acqClient)
	initAcqTcpClient ();
    status = (int) clnt_call (acqClient, command, inProc, inArg, outProc, outArg, acqTimeout);
    if ((status != (int) RPC_SUCCESS) && (errno != EINTR))
    {
	clnt_perror (acqClient, "acqClient_error");
	return (FALSE);
    }
    return (TRUE);
}


int initAcqTcpClient ()
{
    struct sockaddr_in sock;
    struct hostent *hp;
    char	*host, *getenv() ;

    int channel, status, response;
    u_long program = RTS_SERVER, version = RTS_SERVER_VERSION;
/*
    hp = gethostbyname (defaults_get_string ("/Ecat/EcatAcqServer", "localhost", 0));
*/
    host = getenv("VXHOST") ;
    if (!host) host="acs" ;    
    hp = gethostbyname (host);
    memcpy ((caddr_t) & sock.sin_addr, hp->h_addr, hp->h_length);
    sock.sin_family = AF_INET;
    sock.sin_port = 0;
    channel = RPC_ANYSOCK;
    acqTimeout.tv_sec = 60;
    acqTimeout.tv_usec = 0;

    if ((acqClient = clnttcp_create (&sock, program, version, &channel,
		1024, 1024)) == NULL)
    {
	clnt_pcreateerror ("acqClient_error");
	return (0);
    }
    return (1);
}

int rts_rmhd (file, mh)
char *file;
XMAIN_HEAD *mh;
{
    RMHD_resp rdResp;

    if (!(doAcsAcqCommand (RMHD, xdr_wrapstring, &file, xdr_RMHD_resp, &rdResp)))
	return (FALSE);
    memcpy (mh, &rdResp.xmain_head, sizeof (XMAIN_HEAD));
    return (rdResp.status);
}

int rts_wmhd (file, mh)
char *file;
XMAIN_HEAD *mh;
{
    WMHD_args wrArgs;
    int resp;

    wrArgs.file_name = file;
    memcpy (&wrArgs.xmain_head, mh, sizeof (XMAIN_HEAD));
    if (!(doAcsAcqCommand (WMHD, xdr_WMHD_args, &wrArgs, xdr_int, &resp)))
	resp = FALSE;
    return (resp);
}

int rts_rshd (file, matnum, buffer)
char *file;
int matnum;
caddr_t buffer;
{
    RSHD_args rdArgs;
    RSHD_resp rdResp;

    rdArgs.file_name = file;
    rdArgs.matnum = matnum;
    if (!(doAcsAcqCommand (RSHD, xdr_RSHD_args, &rdArgs, xdr_RSHD_resp, &rdResp)))
	return (FALSE);
    memcpy (buffer, rdResp.rhdat, 512);
    return (rdResp.status);
}

int rts_wshd (file, matnum, buffer)
char *file;
int matnum;
caddr_t buffer;
{
    WSHD_args wrArgs;
    int resp;

    wrArgs.file_name = file;
    wrArgs.matnum = matnum;
    memcpy (wrArgs.whdat, buffer, 512);
    if (!(doAcsAcqCommand (WSHD, xdr_WSHD_args, &wrArgs, xdr_int, &resp)))
	return (FALSE);
    return (resp);
}

int rts_rdat (file, matnum, buffer, bufferSize)
char *file;
int matnum, *bufferSize;
caddr_t buffer;
{
    RDAT_args rdArgs;
    RDAT_resp rdResp;

    rdArgs.file_name = file;
    rdArgs.matnum = matnum;
    rdResp.rdat.rdat_val = (char *) buffer;
    if (!(doAcsAcqCommand (RDAT, xdr_RDAT_args, &rdArgs, xdr_RDAT_resp, &rdResp)))
	return (FALSE);
    *bufferSize = rdResp.rdat.rdat_len;
    return (rdResp.status);
}

int rts_wdat (file, matnum, dataType, buffer, bufferSize)
char *file;
int matnum, dataType, bufferSize;
caddr_t buffer;
{
    WDAT_args wrArgs;
    int resp;

    wrArgs.file_name = file;
    wrArgs.matnum = matnum ;
    wrArgs.data_type = dataType;
    wrArgs.wdat.wdat_len = bufferSize;
    wrArgs.wdat.wdat_val = (char *) calloc (1, bufferSize);
    memcpy (wrArgs.wdat.wdat_val, buffer, bufferSize);
    if (!(doAcsAcqCommand (WDAT, xdr_WDAT_args, &wrArgs, xdr_int, &resp)))
	resp = FALSE;
    free (wrArgs.wdat.wdat_val);
    return (resp);
}

convertScanHeaderFromVax (buf, h)
short int *buf;
Scan_subheader *h;
{
    int i;
    float get_vax_float ();
	short b[MatBLKSIZE/2];

    SWAB (buf, b, MatBLKSIZE);
    h->data_type = b[63];
    h->dimension_1 = b[66];
    h->dimension_2 = b[67];
    h->smoothing = b[68];
    h->processing_code = b[69];
    h->sample_distance = get_vax_float (b, 73);
    h->isotope_halflife = get_vax_float (b, 83);
    h->frame_duration_sec = b[85];
    h->gate_duration = get_vax_long (b, 86);
    h->r_wave_offset = get_vax_long (b, 88);
    h->scale_factor = get_vax_float (b, 91);
    h->scan_min = b[96];
    h->scan_max = b[97];
    h->prompts = get_vax_long (b, 98);
    h->delayed = get_vax_long (b, 100);
    h->multiples = get_vax_long (b, 102);
    h->net_trues = get_vax_long (b, 104);
    for (i = 0; i < 16; i++)
    {
	h->cor_singles[i] = get_vax_float (b, 158 + 2 * i);
	h->uncor_singles[i] = get_vax_float (b, 190 + 2 * i);
    }
    h->tot_avg_cor = get_vax_float (b, 222);
    h->tot_avg_uncor = get_vax_float (b, 224);
    h->total_coin_rate = get_vax_long (b, 226);
    h->frame_start_time = get_vax_long (b, 228);
    h->frame_duration = get_vax_long (b, 230);
    h->loss_correction_fctr = get_vax_float (b, 232);
}

convertImageHeaderFromVax (buf, h)
short int *buf;
Image_subheader *h;
{
    int i;
	short b[MatBLKSIZE/2];
    float get_vax_float ();

    strncpy (h->annotation, b + 420, 40);
    SWAB (buf, b, MatBLKSIZE);
    h->data_type = b[63];
    h->num_dimensions = b[64];
    h->dimension_1 = b[66];
    h->dimension_2 = b[67];
    h->x_origin = get_vax_float (b, 80);
    h->y_origin = get_vax_float (b, 82);
    h->recon_scale = get_vax_float (b, 84);
    h->quant_scale = get_vax_float (b, 86);
    h->image_min = b[88];
    h->image_max = b[89];
    h->pixel_size = get_vax_float (b, 92);
    h->slice_width = get_vax_float (b, 94);
    h->frame_duration = get_vax_long (b, 96);
    h->frame_start_time = get_vax_long (b, 98);
    h->slice_location = b[100];
    h->recon_start_hour = b[101];
    h->recon_start_minute = b[102];
    h->recon_start_sec = b[103];
    h->recon_duration = get_vax_long (b, 104);
    h->filter_code = b[118];
    h->scan_matrix_num = get_vax_long (b, 119);
    h->norm_matrix_num = get_vax_long (b, 121);
    h->atten_cor_matrix_num = get_vax_long (b, 123);
    h->image_rotation = get_vax_float (b, 148);
    h->plane_eff_corr_fctr = get_vax_float (b, 150);
    h->decay_corr_fctr = get_vax_float (b, 152);
    h->loss_corr_fctr = get_vax_float (b, 154);
    h->processing_code = b[188];
    h->quant_units = b[190];
    h->recon_start_day = b[191];
    h->recon_start_month = b[192];
    h->recon_start_year = b[193];
    h->ecat_calibration_fctr = get_vax_float (b, 194);
    h->well_counter_cal_fctr = get_vax_float (b, 196);
    for (i = 0; i < 6; i++)
	h->filter_params[i] = get_vax_float (b, 198 + 2 * i);
}

convertAttnHeaderFromVax (buf, header)
short int *buf;
Attn_subheader *header;
{
    int get_vax_long ();
	short b[MatBLKSIZE/2];
    float get_vax_float ();

    SWAB (buf, b, MatBLKSIZE);
    header->data_type = bufr[63];
    header->attenuation_type = bufr[64];
    header->dimension_1 = bufr[66];
    header->dimension_2 = bufr[67];
    header->scale_factor = get_vax_float (bufr, 91);
    header->x_origin = get_vax_float (bufr, 93);
    header->y_origin = get_vax_float (bufr, 95);
    header->x_radius = get_vax_float (bufr, 97);
    header->y_radius = get_vax_float (bufr, 99);
    header->tilt_angle = get_vax_float (bufr, 101);
    header->attenuation_coeff = get_vax_float (bufr, 103);
    header->sample_distance = get_vax_float (bufr, 105);
}

convertNormHeaderFromVax (buf, h)
short int *buf;
Norm_subheader *h;
{
    int get_vax_long ();
	short b[MatBLKSIZE/2];
    float get_vax_float ();

    SWAB (buf, b, MatBLKSIZE);
    h->data_type = bufr[63];
    h->dimension_1 = bufr[66];
    h->dimension_2 = bufr[67];
    h->scale_factor = get_vax_float (bufr, 91);
    h->norm_hour = bufr[93];
    h->norm_minute = bufr[94];
    h->norm_second = bufr[95];
    h->norm_day = bufr[96];
    h->norm_month = bufr[97];
    h->norm_year = bufr[98];
    h->fov_source_width = get_vax_float (bufr, 99);
}

convertScanHeaderToVax (buf, header)
short int *buf;
Scan_subheader *header;
{
    int i;
	short bufr[MatBLKSIZE/2];

    for (i = 0; i < 256; bufr[i++] = 0);
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
    bufr[63] = header->data_type;
    bufr[66] = header->dimension_1;	/* x dimension */
    bufr[67] = header->dimension_2;	/* y_dimension */
    bufr[68] = header->smoothing;
    bufr[69] = header->processing_code;
    ftovaxf (header->sample_distance, &bufr[73]);
    ftovaxf (header->isotope_halflife, &bufr[83]);
    bufr[85] = header->frame_duration_sec;
    sunltovaxl (header->gate_duration, &bufr[86]);
    sunltovaxl (header->r_wave_offset, &bufr[88]);
    ftovaxf (header->scale_factor, &bufr[91]);
    bufr[96] = header->scan_min;
    bufr[97] = header->scan_max;
    sunltovaxl (header->prompts, &bufr[98]);
    sunltovaxl (header->delayed, &bufr[100]);
    sunltovaxl (header->multiples, &bufr[102]);
    sunltovaxl (header->net_trues, &bufr[104]);
    for (i = 0; i < 16; i++)
    {
	ftovaxf (header->cor_singles[i], &bufr[158 + 2 * i]);
	ftovaxf (header->uncor_singles[i], &bufr[190 + 2 * i]);
    };
    ftovaxf (header->tot_avg_cor, &bufr[222]);
    ftovaxf (header->tot_avg_uncor, &bufr[224]);
    sunltovaxl (header->total_coin_rate, &bufr[226]);	/* total coin rate */
    sunltovaxl (header->frame_start_time, &bufr[228]);
    sunltovaxl (header->frame_duration, &bufr[230]);
    ftovaxf (header->loss_correction_fctr, &bufr[232]);
    SWAB (bufr, buf, MatBLKSIZE);
}

convertImageHeaderToVax (buf, header)
short int *buf;
Image_subheader *header;
{
    char *bbufr;
	short bufr[MatBLKSIZE/2];
    int i;

    for (i = 0; i < 256; bufr[i++] = 0);
    bbufr = (char *) buf;
    bufr[63] = header->data_type;
    bufr[64] = header->num_dimensions;
    bufr[66] = header->dimension_1;
    bufr[67] = header->dimension_2;
    ftovaxf (header->x_origin, &bufr[80]);
    ftovaxf (header->y_origin, &bufr[82]);
    ftovaxf (header->recon_scale, &bufr[84]);
    ftovaxf (header->quant_scale, &bufr[86]);
    bufr[88] = header->image_min;
    bufr[89] = header->image_max;
    ftovaxf (header->pixel_size, &bufr[92]);
    ftovaxf (header->slice_width, &bufr[94]);
    sunltovaxl (header->frame_duration, &bufr[96]);
    sunltovaxl (header->frame_start_time, &bufr[98]);
    bufr[100] = header->slice_location;
    bufr[101] = header->recon_start_hour;
    bufr[102] = header->recon_start_minute;
    bufr[103] = header->recon_start_sec;
    sunltovaxl (header->recon_duration, &bufr[104]);
    bufr[118] = header->filter_code;
    sunltovaxl (header->scan_matrix_num, &bufr[119]);
    sunltovaxl (header->norm_matrix_num, &bufr[121]);
    sunltovaxl (header->atten_cor_matrix_num, &bufr[123]);
    ftovaxf (header->image_rotation, &bufr[148]);
    ftovaxf (header->plane_eff_corr_fctr, &bufr[150]);
    ftovaxf (header->decay_corr_fctr, &bufr[152]);
    ftovaxf (header->loss_corr_fctr, &bufr[154]);
    bufr[188] = header->processing_code;
    bufr[190] = header->quant_units;
    bufr[191] = header->recon_start_day;
    bufr[192] = header->recon_start_month;
    bufr[193] = header->recon_start_year;
    ftovaxf (header->ecat_calibration_fctr, &bufr[194]);
    ftovaxf (header->well_counter_cal_fctr, &bufr[196]);
    for (i = 0; i < 6; i++)
	ftovaxf (header->filter_params[i], &bufr[198 + 2 * i]);
    SWAB (bufr, buf, MatBLKSIZE);
    strcpy (bbufr + 420, header->annotation);
}

convertAttnHeaderToVax (buf, header)
short int *buf;
Attn_subheader *header;
{
	short bufr[MatBLKSIZE/2];
    int i;

    for (i = 0; i < 256; bufr[i++] = 0);
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
    bufr[63] = header->data_type;
    bufr[64] = header->attenuation_type;
    bufr[66] = header->dimension_1;
    bufr[67] = header->dimension_2;
    ftovaxf (header->scale_factor, &bufr[91]);
    ftovaxf (header->x_origin, &bufr[93]);
    ftovaxf (header->y_origin, &bufr[95]);
    ftovaxf (header->x_radius, &bufr[97]);
    ftovaxf (header->y_radius, &bufr[99]);
    ftovaxf (header->tilt_angle, &bufr[101]);
    ftovaxf (header->attenuation_coeff, &bufr[103]);
    ftovaxf (header->sample_distance, &bufr[105]);
    SWAB (bufr, buf, 512);
}

convertNormHeaderToVax (buf, header)
short int *buf;
Norm_subheader *header;
{
    int i;
	short bufr[MatBLKSIZE/2];

    for (i = 0; i < 256; bufr[i++] = 0);
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
    bufr[63] = header->data_type;
    bufr[66] = header->dimension_1;
    bufr[67] = header->dimension_2;
    ftovaxf (header->scale_factor, &bufr[91]);
    bufr[93] = header->norm_hour;
    bufr[94] = header->norm_minute;
    bufr[95] = header->norm_second;
    bufr[96] = header->norm_day;
    bufr[97] = header->norm_month;
    bufr[98] = header->norm_year;
    ftovaxf (header->fov_source_width, &bufr[99]);
    SWAB (bufr, buf, 512);
}

bool_t
xdr_RMHD_resp(xdrs, objp)
	XDR *xdrs;
	RMHD_resp *objp;
{
	if (!xdr_int(xdrs, &objp->status)) {
		return (FALSE);
	}
	if (!xdr_XMAIN_HEAD(xdrs, &objp->xmain_head)) {
		return (FALSE);
	}
	return (TRUE);
}




bool_t
xdr_WMHD_args(xdrs, objp)
	XDR *xdrs;
	WMHD_args *objp;
{
	if (!xdr_string(xdrs, &objp->file_name, ~0)) {
		return (FALSE);
	}
	if (!xdr_XMAIN_HEAD(xdrs, &objp->xmain_head)) {
		return (FALSE);
	}
	return (TRUE);
}




bool_t
xdr_RSHD_args(xdrs, objp)
	XDR *xdrs;
	RSHD_args *objp;
{
	if (!xdr_string(xdrs, &objp->file_name, ~0)) {
		return (FALSE);
	}
	if (!xdr_int(xdrs, &objp->matnum)) {
		return (FALSE);
	}
	return (TRUE);
}




bool_t
xdr_RSHD_resp(xdrs, objp)
	XDR *xdrs;
	RSHD_resp *objp;
{
	if (!xdr_int(xdrs, &objp->status)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->rhdat, 512)) {
		return (FALSE);
	}
	return (TRUE);
}




bool_t
xdr_WSHD_args(xdrs, objp)
	XDR *xdrs;
	WSHD_args *objp;
{
	if (!xdr_string(xdrs, &objp->file_name, ~0)) {
		return (FALSE);
	}
	if (!xdr_int(xdrs, &objp->matnum)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->whdat, 512)) {
		return (FALSE);
	}
	return (TRUE);
}




bool_t
xdr_RDAT_args(xdrs, objp)
	XDR *xdrs;
	RDAT_args *objp;
{
	if (!xdr_string(xdrs, &objp->file_name, ~0)) {
		return (FALSE);
	}
	if (!xdr_int(xdrs, &objp->matnum)) {
		return (FALSE);
	}
	return (TRUE);
}




bool_t
xdr_RDAT_resp(xdrs, objp)
	XDR *xdrs;
	RDAT_resp *objp;
{
	if (!xdr_int(xdrs, &objp->status)) {
		return (FALSE);
	}
	if (!xdr_bytes(xdrs, (char **)&objp->rdat.rdat_val, (u_int *)&objp->rdat.rdat_len, ~0)) {
		return (FALSE);
	}
	return (TRUE);
}




bool_t
xdr_WDAT_args(xdrs, objp)
	XDR *xdrs;
	WDAT_args *objp;
{
	if (!xdr_string(xdrs, &objp->file_name, ~0)) {
		return (FALSE);
	}
	if (!xdr_int(xdrs, &objp->matnum)) {
		return (FALSE);
	}
	if (!xdr_int(xdrs, &objp->data_type)) {
		return (FALSE);
	}
	if (!xdr_bytes(xdrs, (char **)&objp->wdat.wdat_val, (u_int *)&objp->wdat.wdat_len, ~0)) {
		return (FALSE);
	}
	return (TRUE);
}

bool_t
xdr_XMAIN_HEAD(xdrs, objp)
	XDR *xdrs;
	XMAIN_HEAD *objp;
{
	if (!xdr_opaque(xdrs, objp->original_file_name, 20)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->sw_version)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->data_type)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->system_type)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->file_type)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->node_id, 10)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->scan_start_day)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->scan_start_month)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->scan_start_year)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->scan_start_hour)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->scan_start_minute)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->scan_start_second)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->isotope_code, 8)) {
		return (FALSE);
	}
	if (!xdr_float(xdrs, &objp->isotope_halflife)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->radiopharmaceutical, 32)) {
		return (FALSE);
	}
	if (!xdr_float(xdrs, &objp->gantry_tilt)) {
		return (FALSE);
	}
	if (!xdr_float(xdrs, &objp->gantry_rotation)) {
		return (FALSE);
	}
	if (!xdr_float(xdrs, &objp->bed_elevation)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->rot_source_speed)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->wobble_speed)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->transm_source_type)) {
		return (FALSE);
	}
	if (!xdr_float(xdrs, &objp->axial_fov)) {
		return (FALSE);
	}
	if (!xdr_float(xdrs, &objp->transaxial_fov)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->transaxial_samp_mode)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->coin_samp_mode)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->axial_samp_mode)) {
		return (FALSE);
	}
	if (!xdr_float(xdrs, &objp->calibration_factor)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->calibration_units)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->compression_code)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->study_name, 12)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->patient_id, 16)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->patient_name, 32)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->patient_sex, 1)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->patient_age, 10)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->patient_height, 10)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->patient_weight, 10)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->patient_dexterity, 1)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->physician_name, 32)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->operator_name, 32)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->study_description, 32)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->acquisition_type)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->bed_type)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->septa_type)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->facility_name, 20)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->num_planes)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->num_frames)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->num_gates)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->num_bed_pos)) {
		return (FALSE);
	}
	if (!xdr_float(xdrs, &objp->init_bed_position)) {
		return (FALSE);
	}
	if (!xdr_vector(xdrs, (char *)objp->bed_offset, 15, sizeof(float), xdr_float)) {
		return (FALSE);
	}
	if (!xdr_float(xdrs, &objp->plane_separation)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->lwr_sctr_thres)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->lwr_true_thres)) {
		return (FALSE);
	}
	if (!xdr_short(xdrs, &objp->upr_true_thres)) {
		return (FALSE);
	}
	if (!xdr_float(xdrs, &objp->collimator)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->user_process_code, 10)) {
		return (FALSE);
	}
	return (TRUE);
}

int rtsRblk (file, blockNumber, buffer)
char *file;
int blockNumber;
caddr_t buffer;
{
    RBLK_args rbArgs;
    RBLK_resp rbResp;

    rbArgs.filename = file;
    rbArgs.block_number = blockNumber;
    if (!(doAcsAcqCommand (RBLK, xdr_RBLK_args, &rbArgs, xdr_RBLK_resp, &rbResp)))
	return (FALSE);
    memcpy (buffer, rbResp.rblk, 512);
    return (rbResp.status);
}

int rtsWblk (file, blockNumber, buffer)
char *file;
int blockNumber;
caddr_t buffer;
{
    WBLK_args wbArgs;
    int resp;

    wbArgs.filename = file;
    wbArgs.block_number = blockNumber;
    memcpy (wbArgs.wblk, buffer, 512);
    if (!(doAcsAcqCommand (WBLK, xdr_WBLK_args, &wbArgs, xdr_int, &resp)))
	return (FALSE);
    return (resp);
}


bool_t
xdr_RBLK_args(xdrs, objp)
	XDR *xdrs;
	RBLK_args *objp;
{
	if (!xdr_string(xdrs, &objp->filename, ~0)) {
		return (FALSE);
	}
	if (!xdr_int(xdrs, &objp->block_number)) {
		return (FALSE);
	}
	return (TRUE);
}




bool_t
xdr_WBLK_args(xdrs, objp)
	XDR *xdrs;
	WBLK_args *objp;
{
	if (!xdr_string(xdrs, &objp->filename, ~0)) {
		return (FALSE);
	}
	if (!xdr_int(xdrs, &objp->block_number)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->wblk, 512)) {
		return (FALSE);
	}
	return (TRUE);
}

bool_t
xdr_RBLK_resp(xdrs, objp)
	XDR *xdrs;
	RBLK_resp *objp;
{
	if (!xdr_int(xdrs, &objp->status)) {
		return (FALSE);
	}
	if (!xdr_opaque(xdrs, objp->rblk, 512)) {
		return (FALSE);
	}
	return (TRUE);
}
