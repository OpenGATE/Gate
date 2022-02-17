/* @(#)rtsacs.c	1.2 2/19/93 */

#ifndef	lint
static char sccsid[]="@(#)rtsacs.c	1.2 2/19/93 Copyright 1991 CTI Pet Systems, Inc.";
#endif	/* lint */

#include <sys/types.h>
#include <sys/socket.h>
#include <rpc/rpc.h>
#if defined(__sun) && defined(__SVR4)       /* solaris 2.x*/
#include <rpc/clnt_soc.h>
#endif
#include <netdb.h>
#include <sys/time.h>
#include <stdio.h>
#include <errno.h>
#include "matrix.h"
#include "matrix_xdr.h"
#include "rfa_xdr.h"
/*
 * ACS/1 access
 */

#define RTS_SERVER 600000032
#define RTS_SERVER_VERSION 1
#define RTS_INFO_SERVER 600000036
#define RTS_INFO_VERSION 1

extern	int	errno ;

static CLIENT *acqClient;
static struct timeval acqTimeout;

bool_t doAcsAcqCommand (command, inProc, inArg, outProc, outArg)
int command;
xdrproc_t inProc, outProc;
caddr_t inArg, outArg;
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
/* ACS/1
    u_long program = RTS_SERVER, version = RTS_SERVER_VERSION;
*/
    u_long program = MATRIX_SERVER, version = MATRIX_SERVER_VERSION;
/*
    hp = gethostbyname (defaults_get_string ("/Ecat/EcatAcqServer", "localhost", 0));
*/
    if ((host = getenv("VXHOST")) == NULL)  host = "acs";
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
    memcpy (wrArgs.whdat,buffer, 512);
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

byte_acs_read(file,file_pos,buffer,size)
char *file;
int file_pos, size;
caddr_t buffer;
{
	int blk, startblk, offset, endblk;
	int cc=0, count=0;
    RBLK_args rbArgs;
    RBLK_resp rbResp;

	startblk = file_pos/MatBLKSIZE+1;
	offset = file_pos%MatBLKSIZE;
	endblk = ((file_pos+size+(MatBLKSIZE-1))/MatBLKSIZE)+1;
    rbArgs.filename = file;
	for (blk=startblk; blk<endblk; blk++) {
		if ((size-count)>MatBLKSIZE) cc = MatBLKSIZE - offset;
		else cc = (size-count) - offset;
    	rbArgs.block_number = blk;
    	if (!(doAcsAcqCommand (RBLK, xdr_RBLK_args, &rbArgs, xdr_RBLK_resp,
			&rbResp))) return -1;
    	memcpy (buffer+count, rbResp.rblk+offset, cc);
		offset = 0;
		count += cc;
	}
	return count;
}
