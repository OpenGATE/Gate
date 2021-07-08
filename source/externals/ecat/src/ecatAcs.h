/* @(#)ecatAcs.h	1.1 4/13/91 */

#ifndef ecatAcs_defined
#define ecatAcs_defined

#define ECAT_ACS_SERVER 600000035
#define ECAT_ACS_VERSION 1
#define FrameServer 600000037
#define FrameServerVersion 1

enum ECAT_ACS_FUNCTIONS {
	ACS_MAGPIE = 1,
	ACS_ALLOCATE = 1 + 1,
	ACS_DEALLOCATE = 1 + 2,
	MCS_ALLOCATE = 1 + 3,
	MCS_DEALLOCATE = 1 + 4,
	ACS_WHAT_TIME_IS_IT = 1 + 5,
	ACS_FRAME_COMPLETE = 1 + 6,
	ACS_FRAME_TRANSFERRED = 1 + 7,
	IS_THIS_FRAME_COMPLETE = 1 + 8,
	IS_THIS_FRAME_TRANSFERRED = 1 + 9,
	RESET_FRAME_TRANSFERRED = 1 + 10,
	ACS_IMAGE_COMPLETE = 1 + 11,
	TELL_ME_ABOUT_STUFF = 1 + 12,
	STOP_TELLING_ME = 1 + 13,
	INIT_MY_MCS = 1 + 14,
	ACQ_STARTED = 1 + 15,
	RFA_FORMAT_COMPLETE = 1 + 16,
	RFA_FORMAT_STARTED = 1 + 17,
	IS_FORMAT_COMPLETE = 1 + 18,
};
typedef enum ECAT_ACS_FUNCTIONS ECAT_ACS_FUNCTIONS;
bool_t xdr_ECAT_ACS_FUNCTIONS();


struct ButtonId {
	int requestorPid;
	char *requestedNode;
};
typedef struct ButtonId ButtonId;
bool_t xdr_ButtonId();


struct TIMEX_resp {
	int seconds;
	int minute;
	int hour;
	int day;
	int month;
	int year;
};
typedef struct TIMEX_resp TIMEX_resp;
bool_t xdr_TIMEX_resp();


struct TELL_ME_args {
	char *whoIam;
	int tellMeAboutFrames;
	int tellMeAboutImages;
};
typedef struct TELL_ME_args TELL_ME_args;
bool_t xdr_TELL_ME_args();


struct STOP_TELLING_ME_args {
	char *whoIam;
	int tellMeAboutFrames;
	int tellMeAboutImages;
};
typedef struct STOP_TELLING_ME_args STOP_TELLING_ME_args;
bool_t xdr_STOP_TELLING_ME_args();

#endif ecatAcs_defined
