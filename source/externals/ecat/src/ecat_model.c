#include "ecat_model.h"
#include <stdio.h>

/* 

If you need to complete this file (or update the structures), 
... go on the scanner console and type:

> capp
IDL. Version 4.0 (sunos sparc).
Copyright 1989-1995, Research Systems, Inc.
All rights reserved.  Unauthorized reproduction prohibited.
% LICENSE MANAGER: No such feature exists (-5,116).		(because you have a run time (RT) license only)
  License file: /home/idl/license.dat
% Machine not licensed for IDL.  Entering 7 minute Demo mode.
  This Demo mode is for short-term product evaluation purposes only
CAPP>st=db_getmodelinfo(model_info,model='961')
% Restored file: DB_GETMODELINFO.
% Restored file: SYS_BIN.
% Restored file: GET_BIN_DIR_.
CAPP>help,/str,model_info

Please note the intrinsic tilt integer in capp is float

January 1997:
----------
For 961, remove old bump correction 6 parameters for geometrical efficiency: 1, 0., 1.2e-6, 1.02, 71, 79};
Add data from other scanners and update parameters using 70Rel3 info (Johan Nuyts)

Dec 2001, update for 921-922-923 ... CM

*/
static EcatModel _ecat_model_921= {
	"921", 3, 36, 4, 1, 4, 4, 8, 8, 61440, 1536,
	 2, 0, 0, 24, 11, 7, 17, 2, 30.9, 2, 192, 192, 0,
	16.2, 58.3, 41.25, 236.2, 650, 250, 0, 0.3375, 0.3375, 3.e-7, 1.0,
	12, 1.0, 0.0, 15., 0, -1, -1, -1, 5, 0, 2., 0, 7, 0, 0};

static EcatModel _ecat_model_922= {
	"922", 1, 12, 4, 3, 12, 4, 8, 8, 8192, 5632,
	4, 1, 1, 24, 11, 7, 17, 2, 30.9, 2, 192, 192, 0,
	16.2, 58.3, 41.25, 236.2, 650, 350, 0, 0.3375, 0.3375, 3.e-7, 1.0,
	12, 1.0, 1.0, 15., 0, -1, -1, -1, 5, 0, 2., 192, 7, 0, 1};

static EcatModel _ecat_model_923= {
	"923", 1, 12, 4, 3, 12, 4, 8, 8, 8192, 5632,
	4, 1, 1, 24, 11, 7, 17, 2, 30.9, 2, 192, 192, 0,
	16.2, 58.3, 41.25, 236.2, 650, 350, 0, 0.3375, 0.3375, 3.e-7, 1.0,
	12, 1.0, 1.0, 15., 0, -1, -1, -1, 5, 0, 2., 192, 7, 0, 1};

static EcatModel _ecat_model_925 = {
	"925", 1, 6, 4, 3, 12, 4, 8, 8, 8192, 5632,
	4, 1, 1, 24, 35, 7, 17, 3, 30., 0, 192, 192, 0,
	16.2, 58.3, 41.25, 222.7, 650, 350, 0, 0.3375, 0.3375, 3.e-7, 1.0,
	0, 1.0, 1.0, 0., 0, -1, -1, -1, 11, 1, 2., 192, 3, 1, 1};

static EcatModel _ecat_model_961 = {
	"961", 3, 42, 8, 1, 8, 4, 8, 7, 8192, 3072,
	4, 1, 1, 24, 11, 7, 17, 2, 30.9, 2, 336, 392, 1,
	15.0, 51.4, 41.2, 235.0, 650, 350, 0, 0.3125, 0.165, 3.e-7, 1.0,
	12, 1.0, 0.0, 13., 0, -1, -1, -1, 5, 0, 3., 336, 7, 0, 0}; 

static EcatModel _ecat_model_962 = {
 	"962", 1, 24, 3, 4, 12, 4, 8, 8, 8192, 5632,
 	4, 1, 1, 32, 15, 9, 22, 2, 30.9, 2, 288, 288, 1,
 	15.52, 58.3, 41.2, 235.52, 650, 350, 0, 0.2425, 0.225, 3.0e-07, 1.00,
 	12, 1.0, 0.0, 0., 0, -1, -1, -1, 7, 0, 3., 288, 13, 2, 1};

static EcatModel _ecat_model_966 = {
 	"966", 1, 36, 2, 6, 12, 4, 8, 8, 8192, 5632,
 	4, 1, 1, 48, 15, 9, 22, 3, 39.75, 0, 288, 288, 1,
 	23.28, 58.3, 41.2, 243.28, 650, 350, 0, 0.2425, 0.225, 3.0e-07, 1.00,
 	12, 1.0, 1.0, 0., 0, -1, -1, -1, 7, 0, 3., 288, 19, 2, 1}; 

EcatModel *ecat_model(system_type)
int system_type;
{
	switch(system_type) {
		case 921:
			return (&_ecat_model_921);
		case 922:
			return (&_ecat_model_922);
		case 923:
			return (&_ecat_model_923);
		case 925:
			return (&_ecat_model_925);
		case 961:
			return (&_ecat_model_961);
		case 962:
			return (&_ecat_model_962);
		case 966:
			return (&_ecat_model_966);
	}

	fprintf(stderr," Sorry, this model is not considered in this SW \n");
	return 0;
}

