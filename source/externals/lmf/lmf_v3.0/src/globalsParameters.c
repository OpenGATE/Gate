/*-------------------------------------------------------

           List Mode Format 
                        
     --  globalsParameters.c  --                      
   
     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of globalsParameters.c:
	 Fill in the LMF record carrier with the data contained 
	 in the LMF ASCII header file
	 Functions used for the ascii part of LMF:
	 ->globalsParameters - Globals (extern) variables declarations

-------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lmf.h"

/* globalsParameters - Globals (extern) variables declarations */

struct tm *structCchTimeDate;

i8 fileName[charNum];
/***************** ccs files ********************************/
i8 ccsFileName[charNum];
i8 coinci_ccsFileName[charNum];
i8 bis_ccsFileName[charNum];

/***************** cch files ********************************/
i8 cchFileName[charNum];
i8 coinci_cchFileName[charNum];
i8 bis_cchFileName[charNum];

/************************************************************/

lmf_header *plist_lmf_header;	/* declares plist_lmf_header to be of type pointer
				   to (structures) lmf_header */
lmf_header *first_lmf_header;	/* store the address of the begin of the structures
				   lmf_header array */

LMF_cch *plist_cch = NULL;	/* declares plist_cch to be of type pointer
				   to (structures) LMF_cch */
LMF_cch *first_cch_list;	/* declares plist_cch to be of type pointer 
				   to (structures) LMF_cch and to store the address of
				   the begin of the structures LMF_cch array */

int last_cch_list = 0;		/* index of the last element of the structures LMF_cch array */

/* Data used to create a list of Shift values */

double **ppShiftValuesList = NULL;
int lastShiftValuesList_index = 0;
