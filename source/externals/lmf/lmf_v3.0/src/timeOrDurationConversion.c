/*-------------------------------------------------------

           List Mode Format 
                        
     --  timeConversion.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of timeConversion.c:


	Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	Function used for the ascii part of LMF:
	Fill in the members value and default_value of the LMF_cch structure 
	with a time


-------------------------------------------------------*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "lmf.h"

/* timeConversion - Fill in the members value and default_value of the LMF_cch structure
      with a time */

int timeConversion(int cch_index)
{

  struct tm result_time = { 0 };

  result_time =
      modifyTimeFormat(plist_cch[cch_index].data,
		       plist_cch[cch_index].field);
  plist_cch[cch_index].value.tps_date.tm_sec = result_time.tm_sec;
  plist_cch[cch_index].value.tps_date.tm_min = result_time.tm_min;
  plist_cch[cch_index].value.tps_date.tm_hour = result_time.tm_hour;
  plist_cch[cch_index].def_unit_value.tps_date.tm_sec =
      plist_cch[cch_index].value.tps_date.tm_sec;
  plist_cch[cch_index].def_unit_value.tps_date.tm_min =
      plist_cch[cch_index].value.tps_date.tm_min;
  plist_cch[cch_index].def_unit_value.tps_date.tm_hour =
      plist_cch[cch_index].value.tps_date.tm_hour;

  return (EXIT_SUCCESS);
}
