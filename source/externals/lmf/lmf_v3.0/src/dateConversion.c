/*-------------------------------------------------------

           List Mode Format 
                        
     --  dateConversion.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of dateConversion.c:

	Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	Function used for the ascii part of LMF:
	Fill in the members value and default_value of the LMF_cch structure with a date
		 
---------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "lmf.h"

/**** dateConversion - Fill in the members value and default_value of the LMF_cch structure
      with a date ****/

int dateConversion(int cch_index)
{

  struct tm result_date = { 0 };

  result_date =
      modifyDateFormat(plist_cch[cch_index].data,
		       plist_cch[cch_index].field);
  if (result_date.tm_mday == 0) {
    printf(ERROR6, plist_cch[cch_index].field, plist_cch[cch_index].data);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }
  plist_cch[cch_index].value.tps_date.tm_mday = result_date.tm_mday;
  plist_cch[cch_index].value.tps_date.tm_mon = result_date.tm_mon;
  plist_cch[cch_index].value.tps_date.tm_year = result_date.tm_year;
  plist_cch[cch_index].def_unit_value.tps_date.tm_mday =
      plist_cch[cch_index].value.tps_date.tm_mday;
  plist_cch[cch_index].def_unit_value.tps_date.tm_mon =
      plist_cch[cch_index].value.tps_date.tm_mon;
  plist_cch[cch_index].def_unit_value.tps_date.tm_year =
      plist_cch[cch_index].value.tps_date.tm_year;

  return (EXIT_SUCCESS);
}
