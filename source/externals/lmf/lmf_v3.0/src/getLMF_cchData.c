/*-------------------------------------------------------

           List Mode Format 
                        
     --  getLMF_cchData.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of getLMF_cchData.c:
	 Functions used for the ascii part of LMF:
	 getLMF_cchData - find a data in structures LMF_cch
	 ->getLMF_cchNumericalValue - find a data = numerical value + unit 
	                              in structures LMF_cch
				      only energy, distance, surface, volume, time, 
				      activity, speed,angle, rotation speed, weigth,
				      temperature, electric and magnetic field, pression

	 ->getLMF_cchInfo - find a data in the  structures LMF_cch
	                    all type of data (date, string, number, energy, distance,
			    surface, volume, time, activity, speed, angle, rotation speed,
			    weigth, temperature, electric and magnetic field, pression)
-------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"


/* getLMF_cchNumericalValue - find a data = numerical value + unit  in structures LMF_cch **/

/* !!!! only numerical value + unit (energy, distance, surface, volume, time,  activity, speed,
   angle,rotation speed, weigth, temperature, electric and magnetic field, pression) *********/

contentLMFdata getLMF_cchNumericalValue(i8 field[charNum])
{

  int cch_index = 0;
  contentLMFdata valueToReconstruction = { 0 };

  if (strlen(field) >= charNum)
    printf(ERROR4, charNum);
  for (cch_index = 0; cch_index <= last_cch_list; cch_index++) {
    if (strcasecmp(plist_cch[cch_index].field, field) == 0)
      break;
  }

  if (strcasecmp(plist_cch[cch_index].field, field) != 0) {
    printf(ERROR31, field);
    printf(ERROR5, cchFileName);
    exit(EXIT_FAILURE);
  }

  valueToReconstruction.numericalValue =
      plist_cch[cch_index].def_unit_value.vNum;
  strcpy(valueToReconstruction.unit, plist_cch[cch_index].def_unit);

  return (valueToReconstruction);
}


/* getLMF_cchInfo - find a data in the  structures LMF_cch *********************************/
/* all type of data (date, string, number, energy, distance, surface, volume, time, activity,
   speed, angle,rotation speed, weigth, temperature, electric and magnetic field, pression) */


int getLMF_cchInfo(i8 field[charNum])
{

  int cch_index = 0;

  if (strlen(field) >= charNum)
    printf(ERROR4, charNum);
  for (cch_index = 0; cch_index <= last_cch_list; cch_index++) {
    if (strcasecmp(plist_cch[cch_index].field, field) == 0)
      break;
  }

  if (strcasecmp(plist_cch[cch_index].field, field) != 0) {
    printf(ERROR31, field);
    printf(ERROR5, cchFileName);
    exit(EXIT_FAILURE);
  }

  return (cch_index);
}
