/*-------------------------------------------------------

           List Mode Format 
                        
     --  durationConversion.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of durationConversion.c:

	Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	Function used for the ascii part of LMF:
	Fill in the members value and default_value of the LMF_cch structure with a duration
		 
---------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "lmf.h"

/**** durationConversion - Fill in the members value and default_value of the LMF_cch structure
      with a duration ****/

int durationConversion(int cch_index)
{

  i8 default_unit_file[charNum];
  int dataType = 4;
  double result_prefix_conversion = 0;
  content_data_unit content_data = { 0 }, unit = {
  0}, default_unit = {
  0};
  result_unit_conversion result_conversion = { 0 };

  initialize(default_unit_file);
  strcpy(default_unit_file, DEFAULT_UNITS_FILE);

  content_data =
      modifyDurationFormat(plist_cch[cch_index].data,
			   plist_cch[cch_index].field);

  if (content_data.unit[0] == '\0') {
    printf(ERROR48, plist_cch[cch_index].field, plist_cch[cch_index].data);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }
  plist_cch[cch_index].value.vNum = content_data.value;
  strcpy(plist_cch[cch_index].unit, content_data.unit);
  unit = definePrefixAndUnit(plist_cch[cch_index].unit, dataType);

  if (unit.unit[0] == '\0') {
    printf(ERROR48, plist_cch[cch_index].field, plist_cch[cch_index].data);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }
  default_unit =
      definePrefixAndUnit(plist_cch[cch_index].def_unit, dataType);

  if (default_unit.unit[0] == '\0') {
    printf(ERROR25, default_unit_file);
    printf(ERROR5, default_unit_file);
    return (EXIT_FAILURE);
  }
  result_prefix_conversion =
      findPrefixConversionFactor(unit, default_unit);

  if ((result_prefix_conversion) == -1) {
    printf(ERROR24, plist_cch[cch_index].field, plist_cch[cch_index].unit);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }

  else if ((result_prefix_conversion) == -2) {
    printf(ERROR24, plist_cch[cch_index].field,
	   plist_cch[cch_index].def_unit);
    printf(ERROR5, default_unit_file);
    return (EXIT_FAILURE);
  }
  result_conversion =
      findUnitConversionFactor(unit, default_unit, dataType);

  plist_cch[cch_index].def_unit_value.vNum =
      (plist_cch[cch_index].value.vNum * result_prefix_conversion *
       result_conversion.factor)
      + result_conversion.constant;


  return (EXIT_SUCCESS);
}
