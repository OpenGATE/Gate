/*-------------------------------------------------------

           List Mode Format 
                        
     --  volumeConversion.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of volumeConversion.c:


	Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	Function used for the ascii part of LMF:
	Fill in the members value and default_value of the LMF_cch structure with a volume 
	(numerical value + unit)

--------------------------------------------------------*/



#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "lmf.h"

/**** volumeConversion - Fill in the members value and default_value of the LMF_cch structure
      with a volume (numerical value + unit) ****/

int volumeConversion(int cch_index, int dataType)
{

  i8 default_unit_file[charNum], surface_volume_unit[charNum],
      surface_volume_default_unit[charNum];
  double result_prefix_conversion = 0;
  result_unit_conversion result_conversion = { 0 };
  content_data_unit content_data = { 0 }, unit = {
  0}, default_unit = {
  0};



  initialize(default_unit_file);
  strcpy(default_unit_file, DEFAULT_UNITS_FILE);

  content_data = defineUnitAndValue(plist_cch[cch_index].data);
  plist_cch[cch_index].value.vNum = content_data.value;
  strcpy(plist_cch[cch_index].unit, content_data.unit);
  initialize(surface_volume_unit);
  initialize(surface_volume_default_unit);
  strcpy(surface_volume_unit,
	 modifySurfaceOrVolumeFormat(plist_cch[cch_index].unit, dataType,
				     "3"));

  if (strcmp(surface_volume_unit, "error") == 0) {
    printf(ERROR14, plist_cch[cch_index].field, plist_cch[cch_index].unit);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }
  strcpy(surface_volume_default_unit,
	 modifySurfaceOrVolumeFormat(plist_cch[cch_index].def_unit,
				     dataType, "3"));

  if (strcmp(surface_volume_default_unit, "error") == 0) {
    printf(ERROR14, plist_cch[cch_index].field,
	   plist_cch[cch_index].def_unit);
    printf(ERROR5, default_unit_file);
    return (EXIT_FAILURE);
  }
  dataType = 13;
  unit = definePrefixAndUnit(surface_volume_unit, dataType);

  if (unit.unit[0] == '\0') {
    printf(ERROR14, plist_cch[cch_index].field, plist_cch[cch_index].data);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }
  default_unit =
      definePrefixAndUnit(surface_volume_default_unit, dataType);

  if (default_unit.unit[0] == '\0') {
    printf(ERROR25, default_unit_file);
    printf(ERROR14, plist_cch[cch_index].field,
	   plist_cch[cch_index].def_unit);
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
      plist_cch[cch_index].value.vNum * (result_prefix_conversion *
					 result_conversion.factor *
					 result_prefix_conversion *
					 result_conversion.factor *
					 result_prefix_conversion *
					 result_conversion.factor)
      + result_conversion.constant;

  return (EXIT_SUCCESS);
}
