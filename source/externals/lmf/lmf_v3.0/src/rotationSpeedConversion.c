/*-------------------------------------------------------

           List Mode Format 
                        
     --  rotationSpeedConversion.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of rotationSpeedConversion.c:


	Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	Function used for the ascii part of LMF:
	Fill in the members value and default_value of the LMF_cch structure 
	with a rotation speed (numerical value + unit)

-------------------------------------------------------*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "lmf.h"

/* rotationSpeedConversion - Fill in the members value and default_value of the LMF_cch structure
   with a rotation speed (numerical value + unit) */

int rotationSpeedConversion(int cch_index, int dataType)
{

  double result_numerator_prefix_conversion =
      0, result_denominator_prefix_conversion = 0;

  content_data_unit content_data = { 0 }, numerator_unit = {
  0}, denominator_unit = {
  0}, numerator_default_unit = {
  0}, denominator_default_unit = {
  0};

  result_unit_conversion result_numerator_conversion = { 0 },
      result_denominator_conversion = {
  0};

  complex_unit_type complex_unit = { "" }, complex_default_unit = {
  ""};


  i8 default_unit_file[charNum], buffer[charNum];


  initialize(default_unit_file);
  strcpy(default_unit_file, DEFAULT_UNITS_FILE);


  content_data = defineUnitAndValue(plist_cch[cch_index].data);
  plist_cch[cch_index].value.vNum = content_data.value;
  strcpy(plist_cch[cch_index].unit, content_data.unit);

  initialize(buffer);
  strcpy(buffer, plist_cch[cch_index].unit);
  complex_unit = modifySpeedFormat(buffer);

  if (strcmp(complex_unit.numerator, "error") == 0) {
    printf(plist_cch[cch_index].field, plist_cch[cch_index].unit);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }

  initialize(buffer);
  strcpy(buffer, plist_cch[cch_index].def_unit);
  complex_default_unit = modifySpeedFormat(buffer);

  if (strcmp(complex_unit.numerator, "error") == 0) {
    printf(ERROR12, plist_cch[cch_index].field,
	   plist_cch[cch_index].def_unit);
    printf(ERROR5, default_unit_file);
    return (EXIT_FAILURE);
  }
  dataType = 14;
  numerator_unit = definePrefixAndUnit(complex_unit.numerator, dataType);

  if (numerator_unit.unit[0] == '\0') {
    printf(ERROR12, plist_cch[cch_index].field, plist_cch[cch_index].data);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }
  numerator_default_unit =
      definePrefixAndUnit(complex_default_unit.numerator, dataType);

  if (numerator_default_unit.unit[0] == '\0') {
    printf(ERROR25, default_unit_file);
    printf(ERROR12, plist_cch[cch_index].field,
	   plist_cch[cch_index].def_unit);
    printf(ERROR5, default_unit_file);
    return (EXIT_FAILURE);
  }
  result_numerator_prefix_conversion =
      findPrefixConversionFactor(numerator_unit, numerator_default_unit);

  if ((result_numerator_prefix_conversion) == -1) {
    printf(ERROR24, plist_cch[cch_index].field, plist_cch[cch_index].unit);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }

  else if ((result_numerator_prefix_conversion) == -2) {
    printf(ERROR24, plist_cch[cch_index].field,
	   plist_cch[cch_index].def_unit);
    printf(ERROR5, default_unit_file);
    return (EXIT_FAILURE);
  }
  result_numerator_conversion =
      findUnitConversionFactor(numerator_unit, numerator_default_unit,
			       dataType);

  dataType = 5;
  denominator_unit =
      definePrefixAndUnit(complex_unit.denominator, dataType);

  if (denominator_unit.unit[0] == '\0') {
    printf(ERROR12, plist_cch[cch_index].field, plist_cch[cch_index].unit);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }

  denominator_default_unit =
      definePrefixAndUnit(complex_default_unit.denominator, dataType);

  if (denominator_default_unit.unit[0] == '\0') {
    printf(ERROR12, plist_cch[cch_index].field,
	   plist_cch[cch_index].def_unit);
    printf(ERROR5, default_unit_file);
    return (EXIT_FAILURE);
  }

  result_denominator_prefix_conversion =
      findPrefixConversionFactor(denominator_unit,
				 denominator_default_unit);

  if ((result_denominator_prefix_conversion) == -1) {
    printf(ERROR24, plist_cch[cch_index].field, plist_cch[cch_index].unit);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }

  else if ((result_denominator_prefix_conversion) == -2) {
    printf(ERROR24, plist_cch[cch_index].field,
	   plist_cch[cch_index].def_unit);
    printf(ERROR5, default_unit_file);
    return (EXIT_FAILURE);
  }
  result_denominator_conversion =
      findUnitConversionFactor(denominator_unit, denominator_default_unit,
			       dataType);

  plist_cch[cch_index].def_unit_value.vNum =
      plist_cch[cch_index].value.vNum *
      (((result_numerator_prefix_conversion *
	 result_numerator_conversion.factor)
	/ (result_denominator_prefix_conversion *
	   result_denominator_conversion.factor)))
      + result_numerator_conversion.constant +
      result_denominator_conversion.constant;

  return (EXIT_SUCCESS);
}
