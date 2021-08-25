/*-------------------------------------------------------

           List Mode Format 
                        
     --  fillInCchDefaultUnitValue.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of fillInCchDefaultUnitValue.c:


	Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	Function used for the ascii part of LMF:
	->fillInCchDefaultUnitValue - set the default unit value in the members 
	                              def_unit and def_unit_value of the structure LMF_cch 
				      after the data conversion 
-------------------------------------------------------*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "lmf.h"

/* fillInCchDefaultUnitValue - set the default unit value in the members 
   def_unit and def_unit_value of the structure LMF_cch after the data conversion */

int fillInCchDefaultUnitValue(int last_lmf_header, int cch_index,
			      ENCODING_HEADER * pEncoHforGeometry)
{

  i8 default_unit_file[charNum];
  int lmf_header_index = 0;
  int dataType = 0;
  i8 error_list[20][500] = { "", "", "", "", "", "",
    ERROR13, ERROR14, ERROR11, ERROR12, ERROR15, ERROR16,
    ERROR17, ERROR18, ERROR19, ERROR20, ERROR21, ERROR22, ERROR23, ERROR37
  };

  i8 symbol_default_units_list[20][7] = { {""},	/*data type isn't defined */
  {""},				/*data = a sting */
  {""},				/*data =a value without unit */
  {""},				/*data = a date */
  {DURATION_UNIT},		/*data = a duration */
  {TIME_UNIT},			/*data = a time */
  {SURFACE_UNIT},		/*data = a surface */
  {VOLUME_UNIT},		/*data = a volume */
  {SPEED_UNIT},			/*data = a speed */
  {ROTATION_SPEED_UNIT},	/*data = a rotation speed */
  {ENERGY_UNIT},		/*data = an energy */
  {ACTIVITY_UNIT},		/*data = an activity */
  {WEIGHT_UNIT},		/*data = a weight */
  {DISTANCE_UNIT},		/*data = a distance */
  {ANGLE_UNIT},			/*data = an angle */
  {TEMPERATURE_UNIT},		/*data = a temperature */
  {ELECTRIC_FIELD_UNIT},	/*data = an electric field */
  {MAGNETIC_FIELD_UNIT},	/*data = a magnetic field */
  {PRESSION_UNIT},		/*data = a pression */
  {DISTANCE_UNIT}
  };				/*data = a shift value */

  double result_prefix_conversion = 0;

  content_data_unit content_data = { 0 }, unit = {
  0}, default_unit = {
  0};

  result_unit_conversion result_conversion = { 0 };

  initialize(default_unit_file);
  strcpy(default_unit_file, DEFAULT_UNITS_FILE);

  for (lmf_header_index = 0; lmf_header_index <= last_lmf_header;
       lmf_header_index++) {
    if (strcasecmp
	(plist_cch[cch_index].field,
	 plist_lmf_header[lmf_header_index].field) != 0)
      continue;
    plist_lmf_header = first_lmf_header;

    dataType = plist_lmf_header[lmf_header_index].type;

    strcpy(plist_cch[cch_index].def_unit,
	   symbol_default_units_list[dataType]);

    switch (dataType) {
    case 0:			/*data = not defined */
    case 1:			/*data = string */
      if (stringConversion(cch_index) != EXIT_SUCCESS)
	return (EXIT_FAILURE);
      break;

    case 2:			/*data = value without unit */
      if (numberConversion(cch_index) != EXIT_SUCCESS)
	return (EXIT_FAILURE);
      break;

    case 3:			/*data is a date */
      if (dateConversion(cch_index) != EXIT_SUCCESS)
	return (EXIT_FAILURE);
      break;
    case 4:			/*data is a duration */
      if (durationConversion(cch_index) != EXIT_SUCCESS)
	return (EXIT_FAILURE);
      break;
    case 5:			/*data is a time */
      if (timeConversion(cch_index) != EXIT_SUCCESS)
	return (EXIT_FAILURE);
      break;

    case 6:			/*data is a surface (value+unit) */
      if (surfaceConversion(cch_index, dataType) != EXIT_SUCCESS)
	return (EXIT_FAILURE);
      break;

    case 7:			/*data is a volume (value+unit) */
      if (surfaceConversion(cch_index, dataType) != EXIT_SUCCESS)
	return (EXIT_FAILURE);
      break;

    case 8:			/*data is a speed (value+unit) */
      if (speedConversion(cch_index, dataType) != EXIT_SUCCESS)
	return (EXIT_FAILURE);
      break;

    case 9:			/*data is a rotation speed (value+unit) */
      if (rotationSpeedConversion(cch_index, dataType) != EXIT_SUCCESS)
	return (EXIT_FAILURE);
      break;

    case 19:			/*data is a shift numerical value (value+unit) */
      content_data = defineUnitAndValue(plist_cch[cch_index].data);

      if (content_data.unit[0] == '\0') {
	printf(ERROR37, plist_cch[cch_index].field,
	       plist_cch[cch_index].data);
	printf(ERROR5, cchFileName);
	return (EXIT_FAILURE);
      }
      plist_cch[cch_index].value.vNum = content_data.value;
      strcpy(plist_cch[cch_index].unit, content_data.unit);
      unit = definePrefixAndUnit(plist_cch[cch_index].unit, dataType);

      if (unit.unit[0] == '\0') {
	printf(ERROR37, plist_cch[cch_index].field,
	       plist_cch[cch_index].data);
	printf(ERROR5, cchFileName);
	return (EXIT_FAILURE);
      }
      default_unit =
	  definePrefixAndUnit(plist_cch[cch_index].def_unit, dataType);

      if (default_unit.unit[0] == '\0') {
	printf(ERROR25, default_unit_file);
	printf(ERROR37, plist_cch[cch_index].field,
	       plist_cch[cch_index].def_unit);
	printf(ERROR5, default_unit_file);
	return (EXIT_FAILURE);
      }
      result_prefix_conversion =
	  findPrefixConversionFactor(unit, default_unit);

      if ((result_prefix_conversion) == -1) {
	printf(ERROR24, plist_cch[cch_index].field,
	       plist_cch[cch_index].unit);
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

      /* remplir le tableau avec valeurs de pitch: function setShiftValues */
      if (setShiftValues(pEncoHforGeometry, cch_index) == EXIT_FAILURE) {
	printf(ERROR5, cchFileName);
	exit(EXIT_FAILURE);
      }
      break;

    default:			/*data with an unit and a value */
      content_data = defineUnitAndValue(plist_cch[cch_index].data);

      if (content_data.unit[0] == '\0') {
	printf(error_list[dataType], plist_cch[cch_index].field,
	       plist_cch[cch_index].data);
	printf(ERROR5, cchFileName);
	return (EXIT_FAILURE);
      }
      plist_cch[cch_index].value.vNum = content_data.value;
      strcpy(plist_cch[cch_index].unit, content_data.unit);
      unit = definePrefixAndUnit(plist_cch[cch_index].unit, dataType);

      if (unit.unit[0] == '\0') {
	printf(error_list[dataType], plist_cch[cch_index].field,
	       plist_cch[cch_index].data);
	printf(ERROR5, cchFileName);
	return (EXIT_FAILURE);
      }
      default_unit =
	  definePrefixAndUnit(plist_cch[cch_index].def_unit, dataType);

      if (default_unit.unit[0] == '\0') {
	printf(ERROR25, default_unit_file);
	printf(error_list[dataType], plist_cch[cch_index].field,
	       plist_cch[cch_index].def_unit);
	printf(ERROR5, default_unit_file);
	return (EXIT_FAILURE);
      }
      result_prefix_conversion =
	  findPrefixConversionFactor(unit, default_unit);

      if ((result_prefix_conversion) == -1) {
	printf(ERROR24, plist_cch[cch_index].field,
	       plist_cch[cch_index].unit);
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

    }				/*end of switch */
    break;
  }				/*end of for */
  return (EXIT_SUCCESS);
}
