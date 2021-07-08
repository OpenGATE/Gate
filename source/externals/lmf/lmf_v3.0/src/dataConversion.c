/*-------------------------------------------------------

           List Mode Format 
                        
     --  dataConversion.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of dataConversion.c:
	 Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	 Functions used for the ascii part of LMF:
	 ->defineUnitAndValue - Define which part of the data is the numerical value 
	                        and which part is the unit
	 ->definePrefiAndUnit - Define which part of the unit is the prefix 
	                        and which part is the real unit
	 ->findPrefixConversionFactor - Convert data in a default format
	 ->findUnitConversionFactor - Convert data in a default format

------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "lmf.h"

/* Function defineUnitAndValue: Define which part of the data is the numerical value 
   and which part is the unit */

content_data_unit defineUnitAndValue(i8 data[charNum])
{

  int lengthIndex = 0, stringbuf_length = 0, unit_length = 0, data_length =
      0;
  i8 *line = NULL;
  i8 *end_word = NULL;
  i8 buffer[charNum], stringbuf[charNum];
  content_data_unit content_data = { 0 };

  initialize(buffer);
  initialize(stringbuf);

  strcpy(buffer, data);
  data_length = strlen(data);

  if (data_length != 0) {
    strcpy(stringbuf, strtok(buffer, " "));

    while ((line = strtok(NULL, " ")) != NULL)
      strcat(stringbuf, line);
    stringbuf_length = strlen(stringbuf);

    if (((isdigit(stringbuf[0])) != 0)
	|| ((strncasecmp(stringbuf, "+", 1)) == 0)
	|| ((strncasecmp(stringbuf, "-", 1)) == 0)) {
      end_word = strchr(stringbuf, '\0');
      end_word = end_word - 2;

      while (lengthIndex < stringbuf_length) {
	if ((isdigit(end_word[0]) != 0)
	    || ((strncasecmp(end_word, ".", 1)) == 0)) {
	  end_word++;
	  strcpy(content_data.unit, end_word);
	  unit_length = strlen(content_data.unit);
	  initialize(buffer);
	  strncpy(buffer, stringbuf, (stringbuf_length - unit_length));
	  content_data.value = atof(buffer);
	  break;
	}

	end_word--;
	lengthIndex++;

      }
    }

    else
      strcpy(content_data.unit, stringbuf);
  }

  return (content_data);
}


/* Function definePrefiAndUnit: Define which part of the unit is the prefix 
   and which part is the real unit */

content_data_unit definePrefixAndUnit(i8 undefined_unit[charNum],
				      int unit_type)
{

  int dataType = 0, unitIndex = 0, unit_length = 0, reply_length = 0;
  content_data_unit unit = { 0 };
  i8 stringbuf[charNum];
  i8 *comp_unit = NULL;

  initialize(stringbuf);
  strcpy(stringbuf, undefined_unit);
  dataType = unit_type;
  unit_length = 0;
  unit_length = strlen(stringbuf);

  for (unitIndex = 0; unitIndex < nb_units[unit_type]; unitIndex++) {
    reply_length = 0;
    comp_unit = strchr(stringbuf, '\0');
    reply_length = strlen(symbol_units_list[dataType][unitIndex]);
    comp_unit = comp_unit - reply_length;

    if ((strncasecmp
	 (comp_unit, symbol_units_list[dataType][unitIndex],
	  reply_length)) == 0) {
      strncpy(unit.unit, comp_unit, reply_length);
      strncpy(unit.prefix, stringbuf, (unit_length - reply_length));
      break;
    }

  }

  return (unit);
}

/* Function findPrefixConversionFactor: Convert data in a default format */

double findPrefixConversionFactor(content_data_unit unit,
				  content_data_unit default_unit)
{

  double result = 0;
  int prefixIndex = 0, defaultPrefixIndex = 0;

  for (prefixIndex = 0; prefixIndex < 16; prefixIndex++) {
    if ((strcmp(unit.prefix, standard_prefix_list[prefixIndex])) == 0)
      break;
  }

  if ((strcmp(unit.prefix, standard_prefix_list[prefixIndex])) != 0) {
    result = -1;
    return (result);
  }

  for (defaultPrefixIndex = 0; defaultPrefixIndex < 16;
       defaultPrefixIndex++) {
    if ((strcmp
	 (default_unit.prefix,
	  standard_prefix_list[defaultPrefixIndex])) == 0)
      break;
  }

  if ((strcmp
       (default_unit.prefix,
	standard_prefix_list[defaultPrefixIndex])) != 0) {
    result = -2;
    return (result);
  }
  //  printf("unit=%s, prefix=%d, def prefix=%d\n",unit.unit,prefixIndex,defaultPrefixIndex);
  result = standard_prefix_factors_list[prefixIndex][defaultPrefixIndex];
  return (result);
}

/* Function findUnitConversionFactor: Convert data in a default format */

result_unit_conversion findUnitConversionFactor(content_data_unit unit,
						content_data_unit
						default_unit,
						int unit_type)
{

  result_unit_conversion result_conversion = { 0 };
  int dataType = 0, unitIndex = 0, defaultUnitIndex = 0;

  dataType = unit_type;

  for (unitIndex = 0; unitIndex < nb_units[dataType]; unitIndex++) {
    if ((strcmp(unit.unit, symbol_units_list[dataType][unitIndex])) == 0)
      break;
  }

  for (defaultUnitIndex = 0; defaultUnitIndex < nb_units[dataType];
       defaultUnitIndex++) {
    if ((strcmp
	 (default_unit.unit,
	  symbol_units_list[dataType][defaultUnitIndex])) == 0)
      break;
  }

  result_conversion.factor =
      units_conversion_factors_list[dataType][unitIndex][defaultUnitIndex];
  result_conversion.constant =
      units_conversion_constants_list[dataType][unitIndex]
      [defaultUnitIndex];

  return (result_conversion);
}
