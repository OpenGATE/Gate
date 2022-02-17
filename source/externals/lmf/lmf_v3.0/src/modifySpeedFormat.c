/*-------------------------------------------------------

           List Mode Format 
                        
     --  modifySpeedFormat.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of modifySpeedFormat.c:


	 Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	 Functions used for the ascii part of LMF:
	 ->modifySpeedFormat.c - Separate in speed and rotation speed, numerator and denominator


-------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include "lmf.h"

/* modifySpeedFormat - Separate in speed and rotation speed, numerator and denominator */

complex_unit_type modifySpeedFormat(i8 undefined_unit[charNum])
{

  i8 buffer[charNum];
  i8 *line = NULL;
  i8 *end_word = NULL;
  int unit_length = 0, denominator_length = 0, unitIndex = 0;
  complex_unit_type complex_unit;

  initialize(complex_unit.numerator);
  initialize(complex_unit.denominator);
  initialize(buffer);

  strcpy(buffer, undefined_unit);
  initialize(undefined_unit);
  strcpy(undefined_unit, strtok(buffer, " "));

  while ((line = strtok(NULL, " ")) != NULL)
    strcat(undefined_unit, line);

  initialize(buffer);
  line = NULL;
  unit_length = strlen(undefined_unit);

  if ((line = strchr(undefined_unit, '/')) == NULL) {
    end_word = strchr(undefined_unit, '\0');
    end_word--;
    if (end_word[0] == '1') {
      end_word--;
      if (end_word[0] == '-') {
	strncpy(buffer, undefined_unit, (unit_length - 2));
	strcpy(complex_unit.numerator, (strtok(buffer, ".*x")));
	while ((line = strtok(NULL, "\0")) != NULL)
	  strcpy(complex_unit.denominator, line);
      }
    } else if (isdigit(end_word[0]) == 0) {
      while (unitIndex < (unit_length - 1)) {
	if (strncmp(end_word, "p", 1) == 0) {
	  end_word++;
	  strcpy(complex_unit.denominator, end_word);
	  denominator_length = strlen(complex_unit.denominator);
	  if (strncmp(complex_unit.denominator, "m", 1) == 0)
	    strcat(complex_unit.denominator, "in");
	  strncpy(complex_unit.numerator, undefined_unit,
		  (unit_length - denominator_length));
	  break;
	}
	end_word--;
	unitIndex++;
      }
    }
  } else if ((line = strchr(undefined_unit, '/')) != NULL) {
    line = NULL;
    strcpy(complex_unit.numerator, (strtok(undefined_unit, "/")));
    while ((line = strtok(NULL, "\0")) != NULL)
      strcpy(complex_unit.denominator, line);
  }

  else
    strcpy(complex_unit.numerator, "error");

  return (complex_unit);
}
