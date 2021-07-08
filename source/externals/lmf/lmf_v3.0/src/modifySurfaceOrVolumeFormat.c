/*-------------------------------------------------------

           List Mode Format 
                        
     --  modifySurfaceOrVolumeFormat.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of modifySurfaceOrVolumeFormat.c:
	 Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	 Functions used for the ascii part of LMF:
	 ->modifySurfaceOrVolumeFormat - Convert surface unit or volume unit in a distance unit
-------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"

/* modifySurfaceOrVolumeFormat - Convert surface unit or volume unit in a distance unit */

i8 *modifySurfaceOrVolumeFormat(i8 unit[charNum], int unit_type,
				i8 power[charNum])
{

  i8 *end_word = NULL;
  i8 *buffer = NULL;
  i8 stringbuf[charNum];
  int unit_length = 0;
  i8 *line = NULL;

  initialize(stringbuf);
  strcpy(stringbuf, unit);
  initialize(unit);
  strcpy(unit, strtok(stringbuf, " *^x."));
  while ((line = strtok(NULL, " *^x.")) != NULL)
    strcat(unit, line);
  unit_length = strlen(unit);

  if (unit_length > 1) {
    end_word = strchr(unit, '\0');
    end_word--;
    initialize(stringbuf);
    if ((strncmp(end_word, power, 1)) == 0)
      strncpy(stringbuf, unit, (unit_length - 1));
  } else
    strcpy(stringbuf, "error");

  buffer = &stringbuf[0];
  return (buffer);
}
