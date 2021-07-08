/*-------------------------------------------------------

           List Mode Format 
                        
     --  numberConversion.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of numberConversion.c:
	Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	Function used for the ascii part of LMF:
	Fill in the members value and default_value of the LMF_cch structure with a number
-------------------------------------------------------*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "lmf.h"

/* numberConversion - Fill in the members value and default_value of the LMF_cch structure
   with a number */

int numberConversion(int cch_index)
{

  i8 *line = NULL;
  i8 stringbuf[charNum], buffer[charNum];

  initialize(buffer);
  initialize(stringbuf);
  strcpy(buffer, plist_cch[cch_index].data);
  strcpy(stringbuf, strtok(buffer, " "));

  while ((line = strtok(NULL, " ")) != NULL) {
    strcat(stringbuf, " ");
    strcat(stringbuf, line);
  }

  if (((isdigit(stringbuf[0])) == 0)
      && ((strncasecmp(stringbuf, "+", 1)) != 0)
      && ((strncasecmp(stringbuf, "-", 1)) != 0)) {
    printf(ERROR27, plist_cch[cch_index].field, plist_cch[cch_index].data);
    printf(ERROR5, cchFileName);
    return (EXIT_FAILURE);
  }
  plist_cch[cch_index].value.vNum = atof(stringbuf);
  plist_cch[cch_index].def_unit_value.vNum = atof(stringbuf);

  return (EXIT_SUCCESS);
}
