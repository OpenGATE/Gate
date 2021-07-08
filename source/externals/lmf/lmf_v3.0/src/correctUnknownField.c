/*-------------------------------------------------------

           List Mode Format 
                        
     --  correctUnknownField.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of correctUnknownField.c:
     Fill in the LMF record carrier with the data contained in the LMF ASCII header file
     correctUnknownField -  Correct unknown field described in the input file
     with reference the lmf data base


-------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"

/****  Function correctUnknowField ****/

int correctUnknownField(i8 field[charNum])
{

  int result = 0;
  i8 answer[charNum], headerFileName[charNum];

  while (result == 0) {
    initialize(answer);

    if (strlen(field) >= charNum) {
      printf(ERROR4, (charNum - 1));
      printf(ERROR5, cchFileName);
      return (EXIT_FAILURE);
    }

    initialize(headerFileName);
    strcpy(headerFileName, HEADER_FILE);

    printf(ERROR34, field, headerFileName);
    printf("Correct this field ? (y/n): ");

    if (*gets(answer) == 'y') {
      result = 1;
      break;
    } else if (answer[0] == 'n') {
      result = 2;
      break;
    }
  }
  return (result);
}
