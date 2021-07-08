/*-------------------------------------------------------

           List Mode Format 
                        
     --  newFieldInCchFile.c  --                      

     Magalie.Krieguer@iphe.unil.ch
 
     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of newFieldInCchFile.c:

	 Functions used for the ascii part of LMF:
	 ->writeNewFieldInDataBase - Add a new field in the lmf_header data base 
	 ->defineFieldType - Define the type of the new field adding in lmf_header data base

-------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"

/* Function writeNewFieldInDataBase: Add new field in lmf_header.db */

int writeNewFieldInDataBase(FILE * lmf_header_infile, i8 field[charNum])
{

  i8 stringbuf[charNum], headerFileName[charNum];
  i8 field_type_num[20][2] =
      { "1", "2", "3", "4", "5", "13", "19", "6", "7", "8", "9", "10",
    "11", "12", "14", "15", "16", "17", "18", "0"
  };

  initialize(stringbuf);
  initialize(headerFileName);
  strcpy(headerFileName, HEADER_FILE);

  if (fseek(lmf_header_infile, 0L, SEEK_END) != 0) {
    printf(ERROR29, field, headerFileName);
    return (EXIT_FAILURE);
  }

  strcpy(stringbuf, field);
  strcat(stringbuf, "&&");
  strncat(stringbuf, field_type_num[defineFieldType(field)], 2);
  fprintf(lmf_header_infile, "%s\n", stringbuf);

  return (EXIT_SUCCESS);
}

/* Function defineFieldType: Define the type of new field adding in lmf_header.db */

int defineFieldType(i8 field[charNum])
{

  int TESTDEF = 0, TESTANSWER = 0, fieldType = 0;
  i8 buffer[charNum];
  i8 *information_type[20] = { "a string",
    "a number (without unit)",
    "a date",
    "a duration (with an unit)",
    "a time",
    "a distance, but isn't a shift value (with an unit)",
    "a shift numerical value (with an unit)",
    "a surface (with an unit)",
    "a volume (with an unit)",
    "a speed (with an unit)",
    "a rotation speed (with an unit)",
    "an energy (with an unit)",
    "an activity (with an unit)",
    "a weight (with an unit)",
    "an angle (with an unit)",
    "a temperature (with an unit)",
    "an electric field (with an unit)",
    "a magnetic field (with an unit)",
    "a pression (with an unit)",
    "another type of information"
  };

  printf
      ("To add the field \"%s\" in your data base, you must select the type of its value.\n",
       field);

  while (TESTDEF == 0) {

    for (fieldType = 0; fieldType < 20; fieldType++) {
      TESTANSWER = 0;
      initialize(buffer);

      while (TESTANSWER == 0) {
	printf("Is the value of the field \"%s\" %s (y/n): ", field,
	       information_type[fieldType]);

	if (*gets(buffer) == 'y') {
	  TESTDEF = 1;
	  TESTANSWER = 1;
	}

	else if (buffer[0] == 'n') {
	  TESTANSWER = 1;
	  continue;
	}
      }
      if (buffer[0] == 'y')
	break;
    }
  }
  return (fieldType);
}
