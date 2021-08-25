/*-------------------------------------------------------

List Mode Format 
                        
--  setShiftValues.c  --                      

Magalie.Krieguer@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of setShiftValues.c:

Fill in the LMF record carrier with the data contained in the LMF ASCII header file
Function used for the ascii part of LMF:
->setShiftValues - fill in the table defined in the LMFcchReader function,
with the shift information contained in the .cch file
->chooseColumnIndex - define if the shift is on the x-axis (x shift or radial shift x)
if the shift is on the y-axis (y shift or radial shift y)
if the shift is on the z-axis (z shift or axial shift)
->findModuloNumber - find in the field description the modulo number,
which defines the shift frequency
example: "x shift ring 0 mod 2", the ring 0 position,
the ring 2 position, the ring 4 position, ... are shifted. 
->findStartNumber - find the first structure, which is shifted
example: "z shift sector 2 mod 3"
the start number is the sector with the number 2
-------------------------------------------------------*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "lmf.h"


int chooseColumnIndex(int cch_index)
{

  int column_index = 0;
  i8 stringbuf[charNum];
  i8 shiftWay[10][7] =
      { {"x"}, {"X"}, {"y"}, {"Y"}, {"z"}, {"Z"}, {"radial"}, {"RADIAL"},
      {"axial"}, {"AXIAL"} };

  for (column_index = 0; column_index < 10; column_index++) {
    initialize(stringbuf);
    strcpy(stringbuf, plist_cch[cch_index].field);
    if ((strstr(stringbuf, shiftWay[column_index])) != NULL) {
      switch (column_index) {
      case 0:
      case 1:
	return (0);
	break;
      case 2:
      case 3:
	return (1);
	break;
      case 4:
      case 5:
      case 8:
      case 9:
	return (2);
	break;
      }
    }
  }
  printf(ERROR37, plist_cch[cch_index].field, plist_cch[cch_index].data);
  printf(ERROR5, cchFileName);
  exit(EXIT_FAILURE);
}


int findModuloNumber(i8 stringbuf[charNum])
{
  i8 *findPos = NULL, *endOfStringbuf = NULL;
  i8 buffer[charNum];
  int moduloNumber = 0;

  endOfStringbuf = strchr(stringbuf, '\0');
  findPos = strstr(stringbuf, "mod");	/* patterns: x shift rsector 0 mod 2, 
					   x shift rsector 0 mod 3, 
					   y shift rsector 1 mod 1 
					   or z shift rsector 6 mod 2 */
  findPos = findPos + 3;
  while (endOfStringbuf != findPos) {
    if ((isdigit(findPos[0])) == 0)
      findPos++;
    else {
      initialize(buffer);
      strcpy(buffer, findPos);
      moduloNumber = atoi(buffer);
      return (moduloNumber);
    }
  }
  return (EXIT_FAILURE);
}

int findStartNumber(i8 stringbuf[charNum],
		    ENCODING_HEADER * pEncoHforGeometry,
		    int structure_index, int cch_index)
{
  i8 *findPos = NULL;
  i8 buffer[charNum], endOfStringbuf[charNum], beginOfStringbuf[charNum];
  int startNumber = 0, stringbufLength = 0, endOfStringbufLength =
      0, index = 0;

  stringbufLength = strlen(stringbuf);
  findPos = (strstr(stringbuf, "mod"));
  findPos--;
  strcpy(endOfStringbuf, findPos);
  endOfStringbufLength = strlen(endOfStringbuf);
  strncpy(beginOfStringbuf, stringbuf,
	  (stringbufLength - endOfStringbufLength));

  findPos = NULL;
  findPos = strchr(beginOfStringbuf, '\0');

  index = 0;
  while (&beginOfStringbuf[index] != findPos) {
    if ((isdigit(beginOfStringbuf[index])) == 0)
      index++;
    else if ((isdigit(beginOfStringbuf[index])) != 0) {
      initialize(buffer);
      strcpy(buffer, &beginOfStringbuf[index]);
      startNumber = atoi(buffer);
      switch (structure_index) {
      case 0:			/* ring */
	if (startNumber >
	    ((int) pEncoHforGeometry->scannerTopology.numberOfRings - 1)) {
	  printf(ERROR40,
		 ((int) pEncoHforGeometry->scannerTopology.numberOfRings -
		  1));
	  printf(ERROR5, cchFileName);
	  exit(EXIT_FAILURE);
	}
	break;
      case 1:			/* rsector */
	if (startNumber >
	    ((int) pEncoHforGeometry->scannerTopology.
	     totalNumberOfRsectors - 1)) {
	  printf(ERROR42,
		 ((int) pEncoHforGeometry->scannerTopology.
		  totalNumberOfRsectors - 1));
	  printf(ERROR5, cchFileName);
	  exit(EXIT_FAILURE);
	}
	break;
      case 2:			/* sector */
	if (startNumber >
	    ((int) pEncoHforGeometry->scannerTopology.numberOfSectors -
	     1)) {
	  printf(ERROR41,
		 ((int) pEncoHforGeometry->scannerTopology.
		  numberOfSectors - 1));
	  printf(ERROR5, cchFileName);
	  exit(EXIT_FAILURE);
	}
	break;
      }
      return (startNumber);
    }
  }
  printf(ERROR47, plist_cch[cch_index].field);
  printf(ERROR38, plist_cch[cch_index].field);
  printf(ERROR5, cchFileName);
  exit(EXIT_FAILURE);
}

int setShiftValues(ENCODING_HEADER * pEncoHforGeometry, int cch_index)
{

  i8 stringbuf[charNum], buffer[charNum];

  i8 structure[3][8] = { {"ring"}, {"rsector"}, {"sector"} };
  i8 fieldDescription[6][16] =
      { {"x shift"}, {"radial shift x"}, {"y shift"}, {"radial shift y"},
      {"z shift"}, {"axial shift"} };
  int structure_index = 0, description_index = 0, column_index =
      0, row_index = 0, modulo = 0, startNumber = 0, kIndex =
      0, max_kIndex = 0;
  i8 *end_stringbuf = NULL;

  initialize(stringbuf);
  strcpy(stringbuf, plist_cch[cch_index].field);
  for (structure_index = 0; structure_index < 3; structure_index++) {
    if ((strstr(stringbuf, structure[structure_index])) != NULL) {	/* In field description, we have find rsector, sector or ring */
      switch (structure_index) {
      case 0:			/* ring */
	column_index = chooseColumnIndex(cch_index);
	initialize(stringbuf);
	strcpy(stringbuf, plist_cch[cch_index].field);
	if ((strstr(stringbuf, "mod")) != NULL) {	/* patterns: x shift ring 0 mod 2, 
							   x shift ring 0 mod 3, 
							   y shift ring 1 mod 1 
							   or z shift ring 6 mod 2 */
	  modulo = findModuloNumber(plist_cch[cch_index].field);
	  if (modulo == 0) {
	    printf(ERROR46, plist_cch[cch_index].field);
	    printf(ERROR38, plist_cch[cch_index].field);
	    printf(ERROR5, cchFileName);
	    return (EXIT_FAILURE);
	  }
	  startNumber =
	      findStartNumber(plist_cch[cch_index].field,
			      pEncoHforGeometry, structure_index,
			      cch_index);

	  max_kIndex =
	      1 +
	      (((int) pEncoHforGeometry->scannerTopology.numberOfRings -
		startNumber) / modulo);
	  for (kIndex = 0; kIndex < max_kIndex; kIndex++) {
	    for (row_index = 0;
		 row_index <
		 (int) pEncoHforGeometry->scannerTopology.numberOfSectors;
		 row_index++) {
	      if (((int) pEncoHforGeometry->scannerTopology.
		   numberOfSectors * ((kIndex * modulo) + startNumber) +
		   row_index)
		  < (int) pEncoHforGeometry->scannerTopology.
		  totalNumberOfRsectors) {
		ppShiftValuesList[((int) pEncoHforGeometry->
				   scannerTopology.numberOfSectors *
				   ((kIndex * modulo) + startNumber) +
				   row_index)][column_index] =
		    plist_cch[cch_index].def_unit_value.vNum;
	      } else
		break;
	    }
	  }
	  return (EXIT_SUCCESS);
	}
	initialize(stringbuf);
	strcpy(stringbuf, plist_cch[cch_index].field);
	end_stringbuf = NULL;
	end_stringbuf = strchr(stringbuf, '\0');	/* patterns: x shift ring 1, 
							   x shift ring 3, 
							   y shift ring 2 
							   or z shift ring 6 */
	end_stringbuf--;
	if ((isdigit(*end_stringbuf)) == 0) {
	  printf(ERROR39, plist_cch[cch_index].field,
		 plist_cch[cch_index].data);
	  return (EXIT_FAILURE);
	}
	while ((isdigit(end_stringbuf[0])) != 0)
	  end_stringbuf--;
	end_stringbuf++;
	initialize(buffer);
	strcpy(buffer, end_stringbuf);
	row_index = atoi(buffer);
	if (row_index >
	    ((int) pEncoHforGeometry->scannerTopology.numberOfRings - 1)) {
	  printf(ERROR43,
		 ((int) pEncoHforGeometry->scannerTopology.numberOfRings -
		  1));
	  return (EXIT_FAILURE);
	}
	for (kIndex = 0;
	     kIndex < pEncoHforGeometry->scannerTopology.numberOfSectors;
	     kIndex++)
	  ppShiftValuesList[(row_index *
			     pEncoHforGeometry->scannerTopology.
			     numberOfSectors) + kIndex][column_index] =
	      plist_cch[cch_index].def_unit_value.vNum;
	return (EXIT_SUCCESS);
	break;

      case 1:			/* rsector */
	initialize(stringbuf);
	strcpy(stringbuf, plist_cch[cch_index].field);
	column_index = chooseColumnIndex(cch_index);
	if ((strstr(stringbuf, "mod")) != NULL) {	/* patterns: x shift rsector 0 mod 2, 
							   x shift rsector 0 mod 3, 
							   y shift rsector 1 mod 1 
							   or z shift rsector 6 mod 2 */
	  modulo = findModuloNumber(plist_cch[cch_index].field);
	  if (modulo == 0) {
	    printf(ERROR46, plist_cch[cch_index].field);
	    printf(ERROR38, plist_cch[cch_index].field);
	    printf(ERROR5, cchFileName);
	    return (EXIT_FAILURE);
	  }

	  startNumber =
	      findStartNumber(stringbuf, pEncoHforGeometry,
			      structure_index, cch_index);
	  for (row_index = startNumber;
	       row_index <
	       (int) pEncoHforGeometry->scannerTopology.
	       totalNumberOfRsectors; row_index = row_index + modulo)
	    ppShiftValuesList[row_index][column_index] =
		plist_cch[cch_index].def_unit_value.vNum;
	  return (EXIT_SUCCESS);
	}

	initialize(stringbuf);
	strcpy(stringbuf, plist_cch[cch_index].field);
	end_stringbuf = NULL;
	end_stringbuf = strchr(stringbuf, '\0');	/* patterns: x shift rsector 1, 
							   x shift rsector 3, 
							   y shift rsector 2 
							   or z shift rsector 6 */
	end_stringbuf--;
	if ((isdigit(*end_stringbuf)) == 0) {
	  printf(ERROR39, plist_cch[cch_index].field,
		 plist_cch[cch_index].data);
	  return (EXIT_FAILURE);
	}
	while ((isdigit(end_stringbuf[0])) != 0)
	  end_stringbuf--;
	end_stringbuf++;
	initialize(buffer);
	strcpy(buffer, end_stringbuf);
	row_index = atoi(buffer);
	if (row_index >
	    ((int) pEncoHforGeometry->scannerTopology.
	     totalNumberOfRsectors - 1)) {
	  printf(ERROR44,
		 ((int) pEncoHforGeometry->scannerTopology.
		  totalNumberOfRsectors - 1));
	  exit(EXIT_FAILURE);
	}
	ppShiftValuesList[row_index][column_index] =
	    plist_cch[cch_index].def_unit_value.vNum;
	return (EXIT_SUCCESS);
	break;

      case 2:			/* sector */
	initialize(stringbuf);
	strcpy(stringbuf, plist_cch[cch_index].field);
	column_index = chooseColumnIndex(cch_index);

	if ((strstr(stringbuf, "mod")) != NULL) {	/* patterns: x shift sector 0 mod 2, 
							   x shift sector 0 mod 3, 
							   y shift sector 1 mod 1 
							   or z shift sector 6 mod 2 */
	  modulo = findModuloNumber(plist_cch[cch_index].field);
	  if (modulo == 0) {
	    printf(ERROR46, plist_cch[cch_index].field);
	    printf(ERROR38, plist_cch[cch_index].field);
	    printf(ERROR5, cchFileName);
	    return (EXIT_FAILURE);
	  }
	  startNumber =
	      findStartNumber(stringbuf, pEncoHforGeometry,
			      structure_index, cch_index);

	  max_kIndex =
	      1 +
	      (((int) pEncoHforGeometry->scannerTopology.numberOfSectors -
		startNumber) / modulo);
	  for (kIndex = 0; kIndex < max_kIndex; kIndex++) {
	    if (((kIndex * modulo) + startNumber) <
		(int) pEncoHforGeometry->scannerTopology.numberOfSectors) {
	      for (row_index = 0;
		   row_index <
		   (int) pEncoHforGeometry->scannerTopology.numberOfRings;
		   row_index++) {
		ppShiftValuesList[((int) pEncoHforGeometry->
				   scannerTopology.numberOfSectors *
				   row_index) + ((kIndex * modulo) +
						 startNumber)]
		    [column_index] =
		    plist_cch[cch_index].def_unit_value.vNum;
	      }
	    } else
	      break;
	  }
	  return (EXIT_SUCCESS);
	}
	initialize(stringbuf);
	strcpy(stringbuf, plist_cch[cch_index].field);
	end_stringbuf = NULL;
	end_stringbuf = strchr(stringbuf, '\0');	/* patterns: x shift sector 1, 
							   x shift sector 3, 
							   y shift sector 2 
							   or z shift sector 6 */
	end_stringbuf--;
	if ((isdigit(end_stringbuf[0])) == 0) {
	  printf(ERROR39, plist_cch[cch_index].field,
		 plist_cch[cch_index].data);
	  return (EXIT_FAILURE);
	}
	while ((isdigit(end_stringbuf[0])) != 0)
	  end_stringbuf--;
	end_stringbuf++;
	initialize(buffer);
	strcpy(buffer, end_stringbuf);
	row_index = atoi(buffer);
	if (row_index >
	    ((int) pEncoHforGeometry->scannerTopology.numberOfSectors -
	     1)) {
	  printf(ERROR45,
		 ((int) pEncoHforGeometry->scannerTopology.
		  numberOfSectors - 1));
	  return (EXIT_FAILURE);
	}
	for (kIndex = 0;
	     kIndex < pEncoHforGeometry->scannerTopology.numberOfRings;
	     kIndex++)
	  ppShiftValuesList[row_index +
			    (kIndex *
			     pEncoHforGeometry->scannerTopology.
			     numberOfSectors)][column_index] =
	      plist_cch[cch_index].def_unit_value.vNum;
	break;
	return (EXIT_SUCCESS);
      }
    }
  }
  for (description_index = 0; description_index < 6; description_index++) {	/* The data concerns a global shift: x shift, y shift or z shift */
    if ((strcasecmp(stringbuf, fieldDescription[description_index])) == 0) {
      column_index = chooseColumnIndex(cch_index);
      for (row_index = 0;
	   row_index <
	   (int) pEncoHforGeometry->scannerTopology.totalNumberOfRsectors;
	   row_index++)
	ppShiftValuesList[row_index][column_index] =
	    plist_cch[cch_index].def_unit_value.vNum;
      return (EXIT_SUCCESS);
    }
  }
  return (EXIT_FAILURE);
}
