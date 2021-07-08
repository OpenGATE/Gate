/*-------------------------------------------------------

           List Mode Format 
                        
     --  fileNameManager.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of fileNameManager.c:

	 Manages the ccs file name, the coinci_ccs file name
	 and the bis_ccs file name
	 ->setFileName
	 ->get_extension_ccs_FileName
	 ->get_extension_cch_FileName


---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"

int setFileName(i8 inputFile[charNum])
{

  i8 input[charNum], buffer[charNum];
  i8 *line = NULL;
  int TESTNAME = FALSE;

  initialize(fileName);
  initialize(cchFileName);
  initialize(ccsFileName);

  if ((strncmp(inputFile, "\0", 1)) == 0) {
    while (TESTNAME == FALSE) {
      printf("You must choose an input file: ");
      initialize(input);
      if (*gets(input) == '\0')
	continue;
      if ((line = strchr(input, '.')) != NULL) {
	printf(ERROR2, input);
	continue;
      }
      strcpy(fileName, input);
      TESTNAME = TRUE;
    }
  } else {
    strcpy(fileName, inputFile);
    if ((line = strchr(input, '.')) != NULL) {
      printf(ERROR2, input);
      return (EXIT_FAILURE);
    }
  }
  /* create cchFileName = fileName + \".cch\" */
  initialize(buffer);
  strcpy(buffer, fileName);
  strcat(buffer, ".cch");
  strcpy(cchFileName, buffer);

  /* create ccsFileName = fileName + \".ccs\" */
  initialize(buffer);
  strcpy(buffer, fileName);
  strcat(buffer, ".ccs");
  strcpy(ccsFileName, buffer);
  return (EXIT_SUCCESS);
}

int get_extension_ccs_FileName(i8 extension[charNum])
{

  i8 inputFileName[charNum];

  initialize(inputFileName);
  strcpy(inputFileName, fileName);
  strcat(inputFileName, extension);
  strcat(inputFileName, ".ccs");

  if ((strcasecmp(fileName, "_coinci")) == 0) {
    printf(ERROR35, inputFileName);
    return (EXIT_FAILURE);
  }

  if ((strcmp(extension, "_coinci")) == 0) {
    initialize(coinci_ccsFileName);
    strcpy(coinci_ccsFileName, inputFileName);

    if (get_extension_cch_FileName(extension) != EXIT_SUCCESS)
      return (EXIT_FAILURE);
  } else if ((strcmp(extension, "_bis")) == 0) {
    initialize(bis_ccsFileName);
    strcpy(bis_ccsFileName, inputFileName);

    if (get_extension_cch_FileName("_bis") != EXIT_SUCCESS)
      return (EXIT_FAILURE);
  } else {
    printf(ERROR33, fileName, extension, fileName, fileName);
    return (EXIT_FAILURE);
  }
  return (EXIT_SUCCESS);
}


int get_extension_cch_FileName(i8 extension[charNum])
{

  i8 inputFileName[charNum], newData[charNum];

  initialize(inputFileName);
  strcpy(inputFileName, fileName);
  strcat(inputFileName, extension);
  strcat(inputFileName, ".cch");

  if ((strcmp(extension, "_coinci")) == 0) {
    initialize(coinci_cchFileName);
    strcpy(coinci_cchFileName, inputFileName);

    if (copyFile(cchFileName, coinci_cchFileName) != EXIT_SUCCESS)
      return (EXIT_FAILURE);
    initialize(newData);
    strcpy(newData, fileName);
    strcat(newData, extension);
    if (modifyDataInFile("scan file name", newData, coinci_cchFileName) !=
	EXIT_SUCCESS)
      return (EXIT_FAILURE);
  } else if ((strcmp(extension, "_bis")) == 0) {
    initialize(bis_cchFileName);
    strcpy(bis_cchFileName, inputFileName);

    if (copyFile(cchFileName, bis_cchFileName) != EXIT_SUCCESS)
      return (EXIT_FAILURE);
    initialize(newData);
    strcpy(newData, fileName);
    strcat(newData, extension);
    if (modifyDataInFile("scan file name", newData, bis_cchFileName) !=
	EXIT_SUCCESS)
      return (EXIT_FAILURE);
  } else {
    printf(ERROR33, fileName, extension, fileName, fileName);
    return (EXIT_FAILURE);
  }

  return (EXIT_SUCCESS);
}

int copyNewCCHfile(i8 newName[charNum])
{
  i8 inputFileName[charNum];

  initialize(inputFileName);
  strcpy(inputFileName, newName);
  strcat(inputFileName, ".cch");

  if (copyFile(cchFileName, inputFileName) != EXIT_SUCCESS)
    return (EXIT_FAILURE);
  if (modifyDataInFile("scan file name", newName, inputFileName) !=
      EXIT_SUCCESS)
    return (EXIT_FAILURE);

  return (EXIT_SUCCESS);
}
