/*-------------------------------------------------------

           List Mode Format 
                        
     --  usefulOptions.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of usefulOptions.c:


	 Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	 Functions used for the ascii part of LMF:
	 ->allocOfMemoryForLMF_Header - allocation of memory to store the data 
	   contained in the LMF cch data base 
	 
	 ->allocOfMemoryForLMF_cch - allocation of memory to store the data contained
	   in the cch file
	   
	 ->openFile - test a file opening
	 ->initialize - initializing strings
	 ->copyFile - duplicate a file 
	 ->modifyDataInFile - modify a data in a .cch file
-------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"

/* Function allocOfMemoryForLMF_Header */

lmf_header *allocOfMemoryForLMF_Header(int lmf_header_index)
{

  i8 headerFileName[charNum];
  initialize(headerFileName);
  strcpy(headerFileName, HEADER_FILE);

  if ((plist_lmf_header = (lmf_header *) realloc(plist_lmf_header,
						 (lmf_header_index +
						  1) *
						 sizeof(lmf_header))) ==
      NULL) {
    printf(ERROR3, headerFileName);
    lmf_header_index++;
  }


  return (plist_lmf_header);
}

/* Function allocOfMemoryForLMF_cch */

LMF_cch *allocOfMemoryForLMF_cch(int cch_index)
{

  if ((plist_cch =
       (LMF_cch *) realloc(plist_cch,
			   (cch_index + 1) * sizeof(LMF_cch))) == NULL) {
    printf(ERROR3, cchFileName);
    cch_index++;
  }
  return (plist_cch);
}

/* Function openFile: test a file opening */
int openFile(FILE * popnf, i8 fileName[charNum])
{
  if (popnf == NULL) {
    printf(ERROR1, fileName);
    return (EXIT_FAILURE);
  }
  return (EXIT_SUCCESS);
}

/* Function initialize: initializing of the  string */

void initialize(i8 buffer[charNum])
{
  int l;
  for (l = 0; l < charNum; l++)
    buffer[l] = 0;
}

/* Function copyFile: duplicating of the file */

int copyFile(i8 infileName[charNum], i8 copyingFileName[charNum])
{

  i8 buffer[charNum];
  FILE *infile, *outfile;

  infile = fopen(infileName, "r");
  if (openFile(infile, infileName))
    return (EXIT_FAILURE);

  outfile = fopen(copyingFileName, "w+");
  if (openFile(outfile, copyingFileName))
    return (EXIT_FAILURE);

  initialize(buffer);
  while (fgets(buffer, charNum, infile) != NULL) {
    if (buffer[0] == '\0')
      continue;
    fputs(buffer, outfile);
    initialize(buffer);
  }
  fclose(infile);
  fclose(outfile);
  return (EXIT_SUCCESS);
}

/* Function modifyDataInFile: modify a data in a .cch file */

int modifyDataInFile(i8 dataDescription[charNum], i8 newData[charNum],
		     i8 file[charNum])
{

  i8 stringbuf[charNum], buffer[charNum], copyLine[charNum],
      saveInfoFile[10000];
  i8 *fileLine = NULL;
  int dataDescriptionLength = 0, lineLength = 0, dataIndex = 0, TESTFIND =
      FALSE;
  FILE *scanFile;
  i32 dataPosition = 0;

  scanFile = fopen(file, "r+");
  if (openFile(scanFile, file))
    return (EXIT_FAILURE);

  initialize(stringbuf);
  fileLine = NULL;

  while (fgets(stringbuf, charNum, scanFile) != NULL) {

    if ((strncasecmp(stringbuf, "\n", 1)) == 0)
      continue;
    lineLength = strlen(stringbuf);

    initialize(buffer);
    strcpy(buffer, stringbuf);
    if ((strchr(buffer, '\t')) != 0) {
      initialize(copyLine);
      strcpy(copyLine, buffer);
      initialize(buffer);
      strcpy(buffer, strtok(copyLine, " \t"));
      strcat(buffer, " ");
      while ((fileLine = strtok(NULL, " \t")) != NULL) {
	strcat(buffer, fileLine);
	strcat(buffer, " ");
      }
    }

    dataDescriptionLength = strlen(dataDescription);
    if ((strncasecmp(buffer, dataDescription, dataDescriptionLength)) == 0) {
      for (dataIndex = 0; dataIndex < 10000; dataIndex++)
	saveInfoFile[dataIndex] = 0;
      dataPosition = ftell(scanFile);
      fseek(scanFile, 0L, SEEK_CUR);
      initialize(stringbuf);
      while (fgets(stringbuf, charNum, scanFile) != NULL) {
	strcat(saveInfoFile, stringbuf);
	initialize(stringbuf);
      }
      fseek(scanFile, dataPosition, SEEK_SET);
      fseek(scanFile, -lineLength, SEEK_CUR);
      initialize(stringbuf);
      sprintf(stringbuf, "%s: %s", dataDescription, newData);
      fprintf(scanFile, "%s\n", stringbuf);
      fprintf(scanFile, "%s", saveInfoFile);

      fclose(scanFile);
      initialize(stringbuf);
      fileLine = NULL;
      TESTFIND = TRUE;
      break;
    }

    initialize(stringbuf);
    fileLine = NULL;
  }
  if (TESTFIND == FALSE) {
    printf(ERROR34, dataDescription, file);
    fclose(scanFile);
    return (EXIT_FAILURE);
  }
  return (EXIT_SUCCESS);
}
