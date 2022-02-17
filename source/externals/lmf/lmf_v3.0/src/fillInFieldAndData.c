/*-------------------------------------------------------

           List Mode Format 
                        
     --  fillInFieldAndData.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of fillInFieldAndData.c:


	 Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	 Function used for the ascii part of LMF:
	 ->fillInFieldAndData - fill in data struct LMF_cch.field and struct LMF_cch.data 
-------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"

/** fillInFieldAndData - fill in data struct LMF_cch.field and struct LMF_cch.data */

int fillInFieldAndData()
{

  i8 indata[charNum], stringbuf[charNum], buffer[charNum],
      copyLine[charNum];
  i8 *fileLine = NULL;
  int indataLength = 0, cch_index = 0;
  FILE *scanFile;

  scanFile = fopen(cchFileName, "r");
  if (openFile(scanFile, cchFileName))
    exit(EXIT_FAILURE);

  initialize(indata);
  fileLine = NULL;

  while (fgets(indata, charNum, scanFile) != NULL) {

    if ((strncasecmp(indata, "\n", 1)) == 0)
      continue;
    if ((strchr(indata, '\t')) != 0) {
      initialize(copyLine);
      strcpy(copyLine, indata);
      initialize(indata);
      strcpy(indata, strtok(copyLine, " \t"));
      strcat(indata, " ");
      while ((fileLine = strtok(NULL, " \t\n")) != NULL) {
	strcat(indata, fileLine);
	strcat(indata, " ");
      }
    }

    if ((plist_cch = allocOfMemoryForLMF_cch(cch_index)) == NULL)
      break;

    first_cch_list = plist_cch;
    indataLength = strlen(indata);
    initialize(stringbuf);
    initialize(buffer);
    fileLine = NULL;
    strncpy(stringbuf, indata, (indataLength - 1));
    stringbuf[indataLength - 1] = 0;

    if ((strchr(stringbuf, ':')) != 0) {
      strcpy(plist_cch[cch_index].field, strtok(stringbuf, ":"));
      strcat(plist_cch[cch_index].field, "\0");

      while ((fileLine = strtok(NULL, " ,?;\t")) != NULL) {
	strcat(buffer, fileLine);
	strcat(buffer, " ");
      }

      if ((strncasecmp(buffer, ".", 1)) == 0) {
	strcpy(plist_cch[cch_index].data, "0");
	strcat(plist_cch[cch_index].data, buffer);
      }

      else
	strcpy(plist_cch[cch_index].data, buffer);
    }

    else if ((strchr(stringbuf, ':')) == 0) {
      printf(ERROR26, stringbuf, cchFileName);
      printf(ERROR5, cchFileName);
      return (0);
    }

    cch_index++;
    initialize(indata);
    fileLine = NULL;
  }

  fclose(scanFile);
  return (cch_index - 1);
}
