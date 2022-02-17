/*-------------------------------------------------------

           List Mode Format 
                        
     --  readTheLMFcchDataBase.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of readTheLMFcchDataBase.c:


-------------------------------------------------------*/
/*------------------------------------------------------------------------
                          List Mode Format

			  --- readTheLMFcchDataBase.c ---
			  
			  released on july 2002
			  Magalie.Krieguer@iphe.unil.ch
			  Copyright IPHE/UNIL, Lausanne.
			  Crystal Clear Collaboration

			   
	 Description : 
	 Function used for the ascii part of LMF:
	 ->readTheLMFcchDataBase - read the LMF cch data base (lmf_header.db)
	                           and store these informations in the structures lmf_header
				   and close the LMF cch data base
	 
---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"

/* readTheLMFcchDataBase - read the LMF cch data base (lmf_header.db) 
   and store these informations in the structures lmf_header */

int readTheLMFcchDataBase()
{

  i8 infield[charNum], stringbuf[charNum], headerFileName[charNum];
  i8 *fileLine = NULL;
  FILE *lmf_header_infile;
  int infieldLength = 0, lmf_header_index = 0;

  initialize(headerFileName);

  strcpy(headerFileName, HEADER_FILE);
  lmf_header_infile = fopen(headerFileName, "r");

  if (lmf_header_infile == NULL) {
    lmf_header_infile = fopen(headerFileName, "a+");
    fputs("\n", lmf_header_infile);
    fclose(lmf_header_infile);
    lmf_header_infile = fopen(headerFileName, "r");
  }

  initialize(infield);
  while (fgets(infield, charNum, lmf_header_infile) != NULL) {
    if ((strncasecmp(infield, "\n", 1)) == 0)
      continue;
    if ((plist_lmf_header =
	 allocOfMemoryForLMF_Header(lmf_header_index)) == NULL)
      break;

    first_lmf_header = plist_lmf_header;
    infieldLength = strlen(infield);
    initialize(stringbuf);
    fileLine = NULL;
    strncpy(stringbuf, infield, (infieldLength - 1));
    stringbuf[infieldLength - 1] = 0;
    strcpy(plist_lmf_header[lmf_header_index].field,
	   strtok(stringbuf, "&"));
    fileLine = strtok(NULL, " ,;:)");
    plist_lmf_header[lmf_header_index].type = atoi(fileLine + 1);
    lmf_header_index++;
    initialize(infield);
  }

  fclose(lmf_header_infile);
  return (lmf_header_index - 1);
}
