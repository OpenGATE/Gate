/*-------------------------------------------------------

           List Mode Format 
                        
     --  testField.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of testField.c:


	 Fill in the LMF record carrier with the data contained in 
	 the LMF ASCII header file
	 Functions used for the ascii part of LMF:
	 ->testField - Compare the fields described in the input file 
	               and the fields store in the lmf_header data base
-------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmf.h"

/** testField - Compare the fields described in the input file 
    and the fields store in the lmf_header data base **/

int testField(int last_lmf_header, int cch_index)
{

  int result = 0, TESTADD = 0, lmf_header_index = 0;
  i8 reply[charNum], headerFileName[charNum];
  FILE *lmf_header_infile;

  plist_lmf_header = first_lmf_header;
  if (last_lmf_header < 0) {
    while (TESTADD == 0) {
      initialize(reply);
      printf("Add this field \"%s\" in your data base ? (y/n): ",
	     plist_cch[cch_index].field);
      if (*gets(reply) == 'y') {
	strcpy(headerFileName, HEADER_FILE);
	lmf_header_infile = fopen(headerFileName, "a");
	if (openFile(lmf_header_infile, headerFileName) != 0)
	  exit(EXIT_FAILURE);
	if (writeNewFieldInDataBase
	    (lmf_header_infile, plist_cch[cch_index].field) != 0)
	  exit(EXIT_FAILURE);
	fclose(lmf_header_infile);
	if ((last_lmf_header =
	     readTheLMFcchDataBase(lmf_header_index)) < 0)
	  exit(EXIT_FAILURE);
	TESTADD = 1;
      }
    }
    return (last_lmf_header);
  }
  for (lmf_header_index = 0; lmf_header_index <= last_lmf_header;
       lmf_header_index++) {
    if (strcasecmp
	(plist_cch[cch_index].field,
	 plist_lmf_header[lmf_header_index].field) == 0)
      break;
  }
  if (strcasecmp
      (plist_cch[cch_index].field,
       plist_lmf_header[lmf_header_index].field) != 0) {
    while (TESTADD == 0) {
      result = correctUnknownField(plist_cch[cch_index].field);
      if (result == 1) {
	printf(ERROR5, cchFileName);
	exit(EXIT_FAILURE);
      } else if (result == 2) {
	initialize(reply);
	printf("Add this field \"%s\" in your data base ? (y/n): ",
	       plist_cch[cch_index].field);
	if (*gets(reply) == 'y') {
	  strcpy(headerFileName, HEADER_FILE);
	  lmf_header_infile = fopen(headerFileName, "a");
	  if (openFile(lmf_header_infile, headerFileName) != 0)
	    exit(EXIT_FAILURE);
	  if (writeNewFieldInDataBase
	      (lmf_header_infile, plist_cch[cch_index].field) != 0)
	    exit(EXIT_FAILURE);
	  fclose(lmf_header_infile);
	  if ((last_lmf_header =
	       readTheLMFcchDataBase(lmf_header_index)) < 0)
	    exit(EXIT_FAILURE);
	  TESTADD = 1;
	}
      }
    }
  }
  return (last_lmf_header);
}
