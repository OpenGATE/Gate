/*-------------------------------------------------------

           List Mode Format 
                        
     --  editLMF_cchInfo.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of editLMF_cchInfo.c:
	 Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	 Functions used for the ascii part of LMF:
	 ->getToEditLMF_cchInfo - Find and print a value in the cch_file
	 ->editLMF_cchData - Print structures LMF_cch

---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lmf.h"

/** Function getToEditLMF_cchInfo - Find and print a value in the cch_file **/

int getToEditLMF_cchData(int last_lmf_header)
{

  int TESTSEARCH = 0, dataType = 0, cch_index = 0, lmf_header_index = 0;
  i8 *line = NULL;
  i8 buffer[charNum], infield[charNum];

  while (TESTSEARCH == 0) {
    initialize(infield);
    printf
	("To find the value of a field, type the field description (max %d i8acters): ",
	 (charNum - 1));
    if (*gets(infield) == '\0')
      continue;
    if (strlen(infield) >= charNum) {
      printf(ERROR4, charNum);
      initialize(infield);
      continue;
    } else {
      initialize(buffer);
      strcpy(buffer, infield);
      initialize(infield);
      strcpy(infield, strtok(buffer, " -_;,."));
      while ((line = strtok(NULL, " -_;,.")) != NULL) {
	strcat(infield, " ");
	strcat(infield, line);
      }
      plist_lmf_header = first_lmf_header;
      for (cch_index = 0; cch_index <= last_cch_list; cch_index++) {
	if (strcasecmp(plist_cch[cch_index].field, infield) == 0) {
	  for (lmf_header_index = 0; lmf_header_index <= last_lmf_header;
	       lmf_header_index++) {
	    if (strcasecmp
		(plist_lmf_header[lmf_header_index].field,
		 plist_cch[cch_index].field) == 0) {
	      dataType = plist_lmf_header[lmf_header_index].type;
	      break;
	    }
	  }
	  switch (dataType) {
	  case 0:		/*data = not defined */
	  case 1:		/*data = string */
	    printf("\t%s: %s\n", plist_cch[cch_index].field,
		   plist_cch[cch_index].def_unit_value.vChar);
	    break;
	  case 2:		/*data = value without unit */
	    printf("\t%s: %f\n", plist_cch[cch_index].field,
		   plist_cch[cch_index].def_unit_value.vNum);
	    break;
	  case 3:		/*data is a date */
	    {
	      i8 stringbuf[charNum];

	      initialize(stringbuf);
	      strftime(stringbuf, charNum, "%b/%d/%Y",
		       &plist_cch[cch_index].def_unit_value.tps_date);

	      printf("\t%s: %s\n", plist_cch[cch_index].field, stringbuf);
	    }
	    break;
	  case 4:		/*data is a duration */
	  case 5:		/*data is a time */
	    printf("\t%s: %d h %d min %d s\n", plist_cch[cch_index].field,
		   plist_cch[cch_index].def_unit_value.tps_date.tm_hour,
		   plist_cch[cch_index].def_unit_value.tps_date.tm_min,
		   plist_cch[cch_index].def_unit_value.tps_date.tm_sec);
	    break;
	  default:		/*data is a value + an unit */
	    printf("\t%s: %E %s\n", plist_cch[cch_index].field,
		   plist_cch[cch_index].def_unit_value.vNum,
		   plist_cch[cch_index].def_unit);
	  }
	  TESTSEARCH = 1;
	  break;
	}
      }
      if (strcasecmp(plist_cch[cch_index].field, infield) != 0) {
	printf(ERROR34, infield, cchFileName);
	continue;
      }
    }
  }
  return (EXIT_SUCCESS);
}


/* Function editLMF_cchData - Print structures LMF_cch */

int editLMF_cchData(int last_lmf_header)
{

  int TESTEDIT = 0, TESTSEARCH = 0, dataType = 0;
  i8 reply[charNum];
  int cch_index = 0, lmf_header_index = 0;

  while (TESTEDIT == 0) {
    initialize(reply);
    printf("Edit the cch_file ? (y/n): ");
    if (*gets(reply) == 'y') {
      plist_lmf_header = first_lmf_header;
      for (cch_index = 0; cch_index <= last_cch_list; cch_index++) {
	for (lmf_header_index = 0; lmf_header_index <= last_lmf_header;
	     lmf_header_index++) {
	  if (strcasecmp
	      (plist_lmf_header[lmf_header_index].field,
	       plist_cch[cch_index].field) != 0)
	    continue;
	  else {
	    dataType = plist_lmf_header[lmf_header_index].type;
	    switch (dataType) {
	    case 0:		/*data = not defined */
	    case 1:		/*data = string */
	      printf("\t%s: %s\n", plist_cch[cch_index].field,
		     plist_cch[cch_index].def_unit_value.vChar);
	      break;
	    case 2:		/*data = value without unit */
	      printf("\t%s: %f\n", plist_cch[cch_index].field,
		     plist_cch[cch_index].def_unit_value.vNum);
	      break;
	    case 3:		/*data is a date */
	      {
		i8 stringbuf[charNum];

		initialize(stringbuf);
		strftime(stringbuf, charNum, "%b/%d/%Y",
			 &plist_cch[cch_index].def_unit_value.tps_date);

		printf("\t%s: %s\n", plist_cch[cch_index].field,
		       stringbuf);
	      }
	      break;
	    case 4:		/*data is a duration */
	    case 5:		/*data is a time */
	      printf("\t%s: %d h %d min %d s\n",
		     plist_cch[cch_index].field,
		     plist_cch[cch_index].def_unit_value.tps_date.tm_hour,
		     plist_cch[cch_index].def_unit_value.tps_date.tm_min,
		     plist_cch[cch_index].def_unit_value.tps_date.tm_sec);
	      break;
	    default:		/*data is a value + an unit */
	      printf("\t%s: %E %s\n", plist_cch[cch_index].field,
		     plist_cch[cch_index].def_unit_value.vNum,
		     plist_cch[cch_index].def_unit);
	    }
	    break;
	  }
	}
      }
      break;
    } else if (reply[0] == 'n') {
      TESTEDIT = 1;
      initialize(reply);
    }
  }
  while (TESTSEARCH == 0) {
    initialize(reply);
    printf("\nFind information in the cch_file ? (y/n): ");
    if (*gets(reply) == 'y') {
      if (getToEditLMF_cchData(last_lmf_header) != 0)
	return (EXIT_FAILURE);
    } else if (reply[0] == 'n')
      break;
  }
  return (EXIT_SUCCESS);
}
