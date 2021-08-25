/*-------------------------------------------------------

           List Mode Format 
                        
     --  modifyDateFormat.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of modifyDateFormat.c:



			   
	 Description : 
	 Fill in the LMF record carrier with the data contained in the LMF ASCII header file
	 Functions used for the ascii part of LMF:
	 ->modifyDateFormat - Convert the date in a default format

-------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include "lmf.h"

/**** modifyDateFormat - Convert the date in a default format ****/

struct tm modifyDateFormat(i8 date[charNum], i8 field[charNum])
{

  i8 buffer[charNum], stringbuf[charNum], word[charNum];
  int conv_result = 0, monthIndex = 0, result_year = 0;
  struct tm result_date = { 0 };
  i8 *line = NULL;
  i8 *MonthNameTable[] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
    "Aug", "Sep", "Oct", "Nov", "Dec"
  };

  initialize(buffer);
  initialize(stringbuf);

  strcpy(buffer, date);
  strcpy(stringbuf, strtok(buffer, " "));
  while ((line = strtok(NULL, " ")) != NULL) {
    strcat(stringbuf, line);
    line = NULL;
  }
  initialize(buffer);
  strcpy(buffer, stringbuf);

  if (strpbrk(buffer, "/:,") != NULL) {
    if (isdigit(buffer[0]) == 0) {
      initialize(stringbuf);
      strcpy(stringbuf, strtok(buffer, "/:,"));
      for (monthIndex = 0; monthIndex < 12; monthIndex++) {
	if (strncasecmp(stringbuf, MonthNameTable[monthIndex], 3) == 0) {
	  result_date.tm_mon = monthIndex;
	  break;
	}
      }
      if (strncasecmp(stringbuf, MonthNameTable[monthIndex], 3) == 0) {
	initialize(word);
	if ((line = strtok(NULL, "/:,")) == NULL) {
	  printf(ERROR6, field, date);
	  printf(ERROR5, cchFileName);
	  exit(EXIT_FAILURE);
	}
	strcpy(word, line);
	if ((isdigit(word[0]) != 0) && (isdigit(word[1]) != 0) != 0) {
	  conv_result = atoi(word);
	  if ((conv_result >= 1 && conv_result <= 31) != 0) {
	    result_date.tm_mday = conv_result;
	    initialize(word);
	    line = NULL;
	    if ((line = strtok(NULL, "/:,")) == NULL) {
	      printf(ERROR6, field, date);
	      printf(ERROR5, cchFileName);
	      exit(EXIT_FAILURE);
	    }
	    strcpy(word, line);
	    if ((isdigit(word[0]) != 0) && (isdigit(word[1]) != 0) != 0) {
	      result_year = atoi(word);
	      if (result_year < 100)
		result_date.tm_year = result_year;
	      else if (result_year >= 1900)
		result_date.tm_year = result_year - 1900;
	      else if ((result_year >= 100 && result_year < 1900) != 0) {
		printf(ERROR6, field, date);
		printf(ERROR5, cchFileName);
		exit(EXIT_FAILURE);
	      }
	    }
	  }
	}
      }
    }
  } else {
    printf(ERROR6, field, date);
    printf(ERROR5, cchFileName);
    exit(EXIT_FAILURE);
  }
  return (result_date);
}
