/*-------------------------------------------------------

           List Mode Format 
                        
     --  modifyTimeFormat.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of modifyTimeFormat.c:

	 Fill in the LMF record carrier with the data contained 
	 in the LMF ASCII header file
	 Functions used for the ascii part of LMF:
	 ->modifyTimeFormat - Convert time in a default format
-------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include "lmf.h"

/**** modifyTimeFormat - Convert time in a default format ****/

struct tm modifyTimeFormat(i8 time[charNum], i8 field[charNum])
{

  i8 *line = NULL, *end_word = NULL;
  i8 buffer[charNum], stringbuf[charNum], time_info[charNum];
  int time_list[3] = { 0 };
  int unitIndex = 0, formatIndex = 0, data_length = 0, buffer_length = 0;
  int delimiters[3][2] = { {'s', 'S'}, {'m', 'M'}, {'h', 'H'} };
  struct tm result_time = { 0 };

  initialize(stringbuf);
  initialize(buffer);
  strcpy(buffer, time);
  strcpy(stringbuf, strtok(buffer, " "));
  while ((line = strtok(NULL, " ")) != NULL) {
    strcat(stringbuf, line);
    line = NULL;
  }
  if ((strchr(stringbuf, ':')) != NULL) {
    for (unitIndex = 0; unitIndex < 3; unitIndex++) {
      if ((line = strrchr(stringbuf, ':')) != NULL) {
	line++;
	data_length = strlen(stringbuf);
	initialize(buffer);
	strcpy(buffer, line);
	buffer_length = strlen(buffer);
	time_list[unitIndex] = atoi(line);
	initialize(buffer);
	strcpy(buffer, stringbuf);
	initialize(stringbuf);
	strncpy(stringbuf, buffer, (data_length - buffer_length - 1));
	time_list[2] = atoi(stringbuf);
      } else {
	time_list[unitIndex] = atoi(stringbuf);
	break;
      }
    }
  } else if ((line = strpbrk(stringbuf, "hmsHMS")) != NULL) {
    for (unitIndex = 2; unitIndex >= 0; unitIndex--) {
      for (formatIndex = 0; formatIndex < 2; formatIndex++) {
	line = NULL;
	data_length = strlen(stringbuf);
	end_word = strchr(stringbuf, '\0');
	if ((line =
	     strchr(stringbuf,
		    delimiters[unitIndex][formatIndex])) != NULL) {
	  initialize(buffer);
	  strcpy(buffer, line);
	  buffer_length = strlen(buffer);
	  initialize(time_info);
	  strncpy(time_info, stringbuf, (data_length - buffer_length));
	  time_list[unitIndex] = atoi(time_info);
	  while (line != end_word) {
	    if (isdigit(line[0]) != 0) {
	      initialize(buffer);
	      strcpy(buffer, line);
	      initialize(stringbuf);
	      strcpy(stringbuf, buffer);
	      break;
	    }
	    line++;
	  }
	}
      }
    }
    unitIndex = 2;
  } else {
    printf(ERROR7, field, time);
    printf(ERROR5, cchFileName);
    exit(EXIT_FAILURE);
  }
  switch (unitIndex) {
  case 0:
    if ((time_list[0] >= 0 && time_list[0] < 60) != 0)
      result_time.tm_sec = time_list[0];
    else {
      printf(ERROR8, field, time);
      printf(ERROR5, cchFileName);
      exit(EXIT_FAILURE);
    }
    break;
  case 1:
    if ((time_list[0] >= 0 && time_list[0] < 60) != 0)
      result_time.tm_sec = time_list[0];
    else {
      printf(ERROR8, field, time);
      printf(ERROR5, cchFileName);
      exit(EXIT_FAILURE);
    }
    if ((time_list[1] >= 0 && time_list[1] < 60) != 0)
      result_time.tm_min = time_list[1];
    else {
      printf(ERROR9, field, time);
      printf(ERROR5, cchFileName);
      exit(EXIT_FAILURE);
    }
    break;
  case 2:
    if ((time_list[0] >= 0 && time_list[0] < 60) != 0)
      result_time.tm_sec = time_list[0];
    else {
      printf(ERROR8, field, time);
      printf(ERROR5, cchFileName);
      exit(EXIT_FAILURE);
    }
    if ((time_list[1] >= 0 && time_list[1] < 60) != 0)
      result_time.tm_min = time_list[1];
    else {
      printf(ERROR9, field, time);
      printf(ERROR5, cchFileName);
      exit(EXIT_FAILURE);
    }
    if ((time_list[2] >= 0 && time_list[2] < 24) != 0)
      result_time.tm_hour = time_list[2];
    else {
      printf(ERROR10, field, time);
      printf(ERROR5, cchFileName);
      exit(EXIT_FAILURE);
    }
    break;
  }
  return (result_time);
}
