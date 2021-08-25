/*-------------------------------------------------------

           List Mode Format 
                        
     --  modifyDurationFormat.c  --                      

     Magalie.Krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of modifyDurationFormat.c:

	 Fill in the LMF record carrier with the data contained 
	 in the LMF ASCII header file
	 Functions used for the ascii part of LMF:
	 ->modifyDurationFormat - Convert duration in a default format
-------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include "lmf.h"

/**** modifyDurationFormat - Convert duration in a default format ****/

content_data_unit modifyDurationFormat(i8 duration[charNum],
				       i8 field[charNum])
{

  i8 *line = NULL, *end_word = NULL;
  i8 buffer[charNum], stringbuf[charNum], duration_info[charNum];
  double duration_list[3] = { 0 };
  int unitIndex = 0, formatIndex = 0, data_length = 0, stringbuf_length =
      0, buffer_length = 0, lengthIndex = 0, unit_length = 0;
  int delimiters[3][2] = { {'s', 'S'}, {'m', 'M'}, {'h', 'H'} };
  content_data_unit content_data = { 0 };

  initialize(stringbuf);
  initialize(buffer);
  strcpy(buffer, duration);
  data_length = strlen(buffer);

  if (data_length != 0) {
    /* remove the empty space in the string "data" */
    strcpy(stringbuf, strtok(buffer, " "));
    while ((line = strtok(NULL, " ")) != NULL) {
      strcat(stringbuf, line);
      line = NULL;
    }
  }
  stringbuf_length = strlen(stringbuf);

  if ((line = strpbrk(stringbuf, "hmHM")) != NULL) {
    strcpy(content_data.unit, "s");
    for (unitIndex = 2; unitIndex >= 0; unitIndex--) {
      for (formatIndex = 0; formatIndex < 2; formatIndex++) {
	line = NULL;
	data_length = strlen(stringbuf);
	end_word = strchr(stringbuf, '\0');
	if ((line =
	     strchr(stringbuf,
		    delimiters[unitIndex][formatIndex])) != NULL) {
	  initialize(buffer);
	  initialize(duration_info);
	  strcpy(buffer, line);
	  buffer_length = strlen(buffer);
	  strncpy(duration_info, stringbuf, (data_length - buffer_length));
	  if (isdigit(duration_info[(data_length - buffer_length) - 1]) == 0) {	/* duration_info[(data_length-buffer_length)-1] != number */
	    printf(ERROR48, field, duration);
	    printf(ERROR5, cchFileName);
	    exit(EXIT_FAILURE);
	  }
	  duration_list[unitIndex] = atof(duration_info);
	  while (line != end_word) {
	    if (isdigit(line[0]) != 0) {	/* line[0]==number */
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
    if ((duration_list[0] >= 0 && duration_list[0] < 60) != 0)
      content_data.value = duration_list[0];
    else {
      printf(ERROR49, field, duration);
      printf(ERROR5, cchFileName);
      exit(EXIT_FAILURE);
    }
    if ((duration_list[1] >= 0 && duration_list[1] < 60) != 0)
      content_data.value += (duration_list[1] * 60);
    else {
      printf(ERROR50, field, duration);
      printf(ERROR5, cchFileName);
      exit(EXIT_FAILURE);
    }
    if ((duration_list[2] >= 0 && duration_list[2] < 24) != 0)
      content_data.value += (duration_list[2] * 3600);
    else {
      printf(ERROR51, field, duration);
      printf(ERROR5, cchFileName);
      exit(EXIT_FAILURE);
    }
    if ((line = strpbrk(stringbuf, "hmsHMS")) == NULL) {
      printf(ERROR48, field, duration);
      printf(ERROR5, cchFileName);
      exit(EXIT_FAILURE);
    }
  } else if ((line = strpbrk(stringbuf, "sS")) != NULL) {
    end_word = strchr(stringbuf, '\0');
    end_word = end_word - 2;

    while (lengthIndex < stringbuf_length) {
      if ((isdigit(end_word[0]) != 0)
	  || ((strncasecmp(end_word, ".", 1)) == 0)) {
	end_word++;
	strcpy(content_data.unit, end_word);
	unit_length = strlen(content_data.unit);
	initialize(buffer);
	strncpy(buffer, stringbuf, (stringbuf_length - unit_length));
	duration_list[0] = atof(buffer);
	break;
      }
      end_word--;
      lengthIndex++;
    }
    content_data.value = duration_list[0];
  }
  return (content_data);
}
