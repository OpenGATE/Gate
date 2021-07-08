/*-------------------------------------------------------

           List Mode Format 
                        
     --  hardget.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of hardget.c:

      hardgeti16 : ask the user to enter 
      a i16 between min and max

      hardgetyesno : ask the user to enter 
      y or n (that means yes or no)
      
      These functions are robust and don't accept 
      ambigous user's answers like Y or yes instead of y

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

i16 hardgeti16(i16 min, i16 max)
{
  i16 good;
  i8 buffer[80];
  i16 a = 0, b, value;
#define TRUE 1
#define FALSE 0
  good = FALSE;

  for (a = 0; a < 80; a++)
    buffer[a] = 0;

  while (!good) {
    printf(" ( %d - %d ) : ", min, max);
    if(fgets(buffer, sizeof(buffer), stdin))
      sscanf(buffer, "%hd", &value);
    for (b = min; b <= max; b++) {
      if (value == b) {
	a = b;
	good = TRUE;
	return (a);
      }
    }
    printf("%s, **** ERROR !!! TRY AGAIN ****\n", buffer);
  }
  return (0);
}


i16 hardgetyesno()
{

  i16 good;
  i8 buffer[80];
  i16 a = 0;
#define TRUE 1
#define FALSE 0
  good = FALSE;

  for (a = 0; a < 80; a++)
    buffer[a] = 0;

  while (!good) {
    printf("  (y / n) : ");
    fgets(buffer, sizeof(buffer), stdin);


    if ((buffer[0] == ('y')) && (buffer[0 + 1] == '\0')) {
      a = 1;
      good = TRUE;
      return (a);
    } else if ((buffer[0] == ('n')) && (buffer[0 + 1] == '\0')) {
      a = 0;
      good = TRUE;
      return (a);
    } else
      printf("**** ERROR !!! TRY AGAIN ****\n");
  }
  return (0);
}
