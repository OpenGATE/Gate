/*-------------------------------------------------------

           List Mode Format 
                        
     --  askTheMode.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of askTheMode.c:
     Ask what kind of event the user want to generate
     Buil a i16  called chosenmode with the following 
     bit format:
     
     0000 0000 0000  0GCE
     
     where:
     G gate digi record
     C count rate record
     E event record
     
     Example: chosenMode = 5 if the file contains 
     event record & 
     gate digi record

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"


u16 askTheMode()
{
  u16 chosenMode = 0;
/* number of different records*/
  printf("\nWhat do you want to store ?\n");
  chosenMode = 0;

  printf("1 : event record \n");
  if (hardgetyesno()) {
    chosenMode += 1;
    printf
	("---> do you want the extended event record : gate digi record ?\n");
    if (hardgetyesno())
      chosenMode += 4;
  }
  printf("2 : count rate record\n");
  if (hardgetyesno())
    chosenMode += 2;

  if (chosenMode == 0) {
    printf("No record selected : exit\n");
    exit(0);
  } else
    return (chosenMode);

}
