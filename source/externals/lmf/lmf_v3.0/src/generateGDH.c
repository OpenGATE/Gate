/*-------------------------------------------------------

           List Mode Format 
                        
     --  generateGDH.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of generateGDH.c:
     Standard filling of gate digi header


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

static GATE_DIGI_HEADER *pGDH = NULL;

GATE_DIGI_HEADER(*generateGDH(void))
{
  static int allocGDHdone = FALSE;

  if (allocGDHdone == 0) {
    if ((pGDH =
	 (GATE_DIGI_HEADER *) malloc(sizeof(GATE_DIGI_HEADER))) == NULL)
      printf
	  ("\n***ERROR : in generateGDH.c : imposible to do : malloc()\n");
    allocGDHdone = 1;
  }




  /*  C  */

  printf("* Do you want to store number of compton ?");
  pGDH->comptonBool = hardgetyesno();
  /*  D  */

  printf("* Do you want to store number of compton in detector ?");
  pGDH->comptonDetectorBool = hardgetyesno();


  /*  S  */
  printf("* Do you want to store source XYZ position ?");
  pGDH->sourceXYZPosBool = hardgetyesno();

  /*  p  */

  printf("* Do you want to store source ID ?");
  pGDH->sourceIDBool = hardgetyesno();


  /*  e  */

  printf("* Do you want to store the event ID ?");
  pGDH->eventIDBool = hardgetyesno();



  /*  r  */


  printf("*  Do you want to stored run ID ?");
  pGDH->runIDBool = hardgetyesno();

  /*  G  */

  printf("* Do you want to store global position ?");
  pGDH->globalXYZPosBool = hardgetyesno();

  /*  M  */

  printf("* Do you want to store multiple ID ?");
  pGDH->multipleIDBool = hardgetyesno();


  return (pGDH);

}

void generateGDHDestructor()
{
  if (pGDH)
    free(pGDH);
}
