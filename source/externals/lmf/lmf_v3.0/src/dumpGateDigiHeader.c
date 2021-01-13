/*-------------------------------------------------------

           List Mode Format 
                        
     --  dumpGateDigiHeader.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of dumpGateDigiHeader.c:
     This function called by dumpTheRecord()
     dispays on screen the containing of a
     gate digi header structure.



-------------------------------------------------------*/
#include <stdio.h>
#include "lmf.h"

void dumpGateDigiHeader(const GATE_DIGI_HEADER * pGDH)
{

  printf("\n");


  printf("--> comptonBool =%d\n", pGDH->comptonBool);
  printf("--> comptonDetectorBool =%d\n", pGDH->comptonDetectorBool);
  printf("--> sourceIDBool =%d\n", pGDH->sourceIDBool);
  printf("--> sourceXYZposBool =%d\n", pGDH->sourceXYZPosBool);
  printf("--> eventIDBool =%d\n", pGDH->eventIDBool);
  printf("--> runIDBool =%d\n", pGDH->runIDBool);
  printf("--> globalXYZposBool =%d\n", pGDH->globalXYZPosBool);
  printf("--> multipleIDBool =%d\n", pGDH->multipleIDBool);



}
