/*-------------------------------------------------------

           List Mode Format 
                        
     --  fillGDHforGATE.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of fillGDHforGATE.c:

     Standard filling of gate digi header
-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"


GATE_DIGI_HEADER(*fillGDHforGATE(GATE_DIGI_HEADER * pGDH))
{

  pGDH->comptonBool = 0;	/* C number of compton */
  pGDH->comptonDetectorBool = 0;	/* C number of compton */
  pGDH->sourceIDBool = 0;	/* s */
  pGDH->sourceXYZPosBool = 0;	/* S */
  pGDH->eventIDBool = 0;	/* e */
  pGDH->runIDBool = 0;		/* r */
  pGDH->globalXYZPosBool = 0;	/* G global xyz pos */
  pGDH->multipleIDBool = 0;	/* M multilple ID */

  return (pGDH);
}
