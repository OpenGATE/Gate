/*-------------------------------------------------------

           List Mode Format 
                        
     --  fillEHforGATE.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of fillEHforGATE.c:
    This function is used for GATE output (a2lmf)
     


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

/* // first we store the GATE event records as singles */
/* // this is just a automatically filling of Event header */
/* // structure   */
/* // we choose to store singles, with energy, id, no neighbours */
/* // and the both of gantry positions   */
EVENT_HEADER(*fillEHforGATE(EVENT_HEADER * peH))
{

  peH->coincidenceBool = 0;	/* Singles stored  */
  peH->detectorIDBool = 1;	/*The IDs are stored */
  peH->energyBool = 1;		/*Energy stored */
  peH->neighbourBool = 0;	/* // no neighbours  */
  peH->neighbourhoodOrder = 0;
  peH->numberOfNeighbours = 0;
  peH->gantryAngularPosBool = 1;	/*Gantry angular pos.stored */
  peH->gantryAxialPosBool = 1;	/*Gantry axial pos.  stored */
  peH->sourcePosBool = 0;	/* source pos not stored */
  peH->gateDigiBool = 1;
  peH->fpgaNeighBool = 0;

  return (peH);
}
