/*-------------------------------------------------------

           List Mode Format 
                        
     --  fillEHforAscii.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of fillEHforAscii.c:
     This function is used for ascii to lmf tool (a2lmf)
     

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

/* // first we store the GATE event records as singles */
/* // this is just an automatically filling of Event header */
/* // structure   */
/* // we choose to store singles, with energy, id, no neighbours */
/* // and the both of gantry positions   */
EVENT_HEADER(*fillEHforAscii(EVENT_HEADER * peH, int codeSorC))
{

  /*peH->coincidenceBool= 0 Singles stored ; 1 coincidence stored */

  if (codeSorC == 1)		/*  // singles ascii file case */
    peH->coincidenceBool = 0;
  if (codeSorC == 2)		/* // coincidences ascii file case */
    peH->coincidenceBool = 1;


  peH->detectorIDBool = 1;	/*The IDs are stored */
  peH->energyBool = 1;		/*Energy stored */
  peH->neighbourBool = 0;	/* // no neighbours  */
  peH->neighbourhoodOrder = 0;
  peH->numberOfNeighbours = 0;
  peH->gantryAngularPosBool = 0;	/*Gantry angular pos. not stored */
  peH->gantryAxialPosBool = 0;	/*Gantry axial pos.  not stored */
  peH->sourcePosBool = 0;	/* source pos not stored */
  return (peH);
}
