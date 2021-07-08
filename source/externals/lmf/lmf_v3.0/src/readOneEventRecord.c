/*-------------------------------------------------------

List Mode Format 
                        
--  readOneEventRecord.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of readOneEventRecord.c:

Fills the LMF_ccs_eventRecord structure with 
a block of read bytes (block size must be check before)
The block is pointed by *pBufEvent

-------------------------------------------------------*/
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

void readOneEventRecord(u8 * pBufEvent,
			u16 Epattern,
			EVENT_HEADER * pEH,
			u16 encodingIDSize, EVENT_RECORD * pER)
{
  /******************************************************************************************/
  int i;
  u16 size;
  u16 *punch = NULL;
  u8 *puni8 = NULL;
  u64 crystalIDs;

  punch = (u16 *) pBufEvent;	/* initial position of the pointers */
  puni8 = (u8 *) pBufEvent;

  /*-=-=-  TAG, TIME STAMP ET TOF   =-=-=-=-=*/
  if ((Epattern & BIT12) == 0) {	/* if c = 0 *//* Time stamp on 64 bits */
    if ((*punch & BIT16) == 0) {	/* if tag ok */
      for (i = 0; i < 8; i++) {	/* read the 8 bytes of time */
	pER->timeStamp[7 - i] = *puni8;
	puni8++;
      }
    } else {			/* BAD TAG WARNING */
      printf("\n\nWARNING : WRONG TAG OF EVENT !!!! ");
      printf("\nMessage from readOneEventRecord.c\n\n");
      getchar();
    }
  } else {			/* if  c = 1 */
    /* Time stamp on 24 bits & TOF on 8 bits */
    if ((*puni8 & BIT8) == 0) {	/* if tag ok */
      for (i = 0; i < 3; i++) {	/* read the 8 bytes of time */
	pER->timeStamp[i] = *puni8;
	puni8++;
      }
      for (i = 3; i < 8; i++)
	pER->timeStamp[i] = 0;

      pER->timeOfFlight = *puni8;
      puni8++;
    } else {			/* BAD TAG WARNING */
      printf("\n\nWARNING : WRONG TAG OF EVENT !!!! ");
      printf("\nMessage from readOneEventRecord.c\n\n");
      getchar();
    }
  }

  /*=-=-==-=-=*
    |    ID1   |
    *=-=-=-=-==*/
  punch = (u16 *) puni8;

  if ((Epattern & BIT11) == BIT11) {	/* if d=1 */
    pER->crystalIDs[0] = 0;
    for (size = 0; size < encodingIDSize + 1; size++) {
      *punch = ntohs(*punch);
      crystalIDs = *punch;
      punch++;
      pER->crystalIDs[0] |= crystalIDs << (size * 16);
    }
  }


  /* Avance de 1 uns i16 */
 /*=-=-==-=-=*
    |    ID2    |
    *=-=-=-=-==*/
  if (((Epattern & BIT11) == BIT11) && ((Epattern & BIT12) == BIT12)) {	/* if d=1 and c=1 extraction of ID2 */
    pER->crystalIDs[pEH->numberOfNeighbours + 1] = 0;
    for (size = 0; size < encodingIDSize + 1; size++) {
      *punch = ntohs(*punch);
      crystalIDs = *punch;
      punch++;
      pER->crystalIDs[pEH->numberOfNeighbours + 1] |=
	  crystalIDs << (size * 16);
    }
  }

  /*-=-==-=-=-=-=-=-=-=-*
    |    Gantry ang pos  |
    *=-=-=-=-=-=-=-=-=-=-*/

  if ((Epattern & BIT6) == BIT6) {	/* Si d=1 : extraction de gant. ang pos */
    *punch = ntohs(*punch);
    pER->gantryAngularPos = *punch;
    punch++;
  }
  /*-=-==-=-=-=-=-=-=-=-=*
    |    Gantry axi pos   |
    *=-=-=-=-=-=-=-=-=-=-=*/


  if ((Epattern & BIT5) == BIT5) {	/* Si b=1 : extraction du gant. axi pos */
    *punch = ntohs(*punch);
    pER->gantryAxialPos = *punch;
    punch++;
  }
  /*-=-==-=-=-=-=-=-=-=-=-=-=-=*
    |    Source ang et axi  pos  |
    *-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  if ((Epattern & BIT4) == BIT4) {	/* Si s=1 : extraction du source. axi et ang pos */
    *punch = ntohs(*punch);
    pER->sourceAngularPos = *punch;
    punch++;
    *punch = ntohs(*punch);
    pER->sourceAxialPos = *punch;
    punch++;
  }

  puni8 = (u8 *) punch;

  /*-==-=-=-=-=-=-=-==-=-=-=-=*
    |    ENERGY OF 1           |
    *-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  if ((Epattern & BIT10) == BIT10) {	/* Si e=0 : extraction energy du 1 (1 octet) */
    pER->energy[0] = *puni8;
    puni8++;
  }

  /*-==-=-=-=-=-=-=-==-=-=-=-=*
    |    ENERGY OF neigh of 1  |
    *-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  if ((Epattern & BIT9) == BIT9) {	/* extraction energy des voisins du 1 (vnn octets) */
    for (i = 1; i < pEH->numberOfNeighbours + 1; i++) {
      pER->energy[i] = *puni8;
      puni8++;
    }
  }
  /*-==-=-=-=-=-=-=-==-=-=-=-=*
    |    ENERGY OF 2           |
    *-=-=-=-=-=-=-=-=-=-=-=-=-=*/


  if (((Epattern & BIT10) == BIT10) && ((Epattern & BIT12) == BIT12)) {	/* Si e=1 et c=1: extraction energy du 2 (1 octet) */
    pER->energy[pEH->numberOfNeighbours + 1] = *puni8;

    puni8++;
  }
  /*-==-=-=-=-=-=-=-==-=-=-=-=*
    |    ENERGY OF neigh of 2  |
    *-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  /* Energy of neigh. of 2 : Si n=1 et c=1 :extraction energy des voisins du 2 */
  if (((Epattern & BIT9) == BIT9) && ((Epattern & BIT12) == BIT12)) {
    for (i = pEH->numberOfNeighbours + 2;
	 i <= (2 * pEH->numberOfNeighbours) + 1; i++) {
      pER->energy[i] = *puni8;
      puni8++;
    }
  }
  /*-==-=-=-=-=-=-=-==-=-=-=-=*
    FPGA NEIGH. INFO 
    *-=-=-=-=-=-=-=-=-=-=-=-=-=*/

  if ((Epattern & BIT2) == BIT2)	// if Z
  {
    pER->fpgaNeighInfo[0] = *puni8;
    puni8++;
    if ((Epattern & BIT12) == BIT12) {
      pER->fpgaNeighInfo[1] = *puni8;
      puni8++;
    }
  }
  return;
}
