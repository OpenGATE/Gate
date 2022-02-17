/*-------------------------------------------------------

           List Mode Format 
                        
     --  readOneGateDigiRecord.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of readOneGateDigiRecord.c:
     Fills the LMF_ccs_gateDigiRecord structure with 
     a block of read bytes (block size must be check before)
     The block is pointed by *pBufGateDigi

-------------------------------------------------------*/
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"


u8 firstHalfOf(u8 a)
{
  a = a >> 4;
  return (a);			/*  // return the value of the 4 first bits of a byte */
}

u8 secondHalfOf(u8 a)
{
  a = a & 15;
  return (a);			/* // return the value of the 4 last bits of a byte */
}


void readOneGateDigiRecord(u8 * pBufGateDigi,
			   u16 GDpattern,
			   GATE_DIGI_HEADER * pGDH,
			   EVENT_HEADER * pEH, EVENT_RECORD * pER)
{



  /***************************************************************/
  int i;			/* Counter                        */
  #ifdef _64
  u32vsu8 pi32;
  int cnt;
#else
  u32 *pi32 = NULL;		/* i32 pointer (4 bytes)         */
#endif
  u16 *punch = NULL;		/* i16. pointer (2 bytes)       */
  u8 *puni8 = NULL;		/* i8 pointer (1 byte)          */
  i16 *pi16 = NULL;		/* i16 pointer                  */
  /***************************************************************/

  puni8 = (u8 *) pBufGateDigi;  /* initial position of the pointers */
  punch = (u16 *) puni8;

#ifdef _64
  cnt = 0;
#else
  pi32 = (u32 *) puni8;
#endif


  if ((GDpattern & BIT8) == BIT8) {	/* if r=1 */
#ifdef _64
    pi32.w16[1] = ntohs(*punch);
/*     printf("[1] %hu -> %hu\n",*punch,pi32.w16[1]); */
    punch++;
    pi32.w16[0] = ntohs(*punch);
/*     printf("[0] %hu -> %hu\n",*punch,pi32.w16[0]); */
    punch++;
    pER->pGDR->runID = pi32.w32;
/*     printf("\trunID = %lu\n",pER->pGDR->runID); */
#else
    *pi32 = ntohl(*pi32);	/* // byte order */
    pER->pGDR->runID = *pi32;
    pi32++;
#endif
  }


  if ((GDpattern & BIT9) == BIT9) {	/* if e=1 */
#ifdef _64
    pi32.w16[1] = ntohs(*punch);
    punch++;
    pi32.w16[0] = ntohs(*punch);
    punch++;
    pER->pGDR->eventID[0] = pi32.w32;
#else
    *pi32 = ntohl(*pi32);	/* // byte order */
    pER->pGDR->eventID[0] = *pi32;
    pi32++;
#endif
    if (pEH->coincidenceBool) {
#ifdef _64
    pi32.w16[1] = ntohs(*punch);
    punch++;
    pi32.w16[0] = ntohs(*punch);
    punch++;
    pER->pGDR->eventID[1] = pi32.w32;
#else
      *pi32 = ntohl(*pi32);	/* // byte order */
      pER->pGDR->eventID[1] = *pi32;
      pi32++;
#endif
    }
  }


  if ((GDpattern & BIT6) == BIT6) {	/* if M=1 */
    if (pEH->coincidenceBool) {	/*if  c=1 */
#ifdef _64
    pi32.w16[1] = ntohs(*punch);
    punch++;
    pi32.w16[0] = ntohs(*punch);
    punch++;
    pER->pGDR->multipleID = pi32.w32;
#else
      *pi32 = ntohs(*pi32);
      pER->pGDR->multipleID = *pi32;
      pi32++;
#endif
    }
  }

#ifndef _64
  punch = (u16 *) pi32;
#endif

  if ((GDpattern & BIT11) == BIT11) {	/* if S=1 */
    *punch = ntohs(*punch);
    pER->pGDR->sourceID[0] = *punch;
    punch++;			/* Avance de 1 uns i16 */


    if (pEH->coincidenceBool) {	/*if s=1 & c=1 */
      *punch = ntohs(*punch);
      pER->pGDR->sourceID[1] = *punch;
      punch++;			/* Avance de 1 uns i16 */
    }
  }

  pi16 = (i16 *) punch;

  if ((GDpattern & BIT10) == BIT10) {	/* if p=1 */
    *pi16 = ntohs(*pi16);
    pER->pGDR->sourcePos[0].X = *pi16;
    pi16++;			/* Avance de 1 uns i16 */

    *pi16 = ntohs(*pi16);
    pER->pGDR->sourcePos[0].Y = *pi16;
    pi16++;			/* Avance de 1 uns i16 */

    *pi16 = ntohs(*pi16);
    pER->pGDR->sourcePos[0].Z = *pi16;
    pi16++;			/* Avance de 1 uns i16 */



    if (pEH->coincidenceBool) {	/*if p=1 & c= 1 */
      *pi16 = ntohs(*pi16);
      pER->pGDR->sourcePos[1].X = *pi16;
      pi16++;			/* Avance de 1 uns i16 */

      *pi16 = ntohs(*pi16);
      pER->pGDR->sourcePos[1].Y = *pi16;
      pi16++;			/* Avance de 1 uns i16 */

      *pi16 = ntohs(*pi16);
      pER->pGDR->sourcePos[1].Z = *pi16;
      pi16++;			/* Avance de 1 uns i16 */

    }
  }

  if ((GDpattern & BIT7) == BIT7) {	/* if G=1 */
    *pi16 = ntohs(*pi16);
    pER->pGDR->globalPos[0].X = *pi16;
    pi16++;
    *pi16 = ntohs(*pi16);
    pER->pGDR->globalPos[0].Y = *pi16;
    pi16++;
    *pi16 = ntohs(*pi16);
    pER->pGDR->globalPos[0].Z = *pi16;
    pi16++;

    if (pEH->neighbourBool) {
      for (i = 1; i < pEH->numberOfNeighbours + 1; i++) {
	*pi16 = ntohs(*pi16);
	pER->pGDR->globalPos[i].X = *pi16;
	pi16++;
	*pi16 = ntohs(*pi16);
	pER->pGDR->globalPos[i].Y = *pi16;
	pi16++;
	*pi16 = ntohs(*pi16);
	pER->pGDR->globalPos[i].Z = *pi16;
	pi16++;

      }
    }


    if (pEH->coincidenceBool) {
      *pi16 = ntohs(*pi16);
      pER->pGDR->globalPos[pEH->numberOfNeighbours + 1].X = *pi16;
      pi16++;
      *pi16 = ntohs(*pi16);
      pER->pGDR->globalPos[pEH->numberOfNeighbours + 1].Y = *pi16;
      pi16++;
      *pi16 = ntohs(*pi16);
      pER->pGDR->globalPos[pEH->numberOfNeighbours + 1].Z = *pi16;
      pi16++;

      if (pEH->neighbourBool) {
	for (i = pEH->numberOfNeighbours + 2;
	     i <= (2 * pEH->numberOfNeighbours) + 1; i++) {

	  *pi16 = ntohs(*pi16);
	  pER->pGDR->globalPos[i].X = *pi16;
	  pi16++;
	  *pi16 = ntohs(*pi16);
	  pER->pGDR->globalPos[i].Y = *pi16;
	  pi16++;
	  *pi16 = ntohs(*pi16);
	  pER->pGDR->globalPos[i].Z = *pi16;
	  pi16++;
	}
      }
    }
  }


  puni8 = (u8 *) pi16;
  if ((GDpattern & BIT12) == BIT12) {	/*if C = 1 */
    if (pEH->coincidenceBool) {	/*if c = 1 */
      pER->pGDR->numberCompton[0] = firstHalfOf(*puni8);
      pER->pGDR->numberCompton[1] = secondHalfOf(*puni8);
      puni8++;
    } else {
      pER->pGDR->numberCompton[0] = *puni8;
      puni8++;
    }
  }

  if ((GDpattern & BIT5) == BIT5) {	/*if D = 1 */



    if (pEH->coincidenceBool) {	/*if c = 1 */

      pER->pGDR->numberDetectorCompton[0] = firstHalfOf(*puni8);
      pER->pGDR->numberDetectorCompton[1] = secondHalfOf(*puni8);
      puni8++;
    } else {

      pER->pGDR->numberDetectorCompton[0] = *puni8;
      puni8++;
    }
  }

  return;
}
