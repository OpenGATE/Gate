/*-------------------------------------------------------

           List Mode Format 
                        
     --  calculatesizecountrate.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of calculatesizecountrate.c:
     This function computes and returns the size (in bytes) 
     of 1 countrate record
     The parameters are the encoding countrate pattern and a pointer to 
     encoding header Structure   
     

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"



u16 calculatesizecountrate(u16 ep4, const ENCODING_HEADER * pEncoH)
{


  u16 sizecountrate = 0;
  i16 SSpart = 0, SSrepeat = 0, i;


  /* TAG + TIME STAMP */
  sizecountrate = sizecountrate + 4;
  /* TOTAL SINGLE RATE */
  if ((ep4 & BIT12) == BIT12) {
    sizecountrate = sizecountrate + 4;

    SSpart = (ep4 >> 9);	/* Extract SS from TTTT sSSc FrbR RRRR */
    SSpart = (SSpart & 3);
    SSrepeat = calculateSSrepeat(SSpart, pEncoH);	/* number of partial rates */

    for (i = 0; i < SSrepeat; i++)
      sizecountrate = sizecountrate + 2;	/* 2 bytes more for each section */
  }
  /* Total coincidence rate */
  if ((ep4 & BIT9) == BIT9)
    sizecountrate = sizecountrate + 2;

  /* Total random rate */
  if ((ep4 & BIT8) == BIT8)
    sizecountrate = sizecountrate + 2;

  /* rotation speed */
  if ((ep4 & BIT7) == BIT7)
    sizecountrate = sizecountrate + 1;


  /* bed speed */
  if ((ep4 & BIT6) == BIT6)
    sizecountrate = sizecountrate + 1;



  return (sizecountrate);

}

/*
main()
{
  ENCODING_HEADER encodingHeader;
  
  encodingHeader.scannerTopology.nRing = 4;
  encodingHeader.scannerTopology.nSector = 16;
  encodingHeader.scannerTopology.nModule = 2;
  encodingHeader.scannerTopology.nCrystal = 32;
  encodingHeader.scannerTopology.nLayer = 2;
  


}

*/
