/*-------------------------------------------------------

           List Mode Format 
                        
     --  calculateSSrepeat.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of calculateSSrepeat.c:
     This function gives how many singles rate are 
     in one countrate record 
     if SS = 0 --> just the total countrate
     Si SS = 1 --> 1 by ring
     Si SS = 2 --> 1 by sector
     Si SS = 3 --> 1 by module  
     it needs the value of SS in the encoding countrate pattern :
     TTTT sSSc FrbR RRRR
     and a ENCODING_HEADER structure pointer


-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"


i16 calculateSSrepeat(i16 SSpart, const ENCODING_HEADER * pEncoH)
{

  i16 SSrepeat = 0;
  if (SSpart == 1) {
    SSrepeat = (pEncoH->scannerTopology.totalNumberOfRsectors);
  }
  if (SSpart == 2) {
    SSrepeat =
	((pEncoH->scannerTopology.totalNumberOfModules) *
	 (pEncoH->scannerTopology.totalNumberOfRsectors));
  }
  if (SSpart == 3) {
    SSrepeat =
	((pEncoH->scannerTopology.totalNumberOfRsectors) *
	 (pEncoH->scannerTopology.totalNumberOfModules) *
	 (pEncoH->scannerTopology.totalNumberOfSubmodules));
  }
  return (SSrepeat);

}

/* U N C O M M E N T   T H I S   M A I N   F O R   E X A M P L E*/
/* main() */
/* { */
/*   ENCODING_HEADER encodingHeader; */

/*   encodingHeader.scannerTopology.nRing = 4; */
/*   encodingHeader.scannerTopology.nSector = 16; */
/*   encodingHeader.scannerTopology.nModule = 2; */
/*   encodingHeader.scannerTopology.nCrystal = 32; */
/*   encodingHeader.scannerTopology.nLayer = 2; */

/*    printf("\nSS = %d et repeat = %d\n",0,calculateSSrepeat(0,&encodingHeader)); */
/* } */
