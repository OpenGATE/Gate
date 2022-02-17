 /*-------------------------------------------------------

           List Mode Format 
                        
     --  demakeid.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of demakeid.c:
     This function takes an ID and  the       
     encodingHeader structure pointer and it  
     gives the value of Ring,Sector,Module, 
     Crystal, and Layer in a 5 u16 
     table.                                    

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"


u16 *demakeid(u64 id, const ENCODING_HEADER * pEncoH)
{
  u16 *pcrist;
  /* Shift (in bits) */
  u16 bitsShiftSector;
  u16 bitsShiftModule;
  u16 bitsShiftSubmodule;
  u16 bitsShiftCrystal;

  bitsShiftCrystal = pEncoH->scanEncodingID.bitForLayers;
  bitsShiftSubmodule =
      pEncoH->scanEncodingID.bitForCrystals + bitsShiftCrystal;
  bitsShiftModule =
      pEncoH->scanEncodingID.bitForSubmodules + bitsShiftSubmodule;
  bitsShiftSector = pEncoH->scanEncodingID.bitForModules + bitsShiftModule;

  if ((pcrist = (u16 *) malloc(5 * sizeof(u16))) == NULL)
    printf
	("\n*** ERROR : in demakeid.c : imposible to do : malloc(pcrist) \n");


  if (pEncoH->scanEncodingID.maximumLayers)	/* if at least one bit per layer */
    pcrist[0] = (id & (pEncoH->scanEncodingID.maximumLayers - 1));	/*  L  */
  else
    pcrist[0] = 0;

  /*   C  */
  pcrist[1] =
      ((id >> bitsShiftCrystal) &
       (pEncoH->scanEncodingID.maximumCrystals - 1));

  /*  sM  */
  pcrist[2] =
      ((id >> bitsShiftSubmodule) &
       (pEncoH->scanEncodingID.maximumSubmodules - 1));

  /*   M  */
  pcrist[3] =
      ((id >> bitsShiftModule) &
       (pEncoH->scanEncodingID.maximumModules - 1));

  /* S  */
  pcrist[4] =
      ((id >> bitsShiftSector) &
       (pEncoH->scanEncodingID.maximumRsectors - 1));

  return (pcrist);
}


/*

  H O W   T O   U S E   I T 
  
 U N C O M M E N T   T H I S   M A I N

*/


/* main() */
/* { */

/*   u16 *pcrist; */
/*   ENCODING_HEADER pEncoH; */
/*   pEncoH.scanEncodingID.nBitRing = 3; */
/*   pEncoH.scanEncodingID.nBitSector = 4; */
/*   pEncoH.scanEncodingID.nBitModule = 3; */
/*   pEncoH.scanEncodingID.nBitCrystal = 5; */
/*   pEncoH.scanEncodingID.nBitLayer = 1; */
/*   pEncoH.scanEncodingID.nMaxRing = 8; */
/*   pEncoH.scanEncodingID.nMaxSector = 16; */
/*   pEncoH.scanEncodingID.nMaxModule = 8; */
/*   pEncoH.scanEncodingID.nMaxCrystal = 32; */
/*   pEncoH.scanEncodingID.nMaxLayer = 2; */

/* pcrist=demakeid(32769, &pEncoH); */
/* printf("\nR = %d\n",pcrist[4]); */
/* printf("\nS = %d\n",pcrist[3]); */
/* printf("\nM = %d\n",pcrist[2]); */
/* printf("\nC = %d\n",pcrist[1]); */
/* printf("\nL = %d\n",pcrist[0]); */


/* } */
