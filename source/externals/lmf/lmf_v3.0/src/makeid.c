/*-------------------------------------------------------

List Mode Format 
                        
--  makeid.c  --                      

Luc.Simon@iphe.unil.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of makeid.c:

This function builds the id/DOI (u16)      
Its parameters are ring,Sector,Module,Crystal,Layer and
a ccs_encodingHeader structure pointer                 
-------------------------------------------------------*/
#include <stdio.h>

#include "lmf.h"


#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif


u64 makeid(u16 RandS, u16 M, u16 sM, u16 C, u16 L,
	   const ENCODING_HEADER * pEncoH, u16 * errFlag)
{
  u64 buildedID = 0;
  u16 Flag = 0;

  static u8 topologyReadOK = FALSE;
  static u16 maxRS = 0;		/* rsector */
  static u16 maxM = 0;		/* module */
  static u16 maxm = 0;		/* submodule */
  static u16 maxC = 0;		/* crystal */
  static u16 maxL = 0;		/* layer */
  /* Shift (in bits) */
  static u16 shiftSectorRing;
  static u16 shiftModule;
  static u16 shiftSubmodule;
  static u16 shiftCrystal;
  /* static u16 shiftLayer = 0; */
  if (topologyReadOK == FALSE) {
    shiftCrystal = pEncoH->scanEncodingID.bitForLayers;
    shiftSubmodule = pEncoH->scanEncodingID.bitForCrystals + shiftCrystal;
    shiftModule = pEncoH->scanEncodingID.bitForSubmodules + shiftSubmodule;
    shiftSectorRing = pEncoH->scanEncodingID.bitForModules + shiftModule;
    maxRS = pEncoH->scannerTopology.totalNumberOfRsectors;
    maxM = pEncoH->scannerTopology.totalNumberOfModules;
    maxm = pEncoH->scannerTopology.totalNumberOfSubmodules;
    maxC = pEncoH->scannerTopology.totalNumberOfCrystals;
    maxL = pEncoH->scannerTopology.totalNumberOfLayers;
    topologyReadOK = TRUE;
  }

  /*   Ring & Sector */
  if (RandS < maxRS)
    buildedID |= ((u64) RandS << shiftSectorRing);
  else {
    printf("\nR is too big to make the ID : R = %d Max < %d !!!\n", RandS,
	   maxRS);
    Flag = 1;
  }

  /*   Module  */
  if (M < maxM)
    buildedID |= ((u64) M << shiftModule);
  else {
    printf("\nM is too big to make the ID : M = %d Max < %d \n", M, maxM);
    Flag = 2;
  }

  /*  SubModule */
  if (sM < maxm)
    buildedID |= ((u64) sM << shiftSubmodule);
  else {
    printf("\nsM is too big to make the ID : sM = %d Max < %d \n", sM,
	   maxm);
    Flag = 3;
  }

  /*  Crystal */
  if (C < maxC)
    buildedID |= ((u64) C << shiftCrystal);
  else {
    printf("\nC is too big to make the ID : C = %d Max < %d \n", C, maxC);
    Flag = 4;
  }

  /*   Layer */
  if(maxL) {
    if (L < maxL)
      buildedID |= (u64) L;
    else {
      printf("\nL is too big to make the ID : L = %d Max < %d \n", L, maxL);
      Flag = 5;
    }
  }

  *errFlag = Flag;

  return (buildedID);
}
