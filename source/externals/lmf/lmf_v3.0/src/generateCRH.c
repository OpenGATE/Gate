/*-------------------------------------------------------

           List Mode Format 
                        
     --  generateCRH.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of generateCRH.c:
     Standard filling of count rate header

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

static COUNT_RATE_HEADER *pCRH = NULL;

COUNT_RATE_HEADER(*generateCRH(void))
{

  u16 test;			/* get the user's answers */
  u16 testTSR = FALSE;		/* TRUE if totalsingleRateBool = 1 */
  static int allocCRHdone = FALSE;

  if (allocCRHdone == FALSE) {
    allocCRHdone = TRUE;
    if ((pCRH =
	 (COUNT_RATE_HEADER *) malloc(sizeof(COUNT_RATE_HEADER))) == NULL)
      printf
	  ("\n*** ERROR : in generateCRH.c : impossible to do : malloc()\n");
  }

  printf("\n\n\n\n");

/*  s  */

  printf("* Total singles rate ?");

  test = hardgetyesno();
  switch (test) {
  case (1):
    pCRH->singleRateBool = 1;	/*singles rate stored */
    testTSR = TRUE;
    break;
  case (0):
    pCRH->singleRateBool = 0;	/*singles rate not stored */
    pCRH->singleRatePart = 0;
    break;
  }


  if (testTSR == TRUE) {
    /* SS */

    printf("* Singles rate in other parts ?\n");
    printf("0 = total only\t");
    printf("1 = rsector\t");
    printf("2 = module\t");
    printf("3 = submodule\n");
    test = hardgeti16(0, 3);
    switch (test) {
    case (0):
      pCRH->singleRatePart = 0;	/* Total only */
      break;
    case (1):
      pCRH->singleRatePart = 1;	/* rsector */
      break;
    case (2):
      pCRH->singleRatePart = 2;	/*module */
      break;
    case (3):
      pCRH->singleRatePart = 3;	/* submodule */
      break;
    }
  }


  /*  c  */

  printf("* Total coincidence count rate  ?");

  test = hardgetyesno();
  switch (test) {
  case (1):
    pCRH->totalCoincidenceBool = 1;	/* Coincidence Rate Stored */
    break;
  case (0):
    pCRH->totalCoincidenceBool = 0;	/* Coincidence Rate not Stored */
    break;
  }

/*  F  */

  printf("* Total random count rate  ");

  test = hardgetyesno();
  switch (test) {
  case (1):
    pCRH->totalRandomBool = 1;	/* Random Rate Stored */
    break;
  case (0):
    pCRH->totalRandomBool = 0;	/* Random Rate not Stored */
    break;
  }


/*  r  */
  printf("* Gantry angular speed ? ");

  test = hardgetyesno();
  switch (test) {
  case (1):
    pCRH->angularSpeedBool = 1;	/* angular speed Stored */
    break;
  case (0):
    pCRH->angularSpeedBool = 0;	/* angular speed not Stored */
    break;
  }

  /*  b  */
  printf("* Gantry axial speed ? ");

  test = hardgetyesno();
  switch (test) {
  case (1):
    pCRH->axialSpeedBool = 1;	/* axial speed Stored */
    break;
  case (0):
    pCRH->axialSpeedBool = 0;	/* axial speed not Stored */
    break;
  }



  return (pCRH);
}


void generateCRHDestructor()
{
  if (pCRH)
    free(pCRH);
}
