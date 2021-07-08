/*-------------------------------------------------------

           List Mode Format 
                        
     --  generateEH.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of generateEH.c:

     Standard filling of event header

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

static EVENT_HEADER *pEH = NULL;

EVENT_HEADER(*generateEH(ENCODING_HEADER * pEncoH))
{
  static int allocEHdone = FALSE;
  i16 test, testNN = 0;


  if (allocEHdone == 0) {
    if ((pEH = (EVENT_HEADER *) malloc(sizeof(EVENT_HEADER))) == NULL)
      printf
	  ("\n***ERROR : in generateEH.c : imposible to do : malloc()\n");
    allocEHdone = 1;
  }

  printf("\n\n\n");

  /*  c  */

  printf("\n* What kind of event do you want to store ?");
  printf("\n  0 = Singles\t1 = Coincidences\n");
  test = hardgeti16(0, 1);
  switch (test) {
  case (0):
    pEH->coincidenceBool = 0;	/* Singles stored  */
    break;
  case (1):
    pEH->coincidenceBool = 1;	/* Coincidence  stored  */
    break;
  }

  /*  d  */

  printf("* Do you want to store the detector id and DOI ?");
  test = hardgetyesno();
  switch (test) {
  case (1):
    pEH->detectorIDBool = 1;	/*The IDs are stored */
    break;
  case (0):
    pEH->detectorIDBool = 0;	/*The IDs are not stored */
    break;
  }

/*  E  */

  printf("* Do you want to store (any) ENERGY ?");

  test = hardgetyesno();
  switch (test) {
  case (1):
    pEH->energyBool = 1;	/*Energy stored */

/*  n  */

    printf("* Do you want to store ENERGY and DOI");
    printf("  in neighbouring crystals ?");

    test = hardgetyesno();
    switch (test) {
    case (1):
      pEH->neighbourBool = 1;	/* Energy and DOI in neigh crystal stored */
      break;
    case (0):
      pEH->neighbourBool = 0;	/* Energy and DOI in neigh crystal is not stored */
      pEH->neighbourhoodOrder = 0;
      pEH->numberOfNeighbours = 0;
      testNN = 1;
      break;
    }
    if (testNN != 1) {		/* IF neighbourBool = 1 */
      /* NN */

      printf("* Order of neighbourhood :");
      test = hardgeti16(0, 3);
      switch (test) {
      case (0):
	pEH->neighbourBool = 0;
	pEH->neighbourhoodOrder = 0;
	pEH->numberOfNeighbours = 0;
	break;
      case (1):
	pEH->neighbourhoodOrder = 1;
	pEH->numberOfNeighbours = 4;
	break;
      case (2):
	pEH->neighbourhoodOrder = 2;
	pEH->numberOfNeighbours = 8;
	break;
      case (3):
	pEH->neighbourhoodOrder = 3;
	pEH->numberOfNeighbours = 20;
	break;
      }
    }

    break;
  case (0):
    pEH->energyBool = 0;	/* energies are not stored */
    pEH->neighbourBool = 0;
    pEH->neighbourhoodOrder = 0;
    pEH->numberOfNeighbours = 0;
    break;
  }



  /*  g  */

  printf("* Do you want to store the gantry angular position ?");

  test = hardgetyesno();
  switch (test) {
  case (1):
    pEH->gantryAngularPosBool = 1;	/*Gantry angular pos.stored */
    break;
  case (0):
    pEH->gantryAngularPosBool = 0;	/*Gantry angular pos. not stored */
    break;
  }

  /*  b  */

  printf("* Do you want to store gantry axial position :");

  test = hardgetyesno();
  switch (test) {
  case (1):
    pEH->gantryAxialPosBool = 1;	/*Gantry axial pos.  stored */
    break;
  case (0):
    pEH->gantryAxialPosBool = 0;	/*Gantry axial pos. not stored */
    break;
  }

  /*  s  */

  printf("* Do you want to store source position. :");

  test = hardgetyesno();
  switch (test) {
  case (1):
    pEH->sourcePosBool = 1;	/*Source pos. stored */
    break;
  case (0):
    pEH->sourcePosBool = 0;	/*Source pos. not stored */
    break;
  }


  /*  G automatically set */

  test = pEncoH->scanContent.gateDigiRecordBool;
  switch (test) {
  case (1):
    pEH->gateDigiBool = 1;	/*gate info. stored */
    break;
  case (0):
    pEH->gateDigiBool = 0;	/*gate info. not stored */
    break;
  }




  /*  Z  */

  printf("* Do you want to store neighbour info (juelich fpga). :");

  test = hardgetyesno();
  switch (test) {
  case (1):
    pEH->fpgaNeighBool = 1;	/* stored */
    break;
  case (0):
    pEH->fpgaNeighBool = 0;	/* not stored */
    break;
  }





  return (pEH);

}

void generateEHDestructor()
{
  if (pEH)
    free(pEH);
}
