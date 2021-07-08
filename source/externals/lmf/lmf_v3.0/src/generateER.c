/*-------------------------------------------------------

           List Mode Format 
                        
     --  generateER.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of generateER.c:
Example of filling of event record. This filling 
is very artificial, but it is just to explain
how you can use the LMF_ccs_eventRecord structure. 

-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "lmf.h"

static EVENT_RECORD *pER;

EVENT_RECORD *generateER(ENCODING_HEADER * pEncoH, EVENT_HEADER * pEH)
{
  static int verboseLevel;

  static i16 allocERdone = 0;
  i16 j = 0;
  static int oneOrZero = 0;
  static u64 counterEvolution = 300;
  static u8 *bufCharTime;

  static u16 rSector, module, sModule, crystal, layer;
  u16 errFlag;


  if (allocERdone == 0) {	/* the allocation is just done once */
    allocERdone = 1;

    printf("verbose level for artificial LMF builder :\n");
    verboseLevel = hardgeti16(0, 1);

    if ((pER = (EVENT_RECORD *) malloc(sizeof(EVENT_RECORD))) == NULL)
      printf
	  ("\n***ERROR : in generateER.c : impossible to do : malloc()\n");

    if ((pER->crystalIDs =
	 malloc((pEH->coincidenceBool + 1) * (pEH->numberOfNeighbours +
					      1) * sizeof(u64))) == NULL)
      printf
	  ("\n***ERROR : in generateER.c : impossible to do : malloc()\n");

    if ((pER->energy =
	 malloc((pEH->coincidenceBool + 1) * (pEH->numberOfNeighbours +
					      1) * sizeof(u8))) == NULL)
      printf
	  ("\n***ERROR : in generateER.c : impossible to do : malloc()\n");

    bufCharTime = malloc(8 * sizeof(u8));

  }

  /* generate an evoluting random time / just for developement */
  /* 
     t(n) = t(n-1) + 20000*r   where r is 
     randomly taken between 0 and 1
   */
  counterEvolution = counterEvolution + monteCarloInt(0, 20000);


  bufCharTime = u64ToU8(counterEvolution);

  for (j = 0; j < 8; j++)
    pER->timeStamp[j] = bufCharTime[j];


  pER->timeOfFlight = 5;	/*  time of flight */


  if (oneOrZero == 0) {
    oneOrZero = 1;
    for (j = 0;
	 j < (pEH->coincidenceBool + 1) * (pEH->numberOfNeighbours + 1);
	 j++) {
      /* random detector ID */
      rSector = monteCarloInt(0, NUMBER_OF_SECTORS * NUMBER_OF_RINGS - 1);
      module =
	  monteCarloInt(0,
			AXIAL_NUMBER_OF_MODULES *
			TANGENTIAL_NUMBER_OF_MODULES - 1);
      sModule =
	  monteCarloInt(0,
			AXIAL_NUMBER_OF_SUBMODULES *
			TANGENTIAL_NUMBER_OF_SUBMODULES - 1);
      crystal =
	  monteCarloInt(0,
			AXIAL_NUMBER_OF_CRYSTALS *
			TANGENTIAL_NUMBER_OF_CRYSTALS - 1);
      layer =
	  monteCarloInt(0,
			AXIAL_NUMBER_OF_LAYERS * RADIAL_NUMBER_OF_LAYERS -
			1);
      //      pER->crystalIDs[j] = makeid(rSector,module,sModule,crystal,layer,pEncoH);
      pER->crystalIDs[j] = makeid(1, 0, 0, 1, 0, pEncoH, &errFlag);

      pER->energy[j] = 100;
    }
  } else {
    oneOrZero = 0;
    for (j = 0;
	 j < (pEH->coincidenceBool + 1) * (pEH->numberOfNeighbours + 1);
	 j++) {
      /*evry two times we shoot in the opposite rsector */
      rSector = (rSector + (NUMBER_OF_SECTORS / 2)) % NUMBER_OF_SECTORS;

      pER->crystalIDs[j] = makeid(1, 0, 1, 3, 0, pEncoH, &errFlag);


      // pER->crystalIDs[j]=makeid(rSector,module,sModule,crystal,layer,pEncoH);
      pER->energy[j] = 200;
    }
  }
  pER->gantryAngularPos = 0;	/* gantry's angular position */
  pER->gantryAxialPos = 0;	/* gantry's axial position */

  pER->sourceAngularPos = 0;	/* external source's angular position */
  pER->sourceAxialPos = 0;	/* external source's axial position */

  pER->fpgaNeighInfo[0] = 5;
  pER->fpgaNeighInfo[1] = 2;
  if (verboseLevel) {
    printf("\ntime = %llu\n", counterEvolution);
    printf("energy = %d\n", pER->energy[0]);
    printf("id = %llu (rsector = %d)\n", pER->crystalIDs[0], rSector);
  }


  return (pER);

}


void generateERDestructor()
{
  if (pER) {
    if (pER->crystalIDs)
      free(pER->crystalIDs);
    if (pER->energy)
      free(pER->energy);
    free(pER);
  }

}
