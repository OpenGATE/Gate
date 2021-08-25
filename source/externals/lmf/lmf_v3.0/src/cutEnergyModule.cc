/*-------------------------------------------------------

           List Mode Format 
                        
     --  cutEnergyModule.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of cutEnergyModule.c

	 Apply an energy window cut on event records

-------------------------------------------------------*/

#include <stdio.h>

#include "lmf.h"
#include "../includes/Calfactor.hh"

static int upLimit = 650;	// keV
static int downLimit = 350;	// keV
static u64 nRejected = 0;
static u8 useCalfactorTable = 0;

void useCalfTable()
{
  useCalfactorTable = 1;

  return;
}

void askForCalfactorTableUsing(u8 * useCalfactorTable)
{
  char response[10];

  printf("Do you want to use calfactor tables (y/n): ");
  if (fgets(response, sizeof(response), stdin))
    if ((response[0] == 'y') || (response[0] == 'Y'))
      *useCalfactorTable = 1;

  return;
}

EVENT_RECORD *cutEnergy(const ENCODING_HEADER * pEncoH,
			const EVENT_HEADER * pEH,
			const GATE_DIGI_HEADER * pGDH, EVENT_RECORD * pER)
{
  static u8 doneonce = 0;
  static int energyStep = 0;
  static Calfactor *calFact = NULL;

  int keepIT = TRUE;		/*  = FALSE if we dont keep this event */
  int energy = 0;

  u16 *pcrist = NULL;
  u16 sector, module, crystal, layer;

  if (!doneonce) {
//     if(!useCalfactorTable)
//       askForCalfactorTableUsing(&useCalfactorTable);
      
    if (useCalfactorTable) {
      calFact = new Calfactor(pEncoH->scannerTopology.totalNumberOfRsectors,
			      pEncoH->scannerTopology.totalNumberOfModules,
			      pEncoH->scannerTopology.totalNumberOfCrystals,
			      pEncoH->scannerTopology.totalNumberOfLayers);
      calFact->ReadAllCalfactorTables();
    }

    energyStep = getEnergyStepFromCCH();
    doneonce++;
  }


  if (pEH->energyBool == FALSE) {
    printf
	("*** error : cutEnergyModule.c : you can cut energy only if you have energy stored in your file\n");
    exit(0);
  }

  if (useCalfactorTable) {
    pcrist = demakeid(pER->crystalIDs[0], pEncoH);
    sector = pcrist[4];
    module = pcrist[3];
    crystal = pcrist[1];
    layer = pcrist[0];
    free(pcrist);

    pER->energy[0] =
      (u8) (((*calFact) (sector, module, crystal, layer)) * pER->energy[0]);
  }
  energy = pER->energy[0] * energyStep;

/*   printf("upLimit = %d, downLimit = %d. nrj bf = %d step = %d af = %d\n",upLimit,downLimit,pER->energy[0],energyStep,energy); */
/*   getchar(); */

  if (upLimit != 0) {
    if (energy > upLimit)
      keepIT = FALSE;
  }

  if (downLimit != 0) {
    if (energy < downLimit)
      keepIT = FALSE;
  }
  //      printf("%d : (%d-%d)  --> %d\n",energy,downLimit,upLimit, keepIT );


  if (keepIT)
    return (pER);
  else {
    nRejected++;
    return (NULL);
  }

}


void setUpEnergyLimit(int upKeVLimit)
{
  upLimit = upKeVLimit;
  printf("Cut Energy Module : upper limit set to %d\n", upKeVLimit);
}
void setDownEnergyLimit(int downKeVLimit)
{
  downLimit = downKeVLimit;
  printf("Cut Energy Module : lower limit set to %d\n", downKeVLimit);
}
u64 printEnergyModuleRejectedEventNumber(void)
{
  printf("Number of rejected records by cutEnergyModule.c : %llu\n",
	 nRejected);
  return (nRejected);
}
