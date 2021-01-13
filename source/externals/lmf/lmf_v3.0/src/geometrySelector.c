/*-------------------------------------------------------

List Mode Format 
                        
--  geometrySelector.c  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2004 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of geometrySelector

Allows to re-mix events comming from different sectors.
It is usefull when the events are stored by fixed size block
of specific sectors.
You have to specify the number of sectors you use and 
the size of one block

-------------------------------------------------------*/


#include <stdio.h>
#include "lmf.h"

static u16 nbOfSectors = 0, nbOfModules = 0, nbOfCrystals = 0;
static u16 *sector = NULL, *module = NULL, *crystal = NULL;

void setSectors(u16 inlineNbOfSectors, u16 ** inlineSector)
{
  nbOfSectors = inlineNbOfSectors;
  sector = *inlineSector;
}

void setModules(u16 inlineNbOfModules, u16 ** inlineModule)
{
  nbOfModules = inlineNbOfModules;
  module = *inlineModule;
}

void setCrystals(u16 inlineNbOfCrystals, u16 ** inlineCrystal)
{
  nbOfCrystals = inlineNbOfCrystals;
  crystal = *inlineCrystal;
}

void geometrySelector(const ENCODING_HEADER * pEncoH, EVENT_RECORD ** ppER)
{
  u16 sct, mod, cry, nb;
  u16 *pcrist;
  u8 flagR, flagM, flagC;

  pcrist = demakeid((*ppER)->crystalIDs[0], pEncoH);
  sct = pcrist[4];
  mod = pcrist[3];
  cry = pcrist[1];
  free(pcrist);

  flagR = 0;
  if (nbOfSectors) {
    for (nb = 0; nb < nbOfSectors; nb++)
      if (sct == sector[nb]) {
	flagR = 1;
	break;
      }
  } 
  else
    flagR = 1;

  flagM = 0;
  if (nbOfModules) {
    for (nb = 0; nb < nbOfModules; nb++)
      if (mod == module[nb]) {
	flagM = 1;
	break;
      }
  } 
  else
    flagM = 1;

  flagC = 0;
  if (nbOfCrystals) {
    for (nb = 0; nb < nbOfCrystals; nb++)
      if (cry == crystal[nb]) {
	flagC = 1;
	break;
      }
  }
  else
    flagC = 1;

  if (!((flagR == 1) & (flagM == 1) & (flagC == 1)))
    *ppER = NULL;

  return;
}
