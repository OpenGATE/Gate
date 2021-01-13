/*-------------------------------------------------------

List Mode Format 
                        
--  peakPositionDispersion.cc --

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2006 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of peakPositionDispersion


-------------------------------------------------------*/

#include <iostream>
#include <stdio.h>
#include "lmf.h"
#include "Calfactor.hh"
#include "Gaussian.hh"

using namespace::std;

static Calfactor *calFact = NULL;
static bool inputParams = false;
static u16 layerNb = 0;
static mean_std *calBase = NULL;

void setPeakPositionDispersionParams(u16 inlineLayerNb, mean_std *inlineCalBase)
{
  layerNb = inlineLayerNb;
  calBase = inlineCalBase;
  inputParams = true;

  return;
}

void peakPositionDispersion(const ENCODING_HEADER *pEncoH,
			    EVENT_RECORD **ppER)
{
  u16 sctnb, modnb, crynb, laynb;
  char response[10];
  u16 *pcrist = NULL;
  u16 sector, module, crystal, layer;
  u16 i, j, k, l;
  double energy;
  double calfactor;

  if(!calFact) {
    sctnb = pEncoH->scannerTopology.totalNumberOfRsectors;
    modnb = pEncoH->scannerTopology.totalNumberOfModules;
    crynb = pEncoH->scannerTopology.totalNumberOfCrystals;
    laynb = pEncoH->scannerTopology.totalNumberOfLayers;

    calFact = new Calfactor(sctnb, modnb, crynb, laynb);
    if(calBase) {
      if(layerNb != laynb) {
	printf("The number of layers (= %hu) is different than the one you set ( = %hu)\n",
	       laynb,layerNb);
	printf("Re-run with the corret one !\n");
	cout << "Bad number of input params -> exit !" << endl;
	exit(EXIT_FAILURE);
      }
    }
    else {
      for(l=0; l<laynb; l++) {
	printf("Select the 511keV peak position mean for layer %hu: ",l);
	if (fgets(response,sizeof(response),stdin))
	  sscanf(response,"%lg", &(calBase[l].mean));
	printf("Select the 511keV peak position std dev for layer %hu: ",l);
	if (fgets(response,sizeof(response),stdin))
	  sscanf(response,"%lg", &(calBase[l].std));
      }
    }
    for (l = 0; l < laynb; l++) {
      cout << "Mean peak position for layer " << (int)l << ": " << calBase[l].mean << " keV" << endl;
      cout << "Std dev peak position for layer " << (int)l << ": " << calBase[l].std << " keV" << endl;
	
      for(i = 0; i < sctnb; i++)
	for(j = 0; j < modnb; j++)
	  for (k = 0; k < crynb; k++) {
	    calfactor = Gaussian::Shoot(calBase[l].mean, calBase[l].std);
// 	    cout << (int)i << "\t" << (int)j << "\t" << (int)k << "\t" << (int)l << ":\t" << calfactor << endl;
	    if(calfactor < 0)
	      calfactor = 0;
	    (*calFact)(i, j, k, l) = calfactor / ENERGY_REF;
	  }
    }
  }

  pcrist = demakeid((*ppER)->crystalIDs[0], pEncoH);
  sector = pcrist[4];
  module = pcrist[3];
  crystal = pcrist[1];
  layer = pcrist[0];
  free(pcrist);

  energy = ((*calFact) (sector, module, crystal, layer)) * (*ppER)->energy[0] + 0.5;
  if(energy < 0)
    *ppER = NULL;
  else
    (*ppER)->energy[0] = (u8) (energy);

  return;
}

void peakPositionDispersionDestructor()
{
  calFact->WriteAllCalfactorInFiles();
  delete calFact;
  calFact = NULL;

  return;
}
