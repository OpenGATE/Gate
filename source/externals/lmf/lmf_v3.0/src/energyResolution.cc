/*-------------------------------------------------------

List Mode Format 
                        
--  energyResolution.cc  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2006 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of energyResolution


-------------------------------------------------------*/


#include <iostream>
#include <fstream>

#include <stdio.h>
#include <math.h>

#include "lmf.h"
#include "Gaussian.hh"
#include "Calfactor.hh"

using namespace std;

static Calfactor *sigmaTable = NULL;
static u16 layerNb = 0;
static mean_std *energyResolBase = NULL;

void setInitialParamsForEnergyResolution(u16 inlineLayerNb, mean_std *inlineEnergyResolBase)
{
  u16 l;

  layerNb = inlineLayerNb;
  energyResolBase = new mean_std[layerNb];
  for(l=0; l<layerNb; l++)
    energyResolBase[l] = inlineEnergyResolBase[l];

  return;
}

void energyResolution(const ENCODING_HEADER *pEncoH,
		      EVENT_RECORD **ppER)
{
  u16 sctnb, modnb, crynb, laynb;
  char response[10];
  u16 *pcrist = NULL;
  u16 sector, module, crystal, layer;
  u16 i, j, k, l;
  double energyStep, resol2sigma, conv_fact;
  double oldEnergy, newEnergy;
  double sigma;
  ofstream coef_log;

  if(!sigmaTable) {
//     sctnb = 6;
    sctnb = pEncoH->scannerTopology.totalNumberOfRsectors;
    modnb = pEncoH->scannerTopology.totalNumberOfModules;
    crynb = pEncoH->scannerTopology.totalNumberOfCrystals;
    laynb = pEncoH->scannerTopology.totalNumberOfLayers;

    sigmaTable = new Calfactor(sctnb, modnb, crynb, laynb);
    if(energyResolBase) {
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
	printf("Select the energy resolution @ 511keV mean for layer %hu: ",l);
	if (fgets(response,sizeof(response),stdin))
	  sscanf(response,"%lg", &(energyResolBase[l].mean));
	printf("Select the energy resolution @ 511keV std dev for layer %hu: ",l);
	if (fgets(response,sizeof(response),stdin))
	  sscanf(response,"%lg", &(energyResolBase[l].std));
      }
    }

    coef_log.open("coef_sigma.log",ios::out);
    for (l = 0; l < laynb; l++) {
      printf("Energy resolution %g +- %g %% @ 511keV for layer %hu\n",
	     energyResolBase[l].mean * 100,energyResolBase[l].std * 100,l);

      energyStep = (double) getEnergyStepFromCCH();
      resol2sigma = 2.*sqrt(2.*log(2.));
      conv_fact = sqrt(ENERGY_REF / energyStep) / resol2sigma;
      energyResolBase[l].mean *= conv_fact;
      energyResolBase[l].std *= conv_fact;
    

      for(i = 0; i < sctnb; i++)
	for(j = 0; j < modnb; j++)
	  for (k = 0; k < crynb; k++) {
	    sigma = Gaussian::Shoot(energyResolBase[l].mean, energyResolBase[l].std);
	    coef_log << (int)i << "\t" << (int)j << "\t" << (int)k << "\t" << (int)l << ":\t" << sigma << endl;
	    (*sigmaTable)(i, j, k, l) = sigma;
	  }
    }
    coef_log.close();
  }

  pcrist = demakeid((*ppER)->crystalIDs[0], pEncoH);
  sector = pcrist[4];
  module = pcrist[3];
  crystal = pcrist[1];
  layer = pcrist[0];
  free(pcrist);

  oldEnergy = (double)((*ppER)->energy[0]);
  sigma = ((*sigmaTable)(sector, module, crystal, layer)) * sqrt(oldEnergy);
  newEnergy = Gaussian::Shoot(oldEnergy, sigma) + 0.5;
  if(newEnergy < 0)
    *ppER = NULL;
  else
    (*ppER)->energy[0] = (u8)(newEnergy);

  return;
}

void energyResolutionDestructor(void)
{
  delete sigmaTable;
  delete[] energyResolBase;
  layerNb = 0;

  return;
}
