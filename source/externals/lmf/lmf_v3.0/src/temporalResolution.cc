/*-------------------------------------------------------

List Mode Format 
                        
--  temporalResolution.cc  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2006 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of temporalResolution


-------------------------------------------------------*/


#include <stdio.h>
#include <math.h>
#include "lmf.h"
#include "Gaussian.hh"

using namespace std;

static u16 layerNb = 0;
static double *sigma = NULL;


void setInitialParamsForTemporalResolution(void)
{
  char response[10];
  u16 i;
  double resol;
  double resol2sigma;
  double timeStep;

  timeStep = (double) (i64) getTimeStepFromCCH() / 1E6;
  resol2sigma = 2.*sqrt(2.*log(2.));

  printf("Set the number of layors in use (Default %hu): ",
	 layerNb);
  if (fgets(response,sizeof(response),stdin))
    sscanf(response,"%hu",&layerNb);

  sigma = new double[layerNb];

  for (i = 0; i < layerNb; i++) {
    printf("Select the temporal resolution (in ns) for layer %hu: ",i);
    if (fgets(response,sizeof(response),stdin))
      sscanf(response,"%lg", &resol);

    printf("Temporal resolution %g ns for layer %hu\n",resol,i);
    resol = resol / timeStep;
    printf("\t == %g time marks\n",resol);
    sigma[i] = resol / resol2sigma;
    printf("\t\t -> sigma = %g time marks\n",sigma[i]);
  }

  return;
}

void setInitialParamsForTemporalResolutionDirectly(u16 layers, double *resol)
{
  u16 i;
  double resol2sigma;
  double timeStep;

  timeStep = (double) (i64) getTimeStepFromCCH() / 1E6;
  resol2sigma = 2.*sqrt(2.*log(2.));

  layerNb = layers;
  sigma = new double[layerNb];
  for (i = 0; i < layerNb; i++) {
    printf("Temporal resolution %g ns for layer %hu\n",resol[i],i);
    resol[i] = resol[i] / timeStep;
    printf("\t == % g time marks\n",resol[i]);
    sigma[i] = resol[i] / resol2sigma;
    printf("\t\t -> sigma = % g time marks\n",sigma[i]);
  }

  return;
}

void temporalResolution(const ENCODING_HEADER *pEncoH,
		      EVENT_RECORD *pER)
{
  static u8 doneonce = 0;
  u16 layer;
  double oldTime, newTime;
  u64vsu8 timeU64;

  if(!doneonce) {
    if(layerNb != pEncoH->scannerTopology.totalNumberOfLayers) {
      printf("The number of layers (= %hu) is different than the one you set ( = %hu)\n",
	     pEncoH->scannerTopology.totalNumberOfLayers,layerNb);
      printf("Re-run with the corret one !\n");
      exit(0);
    }
    doneonce++;
  }

  layer = getLayerID(pEncoH, pER);
  oldTime = (double)(i64)(u8ToU64(pER->timeStamp));
  newTime = Gaussian::Shoot(oldTime, sigma[layer]);
  timeU64.w64 = (u64)(i64)(newTime + 0.5);

  for (int j = 0; j < 8; j++)
    pER->timeStamp[j] = timeU64.w8[j];

//   printf("oldTime = %0.f -> newTime = %0.f\n",oldTime,newTime);
//   getchar();

  return;
}

void temporalResolutionDestructor(void)
{
  delete[] sigma;
  layerNb = 0;

  return;
}
