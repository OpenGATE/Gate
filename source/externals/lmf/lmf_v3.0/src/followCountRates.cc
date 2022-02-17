/*-------------------------------------------------------

List Mode Format 
                        
--  followCountRates.cc  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2006 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of followCountRates


-------------------------------------------------------*/


#include <stdio.h>
#include "lmf.h"
#include "Gaussian.hh"

using namespace std;

static u16 sctNb = 6;
static u16 *sctName = NULL;
static Gaussian **g = NULL;
static double *min = NULL;
static u64 *lastTimes = NULL;

static u16 mode = 0;

static int nb = 0;
static int rejected = 0;

static double timeStep = 0;

void setInitialParamsForFollowCountRates(void)
{
  char response[10];
  u16 i;
  double mean, sigma;

  timeStep = (double) (i64) getTimeStepFromCCH() / 1E9;

  printf("Set the mode of followCountRates\n");
  printf("\tmode 0 based on gaussian min\n");
  printf("\tmode 1 based on simple min (Default %hu): ",
	 mode);
  if (fgets(response,sizeof(response),stdin))
    sscanf(response,"%hu",&mode);

  printf("Set the number of sectors in use (Default %hu): ",
	 sctNb);
  if (fgets(response,sizeof(response),stdin))
    sscanf(response,"%hu",&sctNb);

  sctName = new u16[sctNb];
  if(mode)
    min = new double[sctNb];
  else
    g = new Gaussian*[sctNb];
  lastTimes = new u64[sctNb];

  for (i = 0; i < sctNb; i++) {
    printf("Select the sector nb to characterize: ");
    if (fgets(response,sizeof(response),stdin))
      sscanf(response,"%hu", &(sctName[i]));

    if(mode) {
      printf("Select the min frequency for sector %d in micros: ",sctName[i]);
      if (fgets(response,sizeof(response),stdin))
	sscanf(response,"%lg", &(min[i]));
      printf("Min sct %d = %g micros -> ",sctName[i],min[i]);
      min[i] = min[i] / timeStep;
      printf("%g ps\n",min[i]);
    }
    else {
      printf("Select the mean frequency for sector %d in micros: ",sctName[i]);
      if (fgets(response,sizeof(response),stdin))
	sscanf(response,"%lg", &mean);
      printf("Mean sct %d = %g micros -> ",sctName[i],mean);
      mean = mean / timeStep;
      printf("%g ps\n",mean);

      printf("Select the std deviation of the frequency for sector %d in micros: ",sctName[i]);
      if (fgets(response,sizeof(response),stdin))
	sscanf(response,"%lg", &sigma);

      printf("Std dev sct %d = %g micros -> ",sctName[i],sigma);
      sigma = sigma / timeStep;
      printf("%g ps\n",sigma);

      g[i] = new Gaussian(mean,sigma);
    }
    lastTimes[i] = (u64) -1;
  }

  return;
}

void followCountRates(const ENCODING_HEADER * pEncoH,
		      EVENT_RECORD **ppER)
{
  u16 sct, index;
  u8 check = 0;

  u64 actualTime;
  double difTime;

  double random, prob;

  nb++;
  sct = getRsectorID(pEncoH, *ppER);

  for (index = 0; index < sctNb; index++)
    if (sct == sctName[index]) {
      check++;
      break;
    }

  if (!check) {
    printf
      ("nb of sector in file is greater than the one introduced\nPlease re-run\n");
    exit(0);
  }

  actualTime = u8ToU64((*ppER)->timeStamp);
  if(lastTimes[index] == (u64) -1)
    lastTimes[index] = actualTime;
  else {
    if(mode) {
      difTime = (double)(i64)(actualTime - lastTimes[index]);
    
      if(difTime < min[index]) {
	rejected++;
	*ppER = NULL;
      }
      else
	lastTimes[index] = actualTime;
    }
    else {
      difTime = (double)(i64)(actualTime - lastTimes[index]);


      if(difTime < (g[index])->GetMean()) {
	prob = (*(g[index]))(difTime);
	random = randd();

	//       printf("sct = %d difTime = %g -> prob = %g\n",sctName[index],difTime,prob);
	//       printf("random = %g\n",random);

	if(random < prob)
	  lastTimes[index] = actualTime;
	else {
	  rejected++;
	  *ppER = NULL;
	}
      }
      else
	lastTimes[index] = actualTime;
      //    exit(0);
    }
  }

  return;
}

void followCountRatesDestructor(void)
{
  u16 i;

  delete[] sctName;
  delete[] lastTimes;

  if(mode)
    delete[] min;
  else {
    for (i = 0; i < sctNb; i++)
      delete g[i];
    delete g;
  }

  printf("%d events were rejected on a total of %d singles\n",rejected, nb);
  printf("%d events are stored in the lmf file\n",nb-rejected);

  return;
}
