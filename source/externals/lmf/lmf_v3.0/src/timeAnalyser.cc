/*-------------------------------------------------------

List Mode Format 
                        
--  timeAnalyser.cc  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2006 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of timeAnalyser


-------------------------------------------------------*/


#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "lmf.h"

using namespace std;

static u16 sctNb = 6;
static double epsilon = 0.01;

static u16 *sctName;
static vector < double > *sectorLists;
static double *meanList;
static u64 *lastTimes;
static double *minList;
static vector < double > globalList;
static double globalMean;
static u64 lastTime;
static double globalMin;

static double timeStep = 0;

void setInitialParamsForTimeAnalyser(void)
{
  char response[10];

  printf("Set the number of sectors in use (Default %hu): ",
	 sctNb);
  if (fgets(response,sizeof(response),stdin))
    sscanf(response,"%hu",&sctNb);

  printf("Set the precision on your mesurement (Default %.0f %%): ",
	 epsilon * 100);
  if (fgets(response,sizeof(response),stdin))
    sscanf(response,"%lg",&epsilon);

  printf("Epsilon = %.02f %% -> \n",epsilon);
  epsilon = epsilon/100;
  printf("%.04f\n",epsilon);

  return;
}

void setInitialParamsForTimeAnalyserDirectly(u16 nbOfSct, double inline_epsilon)
{
  sctNb = nbOfSct;

  epsilon = inline_epsilon / 100;

  printf("Nb of sectors = %hu\n",sctNb);
  printf("Epsilon = %g %%\n",epsilon * 100);

  return;
}

void terminateTimeAnalyser(void) 
{
  vector < double >::iterator iter;

  u16 index;
  int nb;
  double mean, variance, tmp;
  double m_mean = 0, m_var = 0;

  ofstream f("output.dat");
  
  printf("Sct\tMean\tVariance\tStd Dev\tMin\n");
  nb = 0;
  variance = 0;
  mean = globalMean;
  while (globalList.size()) {
    /* Take the first event of the list */
    iter = globalList.begin();
    
    /* variance calculation */
    tmp = *iter - mean;
    tmp = tmp*tmp;
    variance += tmp;     
    nb++;
    /* delete the event and put its ref away from the list */
    globalList.erase(iter);
  }
  variance = variance / nb;
  printf("Global\t%.2g micros\t%.2g micros\t%.2g micros\t%.2g micros\n",mean,variance,sqrt(variance),globalMin);

  m_mean = 0;
  m_var = 0;
  for (index = 0; index < sctNb; index++) {
    nb = 0;
    variance = 0;
    mean = meanList[index];
    while (sectorLists[index].size()) {
      /* Take the first event of the list */
      iter = sectorLists[index].begin();
      /* Write the data in file */      
      f << sctName[index] << "\t" << *iter << endl;
      /* variance calculation */
      tmp = *iter - mean;
      tmp = tmp*tmp;
      variance += tmp;     
      nb++;
      /* delete the event and put its ref away from the list */
      sectorLists[index].erase(iter);
    }
    variance = variance / nb;
    printf("%d\t%.2g micros\t%.2g micros\t%.2g micros\t%.2g micros\n",
	   sctName[index],mean,variance,sqrt(variance),minList[index]);
    m_mean += mean;
    m_var += variance;
  }
  m_mean = m_mean / sctNb;
  m_var = m_var / sctNb;
  printf("->\t%.2g micros\t%.2g micros\t%.2g micros\n",m_mean,m_var,sqrt(m_var));

  delete[]sectorLists;
  delete[]sctName;
  delete[] meanList;
  delete[]lastTimes;

  f.close();

  exit(EXIT_SUCCESS);

  return;
}

void timeAnalyser(const ENCODING_HEADER * pEncoH,
		  const EVENT_RECORD * pER)
{
  static u32 viewOnce = 0;
  static u8 doneonce = 0;
  static u8 check = 0;

  u16 sct, index;
  u64 actualTime;
  double difTime;
  double actualMean, difMean;
  size_t actualSize;

  if (!doneonce) {
    if (!(sctName = new u16[sctNb]))
      printf("\n *** error : timeAnalyser.cc : timeAnalyser : malloc\n");

    if (!(sectorLists = new vector < double >[sctNb]))
      printf("\n *** error : timeAnalyser.cc : timeAnalyser : malloc\n");

    if(!(meanList = new double[sctNb]))
      printf("\n *** error : timeAnalyser.cc : timeAnalyser : malloc\n");

    if (!(lastTimes = new u64[sctNb]))
      printf("\n *** error : timeAnalysery.c : timeAnalyser : malloc\n");

    if (!(minList = new double[sctNb]))
      printf("\n *** error : timeAnalyser.cc : timeAnalyser : malloc\n");

    timeStep = (double) (i64) getTimeStepFromCCH() / 1E9;

    for(sct = 0; sct<sctNb; sct++) {
      lastTimes[sct] = (u64) -1;
      meanList[sct] = 0;
    }
    lastTime = (u64) -1;
    globalMean = 0;

    doneonce++;
  }

  sct = getRsectorID(pEncoH, pER);

  if (!((viewOnce >> sct) & 1)) {
    viewOnce |= 1 << sct;
    check = 0;
    for (index = 0; index < 8 * sizeof(u32); index++)
      check += (viewOnce >> index) & 1;
    sctName[check - 1] = sct;
    if (check > sctNb) {
      printf
	  ("nb of sector in file is greater than the one introduced\nPlease re-run\n");
      exit(0);
    }
  }

  for (index = 0; index < check; index++)
    if (sct == sctName[index])
      break;

  actualTime = u8ToU64(pER->timeStamp);

  if(lastTime == (u64) -1)
    lastTime = actualTime;
  else {
    difTime = (double)(i64)(actualTime - lastTime);
    lastTime = actualTime;
    difTime *= timeStep;

    actualSize = globalList.size();
    globalList.push_back(difTime);

    if(actualSize) {
//       printf("actualSize = %d globalMean = %g difTime = %g",actualSize,globalMean,difTime);
      actualMean = (actualSize * globalMean + difTime) / (actualSize + 1);
//       printf(" actualMean = %g\n",actualMean);getchar();
      difMean = actualMean - globalMean;
      if(difMean < 0)
	difMean = -difMean;

      globalMean = actualMean;

      if(difTime < globalMin)
	globalMin = difTime;
    }
    else {
      globalMean = difTime;
      globalMin = difTime;
    }
  }

  if(lastTimes[index] == (u64) -1)
    lastTimes[index] = actualTime;
  else {
    difTime = (double)(i64)(actualTime - lastTimes[index]);
    lastTimes[index] = actualTime;
    difTime *= timeStep;

    actualSize = sectorLists[index].size();
    sectorLists[index].push_back(difTime);

    if(actualSize) {
      actualMean = (actualSize * meanList[index] + difTime) / (actualSize + 1);
      
      difMean = actualMean - meanList[index];
      if(difMean < 0)
	difMean = -difMean;

//       printf("old mean = %g\t new mean = %g dif mean = %g\n",meanList[index], actualMean, difMean);
      meanList[index] = actualMean;

      if(difMean < epsilon * actualMean)
	terminateTimeAnalyser();

      if(difTime < minList[index])
	minList[index] = difTime;
    }
    else {
      meanList[index] = difTime;
      minList[index] = difTime;
    }
  }
  return;
}
