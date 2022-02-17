/*-------------------------------------------------------

List Mode Format 
                        
--  sigmoidCut.cc --

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2006 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of sigmoidCut


-------------------------------------------------------*/

#include <iostream>
#include <fstream>

#include <stdio.h>
#include "lmf.h"

#include "Sigmoid.hh"
#include "Gaussian.hh"

using namespace::std;

static bool done = false;
static u16 sctNb = 0, modNb = 0;
static Sigmoid ***sigmoidTable = NULL;
static double alpha = 0, x0 = 0;
static double e_alpha = 0, e_x0 = 0;

void setSigmoidCutParams(double inputAlpha, double inputX0, 
			 double inputEAlpha, double inputEX0)

{
  alpha = inputAlpha;
  x0 = inputX0;
  e_alpha = inputEAlpha;
  e_x0 = inputEX0;

  done = true;
  return;
}

void sigmoidCut(const ENCODING_HEADER *pEncoH,
		EVENT_RECORD **ppER)
{
  char response[10];
  double energyStep;

  u16 i,j;
  u16 *pcrist = NULL;

  double n_alpha, n_x0;
  double prob, random;

  ofstream out_sigmo;

  if(!sigmoidTable) {
    if(!done) {
      printf("Select the alpha param mean: ");
      if (fgets(response,sizeof(response),stdin))
	sscanf(response,"%lg", &alpha);
      printf("Select the alpha param std: ");
      if (fgets(response,sizeof(response),stdin))
	sscanf(response,"%lg", &e_alpha);

      printf("Select the sigmoid cut mean in keV: ");
      if (fgets(response,sizeof(response),stdin))
	sscanf(response,"%lg", &x0);
      printf("Select the sigmoid cut std in keV: ");
      if (fgets(response,sizeof(response),stdin))
	sscanf(response,"%lg", &e_x0);

      done = true;
    }
    sctNb = pEncoH->scannerTopology.totalNumberOfRsectors;
    modNb = pEncoH->scannerTopology.totalNumberOfModules;
    energyStep = (double) getEnergyStepFromCCH();

    sigmoidTable = new Sigmoid**[sctNb];
    out_sigmo.open("sigmoidParams.dat",ios::out);
    out_sigmo.setf(ios_base::fixed, ios_base::floatfield);
    for(i=0; i<sctNb; i++) {
      sigmoidTable[i] = new Sigmoid*[modNb];
      for(j=0; j<modNb; j++) {
	n_alpha = Gaussian::Shoot(alpha,e_alpha);
	n_x0 = Gaussian::Shoot(x0,e_x0);
	out_sigmo.precision(0);
	out_sigmo << i << "\t" << j << "\t";
	out_sigmo.precision(2);
	out_sigmo << n_alpha << "\t";
	out_sigmo.precision(0);
	out_sigmo << n_x0 << endl;
	sigmoidTable[i][j] = new Sigmoid(n_alpha * energyStep,n_x0 / energyStep);
      }
    }
    out_sigmo.close();
  }

  pcrist = demakeid((*ppER)->crystalIDs[0], pEncoH);
  i = pcrist[4];
  j = pcrist[3];
  free(pcrist);

  prob = (*(sigmoidTable[i][j]))((*ppER)->energy[0]);
  random = randd();

//   cout << "layer " << l << ": mean = " << (sigmoid[l])->GetX0() 
//        << " alpha = " << (sigmoid[l])->GetAlpha() << " nrj = " << (int)((*ppER)->energy[0])
//        << " -> prob = " << prob << " rdm = " << random << endl;

  if(prob < random) {
//     cout << "-> rejected" << endl;
    *ppER = NULL;
  }
  //   else
//     cout << "-> accepted" << endl;
//   getchar();

  return;
}

void sigmoidCutDestructor()
{
  u16 i,j;

  for(i=0; i<sctNb; i++) {
    for(j=0; j<modNb; j++)
      delete sigmoidTable[i][j];
    delete[] sigmoidTable[i];
  }
  delete[] sigmoidTable;
  sigmoidTable = NULL;

  done = false;
  return;
}
