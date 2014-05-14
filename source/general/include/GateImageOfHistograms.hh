/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifndef GATEIMATEOFHISTOGRAMS_HH
#define GATEIMATEOFHISTOGRAMS_HH

#include "GateImage.hh"
#include <TH1D.h>

//-----------------------------------------------------------------------------
class GateImageOfHistograms:public GateImage
{
public:

  GateImageOfHistograms();
  ~GateImageOfHistograms();

  void SetHistoInfo(int n, double min, double max);
  virtual void Allocate();
  void Reset();
  void AddValue(const int & index, TH1D * h);
  virtual void Write(G4String filename, const G4String & comment = "");

protected:
  //std::vector<TH1D*> mHistoData;
  double min_gamma_energy;
  double max_gamma_energy;
  int gamma_bin;
  std::vector<double> dataDouble;

  // DEBUG
  TH1D * mTotalEnergySpectrum;

};
//-----------------------------------------------------------------------------

#endif // GATEIMAGEOFHISTOGRAMS
