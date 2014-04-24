/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMAENERGYSPECTRUMDATA_HH
#define GATEPROMPTGAMMAENERGYSPECTRUMDATA_HH

#include "GateConfiguration.h"
#include "GateMessageManager.hh"
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

//-----------------------------------------------------------------------------
class GatePromptGammaEnergySpectrumData
{
public:

  GatePromptGammaEnergySpectrumData();
  ~GatePromptGammaEnergySpectrumData();

  void SetProtonEMin(double x);
  void SetProtonEMax(double x);
  void SetGammaEMin(double x);
  void SetGammaEMax(double x);
  void SetProtonNbBins(int x);
  void SetGammaNbBins(int x);

  double GetProtonEMin() { return min_proton_energy; }
  double GetProtonEMax() { return max_proton_energy; }
  double GetGammaEMin()  { return min_gamma_energy; }
  double GetGammaEMax()  { return max_gamma_energy; }
  int GetProtonNbBins()  { return proton_bin; }
  int GetGammaNbBins()   { return gamma_bin; }

  void Initialize(std::string & filename);
  void Read(std::string & filename);
  void SaveData();
  void ResetData();

  // Principal 2D histogram protonE / gammaE
  TH2D * GetHEpEpgNormalized();

  // Optional other data
  TH1D * GetHEp();
  TH1D * GetHEpInelastic();
  TH1D * GetHEpSigmaInelastic();
  TH2D * GetHEpEpg();
  TH1D * GetHEpInelasticProducedGamma();

  // Convenient functions
  TH1D * GetGammaEnergySpectrum(const double & energy);
  int ComputeProtonEnergyBinIndex(const double & energy);

protected:
  std::string mFilename;

  // Histograms limits
  double min_proton_energy;
  double min_gamma_energy;
  double max_proton_energy;
  double max_gamma_energy;
  int proton_bin;
  int gamma_bin;

  // Data
  TFile* pTfile;
  TH2D* pHEpEpg;
  TH2D* pHEpEpgNormalized;
  TH1D* pHEpInelastic;
  TH1D* pHEp;
  TH1D* pHEpInelasticProducedGamma;
  TH1D* pHEpSigmaInelastic;

};
//-----------------------------------------------------------------------------

#endif // GATEPROMPTGAMMAENERGYSPECTRUMDATA
