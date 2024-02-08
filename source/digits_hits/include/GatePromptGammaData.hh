/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMAENERGYSPECTRUMDATA_HH
#define GATEPROMPTGAMMAENERGYSPECTRUMDATA_HH

#include "G4Material.hh"
#include "GateConfiguration.h"
#include "GateMessageManager.hh"
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

//-----------------------------------------------------------------------------
class GatePromptGammaData
{
public:

  GatePromptGammaData();
  ~GatePromptGammaData();

  void SetProtonEMin(double x);
  void SetProtonEMax(double x);
  void SetGammaEMin(double x);
  void SetGammaEMax(double x);
  void SetProtonNbBins(int x);
  void SetGammaNbBins(int x);

  void SetTimeNbBins(int x);
  void SetTimeTMax(double x);
  
  double GetProtonEMin() { return min_proton_energy; }
  double GetProtonEMax() { return max_proton_energy; }
  double GetGammaEMin()  { return min_gamma_energy; }
  double GetGammaEMax()  { return max_gamma_energy; }
  int GetProtonNbBins()  { return proton_bin; }
  int GetGammaNbBins()   { return gamma_bin; }

  double GetTimeTMax() { return max_gamma_time;}
  int GetTimeNbBins() { return time_bin;}
  
  void Initialize(std::string & filename, const G4Material * m);
  void InitializeMaterial(bool DebugOutputEnabled);
  void Read(std::string & filename);
  void SaveData();
  void ResetData();

  // Principal 2D histograms, GetNgamma only used for TLE uncertainty.
  TH2D * GetGammaZ(); //IMPORTANT NOTE: this actually stores GammaZ/rho(Z). Then we don't have to multiply with it in the TLE.
  TH2D * GetNgamma();

  TH2D * GetHEpEpgNormalized();
  TH1D * GetHEp();
  TH1D * GetHEpInelastic();
  TH1D * GetHEpSigmaInelastic();
  TH2D * GetHEpEpg();
  TH1D * GetHEpInelasticProducedGamma();

  // Return the gamma energy spectrum for the proton at given energy,
  // for the given material
  TH1D * GetGammaEnergySpectrum(const int & matIndex, const double & energy); //backward compat.
  TH2D * GetGammaM(const int & materialIndex);
  TH2D * GetNgammaM(const int & materialIndex);
  bool DataForMaterialExist(const int & materialIndex);

protected:
  std::string mFilename;

  // Histograms limits
  double min_proton_energy;
  double min_gamma_energy;
  double max_proton_energy;
  double max_gamma_energy;
  unsigned int proton_bin;
  unsigned int gamma_bin;

  double max_gamma_time;
  unsigned int time_bin;
  
  void SetCurrentPointerForThisElement(const G4Element * elem);

  //final outputs used to create tle source.
  TH2D* GammaZ; // \Gamma_Z/rho(Z)
  TH2D* Ngamma; // \boldsymbol{N}_gamma

  // Data current pointer to histograms
  TFile* pTfile;

  TH2D* pHEpEpg;
  TH2D* pHEpEpgNormalized;
  TH1D* pHEpInelastic;
  TH1D* pHEp;
  TH1D* pHEpInelasticProducedGamma;
  TH1D* pHEpSigmaInelastic;

  // Data : histo by elements
  std::vector<TH2D*> GammaZList;
  std::vector<TH2D*> NgammaList;

  std::vector<TH2D*> pHEpEpgList;
  std::vector<TH2D*> pHEpEpgNormalizedList;
  std::vector<TH1D*> pHEpInelasticList;
  std::vector<TH1D*> pHEpList;
  std::vector<TH1D*> pHEpInelasticProducedGammaList;
  std::vector<TH1D*> pHEpSigmaInelasticList;
  std::vector<bool>  ElementIndexList;

  // pre-computed list of gamma energy histogram
  std::vector<std::vector<TH1D *> > mGammaEnergyHistoByMaterialByProtonEnergy;
  // pre-computed list of histogram
  std::vector<TH2D *> GammaM; //NOT GammaZ
  std::vector<TH2D *> NgammaM; //NOT Ngamma(Z,E)

};
//-----------------------------------------------------------------------------

#endif // GATEPROMPTGAMMAENERGYSPECTRUMDATA
