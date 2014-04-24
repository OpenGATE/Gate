/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GatePromptGammaEnergySpectrumData.hh"
#include "G4UnitsTable.hh"

//-----------------------------------------------------------------------------
GatePromptGammaEnergySpectrumData::GatePromptGammaEnergySpectrumData()
{
  SetProtonEMin(0);
  SetProtonEMax(150*MeV);
  SetGammaEMin(0);
  SetGammaEMax(10*MeV);
  SetProtonNbBins(250);
  SetGammaNbBins(250);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaEnergySpectrumData::~GatePromptGammaEnergySpectrumData()
{
  DD("GatePromptGammaEnergySpectrumData destructor");
  delete pHEpEpg;
  delete pHEpEpgNormalized;
  delete pHEpInelastic;
  delete pHEp;
  delete pHEpInelasticProducedGamma;
  delete pHEpSigmaInelastic;
  delete pTfile;
  DD("GatePromptGammaEnergySpectrumData END destructor");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaEnergySpectrumData::SetProtonEMin(double x) { min_proton_energy = x; }
void GatePromptGammaEnergySpectrumData::SetProtonEMax(double x) { max_proton_energy = x; }
void GatePromptGammaEnergySpectrumData::SetGammaEMin(double x)  { min_gamma_energy = x; }
void GatePromptGammaEnergySpectrumData::SetGammaEMax(double x)  { max_gamma_energy = x; }
void GatePromptGammaEnergySpectrumData::SetProtonNbBins(int x)  { proton_bin = x; }
void GatePromptGammaEnergySpectrumData::SetGammaNbBins(int x)   { gamma_bin = x; }
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
TH2D * GatePromptGammaEnergySpectrumData::GetHEpEpgNormalized()          { return pHEpEpgNormalized; }
TH1D * GatePromptGammaEnergySpectrumData::GetHEp()                       { return pHEp; }
TH1D * GatePromptGammaEnergySpectrumData::GetHEpInelastic()              { return pHEpInelastic; }
TH1D * GatePromptGammaEnergySpectrumData::GetHEpSigmaInelastic()         { return pHEpSigmaInelastic; }
TH2D * GatePromptGammaEnergySpectrumData::GetHEpEpg()                    { return pHEpEpg; }
TH1D * GatePromptGammaEnergySpectrumData::GetHEpInelasticProducedGamma() { return pHEpInelasticProducedGamma; }
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaEnergySpectrumData::ResetData()
{
  pHEpEpg->Reset();
  pHEpEpgNormalized->Reset();
  pHEpInelastic->Reset();
  pHEp->Reset();
  pHEpInelasticProducedGamma->Reset();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaEnergySpectrumData::SaveData()
{
  pTfile->Write();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaEnergySpectrumData::Read(std::string & filename)
{
  DD("Read");
  mFilename = filename;
  pTfile = new TFile(filename.c_str(),"READ");

  // Get pointers to histograms
  pHEpEpg = dynamic_cast<TH2D*>(pTfile->Get("EpEpg"));
  pHEpEpgNormalized = dynamic_cast<TH2D*>(pTfile->Get("EpEpgNorm"));
  pHEpInelastic = dynamic_cast<TH1D*>(pTfile->Get("EpInelastic"));
  pHEp = dynamic_cast<TH1D*>(pTfile->Get("Ep"));
  pHEpSigmaInelastic = dynamic_cast<TH1D*>(pTfile->Get("SigmaInelastic"));
  pHEpInelasticProducedGamma = dynamic_cast<TH1D*>(pTfile->Get("EpInelasticProducedGamma"));

  SetProtonNbBins(pHEpEpgNormalized->GetXaxis()->GetNbins());
  SetGammaNbBins(pHEpEpgNormalized->GetYaxis()->GetNbins());
  DD(proton_bin);
  DD(gamma_bin);

  SetProtonEMin(pHEpEpgNormalized->GetXaxis()->GetXmin());
  SetProtonEMax(pHEpEpgNormalized->GetXaxis()->GetXmax());
  SetGammaEMin(pHEpEpgNormalized->GetYaxis()->GetXmin());
  SetGammaEMax(pHEpEpgNormalized->GetYaxis()->GetXmax());

  DD(min_proton_energy);
  DD(max_proton_energy);
  DD(min_gamma_energy);
  DD(max_gamma_energy);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaEnergySpectrumData::Initialize(std::string & filename)
{
  DD("Initialize");
  DD(filename);
  mFilename = filename;
  pTfile = new TFile(filename.c_str(),"RECREATE");

  pHEpEpg = new TH2D("EpEpg","PG count",
                     proton_bin, min_proton_energy/MeV, max_proton_energy/MeV,
                     gamma_bin, min_gamma_energy/MeV, max_gamma_energy/MeV);
  pHEpEpg->SetXTitle("E_{proton} [MeV]");
  pHEpEpg->SetYTitle("E_{gp} [MeV]");

  pHEpEpgNormalized = new TH2D("EpEpgNorm","PG normalized by mean free path in [m]",
                               proton_bin, min_proton_energy/MeV, max_proton_energy/MeV,
                               gamma_bin, min_gamma_energy/MeV, max_gamma_energy/MeV);
  pHEpEpgNormalized->SetXTitle("E_{proton} [MeV]");
  pHEpEpgNormalized->SetYTitle("E_{gp} [MeV]");

  pHEpInelastic = new TH1D("EpInelastic","proton energy for each inelastic interaction",
                           proton_bin, min_proton_energy/MeV, max_proton_energy/MeV);
  pHEpInelastic->SetXTitle("E_{proton} [MeV]");

  pHEp = new TH1D("Ep","proton energy",proton_bin, min_proton_energy/MeV, max_proton_energy/MeV);
  pHEp->SetXTitle("E_{proton} [MeV]");

  pHEpSigmaInelastic = new TH1D("SigmaInelastic","Sigma inelastic Vs Ep",
                                proton_bin, min_proton_energy/MeV, max_proton_energy/MeV);
  pHEpSigmaInelastic->SetXTitle("E_{proton} [MeV]");

  pHEpInelasticProducedGamma = new TH1D("EpInelasticProducedGamma",
                                        "proton energy for each inelastic interaction if gamma production",
                                        proton_bin, min_proton_energy/MeV, max_proton_energy/MeV);
  pHEpInelasticProducedGamma->SetXTitle("E_{proton} [MeV]");

  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
int GatePromptGammaEnergySpectrumData::ComputeProtonEnergyBinIndex(const double & energy)
{
  int temp = pHEp->FindFixBin(energy);

  return temp;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
TH1D * GatePromptGammaEnergySpectrumData::GetGammaEnergySpectrum(const double & energy)
{
  int binX = ComputeProtonEnergyBinIndex(energy);
  TH1D * h = pHEpEpgNormalized->ProjectionY("PhistoEnergy", binX, binX);
  return h;
}
//-----------------------------------------------------------------------------
