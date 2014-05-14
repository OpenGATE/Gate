/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GatePromptGammaData.hh"
#include "G4UnitsTable.hh"

//-----------------------------------------------------------------------------
GatePromptGammaData::GatePromptGammaData()
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
GatePromptGammaData::~GatePromptGammaData()
{
  DD("GatePromptGammaData destructor");
  delete pHEpEpg;
  delete pHEpEpgNormalized;
  delete pHEpInelastic;
  delete pHEp;
  delete pHEpInelasticProducedGamma;
  delete pHEpSigmaInelastic;
  delete pTfile;
  DD("GatePromptGammaData END destructor");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaData::SetProtonEMin(double x) { min_proton_energy = x; }
void GatePromptGammaData::SetProtonEMax(double x) { max_proton_energy = x; }
void GatePromptGammaData::SetGammaEMin(double x)  { min_gamma_energy = x; }
void GatePromptGammaData::SetGammaEMax(double x)  { max_gamma_energy = x; }
void GatePromptGammaData::SetProtonNbBins(int x)  { proton_bin = x; }
void GatePromptGammaData::SetGammaNbBins(int x)   { gamma_bin = x; }
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
TH2D * GatePromptGammaData::GetHEpEpgNormalized()          { return pHEpEpgNormalized; }
TH1D * GatePromptGammaData::GetHEp()                       { return pHEp; }
TH1D * GatePromptGammaData::GetHEpInelastic()              { return pHEpInelastic; }
TH1D * GatePromptGammaData::GetHEpSigmaInelastic()         { return pHEpSigmaInelastic; }
TH2D * GatePromptGammaData::GetHEpEpg()                    { return pHEpEpg; }
TH1D * GatePromptGammaData::GetHEpInelasticProducedGamma() { return pHEpInelasticProducedGamma; }
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaData::ResetData()
{
  pHEpEpg->Reset();
  pHEpEpgNormalized->Reset();
  pHEpInelastic->Reset();
  pHEp->Reset();
  pHEpInelasticProducedGamma->Reset();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaData::SaveData()
{
  pTfile->Write();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaData::Read(std::string & filename)
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

  // Normalisation of 2D histo (pHEpEpgNormalized = E proton vs E gamma) according to
  // proba of inelastic interaction by E proton (pHEpInelastic);
  double temp=0.0;
  for( int i=1; i <= pHEpEpgNormalized->GetNbinsX(); i++) {
      for( int j = 1; j <= pHEpEpgNormalized->GetNbinsY(); j++) {
        if(pHEpInelastic->GetBinContent(i) != 0) {
          temp = pHEpEpgNormalized->GetBinContent(i,j)/pHEpInelastic->GetBinContent(i);
        }
        else {
          if (pHEpEpgNormalized->GetBinContent(i,j)> 0.) {
            DD(i);
            DD(j);
            DD(pHEpInelastic->GetBinContent(i));
            DD(pHEpEpgNormalized->GetBinContent(i,j));
            GateError("ERROR in in histograms, pHEpInelastic is zero and not pHEpEpgNormalized");
          }
        }
        pHEpEpgNormalized->SetBinContent(i,j,temp);
      }
  }
  DD("end normalisation");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaData::Initialize(std::string & filename)
{
  DD("Initialize");
  DD(filename);
  DD(min_proton_energy);
  DD(max_proton_energy);
  DD(proton_bin);
  DD(min_gamma_energy);
  DD(max_gamma_energy);
  DD(gamma_bin);

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


  // FIXME
  DD(pHEpEpgNormalized->GetDefaultSumw2());

  // // FIXME
  // cs = new TH2D("sigma by E","cross section by proton E",
  //               proton_bin, min_proton_energy/MeV, max_proton_energy/MeV,
  //               50, 0, 0.0019);

  ResetData();
}
//-----------------------------------------------------------------------------


// //-----------------------------------------------------------------------------
// int GatePromptGammaData::ComputeProtonEnergyBinIndex(const double & energy)
// {
//   int temp = pHEp->FindFixBin(energy);
//   return temp;
// }
// //-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
TH1D * GatePromptGammaData::GetGammaEnergySpectrum(const double & energy)
{
  int binX = pHEp->FindFixBin(energy); //ComputeProtonEnergyBinIndex(energy);
  TH1D * h = pHEpEpgNormalized->ProjectionY("PhistoEnergy", binX, binX);
  return h;
}
//-----------------------------------------------------------------------------
