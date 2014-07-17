/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GatePromptGammaData.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "G4Material.hh"
#include "G4MaterialTable.hh"
#include "G4SystemOfUnits.hh"

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
  delete pHEpEpg;
  delete pHEpEpgNormalized;
  delete pHEpInelastic;
  delete pHEp;
  delete pHEpInelasticProducedGamma;
  delete pHEpSigmaInelastic;
  delete pTfile;
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
  // Normalisation by the number of primaries. This normalisation must
  // be performed only once so we check we only go here once.
  static bool alreadyHere = false;
  if (alreadyHere) {
    GateError("The PromptGammaStatisticActor has already been saved and normalized. However, it must write its results only once. Remove all 'SaveEvery' for this actor. Abort.");
  }
  // Normalisation
  int n = GateActorManager::GetInstance()->GetCurrentEventId()+1; // +1 because start at zero
  double f = 1.0/n;
  pHEpEpg->Scale(f);
  pHEpEpgNormalized->Scale(f);
  pHEpInelastic->Scale(f);
  pHEp->Scale(f);
  pHEpInelasticProducedGamma->Scale(f);

  pTfile->Write();
  alreadyHere = true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaData::Read(std::string & filename)
{
  mFilename = filename;
  pTfile = new TFile(filename.c_str(),"READ");

  // How many material in this file
  TList * l = pTfile->GetListOfKeys();
  unsigned int n = l->GetSize();
  GateMessage("Actor", 1, "Reading " << n << " elements in " << filename << std::endl);
  std::string s;
  for(unsigned int i=0; i<n; i++) {
    s += std::string(l->At(i)->GetName())+std::string(" ");
  }
  GateMessage("Actor", 1, "Elements are : " << s << std::endl);

  // Prepare vector of histo
  const G4ElementTable & table = *G4Element::GetElementTable();
  unsigned int m = table.size();
  GateMessage("Actor", 1, "G4Elements table has " << m << " elements." << std::endl);

  pHEpEpgList.resize(m);
  pHEpEpgNormalizedList.resize(m);
  pHEpInelasticList.resize(m);
  pHEpList.resize(m);
  pHEpInelasticProducedGammaList.resize(m);
  pHEpSigmaInelasticList.resize(m);
  ElementIndexList.resize(m);
  std::fill(ElementIndexList.begin(), ElementIndexList.end(), false);

  for(unsigned int i=0; i<n; i++) {
    int index = -1;
    for(unsigned int e=0; e<table.size(); e++) {
      if (l->At(i)->GetName() == table[e]->GetName()) index = e;
    }
    if (index == -1) {
      GateMessage("Actor", 1, "Skipping " << l->At(i)->GetName()
                  << ", not used by any material (not in G4Elements table)." << std::endl);
      continue;
    }

    // Set current pointer
    const G4Element * elem = table[index];

    if (elem->GetZ() == 1) {
      GateMessage("Actor", 1, "Skipping Hydrogen (no prompt gamma)." << std::endl);
      continue; // proton (Hydrogen) skip
    }

    SetCurrentPointerForThisElement(elem);
    ElementIndexList[elem->GetIndex()] = true;
    std::string f = elem->GetName();
    TDirectory * dir = pTfile->GetDirectory(f.c_str());
    dir->cd();
    dir->GetObject("EpEpg", pHEpEpg);
    dir->GetObject("EpEpgNorm", pHEpEpgNormalized);
    dir->GetObject("EpInelastic", pHEpInelastic);
    dir->GetObject("Ep", pHEp);
    dir->GetObject("SigmaInelastic", pHEpSigmaInelastic);
    dir->GetObject("EpInelasticProducedGamma", pHEpInelasticProducedGamma);

    // Set the pointers in the lists (have been changed by GetObject)
    pHEpEpgNormalizedList[elem->GetIndex()] = pHEpEpgNormalized;
    pHEpEpgList[elem->GetIndex()] = pHEpEpg;
    pHEpInelasticList[elem->GetIndex()] = pHEpInelastic;
    pHEpList[elem->GetIndex()] = pHEp;
    pHEpSigmaInelasticList[elem->GetIndex()] = pHEpSigmaInelastic;
    pHEpInelasticProducedGammaList[elem->GetIndex()] = pHEpInelasticProducedGamma;

    SetProtonNbBins(pHEpEpgNormalized->GetXaxis()->GetNbins());
    SetGammaNbBins(pHEpEpgNormalized->GetYaxis()->GetNbins());

    SetProtonEMin(pHEpEpgNormalized->GetXaxis()->GetXmin());
    SetProtonEMax(pHEpEpgNormalized->GetXaxis()->GetXmax());
    SetGammaEMin(pHEpEpgNormalized->GetYaxis()->GetXmin());
    SetGammaEMax(pHEpEpgNormalized->GetYaxis()->GetXmax());

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
  }

  for(unsigned int i=0; i<m; i++) {
    if (ElementIndexList[i]) {
      GateMessage("Actor", 1, "Element " << table[i]->GetName()
                  << " is read (and normalized)." << std::endl);
    }
  }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaData::SetCurrentPointerForThisElement(const G4Element * elem)
{
  int i = elem->GetIndex();
  pHEpEpg = pHEpEpgList[i];
  pHEpEpgNormalized = pHEpEpgNormalizedList[i];
  pHEpInelastic = pHEpInelasticList[i];
  pHEp = pHEpList[i];
  pHEpSigmaInelastic = pHEpSigmaInelasticList[i];
  pHEpInelasticProducedGamma = pHEpInelasticProducedGammaList[i];
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// FIXME change the name of this function
void GatePromptGammaData::Initialize(std::string & filename, const G4Material * material)
{
  // Open the file in update mode, part will be overwriten, part will
  // be keep as is.
  mFilename = filename;
  pTfile = new TFile(filename.c_str(),"UPDATE");

  // Check if a directory for this material already exist
  std::string name = material->GetName();
  TDirectory * dir = pTfile->GetDirectory(name.c_str());
  if (dir == 0) {
    // create a subdirectory for the material in this file.
    dir = pTfile->mkdir(name.c_str());
    dir->cd();
  }
  else {
    // If already exist -> will be replaced.
    dir->cd();
    dir->Delete("*;*");
  }

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
// FIXME change the name of this function
void GatePromptGammaData::InitializeMaterial()
{
  GateMessage("Actor", 1, "Create DB for used material. " << std::endl);
  const G4MaterialTable & matTable = *G4Material::GetMaterialTable();
  unsigned int n = G4Material::GetNumberOfMaterials();
  GateMessage("Actor", 1, "Number of materials : " << n << std::endl);

  mGammaEnergyHistoByMaterialByProtonEnergy.resize(n);

  for(unsigned int i=0; i<n; i++) {
    bool stop = false;
    const G4Material * m = matTable[i];

    // Check material
    for(unsigned int e=0; e<m->GetNumberOfElements(); e++) {
      const G4Element * elem = m->GetElement(e);
      unsigned int elementIndex = elem->GetIndex();
      if (elem->GetZ() != 1) { // skip
        if (ElementIndexList[elementIndex] == false) {
          GateMessage("Actor", 1, "Skipping " << m->GetName()
                      << " because " << elem->GetName()
                      << " is missing in the DB." << std::endl);
          stop = true;
        }
      }
    }

    if (!stop) {
      GateMessage("Actor", 1, "Create DB for " << m->GetName()
                  << " (d = " << m->GetDensity()/(g/cm3)
                  << ")" << std::endl);
      mGammaEnergyHistoByMaterialByProtonEnergy[i].resize(proton_bin+1); // from [1 to n]

      for(unsigned int j=1; j<proton_bin+1; j++) {
        TH1D * h = new TH1D();
        h->SetBins(gamma_bin, min_gamma_energy, max_gamma_energy);

        // Loop over element
        for(unsigned int e=0; e<m->GetNumberOfElements(); e++) {
          const G4Element * elem = m->GetElement(e);
          double f = m->GetFractionVector()[e];

          if (elem->GetZ() != 1) { // if not Hydrogen.
            // (If hydrogen probability is zero)
            // Get histogram for the current bin
            SetCurrentPointerForThisElement(elem);
            TH1D * he = new TH1D(*pHEpEpgNormalized->ProjectionY("", j, j));

            // Scale it according to the fraction of this element in the material
            he->Scale(f);

            // Add it to the current total histo
            h->Add(he);
          }
        }
        mGammaEnergyHistoByMaterialByProtonEnergy[i][j] = h;
      }
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePromptGammaData::DataForMaterialExist(const int & materialIndex)
{
  if (mGammaEnergyHistoByMaterialByProtonEnergy[materialIndex].size() == 0) return false;
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
TH1D * GatePromptGammaData::GetGammaEnergySpectrum(const int & materialIndex,
                                                   const double & energy)
{
  // Check material
  if (!DataForMaterialExist(materialIndex)) {
    GateError("Error in GatePromptGammaData for TLE, the material " <<
              (*G4Material::GetMaterialTable())[materialIndex]->GetName()
              << " is not in the DB");
  }

  // Get the index of the energy bin
  int binX = pHEp->FindFixBin(energy);

  // Get the projected histogram of the material
  TH1D * h = mGammaEnergyHistoByMaterialByProtonEnergy[materialIndex][binX];

  return h;
}
//-----------------------------------------------------------------------------
