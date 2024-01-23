/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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
  SetTimeNbBins(250);
  SetTimeTMax(5*ns);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaData::~GatePromptGammaData()
{
  delete GammaZ;
  delete Ngamma;

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
void GatePromptGammaData::SetTimeNbBins(int x)    { time_bin = x; }
void GatePromptGammaData::SetTimeTMax(double x)   { max_gamma_time = x; }
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
TH2D * GatePromptGammaData::GetGammaZ() { return GammaZ; }
TH2D * GatePromptGammaData::GetNgamma() { return Ngamma; }

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
  GammaZ->Reset();
  Ngamma->Reset();
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
  mFilename = filename;
  pTfile = new TFile(filename.c_str(),"READ");
  //ROOT does halt if a file doesnt exist, so we must test
  if (pTfile->IsZombie()) {
      GateError("Prompt Gamma database file '" << filename << "' not found. Aborting...");
  }

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

  GammaZList.resize(m);
  NgammaList.resize(m);
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
    dir->GetObject("GammaZ", GammaZ);
    dir->GetObject("Ngamma", Ngamma);
    dir->GetObject("EpEpg", pHEpEpg);
    dir->GetObject("EpEpgNorm", pHEpEpgNormalized);
    dir->GetObject("EpInelastic", pHEpInelastic);
    dir->GetObject("Ep", pHEp);
    dir->GetObject("SigmaInelastic", pHEpSigmaInelastic);
    dir->GetObject("EpInelasticProducedGamma", pHEpInelasticProducedGamma);

    if (GammaZ == NULL) GateError("No branch 'GammaZ' in the database '" << mFilename << "'. Wrong root file ?");
    if (Ngamma == NULL) GateError("No branch 'Ngamma' in the database '" << mFilename << "'. Wrong root file ?");
    if (pHEpEpg == NULL) GateError("No branch 'EpEpg' in the database '" << mFilename << "'. Wrong root file ?");
    if (pHEpEpgNormalized == NULL) GateError("No branch 'EpEpgNorm' in the database '" << mFilename << "'. Wrong root file ?");
    if (pHEpInelastic == NULL) GateError("No branch 'EpInelastic' in the database '" << mFilename << "'. Wrong root file ?");
    if (pHEp == NULL) GateError("No branch 'Ep' in the database '" << mFilename << "'. Wrong root file ?");
    if (pHEpSigmaInelastic == NULL) GateError("No branch 'SigmaInelastic' in the database '" << mFilename << "'. Wrong root file ?");
    if (pHEpInelasticProducedGamma == NULL) GateError("No branch 'EpInelasticProducedGamma' in the database '" << mFilename << "'. Wrong root file ?");

    // Set the pointers in the lists (have been changed by GetObject)
    GammaZList[elem->GetIndex()] = GammaZ;
    NgammaList[elem->GetIndex()] = Ngamma;
    pHEpEpgNormalizedList[elem->GetIndex()] = pHEpEpgNormalized;
    pHEpEpgList[elem->GetIndex()] = pHEpEpg;
    pHEpInelasticList[elem->GetIndex()] = pHEpInelastic;
    pHEpList[elem->GetIndex()] = pHEp;
    pHEpSigmaInelasticList[elem->GetIndex()] = pHEpSigmaInelastic;
    pHEpInelasticProducedGammaList[elem->GetIndex()] = pHEpInelasticProducedGamma;

    SetProtonNbBins(GammaZ->GetXaxis()->GetNbins());
    SetGammaNbBins(GammaZ->GetYaxis()->GetNbins());
    SetProtonEMin(GammaZ->GetXaxis()->GetXmin());
    SetProtonEMax(GammaZ->GetXaxis()->GetXmax());
    SetGammaEMin(GammaZ->GetYaxis()->GetXmin());
    SetGammaEMax(GammaZ->GetYaxis()->GetXmax());    
  }

  for(unsigned int i=0; i<m; i++) {
    if (ElementIndexList[i]) {
      GateMessage("Actor", 1, "Element " << table[i]->GetName()
                  << " is read." << std::endl);
    }
  }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaData::SetCurrentPointerForThisElement(const G4Element * elem)
{
  int i = elem->GetIndex();
  GammaZ = GammaZList[i];
  Ngamma = NgammaList[i];

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
  // be kept as is.
  mFilename = filename;
  pTfile = new TFile(filename.c_str(),"UPDATE");

  // Check if a directory for this material already exists
  std::string name = material->GetName();
  TDirectory * dir = pTfile->GetDirectory(name.c_str());
  if (dir == 0) {
    // create a subdirectory for the material in this file.
    dir = pTfile->mkdir(name.c_str());
    dir->cd();
  }
  else {
    // If already exists -> will be replaced.
    dir->cd();
    dir->Delete("*;*");
  }

  GammaZ = new TH2D("GammaZ","Gamma_Z as found in JMs paper.",
                               proton_bin, min_proton_energy/MeV, max_proton_energy/MeV,
                               gamma_bin, min_gamma_energy/MeV, max_gamma_energy/MeV);
  GammaZ->SetXTitle("E_{proton} [MeV]");
  GammaZ->SetYTitle("E_{gp} [MeV]");

  Ngamma = new TH2D("Ngamma","PG count per energy bin",
                     proton_bin, min_proton_energy/MeV, max_proton_energy/MeV,
                     gamma_bin, min_gamma_energy/MeV, max_gamma_energy/MeV);
  Ngamma->SetXTitle("E_{proton} [MeV]");
  Ngamma->SetYTitle("E_{gp} [MeV]");

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
void GatePromptGammaData::InitializeMaterial(bool DebugOutputEnabled)
{
  GateMessage("Actor", 1, "Create DB for used material. " << std::endl);
  const G4MaterialTable & matTable = *G4Material::GetMaterialTable();
  unsigned int n = G4Material::GetNumberOfMaterials();
  GateMessage("Actor", 1, "Number of materials : " << n << std::endl);

  mGammaEnergyHistoByMaterialByProtonEnergy.resize(n);
  GammaM.resize(n);
  NgammaM.resize(n);

  for(unsigned int i=0; i<n; i++) {
    const G4Material * m = matTable[i];
    if (m->GetName() == "worldDefaultAir") continue; //skip, should not occur in phantom.

    GateMessage("Actor", 3,"Material: " << m->GetName() << std::endl);

    // Check existence of materials and elements
    for(unsigned int e=0; e<m->GetNumberOfElements(); e++) {
      const G4Element * elem = m->GetElement(e);

      GateMessage("Actor", 3,"Element: " << elem->GetName() << std::endl);

      unsigned int elementIndex = elem->GetIndex();
      if (elem->GetZ() != 1) { // skip
        if (ElementIndexList[elementIndex] == false) {
          GateError("Aborting, because in material " << m->GetName()
                      << " element " << elem->GetName()
                      << " is missing from the PGDB." << std::endl);
        }
      }
    }

    GateMessage("Actor", 1, "Create DB for " << m->GetName()
              << " (d = " << m->GetDensity()/(g/cm3)
              << ")" << std::endl);
    mGammaEnergyHistoByMaterialByProtonEnergy[i].resize(proton_bin+1); // from [1 to n]

    //Build GammaZ -> GammaM, EpEpg=Ngamma(z,E) -> Ngamma(m,E)
    TH2D * hgammam = new TH2D();
    TH2D * hngammam = new TH2D();
    hgammam->SetBins(proton_bin, min_proton_energy, max_proton_energy,
                   gamma_bin, min_gamma_energy, max_gamma_energy); //same arrangement as GammaZ
    hngammam->SetBins(proton_bin, min_proton_energy, max_proton_energy,
                    gamma_bin, min_gamma_energy, max_gamma_energy);
    //Build tmpmat for mGammaEnergyHistoByMaterialByProtonEnergy
    TH2D * tmpmat = new TH2D("tmpmat","tmpmat",
                           proton_bin, min_proton_energy/MeV, max_proton_energy/MeV,
                           gamma_bin, min_gamma_energy/MeV, max_gamma_energy/MeV);

    // Loop over element
    for(unsigned int e=0; e<m->GetNumberOfElements(); e++) {
      const G4Element * elem = m->GetElement(e);
      if (elem->GetZ() == 1) continue; // if Hydrogen, probability is zero => skip

      double f = m->GetFractionVector()[e];
      //double f = m->GetAtomsVector()[e];

      // Get histogram for the current bin
      SetCurrentPointerForThisElement(elem);
      TH2D * hngammam2 = (TH2D*) Ngamma->Clone();
      TH2D * hgammam2 = (TH2D*) GammaZ->Clone();

      // Scale it according to the fraction of this element in the material, multiplied with density ratio
      hgammam2->Scale(f * m->GetDensity() / (g / cm3) );// GammaZ=GammaZ/rho(Z), so dont need to divide by rho(Z)
      hngammam2->Scale(f * m->GetDensity() / (g / cm3) );

      // Add it to the current total histo
      hgammam->Add(hgammam2);
      hngammam->Add(hngammam2);

      // delete temporary TH2D
      delete hngammam2;
      delete hgammam2;

      //Fill tmpmat will this element
      TH2D * tmpelem = (TH2D*) pHEpEpgNormalized->Clone();
      tmpelem->Scale(f);
      tmpmat->Add(tmpelem);
      delete tmpelem;
    }
    //update GammaM,NgammaM
    GammaM[i] = hgammam; //Now it's no longer modulo rho(Z), or rho(M)!!!
    NgammaM[i] = hngammam;

    if(DebugOutputEnabled){
        G4String fn1 = "output/debug.GammaM"+m->GetName()+".root";
        G4String fn2 = "output/debug.NgammaM"+m->GetName()+".root";
        TFile f1(fn1.c_str(),"new");
        hgammam->Write();
        TFile f2(fn2.c_str(),"new");
        hngammam->Write();
      }

    //tmpmat is complete, so we slice it up and copy it into mGammaEnergyHistoByMaterialByProtonEnergy
    for(unsigned int j=1; j<proton_bin+1; j++) {
      TH1D * h = new TH1D(*tmpmat->ProjectionY("", j, j)); //without a new it gives wrong results.
      h->ResetStats(); //necessary for non simple elements: not sure why but it works ! 
      mGammaEnergyHistoByMaterialByProtonEnergy[i][j] = h;
      //delete h; DO NOT DELETE h!!!! Because mGammaEnergyHistoByMaterialByProtonEnergy only holds to pointer to h, not h itself.
    }
    delete tmpmat;
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
  if ((*G4Material::GetMaterialTable())[materialIndex]->GetName() == "worldDefaultAir") return NULL;
  
  if (!DataForMaterialExist(materialIndex)) {
    GateError("Error in GatePromptGammaData for TLE, the material " <<
              (*G4Material::GetMaterialTable())[materialIndex]->GetName()
              << " is not in the DB. materialIndex: " << materialIndex);
  }

  // Get the index of the energy bin
  int binX = pHEp->FindFixBin(energy/MeV);

  // Get the projected histogram of the material
  TH1D * h = mGammaEnergyHistoByMaterialByProtonEnergy[materialIndex][binX];

  /* //DEBUG: verified that these two output the same (modulus density)
  TFile f1("histos1.root","new");
  TH1D * h = mGammaEnergyHistoByMaterialByProtonEnergy[materialIndex][binX];
  h->Write();

  TFile f2("histos2.root","new");
  TH1D * h2 = new TH1D(*GammaM[materialIndex]->ProjectionY("", binX, binX));
  h2->Write();
  */

  return h;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
TH2D* GatePromptGammaData::GetNgammaM(const int & materialIndex)
{
  return NgammaM[materialIndex];
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
TH2D* GatePromptGammaData::GetGammaM(const int & materialIndex)
{
  return GammaM[materialIndex];
}
//-----------------------------------------------------------------------------
