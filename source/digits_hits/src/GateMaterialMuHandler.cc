#ifndef GATEMATERIALMUHANDLER_CC
#define GATEMATERIALMUHANDLER_CC

#include "GateMaterialMuHandler.hh"
#include "GateMuTables.hh"
#include "GateMiscFunctions.hh"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>

#include "G4ParticleTable.hh"
#include "GatePhysicsList.hh"

using std::map;
using std::string;

GateMaterialMuHandler::GateMaterialMuHandler(int /*nbOfElements*/)
{
//   mElementsTable = new GateMuTable*[nbOfElements+1];
//   mNbOfElements = nbOfElements;
  isInitialized = false;
//   InitElementTable();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateMaterialMuHandler::~GateMaterialMuHandler()
{
//   delete[] mElementsTable;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
inline double interpolation(double Xa,double Xb,double Ya,double Yb,double x){
  return exp(log(Ya) + log(Yb/Ya) / log(Xb/Xa)* log(x/Xa) );
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMaterialMuHandler::AddMaterial(G4Material* material)
{  
  //const G4ElementVector* elements = material->GetElementVector();
  int nb_e = 0;
  int nb_of_elements = material->GetNumberOfElements();
  for(int i = 0; i < nb_of_elements; i++)
    nb_e += mElementsTable[(int) material->GetElement(i)->GetZ()]->GetSize();
  
  double* energies = new double[nb_e];
  int *index = new int[nb_of_elements];
  double **e_tables = new double*[nb_of_elements];
  double **mu_tables = new double*[nb_of_elements];
  double **muen_tables = new double*[nb_of_elements];
  //  int min_index;
  
  const G4double* FractionMass = material->GetFractionVector();

  for(int i = 0; i < nb_of_elements; i++){
    e_tables[i] = mElementsTable[(int) material->GetElement(i)->GetZ()]->GetEnergies();
    mu_tables[i] = mElementsTable[(int) material->GetElement(i)->GetZ()]->GetMuTable();
    muen_tables[i] = mElementsTable[(int) material->GetElement(i)->GetZ()]->GetMuEnTable();
    index[i] = 0;
  }
  for(int i = 0; i < nb_e; i++){
    int min_table = 0;
    while(index[min_table] >= mElementsTable[(int) material->GetElement(min_table)->GetZ()]->GetSize())
      min_table++;
    for(int j = min_table + 1; j < nb_of_elements; j++)
      if(e_tables[j][index[j]] < e_tables[min_table][index[min_table]])
  	min_table = j;
    energies[i] = e_tables[min_table][index[min_table]];
    
    if(i > 0){
      if(energies[i] == energies[i-1]){
  	if(index[min_table] > 0 && e_tables[min_table][index[min_table]] == 
  	   e_tables[min_table][index[min_table]-1])
  	  ;
  	else{
  	  i--;
  	  nb_e--;
  	}
      }
    }
    index[min_table]++;
  }
  
  //And now computing mu_en
  double *MuEn = new double[nb_e];
  double *Mu = new double[nb_e];
  for(int i = 0; i < nb_of_elements; i++){
    index[i] = 0;
  }
  

  //Assume that all table begin with the same energy
  for(int i = 0; i < nb_e; i++){
    MuEn[i] = 0.0;
    Mu[i] = 0.0;
    double current_e = energies[i];
    for(int j = 0; j < nb_of_elements; j++){
      //You never need to advance twice
      if(e_tables[j][index[j]] < current_e)
  	index[j]++;
      if(e_tables[j][index[j]] == current_e){
  	Mu[i] += FractionMass[j]*mu_tables[j][index[j]];
  	MuEn[i] += FractionMass[j]*muen_tables[j][index[j]];
	if(i != nb_e-1)
	  if(e_tables[j][index[j]] == e_tables[j][index[j]+1])
	    index[j]++;
      }
      else{
  	Mu[i] += FractionMass[j]*interpolation(e_tables[j][index[j]-1],
						 e_tables[j][index[j]],
						 mu_tables[j][index[j]-1],
						 mu_tables[j][index[j]],
						 current_e);
  	MuEn[i] += FractionMass[j]*interpolation(e_tables[j][index[j]-1],
						 e_tables[j][index[j]],
						 muen_tables[j][index[j]-1],
						 muen_tables[j][index[j]],
						 current_e);

      }
    }
  }
  
  GateMuTable * table = new GateMuTable(material->GetName(), nb_e);
  
  for(int i = 0; i < nb_e; i++){
    table->PutValue(i, energies[i], Mu[i], MuEn[i]);
  }
  
  mMaterialTable.insert(std::pair<G4String, GateMuTable*>(material->GetName(),table));
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMaterialMuHandler::ReadElementFile(int z)
{
  std::ostringstream stream;
  stream << z;
  string filenameMu = "Mu-"+ stream.str() +".dat";
  string filenameMuEn = "Muen-"+ stream.str() +".dat";
  
  std::ifstream fileMu, fileMuEn;
  fileMu.open(filenameMu.c_str());
  fileMuEn.open(filenameMuEn.c_str());
  int nblines;
  fileMu >> nblines;
  fileMuEn >> nblines;
  GateMuTable* table = new GateMuTable(string(), nblines);
  mElementsTable[z] = table;
  for(int j = 0; j < nblines; j++){
    double e, mu, muen;
    fileMu >> e >> mu;
    fileMuEn >> e >> muen;
    table->PutValue(j, e, mu, muen);
  }
  fileMu.close();
  fileMuEn.close();  
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMaterialMuHandler::InitElementTable()
{
  for(int i = 1; i <= mNbOfElements; i++)
    ReadElementFile(i);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateMaterialMuHandler::GetAttenuation(G4Material* material, double energy)
{
  if(!isInitialized) {
    InitMaterialTable();
  }
  
  return mMaterialTable[material->GetName()]->GetMuEn(energy);
  
//   map<G4String, GateMuTable*>::iterator it = mMaterialTable.find(material->GetName());
//   if(it == mMaterialTable.end()){
//     AddMaterial(material);
//   }
//   
//   return mMaterialTable[material->GetName()]->GetMuEn(energy);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateMaterialMuHandler::GetMu(G4Material* material, double energy)
{
  if(!isInitialized) {
    InitMaterialTable();
  }

  return mMaterialTable[material->GetName()]->GetMu(energy);
  
//   map<G4String, GateMuTable*>::iterator it = mMaterialTable.find(material->GetName());
//   if(it == mMaterialTable.end()){
//     AddMaterial(material);
//   }
//   return mMaterialTable[material->GetName()]->GetMu(energy);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMaterialMuHandler::InitMaterialTable()
{
  // Get process list for gamma
  G4ProcessVector *processListForGamma = G4Gamma::Gamma()->GetProcessManager()->GetProcessList();
  G4ParticleDefinition *gamma = G4Gamma::Gamma();
  
  // Find photoelectric, compton (+ particleChange) and rayleigh models
  G4VEmModel *photoElectricModel = 0;
  G4VEmModel *comptonModel = 0;
  G4VEmModel *rayleighModel = 0;
  G4ParticleChangeForGamma *particleChangeCompton = 0;
  
  for(int i=0; i<processListForGamma->size(); i++)
  {
    G4String processName = (*processListForGamma)[i]->GetProcessName();
    if(processName == "PhotoElectric" || processName == "phot") {
      photoElectricModel = (dynamic_cast<G4VEmProcess *>((*processListForGamma)[i]))->Model(1);
    }
    else if(processName == "Compton" || processName == "compt") {
      G4VEmProcess *comptonProcess = dynamic_cast<G4VEmProcess *>((*processListForGamma)[i]);
      comptonModel = comptonProcess->Model(1);
      
      // Get the G4VParticleChange of comptonModel by running a fictive step (no simple 'get' function available)
      G4Track myTrack(new G4DynamicParticle(gamma,G4ThreeVector(1.,0.,0.),0.01),0.,G4ThreeVector(0.,0.,0.));
      myTrack.SetTrackStatus(fStopButAlive); // to get a fast return (see G4VEmProcess::PostStepDoIt(...))
      G4Step myStep;
      particleChangeCompton = dynamic_cast<G4ParticleChangeForGamma *>(comptonProcess->PostStepDoIt((const G4Track)(myTrack), myStep));
    }
    else if(processName == "RayleighScattering" || processName == "Rayl") {
      rayleighModel = (dynamic_cast<G4VEmProcess *>((*processListForGamma)[i]))->Model(1);
    } 
  }
  
//   DD(photoElectricModel->GetName());
//   DD(comptonModel->GetName());

  // Useful members for the loops
  // - cuts and materials
  G4ProductionCutsTable *productionCutList = G4ProductionCutsTable::GetProductionCutsTable();
  G4String materialName;
  
  // - particles
  G4DynamicParticle primary(gamma,G4ThreeVector(1.,0.,0.));
  std::vector<G4DynamicParticle *> secondaries; 
  double incidentEnergy;
  double deltaEnergy = 50.0 * eV;
  
  // - (mu ; muen) calculations 
  map<G4String, GateMuTable*>::iterator it;
  double totalFluoEnergyPhotoElectric;
  double totalFluoEnergyCompton;
  double totalScatterEnergyCompton;
  double crossSectionPhotoElectric;
  double crossSectionCompton;
  double crossSectionRayleigh;
  double fPhotoElectric;
  double fCompton;
  double mu;
  double muen;
  
  // - loops options
  double energyMin = 250. * eV;
  double energyMax = 1. * MeV;
  int energyNumber = 25;
  double energyStep = exp( (log(energyMax) - log(energyMin)) / double(energyNumber) );
  std::vector<double> energyList;

  int shotNumber = 10000;
  
  // Loop on material
  for(unsigned int m=0; m<productionCutList->GetTableSize(); m++)
  {
    const G4MaterialCutsCouple *couple = productionCutList->GetMaterialCutsCouple(m);
    const G4Material *material = couple->GetMaterial();
    materialName = material->GetName();
    it = mMaterialTable.find(materialName);
    if(it == mMaterialTable.end())
    {
      GateMessage("MuHandler",0,"Construction of material : " << materialName << G4endl);

      // construct energyList
      energyList.clear();

      // - basic list
      energyList.push_back(energyMin);
      for(int e = 0; e<energyNumber; e++) { energyList.push_back(energyList[e] * energyStep); }
      
      // - add atomic shell energies
      int elementNumber = material->GetNumberOfElements();
      for(int i = 0; i < elementNumber; i++)
      {
	const G4Element *element = material->GetElement(i);
	for(int j=0; j<material->GetElement(i)->GetNbOfAtomicShells(); j++)
	{
	  double atomicShellEnergy = element->GetAtomicShell(j);
	  if(atomicShellEnergy > energyMin)
	  {
	    energyList.push_back(atomicShellEnergy - deltaEnergy);
	    energyList.push_back(atomicShellEnergy + deltaEnergy);
	  }
	}
      }
      std::sort(energyList.begin(), energyList.end());
      

//       for(unsigned int e = 0; e<energyList.size(); e++)
//       {
// 	DD(energyList[e]);
//       }
      
      
      GateMuTable *table = new GateMuTable(materialName, energyList.size());
      
      
      
      // Loop on energy
      for(unsigned int e=0; e<energyList.size(); e++)
      {
// 	GateMessage("MuHandler",0,"  energy = " << e*energyStep << " MeV" << G4endl);

	incidentEnergy = energyList[e];
	primary.SetKineticEnergy(incidentEnergy);

	crossSectionPhotoElectric = 0.;
	crossSectionCompton = 0.;
	crossSectionRayleigh = 0.;
	totalFluoEnergyPhotoElectric = 0.;
	totalFluoEnergyCompton = 0.;
	totalScatterEnergyCompton = 0.;
	
	// Loop on shot
	for(int i=0; i<shotNumber; i++)
	{
	  double trialFluoEnergy;
	  
	  // photoElectric shots to get the mean fluorescence photon energy
	  if(photoElectricModel)
	  {
	    secondaries.clear(); 
	    photoElectricModel->SampleSecondaries(&secondaries,couple,&primary,0.,0.);
// 	    GateMessage("MuHandler",0,"    shot " << i+1 << " composed with " << secondaries.size() << " particle(s)" << G4endl);
	    trialFluoEnergy = 0.;
	    for(unsigned s=0; s<secondaries.size(); s++) {
// 	      GateMessage("MuHandler",0,"      " << secondaries[s]->GetParticleDefinition()->GetParticleName() << " of " << secondaries[s]->GetKineticEnergy() << " MeV" << G4endl);
	      if(secondaries[s]->GetParticleDefinition()->GetParticleName() == "gamma") { trialFluoEnergy += secondaries[s]->GetKineticEnergy(); }
	    }
	    totalFluoEnergyPhotoElectric += trialFluoEnergy;
// 	    GateMessage("MuHandler",0,"      meanFluoEnergy = " << (trialFluoEnergyPhotoElectric / (double)fluoGammaNumber) << " MeV" << G4endl);
	  }

	  // compton shots to get the mean fluorescence and scatter photon energy
	  if(comptonModel)
	  {
	    secondaries.clear(); 
	    comptonModel->SampleSecondaries(&secondaries,couple,&primary,0.,0.);
// 	    GateMessage("MuHandler",0,"    shot " << i+1 << " composed with " << secondaries.size() << " particle(s)" << G4endl);
	    trialFluoEnergy = 0.;
	    for(unsigned s=0; s<secondaries.size(); s++) {
// 	      GateMessage("MuHandler",0,"      " << secondaries[s]->GetParticleDefinition()->GetParticleName() << " of " << secondaries[s]->GetKineticEnergy() << " MeV" << G4endl);
	      if(secondaries[s]->GetParticleDefinition()->GetParticleName() == "gamma") { trialFluoEnergy += secondaries[s]->GetKineticEnergy(); }
	    }
	    totalFluoEnergyCompton += trialFluoEnergy;
	    totalScatterEnergyCompton += particleChangeCompton->GetProposedKineticEnergy();
	    
// 	    GateMessage("MuHandler",0,"      incidentEnergy = " << incidentEnergy << " scatterEnergy = " << particleChangeCompton->GetProposedKineticEnergy() << G4endl);
	  }
	}
	
	// Mean energy photon calculation
	totalFluoEnergyPhotoElectric = totalFluoEnergyPhotoElectric / (double)shotNumber;
	totalFluoEnergyCompton = totalFluoEnergyCompton / (double)shotNumber;
	totalScatterEnergyCompton = totalScatterEnergyCompton / (double)shotNumber;
	
	// Cross section calculation
	double density = material->GetDensity() / (g/cm3);
	double energyCutForGamma = productionCutList->ConvertRangeToEnergy(gamma,material,couple->GetProductionCuts()->GetProductionCut("gamma"));
	if(photoElectricModel) {
	  crossSectionPhotoElectric = photoElectricModel->CrossSectionPerVolume(material,gamma,incidentEnergy,energyCutForGamma,10.) * cm / density;
	}
	if(comptonModel) {
	  crossSectionCompton = comptonModel->CrossSectionPerVolume(material,gamma,incidentEnergy,energyCutForGamma,10.) * cm / density;
	}
	if(rayleighModel) {
	  crossSectionRayleigh = rayleighModel->CrossSectionPerVolume(material,gamma,incidentEnergy,energyCutForGamma,10.) * cm / density;
	}

	// average fractions of the incident energy E that is transferred to kinetic energy of charged particles (for muen)
	fPhotoElectric = 1. - (totalFluoEnergyPhotoElectric/incidentEnergy);
	fCompton = 1. - ((totalScatterEnergyCompton + totalFluoEnergyCompton) / incidentEnergy);
	
	// mu/rho and muen/rho calculation
	mu   = crossSectionPhotoElectric + crossSectionCompton + crossSectionRayleigh;
	muen = fPhotoElectric * crossSectionPhotoElectric + fCompton * crossSectionCompton;

// 	GateMessage("MuHandler",0,"    csPE = " << crossSectionPhotoElectric    << "   csCo = " << crossSectionCompton << " csRa = " << crossSectionRayleigh << " cm2.g-1" << G4endl);
// 	GateMessage("MuHandler",0,"  fluoPE = " << totalFluoEnergyPhotoElectric << " fluoCo = " << totalFluoEnergyCompton << " scCo = " << totalScatterEnergyCompton << " MeV" << G4endl);
// 	GateMessage("MuHandler",0,"     fPE = " << fPhotoElectric               << "    fCo = " << fCompton << G4endl);
// 	GateMessage("MuHandler",0,"     cut = " << energyCutForGamma            << G4endl);
// 	GateMessage("MuHandler",0,"      mu = " << mu << " muen = " << muen << " cm2.g-1" << G4endl);
	GateMessage("MuHandler",0," " << incidentEnergy << " " << mu << " " << muen << " " << G4endl);

	table->PutValue(e, incidentEnergy, mu, muen);

// 	GateMessage("MuHandler",0," " << G4endl);
      }

//       GateMessage("MuHandler",0," -------------------------------------------------------- " << G4endl);
//       GateMessage("MuHandler",0," " << G4endl);
      
      mMaterialTable.insert(std::pair<G4String, GateMuTable*>(materialName,table));
    }
  }
  
  isInitialized = true;  
}

#endif
