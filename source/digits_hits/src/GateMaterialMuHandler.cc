#ifndef GATEMATERIALMUHANDLER_CC
#include "GateMaterialMuHandler.hh"
#include "GateMuTables.hh"
#include "GateMiscFunctions.hh"
#include "GateConfiguration.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>

using std::map;
using std::string;

//-----------------------------------------------------------------------------
GateMaterialMuHandler *GateMaterialMuHandler::singleton_MaterialMuHandler = 0;
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateMaterialMuHandler::GateMaterialMuHandler()
{
  mIsInitialized = false;  
  mNbOfElements = -1;
  mElementsTable = 0;
  mElementsFolderName = "NULL";
  mEnergyMin = 250. * eV;
  mEnergyMax = 1. * MeV;
  mEnergyNumber = 25;
  mAtomicShellEnergyMin = 1. * keV;
  mPrecision = 0.01;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateMaterialMuHandler::~GateMaterialMuHandler()
{
  if(mElementsTable) delete [] mElementsTable;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateMaterialMuHandler::GetAttenuation(G4Material* material, double energy)
{
  if(!mIsInitialized) { Initialize(); }  
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
  if(!mIsInitialized) { Initialize(); }
  return mMaterialTable[material->GetName()]->GetMu(energy);
  
//   map<G4String, GateMuTable*>::iterator it = mMaterialTable.find(material->GetName());
//   if(it == mMaterialTable.end()){
//     AddMaterial(material);
//   }
//   return mMaterialTable[material->GetName()]->GetMu(energy);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateMuTable *GateMaterialMuHandler::GetMuTable(G4Material *material)
{
  if(!mIsInitialized) { Initialize(); }
  return mMaterialTable[material->GetName()];
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
inline double interpolation(double Xa,double Xb,double Ya,double Yb,double x){
  return exp(log(Ya) + log(Yb/Ya) / log(Xb/Xa)* log(x/Xa) );
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMaterialMuHandler::Initialize()
{
  if(mElementsFolderName == "NULL")
  {
//     DD("Simulation");
    SimulateMaterialTable();
  }
  else
  {
//     DD("Precalculated");
    mNbOfElements = 100;
    mElementsTable = new GateMuTable*[mNbOfElements+1];
    InitElementTable();
    
    G4ProductionCutsTable *productionCutList = G4ProductionCutsTable::GetProductionCutsTable();
    G4String materialName;
    map<G4String, GateMuTable*>::iterator it;
    
    for(unsigned int m=0; m<productionCutList->GetTableSize(); m++)
    {
      const G4Material *material = productionCutList->GetMaterialCutsCouple(m)->GetMaterial();
      materialName = material->GetName();
      it = mMaterialTable.find(materialName);
      if(it == mMaterialTable.end()) { ConstructMaterial(material); }
    }
  }

  mIsInitialized = true;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMaterialMuHandler::ConstructMaterial(const G4Material *material)
{
  GateMessage("Physic",1,"Construction of mu/mu_en table for material : " << material->GetName() << G4endl);

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
//     GateMessage("Physic",2," " << energies[i] << " " << Mu[i] << " " << MuEn[i] << " " << G4endl);
    table->PutValue(i, log(energies[i]), log(Mu[i]), log(MuEn[i]));
  }
  
  mMaterialTable.insert(std::pair<G4String, GateMuTable*>(material->GetName(),table));

  delete [] energies;
  delete [] index;
  delete [] e_tables;
  delete [] mu_tables;
  delete [] muen_tables;
  delete [] MuEn;
  delete [] Mu;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMaterialMuHandler::ReadElementFile(int z)
{
  std::ostringstream stream;
  stream << z;
  string filenameMu = mElementsFolderName+"/Mu-"+ stream.str() +".dat";
  string filenameMuEn = mElementsFolderName+"/Muen-"+ stream.str() +".dat";
  
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
void GateMaterialMuHandler::SimulateMaterialTable()
{
  // Get process list for gamma
  G4ProcessVector *processListForGamma = G4Gamma::Gamma()->GetProcessManager()->GetProcessList();
  G4ParticleDefinition *gamma = G4Gamma::Gamma();
  // Check if fluo is active
  bool isFluoActive = false;
  if(G4LossTableManager::Instance()->AtomDeexcitation()) { isFluoActive = G4LossTableManager::Instance()->AtomDeexcitation()->IsFluoActive();}
  
  // Find photoelectric (PE), compton scattering (CS) (+ particleChange) and rayleigh scattering (RS) models
  G4VEmModel *modelPE = 0;
  G4VEmModel *modelCS = 0;
  G4VEmModel *modelRS = 0;
  G4ParticleChangeForGamma *particleChangeCS = 0;
  
  for(int i=0; i<processListForGamma->size(); i++)
  {
    G4String processName = (*processListForGamma)[i]->GetProcessName();
    if(processName == "PhotoElectric" || processName == "phot") {
      #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))
         modelPE = (dynamic_cast<G4VEmProcess *>((*processListForGamma)[i]))->EmModel(1); 
      #else
         modelPE = (dynamic_cast<G4VEmProcess *>((*processListForGamma)[i]))->Model(1);
      #endif
    }
    else if(processName == "Compton" || processName == "compt") {
      G4VEmProcess *processCS = dynamic_cast<G4VEmProcess *>((*processListForGamma)[i]);
      #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))
         modelCS = processCS->EmModel(1); 
      #else
         modelCS = processCS->Model(1);
      #endif
      
      // Get the G4VParticleChange of compton scattering by running a fictive step (no simple 'get' function available)
      G4Track myTrack(new G4DynamicParticle(gamma,G4ThreeVector(1.,0.,0.),0.01),0.,G4ThreeVector(0.,0.,0.));
      myTrack.SetTrackStatus(fStopButAlive); // to get a fast return (see G4VEmProcess::PostStepDoIt(...))
      G4Step myStep;
      particleChangeCS = dynamic_cast<G4ParticleChangeForGamma *>(processCS->PostStepDoIt((const G4Track)(myTrack), myStep));
    }
    else if(processName == "RayleighScattering" || processName == "Rayl") {
      #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))
         modelRS = (dynamic_cast<G4VEmProcess *>((*processListForGamma)[i]))->EmModel(1); 
      #else
         modelRS = (dynamic_cast<G4VEmProcess *>((*processListForGamma)[i]))->Model(1);
      #endif
    } 
  }
  
  // Useful members for the loops
  // - cuts and materials
  G4ProductionCutsTable *productionCutList = G4ProductionCutsTable::GetProductionCutsTable();
  G4String materialName;
  
  // - particles
  G4DynamicParticle primary(gamma,G4ThreeVector(1.,0.,0.));
  std::vector<G4DynamicParticle *> secondaries; 
  double incidentEnergy;
  
  // - (mu ; muen) calculations 
  map<G4String, GateMuTable*>::iterator it;
  double totalFluoPE;
  double totalFluoCS;
  double totalScatterCS;
  double crossSectionPE;
  double crossSectionCS;
  double crossSectionRS;
  double fPE;
  double fCS;
  double mu(0.);
  double muen(0.);
  
  // - muen uncertainty
  double squaredFluoPE;    // sum of squared PE fluorescence energy measurement
  double squaredFluoCS;    // sum of squared CS fluorescence energy measurement
  double squaredScatterCS; // sum of squared CS scattered photon energy measurement
  int shotNumberPE;
  int shotNumberCS;
  double squaredSigmaPE;        // squared PE uncertainty weighted by corresponding squared cross section
  double squaredSigmaCS;        // squared CS uncertainty weighted by corresponding squared cross section
  double squaredSigmaMuen(0.);  // squared muen uncertainty
  
  // - loops options
  std::vector<MuStorageStruct> muStorage;

  // Loop on material
  for(unsigned int m=0; m<productionCutList->GetTableSize(); m++)
  {
    const G4MaterialCutsCouple *couple = productionCutList->GetMaterialCutsCouple(m);
    const G4Material *material = couple->GetMaterial();
    materialName = material->GetName();
    it = mMaterialTable.find(materialName);
    if(it == mMaterialTable.end())
    {
      GateMessage("Physic",1,"Construction of mu/mu_en table for material : " << material->GetName() << G4endl);

      // Construc energy list (energy, atomicShellEnergy)
      ConstructEnergyList(&muStorage,material);
            
      // Loop on energy
      for(unsigned int e=0; e<muStorage.size(); e++)
      {
// 	GateMessage("Physic",2,"  energy = " << e*energyStep << " MeV" << G4endl);

	incidentEnergy = muStorage[e].energy;
	primary.SetKineticEnergy(incidentEnergy);

	// Cross section calculation
	double density = material->GetDensity() / (g/cm3);
	double energyCutForGamma = productionCutList->ConvertRangeToEnergy(gamma,material,couple->GetProductionCuts()->GetProductionCut("gamma"));
	crossSectionPE = 0.;
	crossSectionCS = 0.;
	crossSectionRS = 0.;
	if(modelPE) { crossSectionPE = modelPE->CrossSectionPerVolume(material,gamma,incidentEnergy,energyCutForGamma,10.) * cm / density; }
	if(modelCS) { crossSectionCS = modelCS->CrossSectionPerVolume(material,gamma,incidentEnergy,energyCutForGamma,10.) * cm / density; }
	if(modelRS) { crossSectionRS = modelRS->CrossSectionPerVolume(material,gamma,incidentEnergy,energyCutForGamma,10.) * cm / density; }
	
	// muen and uncertainty calculation
	squaredFluoPE = 0.;
	squaredFluoCS = 0.;
	squaredScatterCS = 0.;
	totalFluoPE = 0.;
	totalFluoCS = 0.;
	totalScatterCS = 0.;
	shotNumberPE = 0;
	shotNumberCS = 0;
	squaredSigmaPE = 0.;
	squaredSigmaCS = 0.;
	fPE = 1.;
	fCS = 1.;
	double trialFluoEnergy;
	double precision = 10e6;

	int variableShotNumberPE = 0;
	if(modelPE and isFluoActive) { variableShotNumberPE = 50; }

	int variableShotNumberCS = 0;
	if(modelCS) { variableShotNumberCS = 100 - variableShotNumberPE; }
	
	// Loop on shot
	while(precision > mPrecision)
	{
	  // photoElectric shots to get the mean fluorescence photon energy
	  for(int iPE = 0; iPE<variableShotNumberPE; iPE++)
	  {
	    trialFluoEnergy = ProcessOneShot(modelPE,&secondaries,couple,&primary);
	    shotNumberPE++;
	    
	    totalFluoPE += trialFluoEnergy;
	    squaredFluoPE += (trialFluoEnergy * trialFluoEnergy);
	  }

	  // compton shots to get the mean fluorescence and scatter photon energy
	  for(int iCS = 0; iCS<variableShotNumberCS; iCS++)
	  {
	    trialFluoEnergy = ProcessOneShot(modelCS,&secondaries,couple,&primary);
	    shotNumberCS++;

	    totalFluoCS += trialFluoEnergy;
	    squaredFluoCS += (trialFluoEnergy * trialFluoEnergy);
	    double trialScatterEnergy = particleChangeCS->GetProposedKineticEnergy();
	    totalScatterCS += trialScatterEnergy;
	    squaredScatterCS += (trialScatterEnergy * trialScatterEnergy);
	  }
	  
	  // average fractions of the incident energy E that is transferred to kinetic energy of charged particles (for muen)
	  if(shotNumberPE) {
	    fPE = 1. - ((totalFluoPE / double(shotNumberPE)) / incidentEnergy);
	    squaredSigmaPE = SquaredSigmaOnMean(squaredFluoPE,totalFluoPE,shotNumberPE) * crossSectionPE * crossSectionPE;
	  }
	  if(shotNumberCS) {
	    fCS = 1. - (((totalScatterCS + totalFluoCS) / double(shotNumberCS)) / incidentEnergy);
	    squaredSigmaCS = (SquaredSigmaOnMean(squaredFluoCS,totalFluoCS,shotNumberCS) + SquaredSigmaOnMean(squaredScatterCS,totalScatterCS,shotNumberCS)) * crossSectionCS * crossSectionCS;
	  }

	  // mu/rho and muen/rho calculation
	  muen = fPE * crossSectionPE + fCS * crossSectionCS;

	  // uncertainty calculation
	  squaredSigmaMuen = (squaredSigmaPE + squaredSigmaCS) / (incidentEnergy * incidentEnergy);
	  precision = sqrt(squaredSigmaMuen) / muen;

	  if(modelPE and isFluoActive) {
	    if(squaredSigmaPE > 0) { variableShotNumberPE = (int)floor(0.5 + 100. * sqrt(squaredSigmaPE / (squaredSigmaPE + squaredSigmaCS))); }
	    else { variableShotNumberPE = 50.; }
	  }
	  if(modelCS) { variableShotNumberCS = 100 - variableShotNumberPE; }
	}
	
	mu = crossSectionPE + crossSectionCS + crossSectionRS;
	
// 	GateMessage("Physic",2,"    csPE = " << crossSectionPE << "   csCo = " << crossSectionCS << " csRa = " << crossSectionRS << " cm2.g-1" << G4endl);
// 	GateMessage("Physic",2,"  fluoPE = " << totalFluoPE    << " fluoCo = " << totalFluoCS    << " scCo = " << totalScatterCS << " MeV" << G4endl);
// 	GateMessage("Physic",2,"     fPE = " << fPE            << "    fCo = " << fCS << G4endl);
// 	GateMessage("Physic",2,"     cut = " << energyCutForGamma << "    iPE = " << shotNumberPE << " iCS = " << shotNumberCS << G4endl);
// 	GateMessage("Physic",2," " << incidentEnergy << " MeV - muen = " << muen << " +/- " << sqrt(squaredSigmaMuen) << " (" << precision * 100. << " %)" << G4endl);
// 	GateMessage("Physic",2,"   sigPE = " << sqrt(squaredSigmaPE) << "    sigCS = " << sqrt(squaredSigmaCS) << G4endl);
// 	GateMessage("Physic",2," " << incidentEnergy << " " << mu << " " << muen << " " << sqrt(squaredSigmaMuen) << " " << precision << G4endl);
// 	GateMessage("Physic",2,"   nPE = " << variableShotNumberPE << " nCS = " << variableShotNumberCS << " nPEtot = " << shotNumberPE << " nCStot = " << shotNumberCS << G4endl);

	muStorage[e].mu = mu;
	muStorage[e].muen = muen;
      }

      // Interpolation of mu,muen for energy bordering an atomic transition (see ConstructEnergyList(...))
      MergeAtomicShell(&muStorage);
      
      // Fill mu,muen table for this material
      GateMuTable *table = new GateMuTable(materialName, muStorage.size());
      for(unsigned int e=0; e<muStorage.size(); e++) {
	table->PutValue(e, log(muStorage[e].energy), log(muStorage[e].mu), log(muStorage[e].muen));
      }
      mMaterialTable.insert(std::pair<G4String, GateMuTable*>(materialName,table));      
      
//       GateMessage("Physic",2," -------------------------------------------------------- " << G4endl);
//       GateMessage("Physic",2," " << G4endl);
    }
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateMaterialMuHandler::ProcessOneShot(G4VEmModel *model,std::vector<G4DynamicParticle*> *secondaries, const G4MaterialCutsCouple *couple, const G4DynamicParticle *primary)
{
  secondaries->clear();
  model->SampleSecondaries(secondaries,couple,primary,0.,0.);
  double energy = 0.;
  for(unsigned s=0; s<secondaries->size(); s++) {
    if((*secondaries)[s]->GetParticleDefinition()->GetParticleName() == "gamma") { energy += (*secondaries)[s]->GetKineticEnergy(); }
    delete (*secondaries)[s];
  }
  
  return energy;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateMaterialMuHandler::SquaredSigmaOnMean(double sumOfSquaredMeasurement, double sumOfMeasurement, double numberOfMeasurement)
{
  sumOfMeasurement = sumOfMeasurement / numberOfMeasurement;
  return ((sumOfSquaredMeasurement / numberOfMeasurement) - (sumOfMeasurement * sumOfMeasurement)) / numberOfMeasurement;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMaterialMuHandler::ConstructEnergyList(std::vector<MuStorageStruct> *muStorage, const G4Material *material)
{
  muStorage->clear();
  double energyStep = exp( (log(mEnergyMax) - log(mEnergyMin)) / double(mEnergyNumber) );
  double deltaEnergy = 50.0 * eV;
  
  // - basic list
  muStorage->push_back(MuStorageStruct(mEnergyMin,0,0.));
  for(int e = 0; e<mEnergyNumber; e++) { muStorage->push_back(MuStorageStruct((*muStorage)[e].energy*energyStep,0,0.)); }

  // - add atomic shell energies
  int elementNumber = material->GetNumberOfElements();
  for(int i = 0; i < elementNumber; i++)
  {
    const G4Element *element = material->GetElement(i);
    for(int j=0; j<material->GetElement(i)->GetNbOfAtomicShells(); j++)
    {
      double atomicShellEnergy = element->GetAtomicShell(j);
      if(atomicShellEnergy > mAtomicShellEnergyMin && atomicShellEnergy > mEnergyMin)
      {
	muStorage->push_back(MuStorageStruct(atomicShellEnergy-deltaEnergy,-1,atomicShellEnergy)); // inf
	muStorage->push_back(MuStorageStruct(atomicShellEnergy+deltaEnergy,+1,atomicShellEnergy)); // sup
      }
    }
  }
  std::sort(muStorage->begin(), muStorage->end());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMaterialMuHandler::MergeAtomicShell(std::vector<MuStorageStruct> *muStorage)
{
  for(unsigned int e=0; e<muStorage->size(); e++)
  {
    int isAtomicShell = (*muStorage)[e].isAtomicShell;    
    if(isAtomicShell != 0)
    {
      int neighbourIndex = e + isAtomicShell;
      if((neighbourIndex > -1) and (neighbourIndex < (int)muStorage->size()))
      {
	double mu = interpolation((*muStorage)[neighbourIndex].energy,
				  (*muStorage)[e].energy,
				  (*muStorage)[neighbourIndex].mu,
				  (*muStorage)[e].mu,
				  (*muStorage)[e].atomicShellEnergy);

	double muen = interpolation((*muStorage)[neighbourIndex].energy,
				    (*muStorage)[e].energy,
				    (*muStorage)[neighbourIndex].muen,
				    (*muStorage)[e].muen,
				    (*muStorage)[e].atomicShellEnergy);

	(*muStorage)[e].mu = mu;
	(*muStorage)[e].muen = muen;
      }
      (*muStorage)[e].energy = (*muStorage)[e].atomicShellEnergy;
    }
    
//     GateMessage("Physic",2," " << (*muStorage)[e].energy << " " << (*muStorage)[e].mu << " " << (*muStorage)[e].muen << G4endl);
  }
}
//-----------------------------------------------------------------------------

#endif
