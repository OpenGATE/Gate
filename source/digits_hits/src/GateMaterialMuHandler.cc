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
#include "G4LossTableManager.hh"
#include "GatePhysicsList.hh"

using std::map;
using std::string;

//-----------------------------------------------------------------------------
GateMaterialMuHandler *GateMaterialMuHandler::singleton_MaterialMuHandler = 0;
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateMaterialMuHandler::GateMaterialMuHandler()
{
  mNbOfElements = 100;
  mElementsTable = new GateMuTable*[mNbOfElements+1];
//   InitElementTable();
  
  mIsInitialized = false;  
  mNbOfElements = -1;
  mElementsTable = 0;
  mElementsFolderName = "NULL";
  mEnergyMin = 250. * eV;
  mEnergyMax = 1. * MeV;
  mEnergyNumber = 25;
  mAtomicShellEnergyMin = 1. * keV;
  mShotNumber = 10000;
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
inline double interpolation(double Xa,double Xb,double Ya,double Yb,double x){
  return exp(log(Ya) + log(Yb/Ya) / log(Xb/Xa)* log(x/Xa) );
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMaterialMuHandler::Initialize()
{
  DD(mElementsFolderName);
  if(mElementsFolderName == "NULL")
  {
    DD("Simulation");
    SimulateMaterialTable();
  }
  else
  {
    DD("Precalculated");
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
  GateMessage("MuHandler",0,"Construction of material : " << material->GetName() << G4endl);

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
    GateMessage("MuHandler",0," " << energies[i] << " " << Mu[i] << " " << MuEn[i] << " " << G4endl);
    table->PutValue(i, log(energies[i]), log(Mu[i]), log(MuEn[i]));
  }
  
  mMaterialTable.insert(std::pair<G4String, GateMuTable*>(material->GetName(),table));
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
      modelPE = (dynamic_cast<G4VEmProcess *>((*processListForGamma)[i]))->Model(1);
    }
    else if(processName == "Compton" || processName == "compt") {
      G4VEmProcess *processCS = dynamic_cast<G4VEmProcess *>((*processListForGamma)[i]);
      modelCS = processCS->Model(1);
      
      // Get the G4VParticleChange of compton scattering by running a fictive step (no simple 'get' function available)
      G4Track myTrack(new G4DynamicParticle(gamma,G4ThreeVector(1.,0.,0.),0.01),0.,G4ThreeVector(0.,0.,0.));
      myTrack.SetTrackStatus(fStopButAlive); // to get a fast return (see G4VEmProcess::PostStepDoIt(...))
      G4Step myStep;
      particleChangeCS = dynamic_cast<G4ParticleChangeForGamma *>(processCS->PostStepDoIt((const G4Track)(myTrack), myStep));
    }
    else if(processName == "RayleighScattering" || processName == "Rayl") {
      modelRS = (dynamic_cast<G4VEmProcess *>((*processListForGamma)[i]))->Model(1);
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
  double deltaEnergy = 50.0 * eV;
  
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
  double mu;
  double muen;
  
  // - muen uncertainty
  double squaredFluoPE;    // sum of squared PE fluorescence energy measurement
  double squaredFluoCS;    // sum of squared CS fluorescence energy measurement
  double squaredScatterCS; // sum of squared CS scattered photon energy measurement
  double meanFluoPE;       // PE mean fluorescence energy
  double meanFluoCS;       // CS mean fluorescence energy
  double meanScatterCS;    // CS mean scattered photon energy
  double squaredSigmaFluoPE;    // squared mean PE fluorescence energy uncertainty
  double squaredSigmaFluoCS;    // squared mean CS fluorescence energy uncertainty
  double squaredSigmaScatterCS; // squared mean CS scattered photon energy uncertainty
  double squaredSigmaPE;        // squared PE uncertainty weighted by corresponding squared cross section
  double squaredSigmaCS;        // squared CS uncertainty weighted by corresponding squared cross section
  double squaredSigmaMuen;      // squared muen uncertainty
  
  
  // - loops options
  double energyStep = exp( (log(mEnergyMax) - log(mEnergyMin)) / double(mEnergyNumber) );
  std::vector<double> energyList;

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
      energyList.push_back(mEnergyMin);
      for(int e = 0; e<mEnergyNumber; e++) { energyList.push_back(energyList[e] * energyStep); }
      
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
	    energyList.push_back(atomicShellEnergy - deltaEnergy);
	    energyList.push_back(atomicShellEnergy + deltaEnergy);
	  }
	}
      }
      std::sort(energyList.begin(), energyList.end());
      
      GateMuTable *table = new GateMuTable(materialName, energyList.size());
            
      // Loop on energy
      for(unsigned int e=0; e<energyList.size(); e++)
      {
// 	GateMessage("MuHandler",0,"  energy = " << e*energyStep << " MeV" << G4endl);

	incidentEnergy = energyList[e];
	primary.SetKineticEnergy(incidentEnergy);

	// Cross section calculation
	double density = material->GetDensity() / (g/cm3);
	double energyCutForGamma = productionCutList->ConvertRangeToEnergy(gamma,material,couple->GetProductionCuts()->GetProductionCut("gamma"));
	crossSectionPE = 0.;
	crossSectionCS = 0.;
	crossSectionRS = 0.;
	if(modelPE) {
	  crossSectionPE = modelPE->CrossSectionPerVolume(material,gamma,incidentEnergy,energyCutForGamma,10.) * cm / density;
	}
	if(modelCS) {
	  crossSectionCS = modelCS->CrossSectionPerVolume(material,gamma,incidentEnergy,energyCutForGamma,10.) * cm / density;
	}
	if(modelRS) {
	  crossSectionRS = modelRS->CrossSectionPerVolume(material,gamma,incidentEnergy,energyCutForGamma,10.) * cm / density;
	}
	
	// uncertainty calculation
	squaredFluoPE = 0.;
	squaredFluoCS = 0.;
	squaredScatterCS = 0.;
	meanFluoPE = 0.;
	meanFluoCS = 0.;
	meanScatterCS = 0.;

	// muen calculation
	totalFluoPE = 0.;
	totalFluoCS = 0.;
	totalScatterCS = 0.;

	// Loop on shot
	for(int i=0; i<mShotNumber; i++)
	{
	  double trialFluoEnergy;
	  
	  // photoElectric shots to get the mean fluorescence photon energy
	  if(modelPE and isFluoActive)
	  {
	    secondaries.clear(); 
	    modelPE->SampleSecondaries(&secondaries,couple,&primary,0.,0.);
// 	    GateMessage("MuHandler",0,"    shot " << i+1 << " composed with " << secondaries.size() << " particle(s)" << G4endl);
	    trialFluoEnergy = 0.;
	    for(unsigned s=0; s<secondaries.size(); s++) {
// 	      GateMessage("MuHandler",0,"      " << secondaries[s]->GetParticleDefinition()->GetParticleName() << " of " << secondaries[s]->GetKineticEnergy() << " MeV" << G4endl);
	      if(secondaries[s]->GetParticleDefinition()->GetParticleName() == "gamma") { trialFluoEnergy += secondaries[s]->GetKineticEnergy(); }
	    }
	    totalFluoPE += trialFluoEnergy;

	    squaredFluoPE += (trialFluoEnergy * trialFluoEnergy);
	    meanFluoPE = totalFluoPE / double(i+1);
	    
// 	    GateMessage("MuHandler",0,"      meanFluoEnergy = " << (trialFluoEnergyPhotoElectric / (double)fluoGammaNumber) << " MeV" << G4endl);
	  }

	  // compton shots to get the mean fluorescence and scatter photon energy
	  if(modelCS)
	  {
	    secondaries.clear(); 
	    modelCS->SampleSecondaries(&secondaries,couple,&primary,0.,0.);
// 	    GateMessage("MuHandler",0,"    shot " << i+1 << " composed with " << secondaries.size() << " particle(s)" << G4endl);

	    trialFluoEnergy = 0.;
	    if(isFluoActive)
	    {
	      for(unsigned s=0; s<secondaries.size(); s++) {
// 		GateMessage("MuHandler",0,"      " << secondaries[s]->GetParticleDefinition()->GetParticleName() << " of " << secondaries[s]->GetKineticEnergy() << " MeV" << G4endl);
		if(secondaries[s]->GetParticleDefinition()->GetParticleName() == "gamma") { trialFluoEnergy += secondaries[s]->GetKineticEnergy(); }
	      }
	      totalFluoCS += trialFluoEnergy;
	    }
	    double trialScatterEnergy = particleChangeCS->GetProposedKineticEnergy();
	    totalScatterCS += trialScatterEnergy;
	    
	    squaredFluoCS += (trialFluoEnergy * trialFluoEnergy);
	    meanFluoCS = totalFluoCS / double(i+1);
	    squaredScatterCS += (trialScatterEnergy * trialScatterEnergy);
	    meanScatterCS = totalScatterCS / double(i+1);
	  }

// 	  double squaredSigmaFluoPE = (squaredFluoPE / double(i+1)) - (meanFluoPE * meanFluoPE);
// 	  double squaredSigmaFluoCS = (squaredFluoCS / double(i+1)) - (meanFluoCS * meanFluoCS);
// 	  double squaredSigmaScatterCS = (squaredScatterCS / double(i+1)) - (meanScatterCS * meanScatterCS);
// 	  
// 	  double squaredSigmaPE = squaredSigmaFluoPE * crossSectionPE * crossSectionPE;
// 	  double squaredSigmaCS = (squaredFluoCS + squaredScatterCS) * crossSectionCS * crossSectionCS;
// 
// 	  if(i%100 == 0) GateMessage("MuHandler",0,"   sigPE = " << sqrt(squaredSigmaPE / double(i+1)) << "    sigCS = " << sqrt(squaredSigmaCS / double(i+1)) << G4endl);
	}
	
	// Mean energy photon calculation
	totalFluoPE = totalFluoPE / (double)mShotNumber;
	totalFluoCS = totalFluoCS / (double)mShotNumber;
	totalScatterCS = totalScatterCS / (double)mShotNumber;

	// average fractions of the incident energy E that is transferred to kinetic energy of charged particles (for muen)
	fPE = 1. - (totalFluoPE / incidentEnergy);
	fCS = 1. - ((totalScatterCS + totalFluoCS) / incidentEnergy);
	
	// mu/rho and muen/rho calculation
	mu   = crossSectionPE + crossSectionCS + crossSectionRS;
	muen = fPE * crossSectionPE + fCS * crossSectionCS;

	// uncertainty calculation
	squaredSigmaFluoPE = ((squaredFluoPE / double(mShotNumber)) - (meanFluoPE * meanFluoPE))  / double(mShotNumber);
	squaredSigmaFluoCS = ((squaredFluoCS / double(mShotNumber)) - (meanFluoCS * meanFluoCS))  / double(mShotNumber);
	squaredSigmaScatterCS = ((squaredScatterCS / double(mShotNumber)) - (meanScatterCS * meanScatterCS)) / double(mShotNumber);
	
	squaredSigmaPE = squaredSigmaFluoPE * crossSectionPE * crossSectionPE;
	squaredSigmaCS = (squaredSigmaFluoCS + squaredSigmaScatterCS) * crossSectionCS * crossSectionCS;
	
	squaredSigmaMuen = (squaredSigmaPE + squaredSigmaCS) / (incidentEnergy * incidentEnergy);
	
// 	GateMessage("MuHandler",0,"    csPE = " << crossSectionPE << "   csCo = " << crossSectionCS << " csRa = " << crossSectionRS << " cm2.g-1" << G4endl);
// 	GateMessage("MuHandler",0,"  fluoPE = " << totalFluoPE    << " fluoCo = " << totalFluoCS    << " scCo = " << totalScatterCS << " MeV" << G4endl);
// 	GateMessage("MuHandler",0,"     fPE = " << fPE            << "    fCo = " << fCS << G4endl);
// 	GateMessage("MuHandler",0,"     cut = " << energyCutForGamma  << G4endl);
// 	GateMessage("MuHandler",0,"  sigFPE = " << sqrt(squaredSigmaFluoPE) << "   sigFCS = " << sqrt(squaredFluoCS) << " sigSCS = " << sqrt(squaredScatterCS) << G4endl);
// 	GateMessage("MuHandler",0,"   sigPE = " << sqrt(squaredSigmaPE) << "    sigCS = " << sqrt(squaredSigmaCS) << G4endl);
// 	GateMessage("MuHandler",0,"    muen = " << muen << " +/- " << sqrt(squaredSigmaMuen) << " (" << sqrt(squaredSigmaMuen) * 100. / muen << " %)" << G4endl);
// 	DD(isFluoActive);
	GateMessage("MuHandler",0," " << incidentEnergy << " " << mu << " " << muen << " " << sqrt(squaredSigmaMuen) << " " << sqrt(squaredSigmaMuen) * 100. / muen << G4endl);
	table->PutValue(e, log(incidentEnergy), log(mu), log(muen));

// 	GateMessage("MuHandler",0," " << G4endl);
      }

//       GateMessage("MuHandler",0," -------------------------------------------------------- " << G4endl);
//       GateMessage("MuHandler",0," " << G4endl);
      
      mMaterialTable.insert(std::pair<G4String, GateMuTable*>(materialName,table));
    }
  }
}

#endif
