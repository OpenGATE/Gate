/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATEPHYSICSLIST_CC
#define GATEPHYSICSLIST_CC

#include "G4Version.hh"
#include "GatePhysicsList.hh"
#include "G4ParticleDefinition.hh"
#include "G4Hybridino.hh"
#include "G4ParticleWithCuts.hh"
#include "G4ProcessManager.hh"

#include "GatePhysicsListMessenger.hh"
#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4UserLimits.hh"
#include "G4VPhysicsConstructor.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4DNAGenericIonsManager.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4VModularPhysicsList.hh"
#include "G4ExceptionHandler.hh"
#include "G4StateManager.hh"

#include "G4EmStandardPhysics.hh"
#include "G4EmStandardPhysics_option1.hh"
#include "G4EmStandardPhysics_option2.hh"
#include "G4EmStandardPhysics_option3.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmStandardPhysicsSS.hh"
#include "G4EmStandardPhysicsGS.hh"
#include "G4EmLowEPPhysics.hh"
#include "G4EmLivermorePolarizedPhysics.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4EmPenelopePhysics.hh"
#include "G4EmDNAPhysics.hh"
#include "G4EmDNAPhysics_option1.hh"
#include "G4EmDNAPhysics_option2.hh"
#include "G4EmDNAPhysics_option3.hh"
#include "G4EmDNAPhysics_option4.hh"
#include "G4EmDNAPhysics_option5.hh"
#include "G4EmDNAPhysics_option6.hh"
#include "G4EmDNAPhysics_option7.hh"
#include "G4EmDNAPhysics_option8.hh"
#include "G4EmDNAPhysicsActivator.hh"
#include "G4LossTableManager.hh"
#include "G4UAtomicDeexcitation.hh"
#include "G4RadioactiveDecayPhysics.hh"

#include "GatePhysicsList.hh"
#include "GateUserLimits.hh"
#include "GateConfiguration.h"
#include "GatePhysicsListMessenger.hh"
#include "GateRunManager.hh"
#include "GateObjectStore.hh"
#include "GateMixedDNAPhysics.hh"

#include "G4OpticalPhoton.hh"
#include "G4OpticalPhysics.hh"

#include "GateParaPositronium.hh"
#include "GateOrthoPositronium.hh"


//-----------------------------------------------------------------------------------------
GatePhysicsList::GatePhysicsList(): G4VModularPhysicsList()
{
  // default cut value  (1.0mm)
  defaultCutValue = 1.0*mm;

  ParticleCutType worldCuts;
  worldCuts.gammaCut = -1;
  worldCuts.gammaCutDisabledByDefault = false;
  worldCuts.electronCut = -1;
  worldCuts.electronCutDisabledByDefault = false;
  worldCuts.positronCut = -1;
  worldCuts.positronCutDisabledByDefault = false;
  worldCuts.protonCut = -1;
  worldCuts.protonCutDisabledByDefault = false;
  mapOfRegionCuts["DefaultRegionForTheWorld"] = worldCuts;
  mLoadState=0;
  mDEDXBinning=-1;
  mLambdaBinning=-1;
  mEmin=-1;
  mEmax=-1;
  mUserPhysicListName = "";
  userlimits=0;

  G4double limit=250*eV; // limit for diplay production cuts table
  G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(limit, 100.*GeV);

  // Default lower value. Could be set by user.
  mLowEnergyRangeLimit = G4ProductionCutsTable::GetProductionCutsTable()->GetLowEdgeEnergy();

  pMessenger = new GatePhysicsListMessenger(this);
  pMessenger->BuildCommands("/gate/physics");

  emPar= G4EmParameters::Instance();
#if G4VERSION_MAJOR >= 10 && G4VERSION_MINOR >= 5
  mUseICRU90Data = false;
#endif

	// used to conditionally activate DNA physics list in specific regions
	// while using other em physics list elsewhere
	emDNAActivator = new G4EmDNAPhysicsActivator;
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
GatePhysicsList::~GatePhysicsList()
{
  delete pMessenger;
  for(VolumeUserLimitsMapType::iterator i = mapOfVolumeUserLimits.begin(); i!=mapOfVolumeUserLimits.end(); i++)
    {
      delete (*i).second;
    }
  mapOfVolumeUserLimits.clear();

  delete userlimits;
  for(std::list<G4ProductionCuts*>::iterator i = theListOfCuts.begin(); i!=theListOfCuts.end(); i++)
    {
      delete (*i);
    }
  theListOfCuts.clear();

  mapOfRegionCuts.clear();
  theListOfPBName.clear();
  mListOfStepLimiter.clear();
  mListOfG4UserSpecialCut.clear();
  GateVProcess::Delete();

  // delete the transportation process (should be done in ~G4VUserPhysicsList())
  bool isTransportationDelete = false;
#if G4VERSION_NUMBER >= 1030
  auto theParticleIterator=GetParticleIterator();
#else
  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
  //G4ParticleTable::G4PTblDicIterator *
  theParticleIterator = theParticleTable->GetIterator();
#endif
  theParticleIterator->reset();
  while( (*theParticleIterator)() ){//&& !isTransportationDelete){
    G4ParticleDefinition* particle = theParticleIterator->value();
    G4ProcessVector * vect = particle->GetProcessManager()->GetProcessList();
    for(unsigned int i = 0; i<vect->size();i++)
      {
        if((*vect)[i]->GetProcessName()=="Transportation" )//&& !isTransportationDelete)
          {
            if(!isTransportationDelete) delete (*vect)[i];
            isTransportationDelete = true;
            (*vect)[i]=0;
          }
        /*else {
          if( (*vect)[i] ){
          if((*vect)[i]->GetProcessName()=="Decay" ){
          G4cout<<"test  "<<particle->GetParticleName()<<"   "<<(*vect)[i]<< Gateendl;
          delete (*vect)[i];
	  }
          }
          }*/
        //else if((*vect)[i]) delete (*vect)[i];
        //(*vect)[i]=0;

      }
  }
  // Transportation process deleted

  /*theParticleIterator->reset();
    while( (*theParticleIterator)()){
    G4ParticleDefinition* particle = theParticleIterator->value();
    G4ProcessVector * vect = particle->GetProcessManager()->GetProcessList();
    G4cout<<"Particle= "<< particle->GetParticleName() << Gateendl;

    for(int i = 0; i<vect->size();i++)
    {
    if((*vect)[i]) G4cout<<"Process= "<<(*vect)[i]->GetProcessName()<< Gateendl;
    }
    }*/

	delete emDNAActivator;
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
std::vector<GateVProcess*>* GatePhysicsList::GetTheListOfProcesss()
{
  return GateVProcess::GetTheListOfProcesses();
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
void GatePhysicsList::ConstructProcess()
{
  GateMessage("Physic",2,"GatePhysicsList::ConstructProcess" << Gateendl);
  GateMessage("Physic",3,"mLoadState = " << mLoadState << Gateendl);
  GateMessage("Physic",3,"mListOfStepLimiter.size = " << mListOfStepLimiter.size() << Gateendl);

  // if ((mLoadState == 0) && (mUserPhysicListName == "")) {
  //   DD("direct return : do nothing");
  //   return;
  // }

  if ((mLoadState==1) && (mUserPhysicListName == "")) {
    GateMessage("Physic", 0, "WARNING: manual physic lists are being deprecated." << Gateendl
                << "Please, use physics list builder mechanism instead. Related documentation can be found at:" << Gateendl
                << "http://wiki.opengatecollaboration.org/index.php/Users_Guide:Setting_up_the_physics" << Gateendl );
  }

  if(mLoadState==0)
    {
      // AddTransportation(); // not set here. Set only if no physics list builder is used
      //Does GetTheListOfProcesss() needs to be called every time??
      for(unsigned int i=0; i<GetTheListOfProcesss()->size(); i++)
	{
	  theListOfPBName.push_back((*GetTheListOfProcesss())[i]->GetG4ProcessName());
	  for(unsigned int j=0; j<GetTheListOfProcesss()->size(); j++)
	    if(i != j && (*GetTheListOfProcesss())[i]->GetG4ProcessName()==(*GetTheListOfProcesss())[j]->GetG4ProcessName() )
	      GateWarning("Some processes have the same name: "
			  <<(*GetTheListOfProcesss())[i]->GetG4ProcessName() );
	}
    }
  else if(mLoadState==1)
    {
      if (mUserListOfPhysicList.empty()) { // if a user physic list is set, transportation is already set
          AddTransportation();
      }

      for(unsigned int i=0; i<GetTheListOfProcesss()->size(); i++)
        (*GetTheListOfProcesss())[i]->ConstructProcess();
//
      //opt->SetVerbose(2);
      //if(mDEDXBinning>0)   emPar->SetNumberOfBins(mDEDXBinning);
      //if(mLambdaBinning>0) emPar->SetNumberOfBins(mLambdaBinning);
      if(mEmin>0)          emPar->SetMinEnergy(mEmin);
      if(mEmax>0)          emPar->SetMaxEnergy(mEmax);
    }
  else GateMessage("Physic",1,"GatePhysicsList::Construct() -- Warning: processes already defined!" << Gateendl);

  //SetCuts();

  if(mLoadState>0) DefineCuts();

  /* //to check : move in DefineCuts
     if(mLoadState==1 && mListOfStepLimiter.size()!=0){
     G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
     G4ParticleTable::G4PTblDicIterator * theParticleIterator = theParticleTable->GetIterator();
     theParticleIterator->reset();
     while( (*theParticleIterator)() ){
     G4ParticleDefinition* particle = theParticleIterator->value();
     G4ProcessManager* pmanager = particle->GetProcessManager();
     G4String particleName = particle->GetParticleName();
     for(unsigned int i=0; i<mListOfStepLimiter.size(); i++) {
     if(mListOfStepLimiter[i]==particleName) {
     GateMessage("Cuts", 3, "Activate G4StepLimiter for " << particleName << Gateendl);
     pmanager->AddProcess(new G4StepLimiter, -1,-1,3);
     }
     }
     }
     }

     if(mLoadState==1 && mListOfG4UserSpecialCut.size()!=0){
     G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
     G4ParticleTable::G4PTblDicIterator * theParticleIterator = theParticleTable->GetIterator();
     theParticleIterator->reset();
     while( (*theParticleIterator)() ){
     G4ParticleDefinition* particle = theParticleIterator->value();
     G4ProcessManager* pmanager = particle->GetProcessManager();
     G4String particleName = particle->GetParticleName();
     for(unsigned int i=0; i<mListOfG4UserSpecialCut.size(); i++) {
     if(mListOfG4UserSpecialCut[i]==particleName) {
     GateMessage("Cuts", 3, "Activate G4UserSpecialCuts for " << particleName << Gateendl);
     pmanager-> AddProcess(new G4UserSpecialCuts,   -1,-1,4);
     }
     }
     }
     }
  */

	// Construct G4 DNA activator process
	emDNAActivator->ConstructProcess();

  mLoadState++;
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
// Construction of the physics list from a G4 builder
void GatePhysicsList::ConstructPhysicsList(G4String name)
{
  GateMessage("Physic", 0, "The following Geant4's physic-list is enabled :" << name << Gateendl);

  G4PhysListFactory * l = new G4PhysListFactory(); //  instantiate PhysList by environment variable "PHYSLIST"
  const std::vector<G4String>& list = l->AvailablePhysLists();
  mListOfPhysicsLists = "";
  for(unsigned int i=0; i<list.size(); i++) {
    mListOfPhysicsLists += list[i];
    mListOfPhysicsLists += ", " ;
  }
  const std::vector<G4String>& list_em = l->AvailablePhysListsEM();
  mListOfPhysicsLists += "with the following EM option (or nothing for default) : ";
  for(unsigned int i=0; i<list_em.size(); i++) {
    mListOfPhysicsLists += list_em[i];
    mListOfPhysicsLists += " " ;
  }

  mUserPhysicListName = name;

  // First, try to create EM only Physic lists
  G4VPhysicsConstructor * pl = nullptr;
  if (mUserPhysicListName == "emstandard") {
    pl = new G4EmStandardPhysics();
  }
  if (mUserPhysicListName == "emstandard_opt1") {
    pl = new G4EmStandardPhysics_option1();
  }
  if (mUserPhysicListName == "emstandard_opt2") {
    pl = new G4EmStandardPhysics_option2();
  }
  if (mUserPhysicListName == "emstandard_opt3") {
    pl = new G4EmStandardPhysics_option3();
  }
  if (mUserPhysicListName == "emstandard_opt4") {
    pl = new G4EmStandardPhysics_option4();
  }
  if (mUserPhysicListName == "emstandard_SS") {
    pl = new G4EmStandardPhysicsSS();
  }
  if (mUserPhysicListName == "emstandard_GS") {
    pl = new G4EmStandardPhysicsGS();
  }
  if (mUserPhysicListName == "emLowEP") {
    pl = new G4EmLowEPPhysics();
  }
  if (mUserPhysicListName == "emlivermore") {
    pl = new G4EmLivermorePhysics();
  }
  if (mUserPhysicListName == "emlivermore_polar") {
    pl = new G4EmLivermorePolarizedPhysics();
  }
  if (mUserPhysicListName == "empenelope") {
    pl = new G4EmPenelopePhysics();
  }
  if (mUserPhysicListName == "emDNAphysics") {
    pl = new G4EmDNAPhysics();
  }
  if (mUserPhysicListName == "emDNAphysics_option1") {
    pl = new G4EmDNAPhysics_option1;
  }
  if (mUserPhysicListName == "emDNAphysics_option2") {
    pl = new G4EmDNAPhysics_option2;
  }
  if (mUserPhysicListName == "emDNAphysics_option3") {
    pl = new G4EmDNAPhysics_option3;
  }
  if (mUserPhysicListName == "emDNAphysics_option4") {
    pl = new G4EmDNAPhysics_option4;
  }
  if (mUserPhysicListName == "emDNAphysics_option5") {
    pl = new G4EmDNAPhysics_option5;
  }
  if (mUserPhysicListName == "emDNAphysics_option6") {
    pl = new G4EmDNAPhysics_option6;
  }
  if (mUserPhysicListName == "emDNAphysics_option7") {
    pl = new G4EmDNAPhysics_option7;
  }
  if (mUserPhysicListName == "emDNAphysics_option8") {
    pl = new G4EmDNAPhysics_option8;
  }

#ifdef GATE_USE_OPTICAL
    if (mUserPhysicListName == "optical") {
    pl = new G4OpticalPhysics();
  }
#endif


    if(pl != nullptr)  {
        mUserListOfPhysicList.push_back(pl);
          pl->ConstructParticle();
          pl->ConstructProcess();
    }

  if(mUserListOfPhysicList.size() == 1)  {
      AddTransportation();
  }

  if (!mUserListOfPhysicList.empty()) {
    GateRunManager::GetRunManager()->SetUserPhysicListName("");
  }
  else {
    // Set the phys list name. It will be build in GateRunManager.
    GateRunManager::GetRunManager()->SetUserPhysicListName(mUserPhysicListName);
  }

  // Fluorescence processes
  // - default activation of deexcitation process
  emPar->SetFluo(true);
  emPar->SetAuger(true);
  emPar->SetPixe(true);
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
// Construction of the physics list from a G4 builder
void GatePhysicsList::ConstructPhysicsListDNAMixed(G4String name)
{
  if (name == "emstandard_opt3_mixed_emdna") {
    emPhysicsListMixed = new GateMixedDNAPhysics("emstandard_opt3_mixed_emdna");
  }
  else {
    if (name== "emlivermore_mixed_emdna") {
      emPhysicsListMixed = new GateMixedDNAPhysics("emlivermore_mixed_dna");
    }
    else {
      GateError("The mixed Physics List "<<name<<" does not exist!");
    }
  }
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
void GatePhysicsList::ConstructProcessMixed()
{
  emPhysicsListMixed->ConstructProcess();
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
void GatePhysicsList::ConstructParticle()
{

  // Construct all bosons
  G4BosonConstructor boson;
  boson.ConstructParticle();

  // Construct all leptons
  G4LeptonConstructor lepton;
  lepton.ConstructParticle();

  //  Construct all mesons
  G4MesonConstructor meson;
  meson.ConstructParticle();

  //  Construct all barions
  G4BaryonConstructor barion;
  barion.ConstructParticle();

  //  Construct light ions
  G4IonConstructor ion;
  ion.ConstructParticle();

  //  Construct  resonaces and quarks
  G4ShortLivedConstructor slive;
  slive.ConstructParticle();

  //  Construct hybridino
  G4Hybridino::HybridinoDefinition();

  //Construct G4DNA particles

  G4DNAGenericIonsManager* dnagenericIonsManager;
  dnagenericIonsManager=G4DNAGenericIonsManager::Instance();
  dnagenericIonsManager->GetIon("hydrogen");
  dnagenericIonsManager->GetIon("alpha+");
  dnagenericIonsManager->GetIon("alpha++");
  dnagenericIonsManager->GetIon("helium");
  dnagenericIonsManager->GetIon("carbon");
  dnagenericIonsManager->GetIon("nitrogen");
  dnagenericIonsManager->GetIon("iron");
  dnagenericIonsManager->GetIon("oxygen");

	// Construct G4 DNA activator particles
	emDNAActivator->ConstructParticle();

 //Construct positroniums
 GateParaPositronium::ParaPositroniumDefinition();
 GateOrthoPositronium::OrthoPositroniumDefinition();
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
void GatePhysicsList::Print(G4String type, G4String particlename)
{

  if(type=="Initialized") std::cout << Gateendl << Gateendl<<"List of initialized processes:" << Gateendl << Gateendl;
  else if(type=="Enabled") std::cout << Gateendl << Gateendl<<"List of Enabled processes:" << Gateendl << Gateendl;
  else if(type=="Available") std::cout << Gateendl << Gateendl<<"List of Available processes:" << Gateendl << Gateendl;

  if(type=="Enabled")
    {
      if(particlename != "All") std::cout<<"   ***  "<<particlename<<"  ***" << Gateendl;
      for(unsigned int i=0; i<GetTheListOfProcesss()->size(); i++)
	(*GetTheListOfProcesss())[i]->PrintEnabledParticles(particlename);

      std::cout<< Gateendl;
    }

  if(type=="Initialized")
    {
      Print(particlename);
      std::cout<< Gateendl;
    }

  if(type=="Available")
    {
      std::vector<G4String> DefaultParticles;
      std::vector<G4String> DataSets;
      std::vector<G4String> Models;

      for(unsigned int i=0; i<GetTheListOfProcesss()->size(); i++)
	{
	  DefaultParticles = (*GetTheListOfProcesss())[i]->GetTheListOfDefaultParticles();
	  DataSets = (*GetTheListOfProcesss())[i]->GetTheListOfDataSets();
	  Models = (*GetTheListOfProcesss())[i]->GetTheListOfModels();
	  if((*GetTheListOfProcesss())[i]->GetProcessInfo()!="")
	    std::cout<<"  * "<<(*GetTheListOfProcesss())[i]->GetG4ProcessName()<<" ("<<(*GetTheListOfProcesss())[i]->GetProcessInfo()<<")" << Gateendl;
	  else std::cout<<"  * "<<(*GetTheListOfProcesss())[i]->GetG4ProcessName()<< Gateendl;

	  if(DefaultParticles.size() > 1) std::cout<<"     - Default particles: " << Gateendl;
	  else if(DefaultParticles.size() == 1) std::cout<<"     - Default particle: " << Gateendl;
	  for(unsigned int i1=0; i1<DefaultParticles.size(); i1++)
	    {
	      std::cout<<"        + "<<DefaultParticles[i1]<< Gateendl;
	    }

	  if(Models.size() > 1) std::cout<<"     - Models: " << Gateendl;
	  else if(Models.size() == 1) std::cout<<"     - Model: " << Gateendl;
	  for(unsigned int i1=0; i1<Models.size(); i1++)
	    {
	      std::cout<<"        + "<<Models[i1]<< Gateendl;
	    }

	  if(DataSets.size() > 1) std::cout<<"     - DataSets: " << Gateendl;
	  if(DataSets.size() == 1) std::cout<<"     - DataSet: " << Gateendl;
	  for(unsigned int i1=0; i1<DataSets.size(); i1++)
	    {
	      std::cout<<"        + "<<DataSets[i1]<< Gateendl;
	    }
	  std::cout<< Gateendl;
	}
      std::cout<< Gateendl;
    }

}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
void GatePhysicsList::Print(G4String name)
{
  G4ParticleDefinition* particle=0;
  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
  G4ProcessManager* manager = 0;
  G4ProcessVector * processvector = 0;
  //G4ParticleTable::G4PTblDicIterator *
  //theParticleIterator;

  int iDisp = 0;

  if(name=="All")
    {
#if G4VERSION_NUMBER >= 1030
      auto theParticleIterator=GetParticleIterator();
#else
      theParticleIterator = theParticleTable->GetIterator();
#endif
      theParticleIterator -> reset();
      while( (*theParticleIterator)() ) {
	particle = theParticleIterator->value();
	manager  = particle->GetProcessManager();
	processvector = manager->GetProcessList();
	if(manager->GetProcessListLength()==0) continue;
	if(manager->GetProcessListLength()==1 && (*processvector)[0]->GetProcessName()== "Transportation") continue;
	// Transportation process is ignored for display;
	std::cout<<"  * "<<particle->GetParticleName()<< Gateendl;
	iDisp++;
	for(int j=0;j<manager->GetProcessListLength();j++)
	  {
	    if( (*processvector)[j]->GetProcessName() !=  "Transportation" )
	      std::cout<<"    - "<<(*processvector)[j]->GetProcessName()<< Gateendl;
	  }
      }
    }
  else
    {
      G4ParticleDefinition* particle = theParticleTable->FindParticle(name);
      if(!particle)
	{
	  GateWarning("Unknown particle: "<<name );
	  return;
	}
      manager  = particle->GetProcessManager();
      processvector = manager->GetProcessList();
      if(manager->GetProcessListLength()==0) return;
      std::cout<<"Particle: "<<particle->GetParticleName()<< Gateendl;
      for(int j=0;j<manager->GetProcessListLength();j++)
	if( (*processvector)[j]->GetProcessName() !=  "Transportation" )
	  std::cout<<"   - "<<(*processvector)[j]->GetProcessName()<< Gateendl;
    }

}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
std::vector<GateVProcess*> GatePhysicsList::FindProcess(G4String name)
{
  std::vector<GateVProcess*> thelist;

  for(unsigned int i=0; i<GetTheListOfProcesss()->size(); i++) {
    if((*GetTheListOfProcesss())[i]->GetG4ProcessName()==name) thelist.push_back((*GetTheListOfProcesss())[i]);
  }
  return thelist;
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
void GatePhysicsList::AddProcesses(G4String processname, G4String particle)
{
  if ( processname == "UHadronElastic")
    G4Exception( "GatePhysicsList::AddProcesses","AddProcesses", FatalException,"####### WARNING: 'HadronElastic' process name replace 'UHadronElastic' process name since Geant4 9.5");

    if ((processname == "Decay") or (processname == "RadioactiveDecay")) {
        G4RadioactiveDecayPhysics p;
        p.ConstructParticle();
        p.ConstructProcess();
        return;
    }



  std::vector<GateVProcess *>  process = FindProcess(processname);
  if(process.size()>0)
    for(unsigned int i=0; i<process.size(); i++) process[i]->CreateEnabledParticle(particle);
  else
    {
      GateWarning("Unknown process: "<<processname );
      return;
    }

}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
void GatePhysicsList::AddAtomDeexcitation()
{
  if(G4LossTableManager::Instance()->AtomDeexcitation() == NULL)
    {
      G4VAtomDeexcitation* de = new G4UAtomicDeexcitation();
      G4LossTableManager::Instance()->SetAtomDeexcitation(de);
    }

  emPar->SetFluo(true);
  emPar->SetAuger(true);
  emPar->SetPixe(true);
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
void GatePhysicsList::RemoveProcesses(G4String processname, G4String particle)
{

  std::vector<GateVProcess *>  process = FindProcess(processname);
  if(process.size()>0)
    for(unsigned int i=0; i<process.size(); i++) process[i]->RemoveElementOfParticleList(particle);
  else
    {
      GateWarning("Unknown process: "<<processname );;
      return;
    }

}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
/*
  This function is called just before the physics initialization.
  It checks if the fictitious process is activated and if so it forced
  all gamma processes to a standard configuration.
*/
void GatePhysicsList::PurgeIfFictitious()
{
  bool isFictitious = false;
  // We first check if fictitious process is activated
  for(unsigned int i=0; i<GetTheListOfProcesss()->size(); i++) {
    if ( (*GetTheListOfProcesss())[i]->GetG4ProcessName() == "Fictitious" &&
         (*GetTheListOfProcesss())[i]->GetTheListOfEnabledParticles().size() != 0)
      {
        isFictitious=true;
        break;
      }
  }
  // If fictitious is activated, we alert the user that gamma processes are forced to:
  // --> Photoelectric: standard
  // --> Compton: standard
  // --> Rayleigh: inactive
  // --> GammaConvertion: inactive
  if (isFictitious) {
    G4cout << "Fictitious interactions are activated, so gamma processes are forced to:" << Gateendl
           << "  --> PhotoElectric:   standard" << Gateendl
           << "  --> Compton:         standard" << Gateendl
           << "  --> Rayleigh:        inactive" << Gateendl
           << "  --> GammaConversion: inactive" << Gateendl;
    for(unsigned int i=0; i<(*GetTheListOfProcesss()).size(); i++) {
      if ( (*GetTheListOfProcesss())[i]->GetG4ProcessName() == "LowEnergyRayleighScattering" ||
           (*GetTheListOfProcesss())[i]->GetG4ProcessName() == "PhotoElectric" ||
           (*GetTheListOfProcesss())[i]->GetG4ProcessName() == "LowEnergyPhotoElectric" ||
           (*GetTheListOfProcesss())[i]->GetG4ProcessName() == "GammaConversion" ||
           (*GetTheListOfProcesss())[i]->GetG4ProcessName() == "PenelopeGammaConversion" ||
           (*GetTheListOfProcesss())[i]->GetG4ProcessName() == "LowEnergyGammaConversion" ||
           (*GetTheListOfProcesss())[i]->GetG4ProcessName() == "PenelopeCompton" ||
           (*GetTheListOfProcesss())[i]->GetG4ProcessName() == "Compton" ||
           (*GetTheListOfProcesss())[i]->GetG4ProcessName() == "LowEnergyCompton"
           )
        {
          if ((*GetTheListOfProcesss())[i]->GetTheListOfEnabledParticles().size() != 0)
            {
              RemoveProcesses((*GetTheListOfProcesss())[i]->GetG4ProcessName(),"gamma");
            }
        }
    }
  }
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
void GatePhysicsList::Write(G4String file)
{
  G4ParticleDefinition* particle=0;
  //G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
  G4ProcessManager* manager = 0;
  G4ProcessVector * processvector = 0;
  //G4ParticleTable::G4PTblDicIterator *
  //theParticleIterator;

  int iDisp = 0;

  std::ofstream os;
  os.open(file.data());
  if(mLoadState<2)  os<<"<!> *** Warning *** <!>  Processes not yet initialized!" << Gateendl << Gateendl;

  os<<"List of particles with their associated processes" << Gateendl << Gateendl;
  if(mLoadState<2)  os<<"<!> *** Warning *** <!>  Processes not yet initialized!" << Gateendl << Gateendl;
#if G4VERSION_NUMBER >= 1030
  auto theParticleIterator=GetParticleIterator();
#else
  G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
  theParticleIterator = theParticleTable->GetIterator();
#endif
  theParticleIterator -> reset();
  while( (*theParticleIterator)() ) {
    particle = theParticleIterator->value();
    manager  = particle->GetProcessManager();
    processvector = manager->GetProcessList();
    if(manager->GetProcessListLength()==0) continue;
    if(manager->GetProcessListLength()==1 && (*processvector)[0]->GetProcessName()== "Transportation") continue;
    os    <<"  * "<<particle->GetParticleName().data()<<Gateendl;
    iDisp++;
    for(int j=0;j<manager->GetProcessListLength();j++)
      {
	if( (*processvector)[j]->GetProcessName() !=  "Transportation" )
	  os<<"    - "<<(*processvector)[j]->GetProcessName().data()<<Gateendl;
      }
  }
  os << Gateendl << Gateendl<<"-----------------------------------------------------------------------------" << Gateendl << Gateendl;

  os<<"List of processes:" << Gateendl << Gateendl;

  os.close();

  for(unsigned int i=0; i<GetTheListOfProcesss()->size(); i++)
    (*GetTheListOfProcesss())[i]->PrintEnabledParticlesToFile(file);
  os.open(file.data(), std::ios_base::app);
  os << Gateendl;
  os.close();
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Cuts
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GatePhysicsList::SetEmProcessOptions()
{
  //if(mDEDXBinning>0)   emPar->SetNumberOfBins(mDEDXBinning);
  //if(mLambdaBinning>0) emPar->SetNumberOfBins(mLambdaBinning);
  if(mEmin>0)          emPar->SetMinEnergy(mEmin);
  if(mEmax>0)          emPar->SetMaxEnergy(mEmax);
#if G4VERSION_MAJOR >= 10 && G4VERSION_MINOR >= 5
  emPar->SetUseICRU90Data(mUseICRU90Data);
#endif
  emPar->SetApplyCuts(true);

  // Fluorescence processes
  // - register all regions in deexcitation process with fluo, auger and PIXE set to "true"
  GateObjectStore *store = GateObjectStore::GetInstance();
  for(GateObjectStore::iterator it=store->begin() ; it!=store->end() ; ++it){
    emPar->SetDeexActiveRegion(it->first,true,true,true); // G4region, fluo, auger, PIXE
  }
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhysicsList::SetCuts()
{
  /* if (verboseLevel >0){
     std::cout << "GatePhysicsList::SetCuts: default cut length : "
     << G4BestUnit(defaultCutValue,"Length") << Gateendl;
     }  */

  // These values are used as the default production thresholds
  // for the world volume.
  // SetCutsWithDefault();

  // This is needed to enable user cuts
  emPar->SetApplyCuts(true);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhysicsList::DefineCuts(G4VUserPhysicsList * phys)
{
    // GateMessage("Cuts",4,"===================================\n");
    // GateMessage("Cuts",4,"GatePhysicsList::SetCuts() -- begin\n");

    //-----------------------------------------------------------------------------
    // Set defaults production cut for the world
    ParticleCutType worldCuts = mapOfRegionCuts["DefaultRegionForTheWorld"];

    if (worldCuts.gammaCut == -1) {
        worldCuts.gammaCut = defaultCutValue;
        worldCuts.gammaCutDisabledByDefault = gammaCutDisabledByDefault;
    }
    if (worldCuts.electronCut == -1) {
        worldCuts.electronCut = defaultCutValue;
        worldCuts.electronCutDisabledByDefault = electronCutDisabledByDefault;
    }
    if (worldCuts.positronCut == -1) {
        worldCuts.positronCut = defaultCutValue;
        worldCuts.positronCutDisabledByDefault = positronCutDisabledByDefault;
    }
    if (worldCuts.protonCut == -1) {
        worldCuts.protonCut = defaultCutValue;
        worldCuts.protonCutDisabledByDefault = protonCutDisabledByDefault;
    }

    GateMessage("Cuts", 3, "Set default production cuts (world) : "
            << worldCuts.gammaCut << " "
            << worldCuts.electronCut << " "
            << worldCuts.positronCut << " "
            << worldCuts.protonCut << " mm" << Gateendl);


    if (!gammaCutDisabledByDefault)
        phys->SetCutValue(worldCuts.gammaCut, "gamma", "DefaultRegionForTheWorld");
    else
        GateMessage("Cuts", 3, "No gamma cut by default " << Gateendl);

    if (!electronCutDisabledByDefault)
        phys->SetCutValue(worldCuts.electronCut, "e-", "DefaultRegionForTheWorld");
    else
        GateMessage("Cuts", 3, "No electron cut by default " << Gateendl);

    if (!positronCutDisabledByDefault)
        phys->SetCutValue(worldCuts.positronCut, "e+", "DefaultRegionForTheWorld");
    else
        GateMessage("Cuts", 3, "No positron cut by default " << Gateendl);

    if (!protonCutDisabledByDefault)
        phys->SetCutValue(worldCuts.protonCut, "proton", "DefaultRegionForTheWorld");
    else
        GateMessage("Cuts", 3, "No proton cut by default " << Gateendl);


    //-----------------------------------------------------------------------------
    // Set default production cut to other regions
    G4RegionStore *RegionStore = G4RegionStore::GetInstance();
    G4RegionStore::const_iterator pi = RegionStore->begin();
    G4RegionStore::const_iterator pe = RegionStore->end();
    while (pi != pe) {
        G4String regionName = (*pi)->GetName();

        if (regionName != "DefaultRegionForTheWorld" && regionName != "world") {
            RegionCutMapType::iterator current = mapOfRegionCuts.find(regionName);
            if (current == mapOfRegionCuts.end()) {
                // GateMessage("Cuts",5, " Cut not set for region " << regionName << " put -1\n");
                mapOfRegionCuts[regionName].gammaCut = -1;
                mapOfRegionCuts[regionName].gammaCutDisabledByDefault = false;
                mapOfRegionCuts[regionName].electronCut = -1;
                mapOfRegionCuts[regionName].electronCutDisabledByDefault = false;
                mapOfRegionCuts[regionName].positronCut = -1;
                mapOfRegionCuts[regionName].positronCutDisabledByDefault = false;
                mapOfRegionCuts[regionName].protonCut = -1;
                mapOfRegionCuts[regionName].protonCutDisabledByDefault = false;
            }
        }
        ++pi;
    }

    //-----------------------------------------------------------------------------
    // Loop over regions with a defined cuts
    RegionCutMapType::iterator it = mapOfRegionCuts.begin();
    while (it != mapOfRegionCuts.end()) {
        // do not apply cut for the world region
        // GateMessage("Cuts", 5, "Region (*it).first : " << (*it).first<< Gateendl);
        if (((*it).first != "DefaultRegionForTheWorld") && ((*it).first != "world")) {
            G4Region *region = RegionStore->GetRegion((*it).first);
            if (!region) {
                GateError("The region '" << (*it).first << "' does not exist !");
            }
            // set default values
            ParticleCutType regionCuts = (*it).second;

            G4Region *parentRegion = region;

            while (regionCuts.gammaCut == -1) {
                G4bool unique;
                parentRegion = parentRegion->GetParentRegion(unique);
                if (parentRegion->GetName() != "DefaultRegionForTheWorld") {
                    ParticleCutType parentRegionCuts = mapOfRegionCuts[parentRegion->GetName()];
                    regionCuts.gammaCut = parentRegionCuts.gammaCut;
                    regionCuts.gammaCutDisabledByDefault = parentRegionCuts.gammaCutDisabledByDefault;
                } else {
                    regionCuts.gammaCut = worldCuts.gammaCut;
                    regionCuts.gammaCutDisabledByDefault = worldCuts.gammaCutDisabledByDefault;
                }

            }

            parentRegion = region;
            while (regionCuts.electronCut == -1) {
                G4bool unique;
                parentRegion = parentRegion->GetParentRegion(unique);
                if (parentRegion->GetName() != "DefaultRegionForTheWorld") {
                    ParticleCutType parentRegionCuts = mapOfRegionCuts[parentRegion->GetName()];
                    regionCuts.electronCut = parentRegionCuts.electronCut;
                    regionCuts.electronCutDisabledByDefault = parentRegionCuts.electronCutDisabledByDefault;
                } else {
                    regionCuts.electronCut = worldCuts.electronCut;
                    regionCuts.electronCutDisabledByDefault = worldCuts.electronCutDisabledByDefault;
                }

            }

            parentRegion = region;
            while (regionCuts.positronCut == -1) {
                G4bool unique;
                parentRegion = parentRegion->GetParentRegion(unique);
                if (parentRegion->GetName() != "DefaultRegionForTheWorld") {
                    ParticleCutType parentRegionCuts = mapOfRegionCuts[parentRegion->GetName()];
                    regionCuts.positronCut = parentRegionCuts.positronCut;
                    regionCuts.positronCutDisabledByDefault = parentRegionCuts.positronCutDisabledByDefault;
                } else {
                    regionCuts.positronCut = worldCuts.positronCut;
                    regionCuts.positronCutDisabledByDefault = worldCuts.positronCutDisabledByDefault;
                }

            }
            parentRegion = region;
            while (regionCuts.protonCut == -1) {
                G4bool unique;
                parentRegion = parentRegion->GetParentRegion(unique);
                if (parentRegion->GetName() != "DefaultRegionForTheWorld") {
                    ParticleCutType parentRegionCuts = mapOfRegionCuts[parentRegion->GetName()];
                    regionCuts.protonCut = parentRegionCuts.protonCut;
                    regionCuts.protonCutDisabledByDefault = parentRegionCuts.protonCutDisabledByDefault;
                } else {
                    regionCuts.protonCut = worldCuts.protonCut;
                    regionCuts.protonCutDisabledByDefault = worldCuts.protonCutDisabledByDefault;
                }

            }
            parentRegion = region;


//            GateMessage("Cuts", 3, "Set production cuts (g, e-, e+, p) for the region '"
//                    << (*it).first << "' : "
//                    << G4BestUnit(regionCuts.gammaCut, "Length") << " "
//                    << G4BestUnit(regionCuts.electronCut, "Length") << " "
//                    << G4BestUnit(regionCuts.positronCut, "Length") << " "
//                    << G4BestUnit(regionCuts.protonCut, "Length") << Gateendl);

            // apply the cut
            /* G4ProductionCuts* cuts = region->GetProductionCuts();
               if (!cuts) cuts = new G4ProductionCuts;
               cuts->SetProductionCut(regionCuts.gammaCut, "gamma");
               cuts->SetProductionCut(regionCuts.electronCut, "e-");
               cuts->SetProductionCut(regionCuts.positronCut, "e+");
               region->SetProductionCuts(cuts);*/

            if (region->GetProductionCuts()) theListOfCuts.push_back(region->GetProductionCuts());
            else theListOfCuts.push_back(new G4ProductionCuts);

            if (!regionCuts.gammaCutDisabledByDefault) {
                GateMessage("Cuts", 3, " set cut regionCuts.gammaCut = " << regionCuts.gammaCut << " for region "
                                                                         << region->GetName() << Gateendl);
                theListOfCuts.back()->SetProductionCut(regionCuts.gammaCut, "gamma");
            } else {
                GateMessage("Cuts", 3,
                            "DONT set cut regionCuts.gammaCut = " << regionCuts.gammaCut << " for region "
                                                                     << region->GetName() << Gateendl);
            }


            if (!regionCuts.electronCutDisabledByDefault) {
                GateMessage("Cuts", 3, " set cut regionCuts.electronCut = " << regionCuts.electronCut << " for region "
                                                                            << region->GetName() << Gateendl);
                theListOfCuts.back()->SetProductionCut(regionCuts.electronCut, "e-");
            } else {
                GateMessage("Cuts", 3,
                            "DONT set cut regionCuts.electronCut = " << regionCuts.electronCut << " for region "
                                                                     << region->GetName() << Gateendl);
            }


            if (!regionCuts.positronCutDisabledByDefault) {
                GateMessage("Cuts", 3, " set cut regionCuts.positronCut = " << regionCuts.positronCut << " for region "
                                                                            << region->GetName() << Gateendl);
                theListOfCuts.back()->SetProductionCut(regionCuts.positronCut, "e+");
            } else {
                GateMessage("Cuts", 3,
                            "DONT set cut regionCuts.positronCut = " << regionCuts.positronCut << " for region "
                                                                     << region->GetName() << Gateendl);
            }


            if (!regionCuts.protonCutDisabledByDefault) {
                GateMessage("Cuts", 3, " set cut regionCuts.protonCut = " << regionCuts.protonCut << " for region "
                                                                          << region->GetName() << Gateendl);
                theListOfCuts.back()->SetProductionCut(regionCuts.protonCut, "proton");
            } else {
                GateMessage("Cuts", 3,
                            "DONT set cut regionCuts.protonCut = " << regionCuts.protonCut << " for region "
                                                                     << region->GetName() << Gateendl);
            }

            region->SetProductionCuts(theListOfCuts.back());

        }
        ++it;
    } // end loop regions


  //-----------------------------------------------------------------------------
  // now set user limits
  //-----------------------------------------------------------------------------

  //G4LogicalVolumeStore * logicalVolumeStore = G4LogicalVolumeStore::GetInstance();

  GateUserLimits *  worldUserLimit = new GateUserLimits();
  if(mapOfVolumeUserLimits["DefaultRegionForTheWorld"] != 0)
    {
      if(mapOfVolumeUserLimits["DefaultRegionForTheWorld"]->GetMaxStepSize()       != -1.)
        worldUserLimit->SetMaxStepSize(mapOfVolumeUserLimits["DefaultRegionForTheWorld"]->GetMaxStepSize());
      if(mapOfVolumeUserLimits["DefaultRegionForTheWorld"]->GetMaxTrackLength()    != -1.)
        worldUserLimit->SetMaxTrackLength(mapOfVolumeUserLimits["DefaultRegionForTheWorld"]->GetMaxTrackLength());
      if(mapOfVolumeUserLimits["DefaultRegionForTheWorld"]->GetMaxToF()            != -1.)
        worldUserLimit->SetMaxToF(mapOfVolumeUserLimits["DefaultRegionForTheWorld"]->GetMaxToF());
      if(mapOfVolumeUserLimits["DefaultRegionForTheWorld"]->GetMinKineticEnergy()  != -1.)
        worldUserLimit->SetMinKineticEnergy(mapOfVolumeUserLimits["DefaultRegionForTheWorld"]->GetMinKineticEnergy());
      if(mapOfVolumeUserLimits["DefaultRegionForTheWorld"]->GetMinRemainingRange() != -1.)
        worldUserLimit->SetMinRemainingRange(mapOfVolumeUserLimits["DefaultRegionForTheWorld"]->GetMinRemainingRange());
    }


  //FIXME
  //DD(mListOfStepLimiter.size());
  if (mListOfStepLimiter.size()!=0) {
#if G4VERSION_NUMBER >= 1030
    auto theParticleIterator=GetParticleIterator();
#else
    G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
    //G4ParticleTable::G4PTblDicIterator *
    theParticleIterator = theParticleTable->GetIterator();
#endif
    theParticleIterator->reset();
    while( (*theParticleIterator)() ){
      G4ParticleDefinition* particle = theParticleIterator->value();
      G4ProcessManager* pmanager = particle->GetProcessManager();
      G4String particleName = particle->GetParticleName();
      for(unsigned int i=0; i<mListOfStepLimiter.size(); i++) {
	if(mListOfStepLimiter[i]==particleName) {
          GateMessage("Cuts", 3, "Activate G4StepLimiter for " << particleName << Gateendl);
          pmanager->AddProcess(new G4StepLimiter, -1,-1,3);
        }
      }
    }
  }

  //DD(mListOfG4UserSpecialCut.size());
  if (mListOfG4UserSpecialCut.size()!=0) {
#if ( G4VERSION_NUMBER >= 1030 )
    auto theParticleIterator=GetParticleIterator();
#else
    G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
    //G4ParticleTable::G4PTblDicIterator *
    theParticleIterator = theParticleTable->GetIterator();
#endif
    theParticleIterator->reset();
    while( (*theParticleIterator)() ){
      G4ParticleDefinition* particle = theParticleIterator->value();
      G4ProcessManager* pmanager = particle->GetProcessManager();
      G4String particleName = particle->GetParticleName();
      for(unsigned int i=0; i<mListOfG4UserSpecialCut.size(); i++) {
	if(mListOfG4UserSpecialCut[i]==particleName) {
          GateMessage("Cuts", 3, "Activate G4UserSpecialCuts for " << particleName << Gateendl);
          pmanager-> AddProcess(new G4UserSpecialCuts,   -1,-1,4);
        }
      }
    }
  }

  //-----------------------------------------------------------------------------
  // Set default production cut to other regions
  pi = RegionStore->begin();
  pe = RegionStore->end();
  while (pi != pe) {
    G4String regionName = (*pi)->GetName();

    if (regionName != "DefaultRegionForTheWorld" && regionName !="world") {
      VolumeUserLimitsMapType::iterator current = mapOfVolumeUserLimits.find(regionName);
      if (current == mapOfVolumeUserLimits.end()) {
	GateMessage("Cuts",5, " UserCuts not set for region " << regionName << " put -1" << Gateendl);
        mapOfVolumeUserLimits[regionName]= new GateUserLimits();
      }
    }
    ++pi;
  }

  VolumeUserLimitsMapType::iterator it2 = mapOfVolumeUserLimits.begin();
  while (it2 != mapOfVolumeUserLimits.end()) {
    // do not apply cut for the world region
    // GateMessage("Cuts", 5, "Region (*it2).first : " << (*it2).first<< Gateendl);
    if (((*it2).first != "DefaultRegionForTheWorld") && ((*it2).first != "world")) {
      G4Region* region = RegionStore->GetRegion((*it2).first);
      if (!region) {
	GateError( "The region '" << (*it2).first << "' does not exist !");
      }
      // set default values
      GateUserLimits *  regionUserLimit =  (*it2).second;
      G4Region* parentRegion =  region;
      G4Region* regionTmp =  region;
      while((regionUserLimit->GetMaxStepSize() == -1) &&
            (regionTmp->GetName() != "DefaultRegionForTheWorld"))
        {
          G4bool unique;
          parentRegion =  regionTmp->GetParentRegion(unique);
          if(parentRegion->GetName() != "DefaultRegionForTheWorld"){
            GateUserLimits * parentRegionUserLimits = mapOfVolumeUserLimits[parentRegion->GetName()];
            regionUserLimit->SetMaxStepSize(  parentRegionUserLimits->GetMaxStepSize()) ;
            GateMessage("Cuts", 5, "Region " << (*it2).first << " maxStepSize " << parentRegionUserLimits->GetMaxStepSize() << Gateendl);
          }
          else regionUserLimit->SetMaxStepSize( worldUserLimit->GetMaxStepSize());
          regionTmp = parentRegion;
        }

      regionTmp =  region;
      while(regionUserLimit->GetMaxTrackLength() == -1 && regionTmp->GetName() != "DefaultRegionForTheWorld")
        {
          G4bool unique;
          parentRegion =  regionTmp->GetParentRegion(unique);
          if(parentRegion->GetName() != "DefaultRegionForTheWorld"){
            GateUserLimits * parentRegionUserLimits = mapOfVolumeUserLimits[parentRegion->GetName()];
            regionUserLimit->SetMaxTrackLength(  parentRegionUserLimits->GetMaxTrackLength()) ;
            GateMessage("Cuts", 5, "Region " << (*it2).first << " maxTrackLength " << parentRegionUserLimits->GetMaxTrackLength() << Gateendl);
          }
          else regionUserLimit->SetMaxTrackLength( worldUserLimit->GetMaxTrackLength());
          regionTmp = parentRegion;
        }


      regionTmp =  region;
      while(regionUserLimit->GetMaxToF() == -1 && regionTmp->GetName() != "DefaultRegionForTheWorld")
        {
          G4bool unique;
          parentRegion =  regionTmp->GetParentRegion(unique);
          if(parentRegion->GetName() != "DefaultRegionForTheWorld"){
            GateUserLimits * parentRegionUserLimits = mapOfVolumeUserLimits[parentRegion->GetName()];
            regionUserLimit->SetMaxToF(  parentRegionUserLimits->GetMaxToF()) ;
          }
          else regionUserLimit->SetMaxToF( worldUserLimit->GetMaxToF());
          regionTmp = parentRegion;
        }

      regionTmp =  region;
      while(regionUserLimit->GetMinKineticEnergy() == -1 && regionTmp->GetName() != "DefaultRegionForTheWorld")
        {
          G4bool unique;
          parentRegion =  regionTmp->GetParentRegion(unique);
          if(parentRegion->GetName() != "DefaultRegionForTheWorld"){
            GateUserLimits * parentRegionUserLimits = mapOfVolumeUserLimits[parentRegion->GetName()];
            regionUserLimit->SetMinKineticEnergy(  parentRegionUserLimits->GetMinKineticEnergy()) ;
          }
          else regionUserLimit->SetMinKineticEnergy( worldUserLimit->GetMinKineticEnergy());
          regionTmp = parentRegion;
        }

      regionTmp =  region;
      while(regionUserLimit->GetMinRemainingRange() == -1 && regionTmp->GetName() != "DefaultRegionForTheWorld")
        {
          G4bool unique;
          parentRegion =  regionTmp->GetParentRegion(unique);
          if(parentRegion->GetName() != "DefaultRegionForTheWorld"){
            GateUserLimits * parentRegionUserLimits = mapOfVolumeUserLimits[parentRegion->GetName()];
            regionUserLimit->SetMinRemainingRange(  parentRegionUserLimits->GetMinRemainingRange()) ;
          }
          else regionUserLimit->SetMinRemainingRange( worldUserLimit->GetMinRemainingRange());
          regionTmp = parentRegion;
        }


      // Set the G4 limits
      G4bool IsULimitDefined = false;
      userlimits = new G4UserLimits(0.1);


      G4String regionName = region->GetName();
      if(regionUserLimit->GetMaxStepSize()       != -1.){
        userlimits->SetMaxAllowedStep(regionUserLimit->GetMaxStepSize());
        GateMessage("Cuts", 3, "Region " << regionName
                    << " maxStepSize " << regionUserLimit->GetMaxStepSize() << Gateendl);
        IsULimitDefined = true;
      }
      if(regionUserLimit->GetMaxTrackLength()    != -1.){
        userlimits->SetUserMaxTrackLength(regionUserLimit->GetMaxTrackLength());
        GateMessage("Cuts", 3, "Region " << regionName
                    << " maxTrackLength " << regionUserLimit->GetMaxTrackLength() << Gateendl);
        IsULimitDefined = true;
      }
      if(regionUserLimit->GetMaxToF()            != -1.){
        userlimits->SetUserMaxTime(regionUserLimit->GetMaxToF());
        GateMessage("Cuts", 3, "Region " << regionName
                    << " MaxToF " << regionUserLimit->GetMaxToF() << Gateendl);
        IsULimitDefined = true;
      }
      if(regionUserLimit->GetMinKineticEnergy()  != -1.){
        userlimits->SetUserMinEkine(regionUserLimit->GetMinKineticEnergy());
        GateMessage("Cuts", 3, "Region " << regionName
                    << " MinEkine " << regionUserLimit->GetMinKineticEnergy() << Gateendl);
        IsULimitDefined = true;
      }
      if(regionUserLimit->GetMinRemainingRange() != -1.){
        userlimits->SetUserMinRange(regionUserLimit->GetMinRemainingRange());
        GateMessage("Cuts", 3, "Region " << regionName
                    << " MinRange " << regionUserLimit->GetMinRemainingRange() << Gateendl);
        IsULimitDefined = true;
      }
      if(IsULimitDefined) region->SetUserLimits(userlimits);
      else {
        GateMessage("Cuts", 3, "Region " << regionName << " : no UserLimit" << Gateendl);
      }
    }
    ++it2;
  }

  // DS
  delete worldUserLimit;

  // FIXME --> does not work
  // DD("here");
  // if (this != phys) {
  //   DD("create opt");
  //   opt = new G4EmProcessOptions();
  //   if(mDEDXBinning>0)   opt->SetDEDXBinning(mDEDXBinning);
  //   if(mLambdaBinning>0) opt->SetLambdaBinning(mLambdaBinning);
  //   if(mEmin>0)          opt->SetMinEnergy(mEmin);
  //   if(mEmax>0)          opt->SetMaxEnergy(mEmax);
  //   opt->SetSplineFlag(mSplineFlag);
  //   opt->SetApplyCuts(true);
  // }

  GateMessageDec("Cuts",4,"GatePhysicsList::SetCuts() -- end" << Gateendl);
}
//-----------------------------------------------------------------------------


void GatePhysicsList::DisableAllCuts(G4String particleName)
{
    if (particleName == "e-")
    {
        this->electronCutDisabledByDefault = true;
    }
    if (particleName == "e+")
    {
        this->positronCutDisabledByDefault = true;
    }
    if (particleName == "gamma")
    {
        this->gammaCutDisabledByDefault = true;
    }
    if (particleName == "proton")
    {
        this->protonCutDisabledByDefault = true;
    }
}

//-----------------------------------------------------------------------------
void GatePhysicsList::SetCutInRegion(G4String particleName, G4String regionName, G4double cutValue)
{

  /*
    GateMessage("Cuts",3,"SetCutInRegion '" << regionName
    << "' for particle '" << particleName
    << "' : " << cutValue << Gateendl);
  */

  if(regionName=="world") regionName="DefaultRegionForTheWorld";

  RegionCutMapType::iterator it = mapOfRegionCuts.find(regionName);
  if (it == mapOfRegionCuts.end()) {
    // first time this region is concerned
    mapOfRegionCuts[regionName].gammaCut = -1;
    mapOfRegionCuts[regionName].electronCut = -1;
    mapOfRegionCuts[regionName].electronCutDisabledByDefault = false;
    mapOfRegionCuts[regionName].positronCut = -1;
    mapOfRegionCuts[regionName].positronCutDisabledByDefault = false;
    mapOfRegionCuts[regionName].protonCut = -1;
  }
  if (particleName == "gamma") mapOfRegionCuts[regionName].gammaCut = cutValue;
  if (particleName == "e-")
  {
      mapOfRegionCuts[regionName].electronCut = cutValue;
      mapOfRegionCuts[regionName].electronCutDisabledByDefault = false;
  }
  if (particleName == "e+") mapOfRegionCuts[regionName].positronCut = cutValue;
  if (particleName == "proton") mapOfRegionCuts[regionName].protonCut = cutValue;

  /*
    GateMessage("Cuts", 3, " Current Cut is g=" <<
    mapOfRegionCuts[regionName].gammaCut << " e-=" <<
    mapOfRegionCuts[regionName].electronCut << " e+=" <<
    mapOfRegionCuts[regionName].positronCut << Gateendl);
  */
  // DS verbose is in DefineCuts

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhysicsList::SetSpecialCutInRegion(G4String cutType, G4String regionName, G4double cutValue)
{
  if(regionName=="world") regionName="DefaultRegionForTheWorld";

  VolumeUserLimitsMapType::iterator it = mapOfVolumeUserLimits.find(regionName);
  if (it == mapOfVolumeUserLimits.end()) {
    // first time this region is concerned
    mapOfVolumeUserLimits[regionName]= new GateUserLimits();
  }

  if (cutType == "MaxStepSize") mapOfVolumeUserLimits[regionName]->SetMaxStepSize( cutValue);
  if (cutType == "MaxTrackLength") mapOfVolumeUserLimits[regionName]->SetMaxTrackLength( cutValue);
  if (cutType == "MaxToF") mapOfVolumeUserLimits[regionName]->SetMaxToF( cutValue);
  if (cutType == "MinKineticEnergy") mapOfVolumeUserLimits[regionName]->SetMinKineticEnergy( cutValue);
  if (cutType == "MinRemainingRange") mapOfVolumeUserLimits[regionName]->SetMinRemainingRange( cutValue);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhysicsList::GetCuts()
{
  DumpCutValuesTable();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhysicsList::SetOptDEDXBinning(G4int val)
{
  mDEDXBinning=val;
  //emPar->SetNumberOfBins(mDEDXBinning);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GatePhysicsList::SetOptLambdaBinning(G4int val)
{
  mLambdaBinning=val;
  //emPar->SetNumberOfBins(mDEDXBinning);
  //emPar->SetNumberOfBins(mLambdaBinning);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhysicsList::SetOptEMin(G4double val)
{
  mEmin=val;
  emPar->SetMinEnergy(mEmin);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhysicsList::SetOptEMax(G4double val)
{
  mEmax=val;
  emPar->SetMaxEnergy(mEmax);
}


//-----------------------------------------------------------------------------

#if G4VERSION_MAJOR >= 10 && G4VERSION_MINOR >= 5
//-----------------------------------------------------------------------------
void GatePhysicsList::SetUseICRU90DataFlag(G4bool val)
{
  GateMessage("Physic",1,"SetUseICRU90DataFlag set to " << (val?"TRUE":"FALSE"));
  mUseICRU90Data = val;
  emPar->SetUseICRU90Data(val);
}
//-----------------------------------------------------------------------------
#endif


//-----------------------------------------------------------------------------
void GatePhysicsList::SetEnergyRangeMinLimit(double val)
{
  //G4double limit=250*eV; // limit for diplay production cuts table
  mLowEnergyRangeLimit = val;
  // G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(val, G4ProductionCutsTable::GetProductionCutsTable()->GetHighEdgeEnergy());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4double GatePhysicsList::GetLowEdgeEnergy()
{
  return mLowEnergyRangeLimit;
}
//-----------------------------------------------------------------------------

//#endif
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePhysicsList *GatePhysicsList::singleton = 0;
//-----------------------------------------------------------------------------


#endif /* end #define GATEPHYSICSLIST_CC */

