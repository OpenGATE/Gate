/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateVProcess
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


//-----------------------------------------------------------------------------
/// \class GateVProcess
/// \brief Base class for all Processes
//-----------------------------------------------------------------------------

#ifndef GATEVPROCESS_HH
#define GATEVPROCESS_HH

#include "globals.hh"
#include <vector>
#include <list>
#include "G4ios.hh"

#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4VProcess.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"

#include "G4HadronicProcess.hh"
#include "G4VEmProcess.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4HadronicInteraction.hh"

#include "GateGenericWrapperProcess.hh"

#include "GateListOfHadronicModels.hh"

#include "GateHadronicModelHeaders.hh"
#include "GateHadronicDataSetHeaders.hh"
#include "GateEmModelHeaders.hh"

#include "GateMessageManager.hh"
#include "GateActorManager.hh"

#include "G4MscStepLimitType.hh"


//#include "GatePhysicsList.hh" 
//class GatePhysicsList;

class GateVProcessMessenger;

class GateVProcess
{
public:
  GateVProcess(G4String name);
  virtual ~GateVProcess();
  
  void ConstructParticle(){;}
  void ConstructProcess();

  virtual void ConstructProcess(G4ProcessManager * manager)=0;
  virtual G4VProcess* CreateProcess(G4ParticleDefinition*)=0;

  virtual void kill(){}

  G4VProcess* GetProcess(){return pFinalProcess;}
  G4VProcess* GetUserModelProcess(){return pProcess;}

  virtual bool IsApplicable(G4ParticleDefinition * particule)=0;
  virtual bool IsModelApplicable(G4String model, G4ParticleDefinition * par){if(!par) model="";return true;}
   //Do nothing and return true if it is not defined in the PB (ie for EM processes)
  virtual bool IsDatasetApplicable(G4String cs, G4ParticleDefinition * par){if(!par) cs="";return true;}
   //Do nothing and return true if it is not defined in the PB (ie for EM processes)

  void AddDataSet(G4String csName);
  void AddModel(GateListOfHadronicModels *model);

  virtual void AddUserDataSet(G4String csName){if(csName) return;}
   //Do nothing if it is not defined in the PB (ie for EM processes)
  virtual void AddUserModel(GateListOfHadronicModels *model){if(model) return;}
   //Do nothing if it is not defined in the PB (ie for EM processes)

  void SetDefaultParticle(G4String particleName){theListOfDefaultParticles.push_back(particleName);}
  void AddToModelList(G4String modelName){theListOfModels.push_back(modelName);}
  void AddToDataSetList(G4String csName){theListOfDataSets.push_back(csName);}

  void SetProcessInfo(G4String processName) {ProcessInfo = processName; }

  void SetDataSet(G4String cs ,G4String particle = "Default");
  void UnSetDataSet(G4String cs ,G4String particle = "Default");
  void DataSetList(G4String particle="Default", G4int level=1, G4String symbol = "*", G4String symbol2 = "-");

  void SetEnergyRange(G4HadronicInteraction * hadInteraction,GateListOfHadronicModels *model);

  //void Verify(G4ParticleDefinition * particule); // *** Not really useful ***

  void Register();

  void PrintEnabledParticles(G4String particle = "All");
  void PrintEnabledParticlesToFile(G4String file);
  void CreateEnabledParticle(G4String particle="Default");
  void RemoveElementOfParticleList(G4String particle="Default");

  void SetModel(G4String model,G4String particle="Default");
  void UnSetModel(G4String model,G4String particle="Default");
  void ModelList(G4String particle="Default", G4int level=1, G4String symbol = "*", G4String symbol2 = "-");

  std::vector<G4String> FindParticleName(G4String name);
  std::vector<G4ParticleDefinition*> GetParticles(G4String param);
  
  void SetModelEnergyMax(G4String model, G4double energy, G4String particle = "All",G4String Option="NoOption");
  void SetModelEnergyMin(G4String model, G4double energy, G4String particle = "All",G4String Option="NoOption");
  void ClearModelEnergyRange(G4String model, G4String particle = "All");

  // void MessengerInitialization();
 
  bool IsEnabled(G4ParticleDefinition * par);
  
  G4String GetG4ProcessName() {return mG4ProcessName;}

  int GetNumberOfParticles() {return theListOfDefaultParticles.size();}

  std::vector<G4String> GetTheListOfDefaultParticles() {return theListOfDefaultParticles ;}
  std::vector<G4String> GetTheListOfDataSets() {return theListOfDataSets;}
  std::vector<G4String> GetTheListOfModels() {return theListOfModels  ;}
  
  std::vector<G4ParticleDefinition*> GetTheListOfEnabledParticles() {return theListOfEnabledParticles;}


  G4String GetProcessInfo() {return ProcessInfo;}


  static std::vector<GateVProcess*>* GetTheListOfProcesses() {
    static std::vector<GateVProcess*> theListOfProcesses;
    return &theListOfProcesses;
  }
  
	
  static void Delete();


  G4bool GetIsWrapperActive(){return mIsWrapperActive;}
  void   SetIsWrapperActive(G4bool b){mIsWrapperActive = b;}
  void   SetWrapperFactor(G4String part,G4double f);
  void SetWrapperCSEFactor(G4String part,G4double f);
  void SetKeepFilteredSec(G4bool b){mKeepSec=b;}
  G4bool AddFilter(G4String, G4String);

  void SetStepFunction(G4String part, G4double ratio, G4double finalRange);
  void SetLinearlosslimit(G4String part,  G4double limit);
  void SetMsclimitation(G4String part, G4String limit );



protected:  
  G4String mG4ProcessName;
  G4VProcess * pProcess;
  G4VProcess * pFinalProcess;
  //GenericWrapperProcess * wrapper; 

  G4bool mIsWrapperActive;
  G4bool mKeepSec;

  std::vector<G4String> theListOfDefaultParticles;
  G4String ProcessInfo;
  std::vector<G4String> theListOfDataSets;
  std::vector<G4String> theListOfModels;
  
  std::vector<G4ParticleDefinition*> theListOfEnabledParticles;

  std::vector<G4ParticleDefinition*> theListOfParticlesWithSelectedDS;
  std::vector<G4String> theListOfSelectedDataSets;

  std::vector<G4ParticleDefinition*> theListOfParticlesWithSelectedModels;
  std::vector<GateListOfHadronicModels *> theListOfSelectedModels;
  
  std::map<G4String,G4double> theListOfWrapperFactor;
  std::map<G4String,G4double> theListOfWrapperCSEFactor;
  std::map<G4String,GenericWrapperProcess*> theListOfWrapper;

  std::map<G4String,G4double> thelistOfRatioForStepFunction;
  std::map<G4String,G4double> thelistOfFinalRangeForStepFunction;

  std::map<G4String,G4double> thelistOfLinearLossLimit;
  
  std::map<G4String,G4MscStepLimitType> thelistOfMscLimitation;

  std::list<G4HadronicInteraction*> theListOfG4HadronicModels;
  G4ExcitationHandler* theHandler;

  GateVProcessMessenger * pMessenger;
  //static std::vector<GateVProcess*> theListOfProcesses;

  std::list<G4VProcess*> theListOfG4Processes;

};


// Macro for headers of process with 
#define MAKE_PROCESS_AUTO_CREATOR(name)			\
class name:public GateVProcess \
{ \
private: \
  name(); \
  virtual ~name(){};			\
public: \
  virtual G4VProcess* CreateProcess(G4ParticleDefinition*); \
  virtual void ConstructProcess(G4ProcessManager * manager); \
  virtual bool IsApplicable(G4ParticleDefinition * par); \
  static name *GetInstance() \
  {   \
    if (singleton_##name == 0) \
    { \
      singleton_##name = new name; \
    } \
    return singleton_##name; \
  } \
  virtual void kill() \
  { \
    if (NULL != singleton_##name) \
      { \
        delete singleton_##name; \
        singleton_##name = NULL; \
      } \
  } \
private: \
  static name * singleton_##name; \
}; \
  class name##Creator {			\
  public:					\
    name##Creator() {  name::GetInstance()->Register();  };  };	\
  static name##Creator name##Creator_var;
  
// Macro for headers of process with customisable models
#define MAKE_PROCESS_AUTO_CREATOR_WITH_MODEL(name)			\
class name:public GateVProcess \
{ \
private: \
  name(); \
  virtual ~name(){};			\
public: \
  virtual G4VProcess* CreateProcess(G4ParticleDefinition*); \
  virtual void ConstructProcess(G4ProcessManager * manager); \
  virtual bool IsApplicable(G4ParticleDefinition * par); \
  virtual bool IsModelApplicable(G4String model, G4ParticleDefinition * par); \
  virtual bool IsDatasetApplicable(G4String cs, G4ParticleDefinition * par);  \
  virtual void AddUserDataSet(G4String csName);				\
  virtual void AddUserModel(GateListOfHadronicModels *model);		\
  static name *GetInstance() \
  {   \
    if (singleton_##name == 0) \
    { \
      singleton_##name = new name; \
    } \
    return singleton_##name; \
  } \
  virtual void kill() \
  { \
    if (NULL != singleton_##name) \
      { \
        delete singleton_##name; \
        singleton_##name = NULL; \
      } \
  } \
private: \
  static name * singleton_##name; \
}; \
  class name##Creator {			\
  public:					\
    name##Creator() {  name::GetInstance()->Register();  }; };	\
  static name##Creator name##Creator_var;


#define MAKE_PROCESS_AUTO_CREATOR_CC(name) name *name::singleton_##name = 0; 

//    bool dummy##name =  GatePhysicsList::GetInstance()->Register(name::GetInstance());


#endif /* end #define GATEVPROCESS_HH */
