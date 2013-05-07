/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/



#ifndef GATEPHYSICSLIST_HH
#define GATEPHYSICSLIST_HH

#include "G4VUserPhysicsList.hh"
#include "G4StepLimiter.hh"
#include "G4UserSpecialCuts.hh"
#include "G4EmCalculator.hh"
#include "G4ProductionCutsTable.hh"
#include "G4UnitsTable.hh"
#include "G4ProductionCuts.hh"
#include "G4EmProcessOptions.hh"

#include "GateMessageManager.hh"
#include "GateVProcess.hh"
#include "GateUserLimits.hh"

//class GateVProcess;
class GatePhysicsListMessenger;
class GatePhysicsList: public G4VUserPhysicsList
{
private:
  GatePhysicsList();

public:
  //GatePhysicsList();
  virtual ~GatePhysicsList();

  void ConstructProcess();
  void ConstructParticle();
  // CLT
  // Declaration: construction of the physics list from a G4 builder
  void ConstructPhysicsList(G4String name);

 
  void Print(G4String type, G4String particlename);
  void Print(G4String name);

  void RemoveProcesses(G4String process, G4String particle);
  void AddProcesses(G4String process, G4String particle);

  void PurgeIfFictitious();
 
  void Write(G4String file);

  std::vector<GateVProcess*> FindProcess(G4String name);  

  std::vector<G4String> GetTheListOfPBName() {return theListOfPBName;}

  // SetCuts() 
  void SetCuts();
  void DefineCuts(G4VUserPhysicsList * phys);
  void DefineCuts() { DefineCuts(this); }
  void SetCutInRegion(G4String particleName, G4String regionName, G4double cutValue);
  void SetSpecialCutInRegion(G4String cutType, G4String regionName, G4double cutValue);

  void GetCuts();

private:

  int mLoadState;
  GatePhysicsListMessenger * pMessenger;
  std::vector<G4String> theListOfPBName;


public:

  static GatePhysicsList *GetInstance()
  {   
    if (singleton == 0)
      {
        //std::cout << "creating PhysicscList..." << std::endl;
        singleton = new GatePhysicsList;
      }
    //else std::cout << "PhysicscList already created!" << std::endl;
    return singleton;
  }

  std::vector<GateVProcess*>* GetTheListOfProcesss()
  {
    return GateVProcess::GetTheListOfProcesses();
  }
  
  std::vector<G4String> mListOfStepLimiter;
  std::vector<G4String> mListOfG4UserSpecialCut;

private:

public: //FIXME
  static GatePhysicsList *singleton;


  struct ParticleCutType {
    G4double gammaCut;
    G4double electronCut;
    G4double positronCut;
    G4double protonCut;
  };

public: // FIXME
  typedef std::map<G4String, ParticleCutType> RegionCutMapType;
  RegionCutMapType mapOfRegionCuts;
  
  typedef std::map<G4String, GateUserLimits*> VolumeUserLimitsMapType;
  VolumeUserLimitsMapType mapOfVolumeUserLimits;

  std::list<G4ProductionCuts*> theListOfCuts;

  int mDEDXBinning;
  int mLambdaBinning;
  double mEmin;
  double mEmax;
  bool mSplineFlag;
  G4UserLimits * userlimits;
  

  G4VUserPhysicsList * mUserPhysicList;
  G4String mUserPhysicListName;

  void CheckPL_EM(const std::string & name);
  void CheckPL_Had(const std::string & name);
  bool m_PhysicList_EM_Flag;
  bool m_PhysicList_Had_Flag;
  std::string m_PhysicList_EM_name;
  std::string m_PhysicList_Had_name;

public:
  void SetOptDEDXBinning(G4int val);
  void SetOptLambdaBinning(G4int val);
  void SetOptEMin(G4double val);
  void SetOptEMax(G4double val);
  void SetOptSplineFlag(G4bool val);
  RegionCutMapType & GetMapOfRegionCuts() { return mapOfRegionCuts; }

private:
  G4EmProcessOptions *opt;

};


#endif /* end #define GATEPHYSICSLIST_HH */



