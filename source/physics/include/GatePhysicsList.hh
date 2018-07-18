/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details

  Modified version 11-10-13 for physics mixed with dna
  Author: LPC Clermont-Fd

  ----------------------*/

#ifndef GATEPHYSICSLIST_HH
#define GATEPHYSICSLIST_HH

#include "G4VModularPhysicsList.hh"
#include "G4StepLimiter.hh"
#include "G4UserSpecialCuts.hh"
#include "G4EmCalculator.hh"
#include "G4ProductionCutsTable.hh"
#include "G4UnitsTable.hh"
#include "G4ProductionCuts.hh"
#include "G4EmParameters.hh"

#include "GateMessageManager.hh"
#include "GateVProcess.hh"
#include "GateUserLimits.hh"

//class GateVProcess;
class GatePhysicsListMessenger;
class GatePhysicsList: public G4VModularPhysicsList
{
private:
  GatePhysicsList();

public:
  // Types
  struct ParticleCutType {
    G4double gammaCut;
    G4double electronCut;
    G4double positronCut;
    G4double protonCut;
  };
  typedef std::map<G4String, ParticleCutType> RegionCutMapType;
  typedef std::map<G4String, GateUserLimits*> VolumeUserLimitsMapType;

  // Functions
  static GatePhysicsList *GetInstance() { // static function must be here or icc, not in cc
    if (singleton == 0)
      {
        //std::cout << "creating PhysicscList...\n";
        singleton = new GatePhysicsList;
      }
    //else std::cout << "PhysicscList already created!\n";
    return singleton;
  }
  virtual ~GatePhysicsList();

  void ConstructProcess();
  void ConstructParticle();
  void ConstructPhysicsList(G4String name);
  void ConstructPhysicsListDNAMixed(G4String name);
  void ConstructProcessMixed();

  void Print(G4String type, G4String particlename);
  void Print(G4String name);

  void RemoveProcesses(G4String process, G4String particle);
  void AddProcesses(G4String process, G4String particle);
  void AddAtomDeexcitation();

  void PurgeIfFictitious();

  void Write(G4String file);

  std::vector<GateVProcess*> FindProcess(G4String name);
  std::vector<G4String> GetTheListOfPBName() {return theListOfPBName;}

  void SetCuts();
  void SetEmProcessOptions();
  void DefineCuts(G4VUserPhysicsList * phys);
  void DefineCuts() { DefineCuts(this); }
  void SetCutInRegion(G4String particleName, G4String regionName, G4double cutValue);
  void SetSpecialCutInRegion(G4String cutType, G4String regionName, G4double cutValue);
  void SetEnergyRangeMinLimit(double val);
  void GetCuts();
  G4String GetListOfPhysicsLists() { return mListOfPhysicsLists; }
  void SetOptDEDXBinning(G4int val);
  void SetOptLambdaBinning(G4int val);
  void SetOptEMin(G4double val);
  void SetOptEMax(G4double val);
  void SetOptSplineFlag(G4bool val);
  RegionCutMapType & GetMapOfRegionCuts() { return mapOfRegionCuts; }
  G4double GetLowEdgeEnergy();

  std::vector<G4String> mListOfStepLimiter;
  std::vector<G4String> mListOfG4UserSpecialCut;
  RegionCutMapType mapOfRegionCuts;

protected:
  int mLoadState;
  GatePhysicsListMessenger * pMessenger;
  std::vector<G4String> theListOfPBName;

  static GatePhysicsList *singleton;

  std::vector<GateVProcess*>* GetTheListOfProcesss();

  VolumeUserLimitsMapType mapOfVolumeUserLimits;
  std::list<G4ProductionCuts*> theListOfCuts;

  int mDEDXBinning;
  int mLambdaBinning;
  double mEmin;
  double mEmax;
  bool mSplineFlag;
  G4UserLimits * userlimits;

  // Physic list management
  G4VModularPhysicsList * mUserPhysicList;
  //Mixed EM and DNA Physics List
  G4VModularPhysicsList* emPhysicsListMixed;

  G4String mUserPhysicListName;
  G4String mListOfPhysicsLists;
  G4double mLowEnergyRangeLimit;

  G4EmParameters *emPar;
};


#endif /* end #define GATEPHYSICSLIST_HH */
