/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*!
  \class  GateMaterialMuHandler
  \author fabien.baldacci@creatis.insa-lyon.fr
 */

#ifndef GATEMATERIALMUHANDLER_HH
#define GATEMATERIALMUHANDLER_HH

#include "GateMuTables.hh"
#include "GatePhysicsList.hh"

#include "G4UnitsTable.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ParticleTable.hh"
#include "G4LossTableManager.hh"

#include <map>


using std::map;
using std::string;

struct MuStorageStruct
{
    double energy;
    int isAtomicShell;
    double atomicShellEnergy;
    double mu;
    double muen;

    MuStorageStruct(double e, int i, double a)
    {
      energy = e;
      isAtomicShell = i;
      atomicShellEnergy = a;
      mu = 0.;
      muen = 0.;
    }

    bool operator < (const MuStorageStruct& str) const
    {
        return (energy < str.energy);
    }
};

class GateMaterialMuHandler
{

public:

  static GateMaterialMuHandler *GetInstance()
  {
    if (singleton_MaterialMuHandler == 0)
    {
      singleton_MaterialMuHandler = new GateMaterialMuHandler();
    }
    return singleton_MaterialMuHandler;
  };


  ~GateMaterialMuHandler();
  double GetDensity(const G4MaterialCutsCouple *);
  double GetMuEnOverRho(const G4MaterialCutsCouple *, double);
  double GetMuEn(const G4MaterialCutsCouple *, double);
  double GetMuOverRho(const G4MaterialCutsCouple *, double);
  double GetMu(const G4MaterialCutsCouple *, double);
  G4String GetDatabaseName() { return mDatabaseName; }

  GateMuTable *GetMuTable(const G4MaterialCutsCouple *);
  
  void SetDatabaseName(G4String name) { mDatabaseName = name; }
  void SetEMin(double e) { mEnergyMin = e; }
  void SetEMax(double e) { mEnergyMax = e; }
  void SetENumber(int n) { mEnergyNumber = n; }
  void SetAtomicShellEMin(double e) { mAtomicShellEnergyMin = e; }
  void SetPrecision(double p) { mPrecision = p; }

private:

  GateMaterialMuHandler();

  // Initialization
  void Initialize();
  // - Precalculated coefficients (by element)
  void InitElementTable();
  void ConstructMaterial(const G4MaterialCutsCouple *);
  // - Complete simulation of coefficients
  void SimulateMaterialTable();
  void ConstructEnergyList(std::vector<MuStorageStruct> *, const G4Material *);
  void MergeAtomicShell(std::vector<MuStorageStruct> *);
  double ProcessOneShot(G4VEmModel *,std::vector<G4DynamicParticle*> *, const G4MaterialCutsCouple *, const G4DynamicParticle *);
  double SquaredSigmaOnMean(double , double , double);

  map<const G4MaterialCutsCouple *, GateMuTable*> mCoupleTable;
  GateMuTable** mElementsTable;
  int mElementNumber;
  G4String mDatabaseName;

  bool mIsInitialized;
  double mEnergyMin;
  double mEnergyMax;
  int mEnergyNumber;
  double mAtomicShellEnergyMin;
  double mPrecision;

  static GateMaterialMuHandler *singleton_MaterialMuHandler;
  
  // - fast acces
  void CheckLastCall(const G4MaterialCutsCouple *);
  const G4MaterialCutsCouple *mLastCouple;
  GateMuTable *mLastMuTable;

};


#endif
