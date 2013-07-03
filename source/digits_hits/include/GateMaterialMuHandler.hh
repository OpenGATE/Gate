/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
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
#include "G4Material.hh"
#include "G4ParticleTable.hh"
#include "G4LossTableManager.hh"

#include <map>


using std::map;
using std::string;

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
  double GetAttenuation(G4Material* material, double energy);
  double GetMu(G4Material* material, double energy);
  
  void SetElementsFolderName(G4String folder) { mElementsFolderName = folder; }
  void SetEMin(double e) { mEnergyMin = e; }
  void SetEMax(double e) { mEnergyMax = e; }
  void SetENumber(int n) { mEnergyNumber = n; }
  void SetAtomicShellEMin(double e) { mAtomicShellEnergyMin = e; }
  void SetShotNumber(int n) { mShotNumber = n; }
  
private:
  
  GateMaterialMuHandler();
  
  // Initialization
  void Initialize();
  // - Precalculated coefficients (by element)
  void InitElementTable();
  void ReadElementFile(int z);
  void ConstructMaterial(const G4Material *material);
  // - Complete simulation of coefficients
  void SimulateMaterialTable();
  double ProcessOneShot(G4VEmModel *,std::vector<G4DynamicParticle*> *, const G4MaterialCutsCouple *, const G4DynamicParticle *);
  double SquaredSigmaOnMean(double , double , double);
  
  map<G4String, GateMuTable*> mMaterialTable;
  GateMuTable** mElementsTable;
  int mNbOfElements;
  G4String mElementsFolderName;

  bool mIsInitialized;
  double mEnergyMin;
  double mEnergyMax;
  int mEnergyNumber;
  double mAtomicShellEnergyMin;
  int mShotNumber;
  
  
  static GateMaterialMuHandler *singleton_MaterialMuHandler;
  
};


#endif
