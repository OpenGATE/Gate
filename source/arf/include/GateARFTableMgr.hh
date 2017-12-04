/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/
#ifndef GateARFTableMgr_h
#define GateARFTableMgr_h
#include "globals.hh"
#include<map>
#include "G4ThreeVector.hh"
class GateARFSD;
class GateARFTable;
class GateARFTableMgrMessenger;
class GateARFTableMgr
  {
private:
  std::map<G4int, GateARFTable*> mArfTableMap;
  GateARFTableMgrMessenger* mMessenger;
  G4int mCurrentIndex;
  G4int mVerboseLevel;
  G4String mTableName;
  G4double mEnergyVector[40];
  GateARFSD* mArfSD;
  G4String mArfDataFileTxt;

  G4double mEnergyThreshHold;
  G4double mEnergyUpHold;
  G4double mEnergyOfReference;
  G4double mEnergyResolution;
  G4double mDistance;

  G4int mSaveArfTables;
  G4int mLoadArfTables;
  G4String mBinaryFilename;
  G4int mNumberOfBins;
public:
  GateARFTableMgr(const G4String & aName, GateARFSD* arfSD);
  ~GateARFTableMgr();
  void SaveARFToBinaryFile();
  void SetBinaryFile(const G4String & binaryFilename)
    {
    mSaveArfTables = 1;
    mBinaryFilename = binaryFilename;
    }
  ;
  void LoadARFFromBinaryFile(const G4String & binaryFilename);
  void SetNBins(const G4int & N);
  G4int GetNBins()
    {
    return mNumberOfBins;
    }
  ;
  void SetEThreshHold(const G4double & aET)
    {
    mEnergyThreshHold = aET;
    }
  ;
  void SetEUpHold(const G4double & aEU)
    {
    mEnergyUpHold = aEU;
    }
  ;
  void SetEReso(const G4double & energyResolution);
  void SetERef(const G4double & energyOfReference);
  G4int GetCurrentIndex()
    {
    return mCurrentIndex;
    }
  ;
  void AddaTable(GateARFTable* aTable);
  void SetVerboseLevel(const G4int & aL)
    {
    mVerboseLevel = aL;
    }
  ;
  void ComputeARFTablesFromEW(const G4String & filename);
  void ListTables();
  G4String GetName()
    {
    return mTableName;
    }
  ;
  void SetName(const G4String & aName)
    {
    mTableName = aName;
    }
  ;
  GateARFSD* GetARFSD()
    {
    return mArfSD;
    }
  ;
  G4int InitializeTables();
  void FillDRFTable(const G4int & iT,
                    const G4double & depositedEnergy,
                    const G4double & projectedX,
                    const G4double & projectedY);
  void SetNSimuPhotons(G4double*);
  void convertDRF2ARF();
  void CloseARFTablesRootFile();
  G4double ScanTables(const G4double & x, const G4double & y, const G4double & energy);
  void SetDistanceFromSourceToDetector(const G4double & aD)
    {
    mDistance = aD;
    }
  ;
  };

#endif

