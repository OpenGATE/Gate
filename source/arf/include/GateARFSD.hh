/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#ifndef GateARFSD_h
#define GateARFSD_h 1

#include "G4VSensitiveDetector.hh"
#include "GateHit.hh"

#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TBranch.h"
#include "GateProjectionSet.hh"
#include <map>

class G4Step;
class G4HCofThisEvent;
class G4TouchableHistory;

class GateVSystem;
class GateARFSDMessenger;

/*! \class  GateARFSD
 \brief  The GateARFSD is a sensitive detector , derived from G4VSensitiveDetector,
 \brief  to be attached to one or more volumes of a scanner

 - GateVolumeID - by Giovanni.Santin@cern.ch

 - A GateGeomColliSD can be attached to one or more volumes of a scanner. These volumes are
 essentially meant to be scintillating elements (crystals) but the GateGeomColliSD can also be
 attached to non-scintillating elements such as collimators, shields or septa.

 - A GateGeomColliSD can be attached only to those volumes that belong to a system (i.e. that
 are connected to an object derived from GateVSystem). Once a GateGeomColliSD has been attached
 to a volume that belongs to a given system, it is considered as attached to this system, and
 can be attached only to volumes that belong to the same system.

 - The GateGeomColliSD generates hits of the class GateHit, which are stored in a regular
 hit collection.
 */
class GateVVolume;
class GateARFTableMgr;

class GateARFData
  {
public:
  G4double mDepositedEnergy; // deposited energy
  G4double mProjectionPositionY;
  G4double mProjectionPositionX; // porjection position on the detection plane
  };

class GateARFSD: public G4VSensitiveDetector
  {

public:
  //! Constructor.
  //! The argument is the name of the sensitive detector
  GateARFSD(const G4String& pathname, const G4String & aName);
  //! Destructor
  ~GateARFSD();

  GateARFSD* Clone() const override;

  //! Method overloading the virtual method Initialize() of G4VSensitiveDetector
  void Initialize(G4HCofThisEvent*HCE) override;

  //! Implementation of the pure virtual method ProcessHits().
  //! This methods generates a GateHit and stores it into the SD's hit collection
  G4bool ProcessHits(G4Step*aStep, G4TouchableHistory*ROhist) override;

  //! Tool method returning the name of the hit-collection where the crystal hits are stored
  static inline const G4String& GetCrystalCollectionName()
    {
    return mArfHitCollectionName;
    }

  //! Returns the system to which the SD is attached
  inline GateVSystem* GetSystem()
    {
    return mSystem;
    }
  //! Set the system to which the SD is attached
  void SetSystem(GateVSystem* aSystem);

  G4String GetName()
    {
    return mName;
    }
  ;

  G4int PrepareCreatorAttachment(GateVVolume* aCreator);

  inline void setEnergyDepositionThreshold(G4double aT)
    {
    mEnergyDepositionThreshold = aT;
    }
  ;
  void SetInserter(GateVVolume* aInserter)
    {
    mInserter = aInserter;
    }
  ;
  GateVVolume* GetInserter()
    {
    return mInserter;
    }
  ;

  void computeTables();

  void AddNewEnergyWindow(const G4String & basename, const G4int & NFiles)
    {
    mEnergyWindows.push_back(basename);
    mEnergyWindowsNumberOfPrimaries.push_back(NFiles);
    }
  ;

  void ComputeProjectionSet(const G4ThreeVector & position,
                            const G4ThreeVector & direction,
                            const G4double & energy,
                            const G4double & weight,
                            bool addEmToArfCount = false,
                            unsigned int newHead = 1);

  void SetDepth(const G4double & aDepth)
    {
    mDetectorXDepth = aDepth;
    }

  G4double GetDepth()
    {
    return mDetectorXDepth;
    }

  void SetShortcutARF(const bool & boolean)
    {
    mShortcutARF = boolean;
    }

  G4int GetCopyNo()
    {
    return mHeadID;
    }
  ;
  void SetCopyNo(const G4int & aID)
    {
    mHeadID = aID;
    }
  ;

  void SetStage(const G4int & I)
    {
    mArfStage = I;
    }
  ;
  G4int GetStage()
    {
    return mArfStage;
    }
  ;

protected:
  GateVSystem* mSystem;                       //! System to which the SD is attached

private:
  GateHitsCollection * mArfHitCollection;  //! Hit collection
  static const G4String mArfHitCollectionName; //! Name of the hit collection
  GateARFSDMessenger* mMessenger;
  G4String mName;
  GateVVolume* mInserter;
  GateARFTableMgr* mArfTableMgr; // this manages the ARF tables for this ARF Sensitive Detector
  TFile* mFile;
  TTree* mSinglesTree;
  TTree* mNbOfPhotonsTree;

  ULong64_t mNbOfSourcePhotons;
  ULong64_t mNbOfSimuPhotons;
  ULong64_t mNbofGoingOutPhotons;
  ULong64_t mNbofGoingInPhotons;
  ULong64_t mNbofStraightPhotons;
  ULong64_t mNbofStoredPhotons;
  ULong64_t mNbOfGoodPhotons;
  ULong64_t mInCamera;
  ULong64_t mOutCamera;

  long unsigned int mNbOfRejectedPhotons;

  GateARFData mArfData;

  GateProjectionSet* mProjectionSet;
  G4int mHeadID;
  G4int mNbOfHeads;

  G4double mDetectorXDepth; // depth of the detector ( x length )

  std::vector<G4String> mEnergyWindows;
  std::vector<G4int> mEnergyWindowsNumberOfPrimaries;
  G4double mEnergyDepositionThreshold;
  G4int mArfStage;
  bool mShortcutARF;
  };

#endif

#endif
