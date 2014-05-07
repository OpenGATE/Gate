/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#ifndef GateARFSD_h
#define GateARFSD_h 1

#include "G4VSensitiveDetector.hh"
#include "GateCrystalHit.hh"

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
      
    - The GateGeomColliSD generates hits of the class GateCrystalHit, which are stored in a regular
      hit collection.
*/      
class GateVVolume;
class GateARFTableMgr;

class GateARFData
{ public :
      G4double m_Edep; // deposited energy
      G4double m_Y, m_X; // porjection position on the detection plane
};

class GateARFSD : public G4VSensitiveDetector
{

  public:
      //! Constructor.
      //! The argument is the name of the sensitive detector
      GateARFSD(const G4String& pathname, G4String aName );
      //! Destructor
      ~GateARFSD();

      //! Method overloading the virtual method Initialize() of G4VSensitiveDetector
      void Initialize(G4HCofThisEvent*HCE);
      
      //! Implementation of the pure virtual method ProcessHits().
      //! This methods generates a GateCrystalHit and stores it into the SD's hit collection
      G4bool ProcessHits(G4Step*aStep,G4TouchableHistory*ROhist);

      //! Tool method returning the name of the hit-collection where the crystal hits are stored
      static inline const G4String& GetCrystalCollectionName()
         { return theARFCollectionName; }

      //! Returns the system to which the SD is attached   
      inline GateVSystem* GetSystem()
         { return m_system;}
      //! Set the system to which the SD is attached
      void SetSystem(GateVSystem* aSystem);

      G4String GetName(){ return m_name; };

      G4int PrepareCreatorAttachment(GateVVolume* aCreator);

      inline void setEnergyDepositionThreshold(G4double aT){m_edepthreshold = aT;};
      void SetInserter( GateVVolume* aInserter ){ m_inserter = aInserter; };
      GateVVolume* GetInserter() { return m_inserter;}; 

      void computeTables();

      void AddNewEnergyWindow( G4String basename, G4int NFiles){ m_EnWin.insert( make_pair( basename,NFiles) ); };

      void ComputeProjectionSet(G4ThreeVector, G4ThreeVector, G4double );

      void SetDepth(G4double aDepth){m_XPlane = aDepth;}

      G4int GetCopyNo(){return headID;};
      void SetCopyNo(G4int aID){ headID = aID; };

      void SetStage( G4int I ){ m_ARFStage = I; };
      G4int GetStage(){ return m_ARFStage; };
      
      protected:
      GateVSystem* m_system;                       //! System to which the SD is attached

  private:
      GateCrystalHitsCollection * ARFCollection;  //! Hit collection
      
      static const G4String theARFCollectionName; //! Name of the hit collection

      GateARFSDMessenger* m_messenger;

      G4String m_name;

      GateVVolume* m_inserter;

      GateARFTableMgr* m_ARFTableMgr; // this manages the ARF tables for this ARF Sensitive Detector

      TFile* m_file;
      TTree* m_singlesTree;
      TTree* m_NbOfPhotonsTree;

      ULong64_t NbOfSourcePhotons, NbOfSimuPhotons, NbofGoingOutPhotons, NbofGoingInPhotons, NbofStraightPhotons;
      ULong64_t NbofStoredPhotons;
      ULong64_t NbOfGoodPhotons,IN_camera,OUT_camera;

      long unsigned int m_NbOfRejectedPhotons;

      ULong64_t nbofGoingIn;

      GateARFData  theData;

      GateProjectionSet* theProjectionSet;
      G4int headID;
      G4int NbOfHeads;

      G4double m_XPlane; // depth of the detector ( x length )

     std::map< G4String, G4int > m_EnWin;
     G4double m_edepthreshold;
     G4int m_ARFStage;
};




#endif

#endif
