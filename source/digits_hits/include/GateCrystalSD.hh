/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateCrystalSD_h
#define GateCrystalSD_h 1

#include "G4VSensitiveDetector.hh"
#include "G4SDManager.hh"

#include "GateHit.hh"
class G4Step;
class G4HCofThisEvent;
class G4TouchableHistory;

class GateVVolume;
class GateVSystem;

//! List of typedefs for the multi-system usage.
typedef std::vector<GateVSystem*> GateSystemList;
typedef GateSystemList::iterator GateSystemIterator;
typedef GateSystemList::const_iterator GateSystemConstIterator;

/*! \class  GateCrystalSD
    \brief  The GateCrystalSD is a sensitive detector , derived from G4VSensitiveDetector,
    \brief  to be attached to one or more volumes of a scanner

    - GateVolumeID - by Giovanni.Santin@cern.ch

    - A GateCrystalSD can be attached to one or more volumes of a scanner. These volumes are
      essentially meant to be scintillating elements (crystals) but the GateCrystalSD can also be
      attached to non-scintillating elements such as collimators, shields or septa.

    - A GateCrystalSD can be attached only to those volumes that belong to a system (i.e. that
      are connected to an object derived from GateVSystem). Once a GateCrystalSD has been attached
      to a volume that belongs to a given system, it is considered as attached to this system, and
      can be attached only to volumes that belong to the same system.

    - The GateCrystalSD generates hits of the class GateHit, which are stored in a regular
      hit collection.
*/
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.
//    Last modification for GND in Feb 2023 by Olga.Kochebina@cea.fr for NewGateDigitizer
//    May 2023: O. Kochebina: variables for ComptonCamera (from GateComptonCameraActor) are added: Ef, Ei, nCurrentCompt, nCurrentConv, nCurrentRayl, sourceEnergy, source PDG
//		       Important!!! Some of functionalities where not reimplemented!!!
//					They are (from  GateComptonCameraActorMessenger): /specifysourceParentID and /parentIDFileName



class GateCrystalSD : public G4VSensitiveDetector
{

  public:
      //! Constructor.
      //! The argument is the name of the sensitive detector
      GateCrystalSD(const G4String& name);
      //! Destructor
      ~GateCrystalSD();

      GateCrystalSD* Clone() const override;

      //! Method overloading the virtual method Initialize() of G4VSensitiveDetector
      void Initialize(G4HCofThisEvent*HCE) override;

      //! Implementation of the pure virtual method ProcessHits().
      //! This methods generates a GateHit and stores it into the SD's hit collection
      G4bool ProcessHits(G4Step*aStep,G4TouchableHistory*ROhist) override;


      //! Returns the system to which the SD is attached
      //! Returns the system to which the SD is attached
      inline GateVSystem* GetSystem()
      { return m_system;}
      //! Set the system to which the SD is attached
      void SetSystem(GateVSystem* aSystem);

      //! Set the system to which the SD is attached
      //! next methods are for the multi-system approach
      inline GateSystemList* GetSystemList() const { return m_systemList; }
      void AddSystem(GateVSystem* aSystem);
      GateVSystem* FindSystem(GateVolumeID volumeID);
      GateVSystem* FindSystem(G4String& systemName);

      G4int PrepareCreatorAttachment(GateVVolume* aCreator);

  protected:
     GateVSystem* m_system;                           //! System to which the SD is attached //mhadi_obso obsollete, because we use the multi-system approach
     GateSystemList* m_systemList = nullptr;          //! System list instead of one system
  private:
      GateHitsCollection * crystalHitsCollection;  //! Hit collection
      G4int collectionID;
      G4int HCID;

      static GateCrystalSD*  theSD;

      G4bool m_IsNewEvent;
      G4double m_Ef_oldPrimary;
      G4bool mParentIDSpecificationFlag;

      G4double m_sourceEnergy;
      G4int  m_sourcePDG;

};




#endif
