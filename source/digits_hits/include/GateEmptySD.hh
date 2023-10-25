/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateEmptySD_h
#define GateEmptySD_h 1

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

/*! \class  GateEmptySD
    \brief  The GateEmptySD is a sensitive detector , derived from G4VSensitiveDetector,
    \brief  to be used for removing existing SD from G4SDManager 

    - This SD is not exposed to the user. It's sole purpose of existing is to fix issue with 
      GateMultiSensitiveDetector, for a volume with both SD and actor attached to it.

    - It will not ask for any dynamic memory allocation.
    
    - It will not be registered in the GateDigitizerMgr.
*/
//    Created by tontyoutoure@gmail.com 2023/10/24



class GateEmptySD : public G4VSensitiveDetector
{

  public:
      //! Constructor.
      //! The argument is the name of the sensitive detector
      GateEmptySD(const G4String& name);
      //! Destructor
      ~GateEmptySD() = default;

      //! Method overloading the virtual method Initialize() of G4VSensitiveDetector
      void Initialize(G4HCofThisEvent*HCE) override;

      //! Implementation of the pure virtual method ProcessHits().
      G4bool ProcessHits(G4Step*aStep,G4TouchableHistory*ROhist) override;


  private:
      GateHitsCollection * crystalHitsCollection;  //! Hit collection
      G4int collectionID;
      G4int HCID;

};




#endif
