/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GatePhantomSD_h
#define GatePhantomSD_h 1

#include "G4VSensitiveDetector.hh"
#include "GatePhantomHit.hh"
class G4Step;
class G4HCofThisEvent;
class G4TouchableHistory;

class GatePhantomSD : public G4VSensitiveDetector
{

  public:
      GatePhantomSD(const G4String& name);
      ~GatePhantomSD();

      GatePhantomSD* Clone() const override;

      void Initialize(G4HCofThisEvent*HCE) override;
      G4bool ProcessHits(G4Step*aStep,G4TouchableHistory*ROhist) override;
      void EndOfEvent(G4HCofThisEvent*HCE) override;
     //! Tool method returning the name of the hit-collection where the phantom hits are stored
      static inline const G4String& GetPhantomCollectionName()
      	  { return thePhantomCollectionName; }
      void clear() override;
      void DrawAll() override;
      void PrintAll() override;

  private:
      GatePhantomHitsCollection * phantomCollection;
      static const G4String thePhantomCollectionName; //! Name of the hit collection

};




#endif
