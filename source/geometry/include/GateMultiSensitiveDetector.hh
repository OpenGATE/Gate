/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateMultiSensitiveDetector
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEMSD_HH
#define GATEMSD_HH

#include "GateNamedObject.hh"

#include "globals.hh"
#include "G4String.hh"
#include <iomanip>   
#include <vector>

#include "GateVActor.hh"

#include "G4SDManager.hh"

#include "G4VSensitiveDetector.hh"
#include "G4MultiFunctionalDetector.hh"

#include "GateMessageManager.hh"

class GateMultiSensitiveDetector : public G4VSensitiveDetector, public GateNamedObject 
{

public:
  GateMultiSensitiveDetector(G4String name);
  ~GateMultiSensitiveDetector();

  virtual void Initialize(G4HCofThisEvent*);
  virtual void EndOfEvent(G4HCofThisEvent*);
  virtual void clear();
  virtual void DrawAll();
  virtual void PrintAll();

  void SetSensitiveDetector(G4VSensitiveDetector * sd) {pSensitiveDetector = sd;}
  void SetMultiFunctionalDetector(G4MultiFunctionalDetector * mfd) {pMultiFunctionalDetector = mfd;}
  void SetMultiFunctionalDetector(G4String detectorName);
  void SetActor(GateVActor * actor);

  G4VSensitiveDetector * GetSensitiveDetector() {return pSensitiveDetector;}
  G4MultiFunctionalDetector* GetMultiFunctionalDetector() {return pMultiFunctionalDetector;}

protected:
  virtual G4bool ProcessHits(G4Step *aStep,G4TouchableHistory *ROhist);
  virtual G4int GetCollectionID(G4int i){return i;}

protected:
  G4VSensitiveDetector * pSensitiveDetector;
  G4MultiFunctionalDetector* pMultiFunctionalDetector;
};

#endif /* end #define GATEMSD_HH */
