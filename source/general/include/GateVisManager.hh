/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
  
#ifndef GateVisManager_h
#define GateVisManager_h 1

#ifdef G4VIS_USE

#include "G4VisManager.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateVisManager: public G4VisManager {

public:

  GateVisManager ();

private:

  void RegisterGraphicsSystems ();

};

#endif

#endif
