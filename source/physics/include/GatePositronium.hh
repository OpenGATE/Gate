/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GatePositronium_hh
#define GatePositronium_hh

#include "G4VDecayChannel.hh"
#include "G4DecayProducts.hh"

class GatePositronium {
public:
  GatePositronium(const G4String& name, G4double life_time,
                  G4int annihilation_gammas_number);
  G4double GetLifeTime() const;
  G4String GetName() const;
  G4int GetAnnihilationGammasNumber() const;
  G4DecayProducts *GetDecayProducts() const;

private:
  G4String fName;
  G4double fLifeTime = 0.0; //[ns]
  G4int fAnnihilationGammasNumber = 0;
  G4VDecayChannel *pDecayChannel = nullptr;
};
#endif
