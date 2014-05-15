/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


//      ------------ GateSourceVoxellizedMessenger  ------
//           by G.Santin (18 Dec 2001)
// ************************************************************


#ifndef GateSourceVoxellizedMessenger_h
#define GateSourceVoxellizedMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "GateMessenger.hh"

class GateSourceVoxellized;

class GateClock;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

#include "GateUIcmdWithAVector.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateSourceVoxellizedMessenger: public GateMessenger
{
public:
  GateSourceVoxellizedMessenger(GateSourceVoxellized* source);
  ~GateSourceVoxellizedMessenger();
  
  virtual void SetNewValue(G4UIcommand*, G4String);
    
private:
  GateSourceVoxellized*                        m_source;
    
  G4UIcmdWith3VectorAndUnit*          PositionCmd;
  GateUIcmdWithAVector<G4String>*     ReaderInsertCmd;
  G4UIcmdWithoutParameter*            ReaderRemoveCmd;

};

#endif

