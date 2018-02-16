/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


//      ------------ GateSourceVoxellizedMessenger  ------
//           by G.Santin (18 Dec 2001)
// ************************************************************


#ifndef GATESOURCEVOXELLIZEDMESSENGER_H
#define GATESOURCEVOXELLIZEDMESSENGER_H 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "GateMessenger.hh"
#include "GateUIcmdWithAVector.hh"

class GateSourceVoxellized;
class GateClock;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//-----------------------------------------------------------------------------
class GateSourceVoxellizedMessenger: public GateMessenger
{
public:
  GateSourceVoxellizedMessenger(GateSourceVoxellized* source);
  ~GateSourceVoxellizedMessenger();

  virtual void SetNewValue(G4UIcommand*, G4String);

protected:
  GateSourceVoxellized*               m_source;
  G4UIcmdWith3VectorAndUnit*          PositionCmd;
  G4UIcmdWith3VectorAndUnit*          translateIsoCenterCmd;
  GateUIcmdWithAVector<G4String>*     ReaderInsertCmd;
  G4UIcmdWithoutParameter*            ReaderRemoveCmd;
};
//-----------------------------------------------------------------------------

#endif
