/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateARFSDMessenger_h
#define GateARFSDMessenger_h 1

#include "globals.hh"
#include "GateMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateARFSD;

class GateARFSDMessenger: public GateMessenger
{
  public:

    GateARFSDMessenger(GateARFSD* ARFSD);
    virtual ~GateARFSDMessenger();
    
    //! UI command interpreter method
    void SetNewValue(G4UIcommand*, G4String);



  protected:

 GateARFSD* m_ARFSD;

 G4UIcmdWithADoubleAndUnit* setDepth; 
 G4UIcmdWithADoubleAndUnit* setEThreshHoldcmd;
 };

#endif

