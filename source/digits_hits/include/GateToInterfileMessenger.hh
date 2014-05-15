/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateToInterfileMessenger_h
#define GateToInterfileMessenger_h 1

#include "GateOutputModuleMessenger.hh"

class GateToInterfile;

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;




class GateToInterfileMessenger: public GateOutputModuleMessenger
{
  public:
    GateToInterfileMessenger(GateToInterfile* gateToInterfile);
   ~GateToInterfileMessenger();

    void SetNewValue(G4UIcommand*, G4String);

  protected:
    GateToInterfile*             m_gateToInterfile;

//    G4UIcmdWithAString*      SetFileNameCmd;
};

#endif
