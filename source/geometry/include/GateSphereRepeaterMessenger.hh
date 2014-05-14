/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSphereRepeaterMessenger_h
#define GateSphereRepeaterMessenger_h 1

#include "globals.hh"
#include "GateObjectRepeaterMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;
class G4UIcmdWithADouble;

class GateSphereRepeater;

/*! \class GateSphereRepeaterMessenger
    \brief Messenger for a GateSphereRepeater
    
    - GateSphereRepeaterMessenger -  by Delphine.Lazaro@imed.jussieu.fr
    
    - The GateSphereRepeaterMessenger inherits from the abilities/responsabilities
      of the GateObjectRepeaterMessenger base-class: creation and management
      of a Gate UI directory for a Gate object; UI commands "describe",
      'enable' and 'disable'.
      
    - In addition, it creates UI commands to manage a sphere repeater:
      'setRepeatNumberY', 'setRepeatNumberZ', 'autoCenter', 'setRadius', 
      'setAlphaAngle', 'setBetaAngle'

*/      
class GateSphereRepeaterMessenger: public GateObjectRepeaterMessenger
{
  public:
    GateSphereRepeaterMessenger(GateSphereRepeater* itsSphereRepeater);
   ~GateSphereRepeaterMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);

  public:
    virtual inline GateSphereRepeater* GetSphereRepeater() 
      { return (GateSphereRepeater*)GetObjectRepeater(); }
    
  private:
    G4UIcmdWithAnInteger*      SetRepeatNumberWithPhiCmd;
    G4UIcmdWithAnInteger*      SetRepeatNumberWithThetaCmd;
    G4UIcmdWithABool* 	       AutoCenterCmd;
    G4UIcmdWithABool* 	        EnableAutoRotationCmd;
    G4UIcmdWithABool* 	        DisableAutoRotationCmd;
    G4UIcmdWithADoubleAndUnit* ThetaAngleCmd;
    G4UIcmdWithADoubleAndUnit* PhiAngleCmd;
    G4UIcmdWithADoubleAndUnit* RadiusCmd;

};

#endif

