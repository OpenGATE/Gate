/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*! \file GateToOpticalRawMessenger.hh
   Created on   2012/07/09  by vesna.cuplov@gmail.com
   Implemented new class GateToOpticalRaw for Optical photons: write result of the projection.
*/


#ifndef GateToOpticalRawMessenger_h
#define GateToOpticalRawMessenger_h 1

#include "GateOutputModuleMessenger.hh"

class GateToOpticalRaw;

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;




class GateToOpticalRawMessenger: public GateOutputModuleMessenger
{
  public:
    GateToOpticalRawMessenger(GateToOpticalRaw* gateToOpticalRaw);
   ~GateToOpticalRawMessenger();

    void SetNewValue(G4UIcommand*, G4String);

  protected:
    GateToOpticalRaw*             m_gateToOpticalRaw;

};

#endif
