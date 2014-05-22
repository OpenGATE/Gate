/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDistributionArrayMessenger_h
#define GateDistributionArrayMessenger_h 1

#include "GateDistributionMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class GateVDistributionArray;

class GateDistributionArrayMessenger: public GateDistributionMessenger
{
  public:
    GateDistributionArrayMessenger(GateVDistributionArray* itsDistribution,
    			     const G4String& itsDirectoryName="");
    virtual ~GateDistributionArrayMessenger();
    inline GateVDistributionArray* GetVDistributionArray() const
    {return (GateVDistributionArray*)(GetDistribution());}
    void SetNewValue(G4UIcommand* aCommand, G4String aString);

  private:
    G4UIcmdWithAString        *setUnitY_Cmd;
    G4UIcmdWithAString        *setUnitX_Cmd;
    G4UIcmdWithAnInteger      *setAutoX_Cmd;
};

#endif
