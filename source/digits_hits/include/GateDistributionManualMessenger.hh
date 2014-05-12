/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDistributionManualMessenger_h
#define GateDistributionManualMessenger_h 1

#include "GateDistributionArrayMessenger.hh"

class G4UIdirectory;
class GateUIcmdWithTwoDouble;
class G4UIcmdWithADouble;
class GateDistributionManual;

class GateDistributionManualMessenger: public GateDistributionArrayMessenger
{
  public:
    GateDistributionManualMessenger(GateDistributionManual* itsDistribution,
    			     const G4String& itsDirectoryName="");
    virtual ~GateDistributionManualMessenger();
    inline GateDistributionManual* GetDistributionManual() const
    {return (GateDistributionManual*)(GetDistribution());}
    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  private:
    GateUIcmdWithTwoDouble              *insPointCmd;
    G4UIcmdWithADouble                  *addPointCmd;
};

#endif
