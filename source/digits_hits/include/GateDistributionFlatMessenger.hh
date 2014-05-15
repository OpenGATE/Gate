/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDistributionFlatMessenger_h
#define GateDistributionFlatMessenger_h 1

#include "GateDistributionMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithADoubleAndUnit;
class GateDistributionFlat;

class GateDistributionFlatMessenger: public GateDistributionMessenger
{
  public:
    GateDistributionFlatMessenger(GateDistributionFlat* itsDistribution,
    			     const G4String& itsDirectoryName="");
    virtual ~GateDistributionFlatMessenger();
    inline GateDistributionFlat* GetDistributionFlat() const
    {return (GateDistributionFlat*)(GetDistribution());}
    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  private:
    G4UIcmdWithADoubleAndUnit *setMinCmd;
    G4UIcmdWithADoubleAndUnit *setMaxCmd;
    G4UIcmdWithADoubleAndUnit *setAmplitudeCmd;
};

#endif
