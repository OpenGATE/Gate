/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDistributionExponentialMessenger_h
#define GateDistributionExponentialMessenger_h 1

#include "GateDistributionMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithADoubleAndUnit;
class GateDistributionExponential;

class GateDistributionExponentialMessenger: public GateDistributionMessenger
{
  public:
    GateDistributionExponentialMessenger(GateDistributionExponential* itsDistribution,
    			     const G4String& itsDirectoryName="");
    virtual ~GateDistributionExponentialMessenger();
    inline GateDistributionExponential* GetDistributionExponential() const
    {return (GateDistributionExponential*)(GetDistribution());}
    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  private:
    G4UIcmdWithADoubleAndUnit *setLambdaCmd;
    G4UIcmdWithADoubleAndUnit *setAmplitudeCmd;
};

#endif
