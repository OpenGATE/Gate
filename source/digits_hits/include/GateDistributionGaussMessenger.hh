/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDistributionGaussMessenger_h
#define GateDistributionGaussMessenger_h 1

#include "GateDistributionMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithADoubleAndUnit;
class GateDistributionGauss;

class GateDistributionGaussMessenger: public GateDistributionMessenger
{
  public:
    GateDistributionGaussMessenger(GateDistributionGauss* itsDistribution,
    			     const G4String& itsDirectoryName="");
    virtual ~GateDistributionGaussMessenger();
    inline GateDistributionGauss* GetDistributionGauss() const
    {return (GateDistributionGauss*)(GetDistribution());}
    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  private:
    G4UIcmdWithADoubleAndUnit *setMeanCmd;
    G4UIcmdWithADoubleAndUnit *setSigmaCmd;
    G4UIcmdWithADoubleAndUnit *setAmplitudeCmd;
};

#endif
