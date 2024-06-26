/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateDistributionFileMessenger_h
#define GateDistributionFileMessenger_h 1

#include "GateDistributionArrayMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithoutParameter;
class GateDistributionFile;

class GateDistributionFileMessenger: public GateDistributionArrayMessenger
{
  public:
    GateDistributionFileMessenger(GateDistributionFile* itsDistribution,
    			     const G4String& itsDirectoryName="");
    virtual ~GateDistributionFileMessenger();
    inline GateDistributionFile* GetDistributionFile() const
    {return (GateDistributionFile*)(GetDistribution());}
    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  private:
    G4UIcmdWithAString        *setFileNameCmd;
    G4UIcmdWithAnInteger      *setColXCmd;
    G4UIcmdWithAnInteger      *setColYCmd;
    G4UIcmdWithoutParameter   *readCmd;
    G4UIcmdWithoutParameter   *read2DCmd;
    G4UIcmdWithoutParameter   *autoXCmd;
};

#endif
