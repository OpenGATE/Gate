/*##########################################
#developed by Zhenjie Cen
#
#CREATIS
#
#May 2016
##########################################
*/
#ifndef GATEXRAYBOUNDARYMESSENGER_HH
#define GATEXRAYBOUNDARYMESSENGER_HH

#include "GateVProcessMessenger.hh"

class GateVProcess;

class GateXrayBoundaryMessenger: public GateVProcessMessenger
{
    public:
        GateXrayBoundaryMessenger(GateVProcess* pb);
        virtual ~GateXrayBoundaryMessenger();

        virtual void BuildCommands(G4String base);
        virtual void SetNewValue(G4UIcommand*, G4String);

    protected:
};

#endif
