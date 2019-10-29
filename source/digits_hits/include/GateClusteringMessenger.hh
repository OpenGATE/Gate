

#ifndef GateClusteringMessenger_h
#define GateClusteringMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include <vector>
class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateClustering;


class GateClusteringMessenger: public GatePulseProcessorMessenger
{
public:
    GateClusteringMessenger(GateClustering* itsPulseClus);
    virtual ~GateClusteringMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateClustering* GetClustering()
    { return (GateClustering*) GetPulseProcessor(); }

private:

    G4UIcmdWithADoubleAndUnit*   pAcceptedDistCmd;
    G4UIcmdWithABool* pRejectionMultipleClustersCmd;

};

#endif
