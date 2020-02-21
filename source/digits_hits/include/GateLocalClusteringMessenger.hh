

#ifndef GateLocalClusteringMessenger_h
#define GateLocalClusteringMessenger_h 1

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

class GateLocalClustering;


class GateLocalClusteringMessenger: public GatePulseProcessorMessenger
{
  public:
    GateLocalClusteringMessenger(GateLocalClustering* itsPulseAdder);
    virtual ~GateLocalClusteringMessenger();


     inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
    inline  void SetNewValue2(G4UIcommand* aCommand, G4String aString);

    inline GateLocalClustering* GetLocalClustering()
      { return (GateLocalClustering*) GetPulseProcessor(); }

 private:



   G4UIcmdWithAString   *newVolCmd;
   std::vector<G4UIdirectory*> m_volDirectory;

   std::vector<G4UIcmdWithADoubleAndUnit*>    pAcceptedDist;
   std::vector<G4UIcmdWithABool*>     pRejectionMultipleClustersCmd;

   std::vector<G4String> m_name;
    G4int m_count;


};

#endif
