

#include "GateConfiguration.h"
//#ifdef G4ANALYSIS_USE_ROOT



#ifndef GATECOMPTONCAMERAACTORMESSENGER_HH
#define GATECOMPTONCAMERAACTORMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"

#include "GateActorMessenger.hh"

class GateComptonCameraActor;

//-----------------------------------------------------------------------------
/// \brief Messenger of GateVImageActor
class GateComptonCameraActorMessenger : public GateActorMessenger
{
public:

  //-----------------------------------------------------------------------------
  /// Constructor with pointer on the associated sensor
  GateComptonCameraActorMessenger(GateComptonCameraActor * v);
  /// Destructor
  virtual ~GateComptonCameraActorMessenger();

  /// Command processing callback
  virtual void SetNewValue(G4UIcommand*, G4String);
  void BuildCommands(G4String base);

protected:

  /// Associated sensor
  GateComptonCameraActor * pActor;

  /// Command objects (These are the parameters that I can include in my macros)

  //G4UIcmdWithADoubleAndUnit * pEdepminCmd;
 // G4UIcmdWithABool          * pSaveAsText;

  G4UIcmdWithABool          * pSaveHitsTree;


}; // end class GateComptonCameraActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATECOMPTONCAMERAACTORMESSENGER_HH */
//#endif
