/*!
  \class  GateDoILawMessenger


  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#ifndef GateVDoILawMessenger_h
#define GateVDoILawMessenger_h

#include "GateNamedObjectMessenger.hh"
#include "GateVDoILaw.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"



class GateDoILawMessenger : public GateNamedObjectMessenger {

	public :
        GateDoILawMessenger(GateVDoILaw* itsDoILaw);
        virtual ~GateDoILawMessenger() {}

        GateVDoILaw* GetDoILaw() const;
		void SetNewValue(G4UIcommand* cmdName, G4String val);


	private :


};

#endif
