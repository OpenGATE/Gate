


#ifndef GateVEffectiveEnergyLawMessenger_h
#define GateVEffectiveEnergyLawMessenger_h

#include "GateNamedObjectMessenger.hh"
#include "GateVEffectiveEnergyLaw.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"



class GateEffectiveEnergyLawMessenger : public GateNamedObjectMessenger {

	public :
        GateEffectiveEnergyLawMessenger(GateVEffectiveEnergyLaw* itsEffectiveEnergyLaw);
        virtual ~GateEffectiveEnergyLawMessenger() {}

        GateVEffectiveEnergyLaw* GetEffectiveEnergyLaw() const;
		void SetNewValue(G4UIcommand* cmdName, G4String val);




};

#endif
