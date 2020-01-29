

#ifndef GateVEffectiveEnergyLaw_h
#define GateVEffectiveEnergyLaw_h 1

#include "globals.hh"
#include "GateNamedObject.hh"
#include "GatePulse.hh"




class GateVEffectiveEnergyLaw : public GateNamedObject {

	public :
         GateVEffectiveEnergyLaw(const G4String& itsName);

        virtual ~ GateVEffectiveEnergyLaw() {}
        virtual G4double ComputeEffectiveEnergy(GatePulse pulse) const = 0;

  		// Implementation of the virtual method in GateNamedObject class
  		void Describe (size_t ident=0);

  		// Pure virtual method called in Describe()
  		virtual void DescribeMyself (size_t ident = 0) const = 0;

};

#endif
