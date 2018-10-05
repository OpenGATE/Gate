

#ifndef GateVDoILaw_h
#define GateVDoILaw_h 1

#include "globals.hh"
#include "GateNamedObject.hh"
#include "GatePulse.hh"




class GateVDoILaw : public GateNamedObject {

	public :
         GateVDoILaw(const G4String& itsName);

        virtual ~ GateVDoILaw() {}
        virtual void ComputeDoI(GatePulse* pulse, G4ThreeVector axis) = 0;

  		// Implementation of the virtual method in GateNamedObject class
  		void Describe (size_t ident=0);

  		// Pure virtual method called in Describe()
  		virtual void DescribeMyself (size_t ident = 0) const = 0;

};

#endif
