


#ifndef GateDepositedEnergyLaw_h
#define GateDepositedEnergyLaw_h 1

#include "GateVEffectiveEnergyLaw.hh"
#include "GateDepositedEnergyLawMessenger.hh"

#include "GateTools.hh"
#include "G4UnitsTable.hh"
#include "G4VoxelLimits.hh"

class GateDepositedEnergyLaw  : public GateVEffectiveEnergyLaw {

public :
    GateDepositedEnergyLaw(const G4String& itsName);
    virtual ~GateDepositedEnergyLaw() {delete m_messenger;}
    virtual G4double ComputeEffectiveEnergy(GatePulse plse) const;
    virtual void DescribeMyself (size_t ident=0) const;

private :

    GateDepositedEnergyLawMessenger* m_messenger;

};

#endif
