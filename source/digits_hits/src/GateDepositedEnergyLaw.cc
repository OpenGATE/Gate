
#include "GateDepositedEnergyLaw.hh"



GateDepositedEnergyLaw::GateDepositedEnergyLaw(const G4String& itsName) :
    GateVEffectiveEnergyLaw(itsName)
{
    m_messenger = new GateDepositedEnergyLawMessenger(this);
}


G4double GateDepositedEnergyLaw::ComputeEffectiveEnergy(GatePulse pulse) const {
    return pulse.GetEnergy();
}


void GateDepositedEnergyLaw::DescribeMyself (size_t indent) const {
    G4cout << GateTools::Indent(indent) << "Deposited energy  law "<< Gateendl;
}
