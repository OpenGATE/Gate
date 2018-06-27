

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
    ///G4cout << "Inverse Square law for energy blurring\n";
    ///G4cout << GateTools::Indent(indent) << "Energy of Reference:\t" << G4BestUnit(GetEnergyRef(),"Energy") << Gateendl;
    ///G4cout << GateTools::Indent(indent) << "Resolution of Reference:\t" << GetResolution() << Gateendl;
}
