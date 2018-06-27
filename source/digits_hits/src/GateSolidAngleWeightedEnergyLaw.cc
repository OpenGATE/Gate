

#include "GateSolidAngleWeightedEnergyLaw.hh"



GateSolidAngleWeightedEnergyLaw::GateSolidAngleWeightedEnergyLaw(const G4String& itsName, G4double its_SzX, G4double its_SzY) :
    GateVEffectiveEnergyLaw(itsName),
    m_szX(its_SzX),
    m_szY(its_SzY)
{
    m_messenger = new GateSolidAngleWeightedEnergyLawMessenger(this);
}



G4double GateSolidAngleWeightedEnergyLaw::ComputeEffectiveEnergy(GatePulse pulse) const {

    if(m_szX < 0. ) {
        G4cerr << 	Gateendl << "[GateSolidAngleWeightedEnergyLaw::ComputeEffectiveEnergy]:\n"
           <<   "Sorry, but the size of the length of the square  in X direction (" << GetRectangleSzX() << ") is invalid\n";
        G4Exception( "GateSolidAngleWeightedEnergyLaw::ComputeEffectiveEnergy", "ComputeEffectiveEnergy", FatalException, "You must set the size od the square to calculate the subtended angles positive  \n");
	}
    else if (m_szY < 0.) {
        G4cerr <<   Gateendl << "[GateSolidAngleWeightedEnergyLaw::ComputeEffectiveEnergy]]:\n"
            <<   "Sorry, but the size of the lentgh of the square  in Y direction  (" << GetRectangleSzY()  << ") is invalid\n";
        G4Exception( "GateSolidAngleWeightedEnergyLaw::ComputeEffectiveEnergy", "ComputeEffectiveEnergy", FatalException, "You must set the size od the square to calculate the subtended angles positive \n");
	}

  ///  double alfa=m_szX/(2*pulse.Get);
    G4VoxelLimits limits;
    G4double min, max;
    G4AffineTransform at;
    pulse.GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, min, max);

    double crystalThicknes=max-min;
    double zProp;
    if(m_zSense4Readout==1){
         zProp=crystalThicknes/2 -pulse.GetLocalPos().getZ();
    }
    else if(m_zSense4Readout==(-1)){
         zProp=crystalThicknes/2 +pulse.GetLocalPos().getZ();
    }
    else{
        G4cerr <<   Gateendl << "[GateSolidAngleWeightedEnergyLaw::ComputeEffectiveEnergy]]:\n"
            <<   "Sorry, but the zSense4Readout must be 1 (same sense as Z axis) or minus 1 (opposite sense) (" << GetZSense()  << ") is invalid\n";
        G4Exception( "GateSolidAngleWeightedEnergyLaw::ComputeEffectiveEnergy", "ComputeEffectiveEnergy", FatalException, "You must set the direction to 1 or -1 \n");
    }


    double alfap=m_szX/ (2*zProp);
     double betap=m_szY/ (2*zProp);
     double fSolidAngle=(acos(sqrt((1+alfap*alfap+betap*betap)/((1+alfap*alfap)*(1+betap*betap)))))/pi;
     G4cout<<"zProp="<<zProp<<"  fsolidA="<<fSolidAngle<<"  Energy="<<pulse.GetEnergy()<<"  effectiveEnergy="<<pulse.GetEnergy()*fSolidAngle<<G4endl;


    return (pulse.GetEnergy())*fSolidAngle;

}

void GateSolidAngleWeightedEnergyLaw::DescribeMyself (size_t indent) const {
    ///G4cout << "Inverse Square law for energy blurring\n";
    ///G4cout << GateTools::Indent(indent) << "Energy of Reference:\t" << G4BestUnit(GetEnergyRef(),"Energy") << Gateendl;
    ///G4cout << GateTools::Indent(indent) << "Resolution of Reference:\t" << GetResolution() << Gateendl;
}
