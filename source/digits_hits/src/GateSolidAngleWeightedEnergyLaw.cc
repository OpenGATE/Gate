/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#include "GateSolidAngleWeightedEnergyLaw.hh"

#include "Randomize.hh"

GateSolidAngleWeightedEnergyLaw::GateSolidAngleWeightedEnergyLaw(const G4String& itsName, G4double its_SzX, G4double its_SzY) :
    GateVEffectiveEnergyLaw(itsName),
    m_szX(its_SzX),
    m_szY(its_SzY)
{
    m_messenger = new GateSolidAngleWeightedEnergyLawMessenger(this);
}



G4double GateSolidAngleWeightedEnergyLaw::ComputeEffectiveEnergy(GateDigi digi) const {

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

    G4VoxelLimits limits;
    G4double min, max;
    G4AffineTransform at;
    digi.GetVolumeID().GetBottomCreator()->GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, min, max);

    double crystalThicknes=max-min;
    double zProp=0;
    if(m_zSense4Readout==1){
         zProp=crystalThicknes/2 -digi.GetLocalPos().getZ();
    }
    else if(m_zSense4Readout==(-1)){
         zProp=crystalThicknes/2 +digi.GetLocalPos().getZ();
    }
    else{
        G4cerr <<   Gateendl << "[GateSolidAngleWeightedEnergyLaw::ComputeEffectiveEnergy]]:\n"
            <<   "Sorry, but the zSense4Readout must be 1 (same sense as Z axis) or minus 1 (opposite sense) (" << GetZSense()  << ") is invalid\n";
        G4Exception( "GateSolidAngleWeightedEnergyLaw::ComputeEffectiveEnergy", "ComputeEffectiveEnergy", FatalException, "You must set the direction to 1 or -1 \n");
    }

    double fSolidAngle;
    if(zProp!=0){
        double alfap=m_szX/ (2*zProp);
        double betap=m_szY/ (2*zProp);
        fSolidAngle=(acos(sqrt((1+alfap*alfap+betap*betap)/((1+alfap*alfap)*(1+betap*betap)))))/pi;
    }
    else{
        fSolidAngle=0.5;
    }


    double energyf=(digi.GetEnergy())*fSolidAngle;
    return energyf;

}

void GateSolidAngleWeightedEnergyLaw::DescribeMyself (size_t indent) const {
    G4cout << "Solid Angle Weighted enegy";
    G4cout << GateTools::Indent(indent) << "size of the pixel in X direction:\t" << G4BestUnit(GetRectangleSzX(),"Length") << Gateendl;
    G4cout << GateTools::Indent(indent) << "size of the pixel in Y direction:\t" << G4BestUnit(GetRectangleSzY(),"Length") << Gateendl;
    G4cout << GateTools::Indent(indent) << "ReadoutSense in z:\t" <<GetZSense()  << Gateendl;
}



