/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateCCCoincidenceSequenceRecon.hh"
#include "G4UnitsTable.hh"
#include "GateCCCoincidenceSequenceReconMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"
#include "GateVVolume.hh"
#include "GateDetectorConstruction.hh"
#include "GateCrystalSD.hh"
#include "GateObjectChildList.hh"
#include "GateMaps.hh"


GateCCCoincidenceSequenceRecon::GateCCCoincidenceSequenceRecon(GateCoincidencePulseProcessorChain* itsChain,
                                                               const G4String& itsName)
    : GateVCoincidencePulseProcessor(itsChain,itsName),
      m_sequencePolicy(kSinglesTime)
{

    m_messenger = new GateCCCoincidenceSequenceReconMessenger(this);

}

GateCCCoincidenceSequenceRecon::~GateCCCoincidenceSequenceRecon()
{
    delete m_messenger;
}




GateCoincidencePulse* GateCCCoincidenceSequenceRecon::ProcessPulse(GateCoincidencePulse* inputPulse, G4int )
{
    if (!inputPulse ){
        if (nVerboseLevel>1)
            G4cout << "[GateCCCoincidenceSequenceRecon::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
        return 0;
    }

    //G4cout << "[GateCCC]"<<inputPulse->GetCoincID()<<"\n\n";
    // unsigned long long int  currentTime = (unsigned long long int)(inputPulse->GetTime()/picosecond);


    GateCoincidencePulse* outputPulse=0;

    switch(m_sequencePolicy) {
    case kSinglesTime :
        outputPulse= new GateCoincidencePulse(*inputPulse);
        break;
    case kLowestEnergyFirst :
        outputPulse= new GateCoincidencePulse(*inputPulse);
        sort( outputPulse->begin( ), outputPulse->end( ), [ ]( const GatePulse& pulse1, const GatePulse& pulse2 )
        {
            return pulse1.GetEnergy() < pulse2.GetEnergy();
        });

        break;
    case kRandomly :
        outputPulse= new GateCoincidencePulse(*inputPulse);
        seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle ( outputPulse->begin( ), outputPulse->end( ), std::default_random_engine(seed));
        break;
    case kSpatially :
    {
        outputPulse= new GateCoincidencePulse(*inputPulse);
        //First the pulses which are axially (z) closer to the source
        sort( outputPulse->begin( ), outputPulse->end( ), [ ]( const GatePulse& pulse1, const GatePulse& pulse2 )
        {
            return fabs(pulse1.GetGlobalPos().getZ() -pulse1.GetSourcePosition().getZ())< fabs(pulse2.GetGlobalPos().getZ() -pulse2.GetSourcePosition().getZ());

        });
    }

        break;
    case kRevanC_CSR:
    {

        GateCoincidencePulse* outputPulseT= new GateCoincidencePulse(*inputPulse);
        if(outputPulseT->size()==2){
            //First the pulses which are axially (z) closer to the source,
            outputPulse= outputPulseT;
            sort( outputPulse->begin( ), outputPulse->end( ), [ ]( const GatePulse& pulse1, const GatePulse& pulse2 )
            {
                return fabs(pulse1.GetGlobalPos().getZ() -pulse1.GetSourcePosition().getZ())< fabs(pulse2.GetGlobalPos().getZ() -pulse2.GetSourcePosition().getZ());

            });
        }
        else if(outputPulseT->size()==3){
            std::vector<GateCoincidencePulse> CoinPermutations;
            std::vector<double> qualityFactors (0);

            sort(outputPulseT->begin( ) ,  outputPulseT->end() );
            do
            {
                CoinPermutations.emplace_back(*outputPulseT);
                //std::cout << outputPulseT->at(0)->GetEnergy()<< ' ' << outputPulseT->at(1)->GetEnergy() << ' ' << outputPulseT->at(2)->GetEnergy() << '\n';
                //std::cout << outputPulseT->at(0)->GetGlobalPos().getZ()<< ' ' << outputPulseT->at(1)->GetGlobalPos().getZ() << ' ' << outputPulseT->at(2)->GetGlobalPos().getZ() << '\n';
            } while ( std::next_permutation(outputPulseT->begin(),outputPulseT->end())  );


            static const double E0 = 0.511044;//Electron rest energy

            double Ei = 0.0;         // energy incoming gamma
            double Eg = 0.0;         // energy scattered gamma
            double Ee = 0.0;         // energy recoil electron( in the second compton scattering)
            double CosPhiE = 0.0;  // cos(phi) computed by energies
            double CosPhiA = 0.0;  // cos(phi) computed by positions

            double dEg = 0.0;         // energy resolution scattered gamma
            double dEe = 0.0;         // energy resolution recoil electron

            double dCosPhiE2 = 0.0;  // square of the error for cos(phi) with energies
            double dCosPhiA2 = 0.0;  // square of the error for cos(phi) with positions

            double qualityf=0.0;


            //size should be 6
            for(unsigned int i=0; i<CoinPermutations.size(); i++){
                GateCoincidencePulse tempPulse=CoinPermutations.at(i);
                //Only enters once  because we have only three singles in the coincidence. So the parameters are obtained from the second interaction
                for(unsigned int j=1; j<tempPulse.size()-1; j++){
                    //deposited energy of the second interaction
                    Ee = tempPulse.at(j)->GetEnergy();
                    dEe=tempPulse.at(j)->GetEnergyError();
                    for (unsigned int k = j+1; k < tempPulse.size(); k++) {
                        //deposited energy in the last interaction
                        Eg += tempPulse.at(k)->GetEnergy();
                        dEg=tempPulse.at(k)->GetEnergyError();
                    }
                }

                if (Eg <= 0 ||Ee<0) {
                    std::cout<<"Eg is not positive!"<<std::endl;
                    qualityFactors.push_back( qualityf);
                    qualityf=INVALID_Qf;
                    break;
                }


                CosPhiE = 1 - E0/Eg + E0/(Ee+Eg);
                //std::cout << CoinPermutations.at(i).at(0)->GetGlobalPos().getZ()<< ' ' << CoinPermutations.at(i).at(1)->GetGlobalPos().getZ() << ' ' << CoinPermutations.at(i).at(2)->GetGlobalPos().getZ() << '\n';

                G4ThreeVector v1G=(CoinPermutations.at(i).at(1)->GetGlobalPos()-CoinPermutations.at(i).at(0)->GetGlobalPos());
                G4ThreeVector v2G=(CoinPermutations.at(i).at(2)->GetGlobalPos()-CoinPermutations.at(i).at(1)->GetGlobalPos());
                std::vector<double> v1;
                std::vector<double> v2;
                v1.push_back(v1G.getX());
                v1.push_back(v1G.getY());
                v1.push_back(v1G.getZ());
                v2.push_back(v2G.getX());
                v2.push_back(v2G.getY());
                v2.push_back(v2G.getZ());
                double v1Norm=sqrt(std::inner_product(v1.begin(), v1.end(), v1.begin(), 0.0));
                double v2Norm=sqrt(std::inner_product(v2.begin(), v2.end(), v2.begin(), 0.0));
                if(v1Norm>0 &&v2Norm>0)
                    CosPhiA=std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0)/(v1Norm*v2Norm);


                // MEGALIB computation method for geometrical angel error This is very time critical!

                dCosPhiA2 = pow(computeGeomScatteringAngleError(tempPulse),2);
                dCosPhiE2 = E0*E0/(Ei*Ei*Ei*Ei)*dEe*dEe+pow(E0/(Eg*Eg)-E0/(Ee+Eg)/(Ee+Eg),2)*dEg*dEg;

                //Needed errors in position and energy
                if(dCosPhiE2==0 && dCosPhiA2==0 ){

                    qualityf=(CosPhiE - CosPhiA)*(CosPhiE - CosPhiA);
                    // G4cout<<"quality factor "<< qualityf<<G4endl;
                }
                else{
                    qualityf=(CosPhiE - CosPhiA)*(CosPhiE - CosPhiA)/(dCosPhiE2 + dCosPhiA2);
                    //This is the error of the quality facotr
                    //2*fabs(CosPhiE - CosPhiA)*sqrt(dCosPhiE2 + dCosPhiA2);
                }

                if (fabs(CosPhiE) >1 ){
                    // G4cout<<"cosphiE out of bounds: "<<CosPhiE<<G4endl;
                    //qualityf=-1;
                    qualityf=INVALID_Qf;
                }
                if (fabs(CosPhiA) >1 ){
                    //G4cout<<"cosphiA out of bounds: "<<CosPhiA<<G4endl;
                    // qualityf=-1;
                    qualityf=INVALID_Qf;
                }

                qualityFactors.push_back( qualityf);
                //IF qualityFActor=-1, INVALID throw the sequence (Do not choose it) To be done

            }


            //find index of the smallest element in the vector or two quality factors equal to zero Must add checks
            std::vector<G4double>::iterator it=std::min_element(qualityFactors.begin(), qualityFactors.end());
            unsigned int posMin=std::distance(qualityFactors.begin(),it);
            //G4cout<<"pos min"<<posMin<<G4endl;

            while ( outputPulseT->size() ) {
                delete outputPulseT->back();
                outputPulseT->erase(outputPulseT->end()-1);
            }
            //G4cout<<"pulse size at min pos"<< CoinPermutations.at(posMin).size()<<G4endl;
            outputPulse= new GateCoincidencePulse(CoinPermutations.at(posMin));


        }
        else{
            outputPulse=0;
            return 0;

        }
    }
        break;
    default:
        std::cout << "default NOT sequence policy taken\n";
        break;
    }


    //G4cout << "returning outpuPulse of size="<<outputPulse->size()<<G4endl;
    return outputPulse;
}


//MEgalib method to compute it from  MERCSR::ComputePositionError Time critical
double GateCCCoincidenceSequenceRecon::computeGeomScatteringAngleError( GateCoincidencePulse coincP){

    double deltaF=0;
    if(coincP.size()==3){
        double Ax = coincP.at(0)->GetGlobalPos().getX();
        double Ay = coincP.at(0)->GetGlobalPos().getY();
        double Az = coincP.at(0)->GetGlobalPos().getZ();

        double Bx = coincP.at(1)->GetGlobalPos().getX();
        double By = coincP.at(1)->GetGlobalPos().getY();
        double Bz = coincP.at(1)->GetGlobalPos().getZ();

        double Cx = coincP.at(2)->GetGlobalPos().getX();
        double Cy = coincP.at(2)->GetGlobalPos().getY();
        double Cz = coincP.at(2)->GetGlobalPos().getZ();


        double dAx = coincP.at(0)->GetGlobalPosError().getX();
        double dAy = coincP.at(0)->GetGlobalPosError().getY();
        double dAz = coincP.at(0)->GetGlobalPosError().getZ();

        double dBx = coincP.at(1)->GetGlobalPosError().getX();
        double dBy = coincP.at(1)->GetGlobalPosError().getY();
        double dBz = coincP.at(1)->GetGlobalPosError().getZ();

        double dCx = coincP.at(2)->GetGlobalPosError().getX();
        double dCy = coincP.at(2)->GetGlobalPosError().getY();
        double dCz = coincP.at(2)->GetGlobalPosError().getZ();


        double Vx = Ax - Bx;
        double Vy = Ay - By;
        double Vz = Az - Bz;
        double Ux = Bx - Cx;
        double Uy = By - Cy;
        double Uz = Bz - Cz;
        double UdotV = Ux*Vx + Uy*Vy + Uz*Vz;


        double lengthV2 = Vx*Vx + Vy*Vy + Vz*Vz;
        double lengthU2 = Ux*Ux + Uy*Uy + Uz*Uz;
        double lengthV = sqrt(lengthV2);
        double lengthU = sqrt(lengthU2);

        double lengthVlengthU = lengthV * lengthU;
        double lengthV3lengthU = lengthV2 * lengthVlengthU;
        double lengthVlengthU3 = lengthVlengthU * lengthU2;
        if(lengthU!=0 && lengthV!=0){
            double DCosThetaDx1 = (Vx-Ux)/lengthVlengthU - Ux*UdotV/lengthVlengthU3+Vx*UdotV/lengthV3lengthU;
            double DCosThetaDx2 = Ux*UdotV/lengthVlengthU3-Vx/lengthVlengthU;
            double DCosThetaDx  =-Vx*UdotV/lengthV3lengthU+Ux/lengthVlengthU;
            double DCosThetaDy1 = (Vy-Uy)/lengthVlengthU - Uy*UdotV/lengthVlengthU3+  Vy*UdotV/lengthV3lengthU;
            double DCosThetaDy2 = Uy*UdotV/lengthVlengthU3-Vy/lengthVlengthU;
            double DCosThetaDy  =-Vy*UdotV/lengthV3lengthU+Uy/lengthVlengthU;
            double DCosThetaDz1 = (Vz-Uz)/lengthVlengthU - Uz*UdotV/lengthVlengthU3+ Vz*UdotV/lengthV3lengthU;
            double DCosThetaDz2 = Uz*UdotV/lengthVlengthU3-Vz/lengthVlengthU;
            double DCosThetaDz  =-Vz*UdotV/lengthV3lengthU+Uz/lengthVlengthU;

            deltaF = sqrt(DCosThetaDx1*DCosThetaDx1 * dBx*dBx + DCosThetaDy1*DCosThetaDy1 * dBy*dBy +
                          DCosThetaDz1*DCosThetaDz1 * dBz*dBz +
                          DCosThetaDx2*DCosThetaDx2 * dCx*dCx + DCosThetaDy2*DCosThetaDy2 * dCy*dCy +
                          DCosThetaDz2*DCosThetaDz2 * dCz*dCz +
                          DCosThetaDx *DCosThetaDx  * dAx*dAx + DCosThetaDy *DCosThetaDy  * dAy*dAy +
                          DCosThetaDz *DCosThetaDz  * dAz*dAz);

            if (deltaF == 0) {
                // In case they are on a straight line the above fails (no problem since this is anyway no good Compton event):
                if (fabs(UdotV/lengthU/lengthV) - 1 < 1E-10) {
                    // Let's do an approximation
                    double AvgResA = sqrt(dAx*dAx + dAy*dAy + dAz*dAz);
                    double AvgResB = sqrt(dBx*dBx + dBy*dBy + dBz*dBz);
                    double AvgResC = sqrt(dCx*dCx + dCy*dCy + dCz*dCz);

                    if(AvgResA!=0 ||AvgResB!=0||AvgResC!=0){

                        deltaF = fabs(cos(atan((AvgResA+AvgResB)/lengthV) + atan((AvgResB+AvgResC)/lengthU)));
                    }

                    //cout<<"Using crude position error approximation for hits on a straight line: "<<deltaF<<endl;
                }
            }
        }


    }

    return deltaF;

}

void GateCCCoincidenceSequenceRecon::DescribeMyself(size_t indent)
{
    G4cout << GateTools::Indent(indent) << "Sequence policy " <<m_sequencePolicy  << Gateendl;
}




void  GateCCCoincidenceSequenceRecon::SetSequencePolicy(const G4String& policy)
{
    //    if (policy=="singlesTime ")
    //        m_sequencePolicy=kSinglesTime;
    //SinglesTime set  in the constructor
    if (policy=="lowestEnergyFirst")
        m_sequencePolicy=kLowestEnergyFirst;
    else if (policy=="randomly")
        m_sequencePolicy=kRandomly;
    else if (policy=="axialDist2Source")
        m_sequencePolicy=kSpatially;
    else if (policy=="revanC_CSR")
        m_sequencePolicy=kRevanC_CSR;
    else {
        if (policy!="singlesTime")
            G4cout<<"WARNING : policy not recognized, using default : singlesTime\n";
        m_sequencePolicy=kSinglesTime;
    }
}
//------------------------------------------------------------------------------------------------------
