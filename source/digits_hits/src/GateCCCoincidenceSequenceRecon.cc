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
      // G4cout << '2';
      //The coincidence sorter has orderer the singles by  time to create the coincidence
      // outputPulse= new GateCoincidencePulse(*inputPulse);
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
      //difference entre random_suffle and shuffle
      seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::shuffle ( outputPulse->begin( ), outputPulse->end( ), std::default_random_engine(seed));
     // G4cout<<"seed "<<seed <<"engine "<<std::default_random_engine(seed)<<G4endl;
      break;
    case kSpatially :
      {
        outputPulse= new GateCoincidencePulse(*inputPulse);

    //First the pulses which are axially (z) closest to the source
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
          //Must be ordered spatially (from source) first scatterer then absorber
          outputPulse= outputPulseT;

          //First the pulses which are axially (z) closest to the source
          sort( outputPulse->begin( ), outputPulse->end( ), [ ]( const GatePulse& pulse1, const GatePulse& pulse2 )
          {
              return fabs(pulse1.GetGlobalPos().getZ() -pulse1.GetSourcePosition().getZ())< fabs(pulse2.GetGlobalPos().getZ() -pulse2.GetSourcePosition().getZ());

          });
      }
      else if(outputPulseT->size()==3){

          std::vector<GateCoincidencePulse> CoinPermutations;
          std::vector<double> qualityFactors (0);


          // std::cout << outputPulse->at(0)->GetGlobalPos().getZ()<< ' ' << outputPulse->at(1)->GetGlobalPos().getZ() << ' ' << outputPulse->at(2)->GetGlobalPos().getZ() << '\n';
          sort(  outputPulseT->begin( ) ,  outputPulseT->end() );
             do
             {
                 CoinPermutations.emplace_back(*outputPulseT);
              //std::cout << outputPulseT->at(0)->GetEnergy()<< ' ' << outputPulseT->at(1)->GetEnergy() << ' ' << outputPulseT->at(2)->GetEnergy() << '\n';
              //std::cout << outputPulseT->at(0)->GetGlobalPos().getZ()<< ' ' << outputPulseT->at(1)->GetGlobalPos().getZ() << ' ' << outputPulseT->at(2)->GetGlobalPos().getZ() << '\n';
             } while ( std::next_permutation(outputPulseT->begin(),outputPulseT->end())  );


          static const double E0 = 0.511044;//Electron rest energy

          double Ei = 0.0;         // energy incoming gamma
          double Eg = 0.0;         // energy scattered gamma
          double Ee = 0.0;         // energy recoil electron
          double CosPhiE = 0.0;  // cos(phi) computed by energies
          double CosPhiA = 0.0;  // cos(phi) computed by positions

          double dEg = 0.0;         // energy resolution scattered gamma
          double dEe = 0.0;         // energy resolution recoil electron

          double dCosPhiE2 = 0.0;  // square of the error for cos(phi) with energies
          double dCosPhiA2 = 0.0;  // square of the error for cos(phi) with positions

          double qualityf=0.0;



          for(unsigned int i=0; i<CoinPermutations.size(); i++){
              GateCoincidencePulse tempPulse=CoinPermutations.at(i);
               //Only enters once in the two loops because we have only three singles. So the parameters are obtained for the middle interaction
              for(unsigned int j=1; j<tempPulse.size()-1; j++){
                  //Only one time for each permutation
                  Ee = tempPulse.at(j)->GetEnergy();
                 // G4cout<<"energy e "<< Ee <<G4endl;
                  for (unsigned int k = j+1; k < tempPulse.size(); k++) {
                      Eg += tempPulse.at(k)->GetEnergy();
                      //only last interaction
                  }
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
              CosPhiA=std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0)/(v1Norm*v2Norm);


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
                   //NOT GOOD I SHOULD BE SOME INVALID THING oR DELETE FROM VECTOR BUT IT CAN NOT BE DONE INSIDE THE LOOP
                   qualityf=INVALID_Qf;
               }
               if (fabs(CosPhiA) >1 ){
                   //G4cout<<"cosphiA out of bounds: "<<CosPhiA<<G4endl;
                   // qualityf=-1;
                    qualityf=INVALID_Qf;
               }



                   qualityFactors.push_back( qualityf);

                 //IF qualityFActor=-1 throw the sequence (Do not choose it)




          }




          //find index os the smallest element in the vector or two quality factors equal to zero Must add checks
         std::vector<G4double>::iterator it=std::min_element(qualityFactors.begin(), qualityFactors.end());
         unsigned int posMin=std::distance(qualityFactors.begin(),it);
         //G4cout<<"pos min"<<posMin<<G4endl;

         while ( outputPulseT->size() ) {
           delete outputPulseT->back();
          outputPulseT->erase(outputPulseT->end()-1);
         }



         //G4cout<<"pulse size at min pos"<< CoinPermutations.at(posMin).size()<<G4endl;
         outputPulse= new GateCoincidencePulse(CoinPermutations.at(posMin));







          //G4cout << "despues de permutacoines"<<G4endl;
      }
      else{
          //reject the pulse ?
          //G4cout << "rejection ?";
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
