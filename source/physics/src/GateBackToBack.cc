/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateBackToBack.hh"

#include "G4PhysicalConstants.hh"
#include "GateConstants.hh"
#include "GateSingletonDebugPositronAnnihilation.hh"

//-------------------------------------------------------------------------------------------------
GateBackToBack::GateBackToBack( GateVSource* source )
{
  m_source = source;
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
GateBackToBack::~GateBackToBack()
{
  ;
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateBackToBack::Initialize()
{
  m_source->SetNumberOfParticles( 2 );
  m_source->SetParticleTime( m_source->GetTime() );
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateBackToBack::GenerateVertex( G4Event* aEvent, G4bool accolinearityFlag)
{
  
  //G4cout<<"m_accoValue  = "<< m_source->GetAccoValue() << Gateendl;
  
  m_source->GeneratePrimaryVertex( aEvent );
  G4PrimaryParticle* particle = aEvent->GetPrimaryVertex( 0 )->GetPrimary( 0 );
  G4PrimaryParticle* particle1 = aEvent->GetPrimaryVertex( 0 )->GetPrimary( 1 );

  if( accolinearityFlag == true )
    {
        
      G4ThreeVector gammaMom = particle->GetMomentum();
      G4double accoValue = m_source->GetAccoValue();
      if (accoValue == 0.0){
        accoValue = 0.5*pi / 180;
      }
    
      G4double dev = CLHEP::RandGauss::shoot( 0.,accoValue / GateConstants::fwhm_to_sigma );
      G4double Phi1 = ( twopi * G4UniformRand() )/2. ;
      
      G4ThreeVector DirectionPhoton( sin( dev ) * cos( Phi1 ),
                                     sin( dev ) * sin( Phi1 ), cos( dev ) );
      
      // Scale vector to unit length before rotation, re-scale to original length after rotation:
      G4double gammaMom_mag = gammaMom.mag();
      gammaMom = gammaMom.unit();
      DirectionPhoton.rotateUz(gammaMom);
      
      particle1->SetMomentum( gammaMom_mag*DirectionPhoton.x(),
                              gammaMom_mag*DirectionPhoton.y(),
                              gammaMom_mag*DirectionPhoton.z() );
      particle->SetMomentum( -gammaMom_mag*gammaMom.x(),
                             -gammaMom_mag*gammaMom.y(),
                             -gammaMom_mag*gammaMom.z() );
      auto debugPositronAnnihilation = GateSingletonDebugPositronAnnihilation::GetInstance();
      if (debugPositronAnnihilation->GetDebugFlag()){
        G4double tmp;
        std::ofstream out;
        out.open(debugPositronAnnihilation->GetOutputFile(), std::ios::app | std::ios::out | std::ios::binary);
        tmp = gammaMom.angle(-DirectionPhoton);
        out.write((char*)&tmp, sizeof(double));
        out.close();
      }

      /*G4cout<<"gammaMom.x() = "<<gammaMom.x() << Gateendl;
      G4cout<<"gammaMom.y() = "<<gammaMom.y() << Gateendl;
      G4cout<<"gammaMom.z() = "<<gammaMom.z() << Gateendl;
      G4cout<<"DirectionPhoton.x() = "<<DirectionPhoton.x() << Gateendl;
      G4cout<<"DirectionPhoton.y() = "<<DirectionPhoton.y() << Gateendl;
      G4cout<<"DirectionPhoton.z() = "<<DirectionPhoton.z() << Gateendl;
      */
    }
  else
    {    

      G4ThreeVector gammaMom = particle->GetMomentum();
      particle1->SetMomentum( -gammaMom.x(),-gammaMom.y(),-gammaMom.z() );

     /* G4ThreeVector gammaMom1 = particle1->GetMomentum();

      if(gammaMom1.x()/gammaMom.x() != -1 ||gammaMom1.y()/gammaMom.y() != -1 || gammaMom1.z()/gammaMom.z() != -1){
      G4cout<<"gammaMom1.x() / gammaMom.x() = "<<gammaMom1.x()/gammaMom.x() << Gateendl;
      G4cout<<"gammaMom1.y() / gammaMom.y() = "<<gammaMom1.y()/gammaMom.y() << Gateendl;
      G4cout<<"gammaMom1.z() / gammaMom.z() = "<<gammaMom1.z()/gammaMom.z() << Gateendl;
      }*/
    }
}
//-------------------------------------------------------------------------------------------------
