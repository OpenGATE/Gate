/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateBackToBack.hh"

#include "G4PhysicalConstants.hh"

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
  
  //G4cout<<"m_accoValue  = "<< m_source->GetAccoValue() <<G4endl;
  
  m_source->GeneratePrimaryVertex( aEvent );
  G4PrimaryParticle* particle = aEvent->GetPrimaryVertex( 0 )->GetPrimary( 0 );
  G4PrimaryParticle* particle1 = aEvent->GetPrimaryVertex( 0 )->GetPrimary( 1 );

  if( accolinearityFlag == true )
    {
        
      G4ThreeVector gammaMom = particle->GetMomentum();

      G4double dev = CLHEP::RandGauss::shoot( 0.,m_source->GetAccoValue() / 2.35 );      
      
      G4double Phi1 = ( twopi * G4UniformRand() )/2. ;
      
      G4ThreeVector DirectionPhoton( sin( dev ) * cos( Phi1 ),
                                     sin( dev ) * sin( Phi1 ), cos( dev ) );
      
      DirectionPhoton.rotateUz(gammaMom);
      
      particle1->SetMomentum( DirectionPhoton.x(),
                              DirectionPhoton.y(), DirectionPhoton.z() );
      particle->SetMomentum( -gammaMom.x(),
                             -gammaMom.y(), -gammaMom.z() );
      
      /*G4cout<<"gammaMom.x() = "<<gammaMom.x() <<G4endl;
      G4cout<<"gammaMom.y() = "<<gammaMom.y() <<G4endl;
      G4cout<<"gammaMom.z() = "<<gammaMom.z() <<G4endl;
      G4cout<<"DirectionPhoton.x() = "<<DirectionPhoton.x() <<G4endl;
      G4cout<<"DirectionPhoton.y() = "<<DirectionPhoton.y() <<G4endl;
      G4cout<<"DirectionPhoton.z() = "<<DirectionPhoton.z() <<G4endl;
      */
    }
  else
    {    

      G4ThreeVector gammaMom = particle->GetMomentum();
      particle1->SetMomentum( -gammaMom.x(),-gammaMom.y(),-gammaMom.z() );

     /* G4ThreeVector gammaMom1 = particle1->GetMomentum();

      if(gammaMom1.x()/gammaMom.x() != -1 ||gammaMom1.y()/gammaMom.y() != -1 || gammaMom1.z()/gammaMom.z() != -1){
      G4cout<<"gammaMom1.x() / gammaMom.x() = "<<gammaMom1.x()/gammaMom.x() <<G4endl;
      G4cout<<"gammaMom1.y() / gammaMom.y() = "<<gammaMom1.y()/gammaMom.y() <<G4endl;
      G4cout<<"gammaMom1.z() / gammaMom.z() = "<<gammaMom1.z()/gammaMom.z() <<G4endl;
      }*/
    }
}
//-------------------------------------------------------------------------------------------------
