/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#include <CLHEP/Vector/ThreeVector.h>
#include <math.h>
#include "Randomize.hh"
#include "G4TransportationManager.hh"
#include "G4VPhysicalVolume.hh"
#include "G4PhysicalVolumeStore.hh"

#include "GateSPSPosDistribution.hh"
#include "GateMessageManager.hh"

//-----------------------------------------------------------------------------
GateSPSPosDistribution::GateSPSPosDistribution()
{
  Forbid = false;
//  VolName = "NULL";
  gNavigator = G4TransportationManager::GetTransportationManager()
    ->GetNavigatorForTracking();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateSPSPosDistribution::~GateSPSPosDistribution()
{ 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSPSPosDistribution::setVerbosity(G4int vL)
{
  verbosityLevel = vL; 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSPSPosDistribution::SetPositronRange( G4String positronType )
{
  positronrange = positronType ;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSPSPosDistribution::GeneratePositronRange()
{
  G4ThreeVector IsotopeRange;

  G4double rangex = 0. ;
  G4double rangey = 0. ;
  G4double rangez = 0. ;

  G4double Z = 0. ;
  G4double Y = 0. ;
  G4double X = 0. ;

  G4double R = 1 ;
  G4double Fit = 0 ;

  if( positronrange == "Fluor18" )
    {   
      while( R >= Fit || R > 2.0 )
        {
          X = CLHEP::RandFlat::shoot( -2.0, 2.0 ) ; // Rangemin ; Rangemax	 
          Y = CLHEP::RandFlat::shoot( -2.0, 2.0 ) ; // Rangemin ; Rangemax	 
          Z = CLHEP::RandFlat::shoot( -2.0, 2.0 ) ; // Rangemin ; Rangemax
     	 
          Fit = CLHEP::RandExponential::shoot( 0.7 ) ;
	 
          R = sqrt( X*X + Y*Y + Z*Z ) ;
	 
          rangex = X ;
          rangey = Y ;
          rangez = Z ;
        }
	  
      IsotopeRange.setX( rangex ) ;
      IsotopeRange.setY( rangey ) ;
      IsotopeRange.setZ( rangez ) ;
  
      particle_position += IsotopeRange ;

    }

  if( positronrange == "Carbon11" )
    {  
      while( R >= Fit || R > 4.0 )
        {
          X = CLHEP::RandFlat::shoot(-4.0, 4.0 ) ; // Rangemin ; Rangemax	 
          Y = CLHEP::RandFlat::shoot(-4.0, 4.0 ) ; // Rangemin ; Rangemax	 
          Z = CLHEP::RandFlat::shoot(-4.0, 4.0 ) ; // Rangemin ; Rangemax	 
	 
          Fit = CLHEP::RandExponential::shoot(1.4) ;
	 
          R = sqrt( X*X + Y*Y + Z*Z ) ;
	 
          rangex = X ;
          rangey = Y ;
          rangez = Z ;
        }
	  
      IsotopeRange.setX( rangex ) ;
      IsotopeRange.setY( rangey ) ;
      IsotopeRange.setZ( rangez ) ;
  
      particle_position += IsotopeRange ;
 
    }
 
  if( positronrange == "Oxygen15" )
    {  
      while( R >= Fit || R > 8.0 )
        {
          X = CLHEP::RandFlat::shoot(-8.0,8.0) ; // Rangemin ; Rangemax	 
          Y = CLHEP::RandFlat::shoot(-8.0,8.0) ; // Rangemin ; Rangemax	 
          Z = CLHEP::RandFlat::shoot(-8.0,8.0) ; // Rangemin ; Rangemax	 
	 
          Fit = CLHEP::RandExponential::shoot( 2.4 ) ;
	 
          R = sqrt( X*X + Y*Y + Z*Z ) ;
	 
          rangex = X ;
          rangey = Y ;
          rangez = Z ;
        }
	  
      IsotopeRange.setX( rangex ) ;
      IsotopeRange.setY( rangey ) ;
      IsotopeRange.setZ( rangez ) ;
 
      particle_position += IsotopeRange ;
  
    }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4ThreeVector GateSPSPosDistribution::GenerateOne()
{
/*
  G4bool srcconf = false;
  // G4int LoopCount = 0;
 
  if(Forbid == true)
    {
      srcconf = IsSourceForbidden();
      // if source in confined srcconf = true terminating the loop
      // if source isnt confined srcconf = false and loop continues
    }
  else
    srcconf = true;
*/

  G4bool shootAgain = true;
  G4int nbShoot = 0;
  G4int limitShoot = 1000000;
  while (shootAgain && nbShoot<limitShoot)
  {	
    if( GetPosDisType() != "NULL" )
      { 
        particle_position = G4SPSPosDistribution::GenerateOne() ;     
      }
    else 
      { 
        SetPosDisType("Point");
        particle_position = G4SPSPosDistribution::GenerateOne() ;
      }      
    if( positronrange != "NULL" )
      {
        GeneratePositronRange() ;
      }
    if (Forbid)
      {
        shootAgain = IsSourceForbidden();
      }
    else
      {
        break;
      }
    nbShoot++;
  }
  // To check if it seems that we are in infinite loop
  // meaning that all sources are forbidden
  if (nbShoot == limitShoot)
  {
    char tmp[20];
    sprintf(tmp,"%d",limitShoot);
		G4String msg = ((G4String)tmp)+" primaries were always generated in forbidden volumes !\n It seems that all sources are forbidden.";
    G4Exception("GateSPSPosDistribution::GenerateOne", "GenerateOne", FatalException, msg );
  }

  return particle_position ;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSPSPosDistribution::ForbidSourceToVolume( const G4String& Vname )
{


  G4VPhysicalVolume *tempPV      = NULL;
  G4PhysicalVolumeStore *PVStore = G4PhysicalVolumeStore::GetInstance();
  G4int      i = 0;
  G4bool found = false;
  while (!found && i<G4int(PVStore->size()))
  {
    tempPV = (*PVStore)[i];
    found  = tempPV->GetName() == Vname;
    if(verbosityLevel == 2)
      G4cout << i << " " << " " << tempPV->GetName() << " " << Vname << " " << found << Gateendl;
    i++;
  }
  // found = true then the volume exists else it doesnt.
  if(found == true)
    {
      if(verbosityLevel >= 1)
	G4cout << "Volume " << Vname << " exists\n";
      Forbid = true;
      ForbidVector.push_back(tempPV);
      // Modif DS: we write a confirmation message 
      G4cout << " Activity forbidden in volume '" << Vname << "' confirmed\n";
    }
  else
    {
      // Modif DS: for volume-name "NULL", we don't write the error message
      // so as not to confuse users
      if ( Vname != "NULL" )
        G4cout << " **** Error: Volume does not exist **** \n";
      G4cout << " Ignoring forbid condition for volume '" << Vname << "'\n";
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4bool GateSPSPosDistribution::IsSourceForbidden()
{
  // Method to check point is within the volume specified
  if(Forbid == false)
    G4cout << "Error: Forbid is false\n";
  G4ThreeVector null(0.,0.,0.);
  G4ThreeVector *ptr = &null;

  // Check particle_position is within VolName, if so true, 
  // else false
  G4VPhysicalVolume *currentVolume = gNavigator->LocateGlobalPointAndSetup(particle_position,ptr,true);

  G4bool isForbidden = false;
  for (std::vector<G4VPhysicalVolume*>::iterator itr=ForbidVector.begin(); itr!=ForbidVector.end(); itr++)
  {
    if (currentVolume==*itr)
    {
      isForbidden = true;
      break;
    }
  }
  return isForbidden;
}
//-----------------------------------------------------------------------------

