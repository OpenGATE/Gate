/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateMuCalculatorActor :
  \brief
*/

#include "G4PhysicalConstants.hh"
#include "G4CompositeEMDataSet.hh"
#include "G4UnitsTable.hh"
#include "GateDiffCrossSectionActor.hh"
#include "GateMiscFunctions.hh"
#include "G4ProcessManager.hh"
#include "G4MaterialTable.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Electron.hh"
#include "G4LogLogInterpolation.hh"
#include <list>

//-----------------------------------------------------------------------------
GateDiffCrossSectionActor::GateDiffCrossSectionActor( G4String name, G4int depth):GateVActor( name, depth), scatterFunctionData(0), formFactorData(0)
{
  GateDebugMessageInc( "Actor", 4, "GateDiffCrossSectionActor() -- begin" << G4endl);
  //scatterFunctionData = 0;
  mUserEnergy = 1.0*keV;
  mUserEnergyList.push_back(1.0*keV);

  mUserAngle = 0.0*radian;
  mUserAngleList.push_back(0.0*radian);

  mUserMaterial = "Water";
  mExitFileNameSF = "ScatterFunction_results.txt";
  mExitFileNameFF = "FormFactor_results.txt";
  mExitFileNameDCScompton = "DiffCrossSectionCompton_results.txt";
  mExitFileNameDCSrayleigh = "DiffCrossSectionRayleigh_results.txt";

  pMessenger = new GateDiffCrossSectionActorMessenger( this);
  GateDebugMessageDec("Actor", 4, "GateDiffCrossSectionActor( ) -- end" << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateDiffCrossSectionActor::~GateDiffCrossSectionActor()
{
  delete pMessenger;
  delete scatterFunctionData;
  delete formFactorData;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDiffCrossSectionActor::ReadListEnergy(G4String energylist)
{
  G4double energy;
  std::ifstream inEnergyFile;
  mUserEnergyList.clear( );

  //Read energy list
  inEnergyFile.open( energylist);
  if( !inEnergyFile ) { // file couldn't be opened
    G4cout << "Error: file could not be opened" << G4endl;
    exit( 1);
  }
  while ( !inEnergyFile.eof( ))
    {
      inEnergyFile >> energy;
      energy = energy*keV;
      mUserEnergyList.push_back( energy);
    }
  inEnergyFile.close( );
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDiffCrossSectionActor::ReadListAngle( G4String anglelist)
{
  G4double angle;
  std::ifstream inAngleFile;
  mUserAngleList.clear( );

  //Read angle list
  inAngleFile.open( anglelist);
  if( !inAngleFile ) { // file couldn't be opened
    G4cout << "Error: file could not be opened" << G4endl;
    exit( 1);
  }
  while ( !inAngleFile.eof( ))
    {
      inAngleFile >> angle;
      angle = angle*radian;
      mUserAngleList.push_back( angle);
    }
  inAngleFile.close( );
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDiffCrossSectionActor::ReadMaterialList(G4String materiallist)
{
  G4String material;
  std::ifstream inMaterialFile;
  mUserMaterialList.clear( );
  inMaterialFile.open( materiallist);
  if( !inMaterialFile ) { // file couldn't be opened
    G4cout << "Error: file could not be opened" << G4endl;
    exit( 1);
  }
  while ( !inMaterialFile.eof( ))
    {
      inMaterialFile >> material;
      mUserMaterialList.push_back( material);
    }
  inMaterialFile.close( );
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDiffCrossSectionActor::SetEnergy( G4double energyValue)
{
  mUserEnergy = energyValue;
  mUserEnergyList.clear( );
  mUserEnergyList.push_back( mUserEnergy);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDiffCrossSectionActor::SetAngle( G4double angleValue)
{
  mUserAngle = angleValue;
  mUserAngleList.clear( );
  mUserAngleList.push_back( mUserAngle);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDiffCrossSectionActor::SetMaterial(G4String material)
{
  mUserMaterial = material;
  mUserMaterialList.clear( );
  mUserMaterialList.push_back( mUserMaterial);
}


//-----------------------------------------------------------------------------
void GateDiffCrossSectionActor::Initialise()
{
  delete scatterFunctionData;
  G4VDataSetAlgorithm* scatterInterpolation = new G4LogLogInterpolation;
  G4String scatterFile = "comp/ce-sf-";
  scatterFunctionData = new G4CompositeEMDataSet( scatterInterpolation, 1/cm, 1.);
  scatterFunctionData->LoadData(scatterFile);

  delete formFactorData;
  G4VDataSetAlgorithm* ffInterpolation = new G4LogLogInterpolation;
  G4String formFactorFile = "rayl/re-ff-";
  formFactorData = new G4CompositeEMDataSet( ffInterpolation, 1/cm, 1.);
  formFactorData->LoadData(formFactorFile);

}


//-----------------------------------------------------------------------------
/// Construct
void GateDiffCrossSectionActor::Construct()
{
  GateDebugMessageInc("Actor", 4, "GateDiffCrossSectionActor -- Construct - begin" << G4endl);
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  //  EnableBeginOfEventAction(true);
  //  EnablePreUserTrackingAction(false);
  //  EnableUserSteppingAction(true);

  // Print information
  GateMessage("Actor", 1,
              "\tDiffCrossSection DiffCrossSectionActor    = '" << GetObjectName() << "'" << G4endl);


  //  ResetData();
  GateMessageDec("Actor", 4, "GateDiffCrossSectionActor -- Construct - end" << G4endl);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDiffCrossSectionActor::BeginOfRunAction(const G4Run*)
{
  G4cout << "*******************************************" << G4endl;
  G4double scatteringFunction, formFactor, Theta, cosT, sinT, sinT2, E_in, Ecompton, eRadiusTerm, DCSKleinNishinaTerm1, DCSKleinNishinaTerm2, DCSKleinNishina, DCSThomsonTerm1, DCSThomson, xi, DCScompton, DCSrayleigh, e0m;
  //****Ration energy (E / Me C^2)
  const G4MaterialTable* matTbl = G4Material::GetMaterialTable();
  G4int Z;

  std::list< std::pair< G4double, G4double > > listAngleSF, listAngleDCScompton, listAngleFF, listAngleDCSrayleigh;
  std::list< std::pair< G4double, G4double > >::iterator iListAngleSF, iListAngleDCScompton, iListAngleFF, iListAngleDCSrayleigh;

  std::map< G4double, std::list< std::pair< G4double, G4double > > > mapEnergySF, mapEnergyDCScompton, mapEnergyFF, mapEnergyDCSrayleigh;
  std::map< G4double, std::list< std::pair< G4double, G4double > > >::iterator iMapEnergySF, iMapEnergyDCScompton, iMapEnergyFF, iMapEnergyDCSrayleigh;

  std::map< G4String, std::map< G4double, std::list< std::pair< G4double, G4double > > > > mapGeneralSF, mapGeneralDCScompton, mapGeneralFF, mapGeneralDCSrayleigh;
  std::map< G4String, std::map< G4double, std::list< std::pair< G4double, G4double > > > >::iterator iMapGeneralSF, iMapGeneralDCScompton, iMapGeneralFF, iMapGeneralDCSrayleigh;
  //std::vector< G4String> nameprocess;
  //std::vector< G4double> muvalue;
  this->Initialise();

  //****Loop for the different materials presents
  for( size_t k = 0; k < G4Material::GetNumberOfMaterials(); k++)
    {
      if( ( *matTbl)[k]->GetName() == mUserMaterial.c_str() )
	{
         //for( size_t n = 0; n < G4Element::GetNumberOfElements(); n++)
         //   {
         //    (*((*matTbl)[k]->GetElementVector()))[n]->GetName()


          //****Selection the first element into the material to take Z value

	  Z = (*matTbl)[k]->GetElement(0)->GetZ();
	  //****Loop for the differents energies choosen
	  for( size_t l = 0; l < mUserEnergyList.size(); l++)
	    {
              //****Incident energy
	      E_in = mUserEnergyList[l];
	      //****Loop for scattering angle emission
	      for( size_t j = 0; j < mUserAngleList.size(); j++)
		{
                  //****Scatter angle in polar coordinates
		  Theta = mUserAngleList[j];
                  cosT = cos(Theta);
                  sinT = sin(Theta);
                  sinT2 = sin(Theta/2);
                  //****xi = q/2h  where q is the magnitud of the momentum transfert
                  xi = ( E_in * sinT2) / ( h_Planck * c_light);
                  //****first member of the Klein Nishina (KN) expression and of the Thomson (Thom) expression: Re² / 2
                  eRadiusTerm = ( classic_electr_radius*classic_electr_radius) / 2;

                  //****************DIFF CROSS SECTION for RAYLEIGH SCATTER process********************************//
                  //****Thomson differential cross section calcul
                  DCSThomsonTerm1 = (1 + cosT * cosT);
                  DCSThomson = eRadiusTerm * DCSThomsonTerm1;
                  //****Form Factor (FF) calcul . NOTICE in FindValue they use (xi,Z-1) i have to understand why !!
                  formFactor = formFactorData->FindValue(xi,Z-1);
                  listAngleFF.push_back( std::make_pair( xi/(1/cm), formFactor));
                  //****Rayleigh differential cross section per solid angle
                  DCSrayleigh = DCSThomson * formFactor * formFactor;
                  listAngleDCSrayleigh.push_back( std::make_pair( Theta, DCSrayleigh/(barn/steradian)));

                  //****************DIFF CROSS SECTION for COMPTON SCATTER process*********************************//
                  //****Klein Nishina differential cross section calcul
                  e0m = E_in / electron_mass_c2;
                  Ecompton = E_in / ( 1 + e0m*(1 - cosT));
                  DCSKleinNishinaTerm1 = ( Ecompton / E_in)*(Ecompton / E_in);
                  DCSKleinNishinaTerm2 = ( ( Ecompton / E_in) + ( E_in / Ecompton) - sinT*sinT);
                  DCSKleinNishina = eRadiusTerm * DCSKleinNishinaTerm1 * DCSKleinNishinaTerm2;
                  //****Scatter Function (SF) calcul . NOTICE in FindValue they use (xi,Z-1) i have to understand why !!
                  scatteringFunction = scatterFunctionData->FindValue(xi,Z-1);
                  listAngleSF.push_back( std::make_pair( xi/(1/cm), scatteringFunction));
                  //****Compton differential cross section per solid angle
                  DCScompton = DCSKleinNishina * scatteringFunction;
                  listAngleDCScompton.push_back( std::make_pair( Theta, DCScompton/(barn/steradian)));
		}
              mapEnergySF.insert( make_pair( mUserEnergyList[l], listAngleSF));
              mapEnergyDCScompton.insert( make_pair( mUserEnergyList[l], listAngleDCScompton));
              mapEnergyFF.insert( make_pair( mUserEnergyList[l], listAngleFF));
              mapEnergyDCSrayleigh.insert( make_pair( mUserEnergyList[l], listAngleDCSrayleigh));
              listAngleSF.clear( );
              listAngleDCScompton.clear( );
              listAngleFF.clear( );
              listAngleDCSrayleigh.clear( );
            }
	}
      ////ATTENTION write map to materials(energy(DCS)))
    }
  //TableSF_co[(*matTbl)[k]->GetName()].push_back( std::make_pair( mUserEnergyList[l], Sigtot_vo/density));


  //****WRITE Rayleigh process data files*********************************
  //****print Form Factor
  for( iMapEnergyFF = mapEnergyFF.begin(); iMapEnergyFF != mapEnergyFF.end(); iMapEnergyFF++)
    {
        //DriverDataOutFF << iMapEnergyFF->first << "\n" ;
        listAngleFF = iMapEnergyFF->second;
        for( iListAngleFF = listAngleFF.begin(); iListAngleFF != listAngleFF.end(); iListAngleFF++)
            {
                DriverDataOutFF << iListAngleFF->first << " " << iListAngleFF->second << "\n" ;
            }
        DriverDataOutFF << " "  << "\n" ;
    }
  //****print Rayleigh Differential Cross Section
  for( iMapEnergyDCSrayleigh = mapEnergyDCSrayleigh.begin(); iMapEnergyDCSrayleigh != mapEnergyDCSrayleigh.end(); iMapEnergyDCSrayleigh++)
    {
      //DriverDataOutDCSrayleigh << iMapEnergyDCSrayleigh->first << "\t" << numero << "\n" ;
      listAngleDCSrayleigh = iMapEnergyDCSrayleigh->second;
      for( iListAngleDCSrayleigh = listAngleDCSrayleigh.begin(); iListAngleDCSrayleigh != listAngleDCSrayleigh.end(); iListAngleDCSrayleigh++)
        {
            DriverDataOutDCSrayleigh << iListAngleDCSrayleigh->first << " " << iListAngleDCSrayleigh->second << "\n" ;
        }
      DriverDataOutDCSrayleigh << " "  << "\n" ;
    }
  //DD("ici2");


  //****WRITE compton process data files*********************************
  //****print Scatttering Function
  for( iMapEnergySF = mapEnergySF.begin(); iMapEnergySF != mapEnergySF.end(); iMapEnergySF++)
    {
        //DriverDataOutSF << iterTableEnergySF_co->first << "\n" ;
        listAngleSF = iMapEnergySF->second;
        for( iListAngleSF = listAngleSF.begin(); iListAngleSF != listAngleSF.end(); iListAngleSF++)
            {
                DriverDataOutSF << iListAngleSF->first << " " << iListAngleSF->second << "\n" ;
            }
        DriverDataOutSF << " "  << "\n" ;
    }
  //****print Compton Differential Cross Section
  for( iMapEnergyDCScompton = mapEnergyDCScompton.begin(); iMapEnergyDCScompton != mapEnergyDCScompton.end(); iMapEnergyDCScompton++)
    {
      //DriverDataOutDCScompton << iMapEnergyDCScompton->first << "\t" << numero << "\n" ;
      listAngleDCScompton = iMapEnergyDCScompton->second;
      for( iListAngleDCScompton = listAngleDCScompton.begin(); iListAngleDCScompton != listAngleDCScompton.end(); iListAngleDCScompton++)
        {
            DriverDataOutDCScompton << iListAngleDCScompton->first << " " << iListAngleDCScompton->second << "\n" ;
        }
      DriverDataOutDCScompton << " "  << "\n" ;
    }
  //DD("ici2");
  G4cout << "*******************************************" << G4endl;



}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
///// Save data
void GateDiffCrossSectionActor::SaveData()
{
  G4String directory = "/home/eromero/Documents/Synergy/output/";
  PointerToFileDataOutSF.open ( (directory + mExitFileNameSF).c_str());
  PointerToFileDataOutFF.open ( (directory + mExitFileNameFF).c_str());
  PointerToFileDataOutDCScompton.open ( (directory + mExitFileNameDCScompton).c_str());
  PointerToFileDataOutDCSrayleigh.open ( (directory + mExitFileNameDCSrayleigh).c_str());

  DataOutSF = DriverDataOutSF.str();
  DataOutFF = DriverDataOutFF.str();
  DataOutDCScompton = DriverDataOutDCScompton.str();
  DataOutDCSrayleigh = DriverDataOutDCSrayleigh.str();

  PointerToFileDataOutSF << DataOutSF;
  PointerToFileDataOutFF << DataOutFF;
  PointerToFileDataOutDCScompton << DataOutDCScompton;
  PointerToFileDataOutDCSrayleigh << DataOutDCSrayleigh;

  PointerToFileDataOutSF.close();
  PointerToFileDataOutFF.close();
  PointerToFileDataOutDCScompton.close();
  PointerToFileDataOutDCSrayleigh.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDiffCrossSectionActor::ResetData()
{
}
//------------------------------------------------------------------------
