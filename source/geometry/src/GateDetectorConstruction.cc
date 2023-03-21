/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#include "GateConfiguration.h"
#include "GateDetectorConstruction.hh"
#include "GateDetectorMessenger.hh"
#include "GateRunManager.hh"
#include "GateVVolume.hh"
#include "GateBox.hh"
#include "GateObjectStore.hh"
#include "GateSystemListManager.hh"
#include "GateMaterialDatabase.hh"
#include "GateCrystalSD.hh"
#include "GatePhantomSD.hh"
#include "GateMessageManager.hh"
#include "GateObjectMoveListMessenger.hh"
#include "GateARFSD.hh"
#include "GateMagTabulatedField3D.hh"
#include "GateElectricTabulatedField3D.hh"
#include "GateElectricMagTabulatedField3D.hh"

#include "globals.hh"
#include "G4Navigator.hh"
#include "G4SDManager.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"

#ifdef GATE_USE_OPTICAL
#include "GateSurfaceList.hh"
#endif

GateDetectorConstruction* GateDetectorConstruction::pTheGateDetectorConstruction=0;

//---------------------------------------------------------------------------------
GateDetectorConstruction::GateDetectorConstruction()
  :  pworld(0),
     pworldPhysicalVolume(0),
     nGeometryStatus(geometry_needs_rebuild),
     flagAutoUpdate(false),
     m_crystalSD(0),
     m_phantomSD(0),
     pdetectorMessenger(0),
     moveFlag(0),
     m_magField(0), m_magFieldValue(0),
     e_electFieldValue(0),
     m_magFieldUniform(false), m_magFieldTabulated(false),
	 e_electFieldUniform(false), e_electFieldTabulated(false),
   em_electmagFieldTabulated(false), 
	 m_MagField(0), e_ElecField(0), em_ElecMagField(0),
	 fEquation_B(0), fEquation_E(0),
	 fFieldMgr(0), fStepper(0),
	 fMinStep(1*um),
	 fDeltaChord(1*um),
	 fDeltaIntersection(1*nm),
	 fDeltaOneStep(1*nm),
	 fMinimumEpsilonStep(1e-11),
	 fMaximumEpsilonStep(1e-10),
	 fIntegratorStepper("ClassicalRK4"),
	 nvarOfIntegratorStepper(8) // The Equation of motion for Electric (or combined Electric/Magnetic)
                                // field requires 8 integration variables
{

  GateMessage("Geometry", 1, "GateDetectorConstruction instantiating...\n");
  GateMessage("Geometry", 5, " GateDetectorConstruction constructor -- begin ");
  GateMessage("Geometry", 5, " nGeometryStatus = " << nGeometryStatus << Gateendl;);

  pTheGateDetectorConstruction = this;

  pcreatorStore = GateObjectStore::GetInstance();
  psystemStore=GateSystemListManager::GetInstance();

  pdetectorMessenger = new GateDetectorMessenger(this);

  m_magFieldValue = G4ThreeVector(0.,0.,0. * tesla);
  e_electFieldValue = G4ThreeVector(0.,0.,0. * keV);

  G4double pworld_x = 50.*cm;
  G4double pworld_y = 50.*cm;
  G4double pworld_z = 50.*cm;

  //-------------------------------------------------------------------------
  // Create default material (air) for the world
  G4String AirName = "worldDefaultAir";
  G4Material* Air = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR"); // Use Air for NIST Manager
  if (Air==NULL)//will never enter here
    {
   	  G4Element* N  = new G4Element("worldDefaultN","N" , 7., 14.01*g/mole );
  	  G4Element* O  = new G4Element("worldDefaultO","O" , 8., 16.00*g/mole);
   	  G4Material* Air = new G4Material(AirName  , 1.290*mg/cm3, 2);
   	  Air->AddElement(N, 0.7);
   	  Air->AddElement(O, 0.3);
    }
  else Air->SetName(AirName);//For compatibility put name of this Air material to "worldDefaultAir"

  //-------------------------------------------------------------------------

  pworld = new GateBox("world", "worldDefaultAir",  pworld_x, pworld_y, pworld_z, true);
  pworld->SetMaterialName("worldDefaultAir");

  G4SDManager* SDman = G4SDManager::GetSDMpointer();
  // OK GND 2022 moved to GateVVolume::AttachCrystalSD()
  /*
  if(!m_crystalSD) {
    G4String crystalSDname = "/gate/crystal";
    m_crystalSD = new GateCrystalSD(crystalSDname);
    SDman->AddNewDetector(m_crystalSD);
  }
*/
  if(!m_phantomSD) {
    G4String phantomSDname = "/gate/phantom";
    m_phantomSD = new GatePhantomSD(phantomSDname);
    SDman->AddNewDetector(m_phantomSD);
  }
  GateMessage("Geometry", 5, "  GateDetectorConstruction constructor -- end ");


  /* instantiate the singleton RTPhantom Manager  - PY Descourt 08/09/2008 */

  m_RTPhantomMgr = GateRTPhantomMgr::GetInstance();
  m_ARFSD = 0;
}
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
GateDetectorConstruction::~GateDetectorConstruction()
{
  if (pworld) {
    DestroyGeometry();
    delete pworld;
    pworld = 0;
  }
  delete pdetectorMessenger;
}
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
G4VPhysicalVolume* GateDetectorConstruction::Construct()
{
  GateMessage("Geometry", 3, "Geometry construction starts. \n");

  pworldPhysicalVolume = pworld->GateVVolume::Construct();
  SetGeometryStatusFlag(geometry_is_uptodate);

  GateMessage("Physic", 1, " \n");
  GateMessage("Physic", 1, "----------------------------------------------------------\n");
  GateMessage("Physic", 1, "                    Ionization potential \n");

  const G4MaterialTable * theTable = G4Material::GetMaterialTable();
  for(unsigned int i =0;i<(*theTable).size();i++){
    if(theListOfIonisationPotential[(*theTable)[i]->GetName()]){
      (*theTable)[i]->GetIonisation()->SetMeanExcitationEnergy(theListOfIonisationPotential[(*theTable)[i]->GetName()]);
      GateMessage("Physic", 1, " - " << (*theTable)[i]->GetName() << "\t defaut value: I = " <<
                  G4BestUnit((*theTable)[i]->GetIonisation()->GetMeanExcitationEnergy(),"Energy") <<
                  "\t-->  new value: I = " <<
                  G4BestUnit((*theTable)[i]->GetIonisation()->GetMeanExcitationEnergy(),"Energy") << Gateendl);
    }
    else {
      GateMessage("Physic", 1, " - " << (*theTable)[i]->GetName() << "\t defaut value: I = " <<
                  G4BestUnit((*theTable)[i]->GetIonisation()->GetMeanExcitationEnergy(),"Energy") << Gateendl);
    }
  }
  GateMessage("Physic", 1, "----------------------------------------------------------\n");

  GateMessage("Geometry", 3, "Geometry has been constructed (status = " << nGeometryStatus << ").\n");

#ifdef GATE_USE_OPTICAL
  BuildSurfaces();
#endif
  BuildField();

  return pworldPhysicalVolume;
}
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
// Adds a Material Database
void GateDetectorConstruction::AddFileToMaterialDatabase(const G4String& f)
{
  mMaterialDatabase.AddMDBFile(f);
}
//---------------------------------------------------------------------------------
void GateDetectorConstruction::SetElectField(G4ThreeVector fieldValue)
{
  e_electFieldValue = fieldValue;
  e_electFieldUniform = true;
}
//---------------------------------------------------------------------------------
void GateDetectorConstruction::SetElectFieldTabulatedFile(G4String filenameFieldTable)
{
  e_electFieldTabulatedFile = filenameFieldTable;
  e_electFieldTabulated = true;
}
//---------------------------------------------------------------------------------
void GateDetectorConstruction::SetElectMagFieldTabulatedFile(G4String filenameFieldTable)
{
  em_electmagFieldTabulatedFile = filenameFieldTable;
  em_electmagFieldTabulated = true;
}
//---------------------------------------------------------------------------------
void GateDetectorConstruction::SetMagField(G4ThreeVector fieldValue)
{
  m_magFieldValue = fieldValue;
  m_magFieldUniform = true;
}
//---------------------------------------------------------------------------------
void GateDetectorConstruction::SetMagFieldTabulatedFile(G4String filenameFieldTable)
{
  m_magFieldTabulatedFile = filenameFieldTable;
  m_magFieldTabulated = true;
}
//---------------------------------------------------------------------------------
void GateDetectorConstruction::SetMagStepMinimum(G4double MinStep){
  fMinStep = MinStep;
}
//---------------------------------------------------------------------------------
void GateDetectorConstruction::SetMagDeltaChord(G4double DeltaChord){
  fDeltaChord = DeltaChord;
}
//---------------------------------------------------------------------------------
void GateDetectorConstruction::SetMagDeltaIntersection(G4double DeltaIntersection){
  fDeltaIntersection = DeltaIntersection;
}
//---------------------------------------------------------------------------------
void GateDetectorConstruction::SetMagDeltaOneStep(G4double DeltaOneStep){
  fDeltaOneStep = DeltaOneStep;
}
//---------------------------------------------------------------------------------
void GateDetectorConstruction::SetMagMinimumEpsilonStep(G4double MinEpsilonStep){
  fMinimumEpsilonStep = MinEpsilonStep;
}
//---------------------------------------------------------------------------------
void GateDetectorConstruction::SetMagMaximumEpsilonStep(G4double MaxEpsilonStep){
  fMaximumEpsilonStep = MaxEpsilonStep;
}

void GateDetectorConstruction::SetMagIntegratorStepper(G4String IntegratorStepper ){
  fIntegratorStepper = IntegratorStepper;
}
//---------------------------------------------------------------------------------

void GateDetectorConstruction::SetField(){

	if (m_magFieldTabulated){

		fEquation_B = new G4Mag_UsualEqRhs (m_MagField);

		if (fIntegratorStepper == "ExplicitEuler"){
		  fStepper  = new G4ExplicitEuler (fEquation_B, nvarOfIntegratorStepper);
		}
		else if (fIntegratorStepper == "ImplicitEuler") {
		  fStepper  = new G4ImplicitEuler (fEquation_B, nvarOfIntegratorStepper);
		}
		else if (fIntegratorStepper == "SimpleRunge") {
		  fStepper  = new G4SimpleRunge (fEquation_B, nvarOfIntegratorStepper);
		}
		else if (fIntegratorStepper == "SimpleHeum") {
		  fStepper  = new G4SimpleHeum (fEquation_B, nvarOfIntegratorStepper);
		}
		else if (fIntegratorStepper == "NystromRK4") {
		  fStepper  = new G4NystromRK4 (fEquation_B);
		}
		else {
		  fStepper  = new G4ClassicalRK4 (fEquation_B,nvarOfIntegratorStepper);
		}

		fFieldMgr = G4TransportationManager::GetTransportationManager()->GetFieldManager();
		G4MagInt_Driver* pIntgrDriver_B = new G4MagInt_Driver(1*mm,fStepper,nvarOfIntegratorStepper);
		G4ChordFinder* fChordFinder_B = new G4ChordFinder(pIntgrDriver_B);
		fFieldMgr->SetChordFinder(fChordFinder_B);
		fFieldMgr->SetDetectorField(m_MagField);

		GateMessage("Core", 0, " THE FOLLOWING INTEGRATOR STEPPER FOR MAGNETIC FIELD HAS BEEN ACTIVATED: "
								<< fIntegratorStepper << Gateendl);
	}

	if (e_electFieldTabulated){

		fEquation_E = new G4EqMagElectricField(e_ElecField);

		if (fIntegratorStepper == "ExplicitEuler"){
		  fStepper  = new G4ExplicitEuler (fEquation_E, nvarOfIntegratorStepper);
		}
		else if (fIntegratorStepper == "ImplicitEuler") {
		  fStepper  = new G4ImplicitEuler (fEquation_E, nvarOfIntegratorStepper);
		}
		else if (fIntegratorStepper == "SimpleRunge") {
		  fStepper  = new G4SimpleRunge (fEquation_E, nvarOfIntegratorStepper);
		}
		else if (fIntegratorStepper == "SimpleHeum") {
		  fStepper  = new G4SimpleHeum (fEquation_E, nvarOfIntegratorStepper);
		}
		else {
		  fStepper  = new G4ClassicalRK4 (fEquation_E,nvarOfIntegratorStepper);
		}

		fFieldMgr = G4TransportationManager::GetTransportationManager()->GetFieldManager();
		G4MagInt_Driver  *pIntgrDriver_E = new G4MagInt_Driver(1*mm, fStepper, nvarOfIntegratorStepper);
		G4ChordFinder *fChordFinder_E = new G4ChordFinder(pIntgrDriver_E);
		fFieldMgr -> SetChordFinder(fChordFinder_E);
		fFieldMgr->SetDetectorField(e_ElecField);

		GateMessage("Core", 0, " THE FOLLOWING INTEGRATOR STEPPER FOR ELECTRIC FIELD HAS BEEN ACTIVATED: "
										<< fIntegratorStepper << Gateendl);
	}

  if (em_electmagFieldTabulated){

		fEquation_E = new G4EqMagElectricField(em_ElecMagField);

		if (fIntegratorStepper == "ExplicitEuler"){
		  fStepper  = new G4ExplicitEuler (fEquation_E, nvarOfIntegratorStepper);
		}
		else if (fIntegratorStepper == "ImplicitEuler") {
		  fStepper  = new G4ImplicitEuler (fEquation_E, nvarOfIntegratorStepper);
		}
		else if (fIntegratorStepper == "SimpleRunge") {
		  fStepper  = new G4SimpleRunge (fEquation_E, nvarOfIntegratorStepper);
		}
		else if (fIntegratorStepper == "SimpleHeum") {
		  fStepper  = new G4SimpleHeum (fEquation_E, nvarOfIntegratorStepper);
		}
		else {
		  fStepper  = new G4ClassicalRK4 (fEquation_E,nvarOfIntegratorStepper);
		}

		fFieldMgr = G4TransportationManager::GetTransportationManager()->GetFieldManager();
		G4MagInt_Driver  *pIntgrDriver_EB = new G4MagInt_Driver(1*mm, fStepper, nvarOfIntegratorStepper);
		G4ChordFinder *fChordFinder_EB = new G4ChordFinder(pIntgrDriver_EB);
		fFieldMgr -> SetChordFinder(fChordFinder_EB);
		fFieldMgr->SetDetectorField(em_ElecMagField);

		GateMessage("Core", 0, " THE FOLLOWING INTEGRATOR STEPPER FOR ELECTROMAGNETIC FIELD HAS BEEN ACTIVATED: "
										<< fIntegratorStepper << Gateendl);
	}

	fFieldMgr->GetChordFinder()->SetDeltaChord(fDeltaChord);
	fFieldMgr->SetDeltaIntersection(fDeltaIntersection);
	fFieldMgr->SetDeltaOneStep(fDeltaOneStep);

	G4PropagatorInField *fPropInField = G4TransportationManager::GetTransportationManager()->GetPropagatorInField();
	fPropInField->SetMinimumEpsilonStep(fMinimumEpsilonStep);
	fPropInField->SetMaximumEpsilonStep(fMaximumEpsilonStep);

	GateMessage("Core", 0, "\n" <<
			  "---> fMinStep " << fMinStep/mm << " mm \n"
			  "---> fDeltaChord "<<fDeltaChord/mm <<" mm \n"
			  "---> fDeltaIntersection "<<fFieldMgr->GetDeltaIntersection()/mm <<" mm \n"
			  "---> fDeltaOneStep "<<fFieldMgr->GetDeltaOneStep()/mm <<" mm \n"
			  "---> fMinimumEpsilonStep "<<fMinimumEpsilonStep << " \n"
			  "---> fMaximumEpsilonStep "<<fMaximumEpsilonStep << " \n"
			  "-----------------------------------------------------------"<< Gateendl);
}


void GateDetectorConstruction::BuildField()
{

  if (m_magFieldUniform){

	  fFieldMgr = G4TransportationManager::GetTransportationManager()->GetFieldManager();
	  if(m_magField) delete m_magField;             //delete the existing mag field
	  if(m_magFieldValue.mag()!=0.){               // create a new one if non null
		  m_magField = new G4UniformMagField(m_magFieldValue);
		  fFieldMgr->SetDetectorField(m_magField);
		  fFieldMgr->CreateChordFinder(m_magField);
	  } else {
		  m_magField = NULL;
		  fFieldMgr->SetDetectorField(m_magField);
	  }

  } else if (m_magFieldTabulated) {

	  if(m_MagField) delete m_MagField;
	  m_MagField = new GateMagTabulatedField3D(m_magFieldTabulatedFile);
	  SetField();

  } else if (e_electFieldUniform){

  	  fFieldMgr = G4TransportationManager::GetTransportationManager()->GetFieldManager();
  	  if(e_ElecField) delete e_ElecField;             //delete the existing elect field
  	  if(e_electFieldValue.mag()!=0.){               // create a new one if non null
  		  e_ElecField = new G4UniformElectricField(e_electFieldValue);
  		  fFieldMgr->SetDetectorField(e_ElecField);
  	  } else {
  		  e_ElecField = NULL;
  		  fFieldMgr->SetDetectorField(m_magField);
  	  }

  } else if (e_electFieldTabulated) {

      fFieldMgr = new G4FieldManager();
	  e_ElecField = new GateElectricTabulatedField3D(e_electFieldTabulatedFile);
	  SetField();
  
  } else if (em_electmagFieldTabulated) {

      fFieldMgr = new G4FieldManager();
	  em_ElecMagField = new GateElectricMagTabulatedField3D(em_electmagFieldTabulatedFile);
	  SetField();
    }
}
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
#ifdef GATE_USE_OPTICAL
void GateDetectorConstruction::BuildSurfaces()
{
  GateObjectStore* store = GateObjectStore::GetInstance();
  for (GateObjectStore::iterator p = store->begin(); p != store->end(); p++)
    {
      p->second->GetSurfaceList()->BuildSurfaces();
    }

}
#endif
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
void GateDetectorConstruction::UpdateGeometry()
{
  GateMessage("Geometry", 3,"UpdateGeometry starts (status = " << nGeometryStatus << "). \n");

  if (nGeometryStatus == geometry_is_uptodate){
    GateMessage("Geometry", 3,"Geometry is uptodate.\n");
    return;
  }

  switch (nGeometryStatus){
  case geometry_needs_update:
    pworld->Construct(true);
    break;

  case geometry_needs_rebuild:
  default:
    DestroyGeometry();
    Construct();
    break;
  }
  GateRunManager::GetRunManager()->DefineWorldVolume(pworldPhysicalVolume);

  nGeometryStatus = geometry_is_uptodate;

  GateMessage("Geometry", 3, "nGeometryStatus = geometry_is_uptodate \n");
  GateMessage("Geometry", 3, "UpdateGeometry finished. \n");
}
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
void GateDetectorConstruction::DestroyGeometry()
{
  GateMessageInc("Geometry", 4,"Geometry is going to be destroyed.\n");

  pworld->DestroyGeometry();
  nGeometryStatus = geometry_needs_rebuild;

  GateMessage("Geometry", 4,"nGeometryStatus = geometry_needs_rebuild\n");
  GateMessageDec("Geometry", 4,"Geometry has been destroyed.\n");
}
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
/*
  void GateDetectorConstruction::GeometryHasChanged(GeometryStatus changeLevel)
  {

  GateMessage("Geometry", 3, "   nGeometryStatus = " << nGeometryStatus << " changeLevel = " << changeLevel << Gateendl;);

  if (flagAutoUpdate == 0)
  GateMessage("Geometry", 3, "   flagAutoUpdate = " << flagAutoUpdate << Gateendl;);

  if ( changeLevel > nGeometryStatus )
  nGeometryStatus = changeLevel;

  if (nGeometryStatus == 0){
  GateMessage("Geometry", 3, "   The geometry is uptodate.\n";);
  }
  else if (nGeometryStatus == 1){
  GateMessage("Geometry", 3, "   The geometry needs to be uptodated.\n";);
  }
  else if (nGeometryStatus == 2){
  GateMessage("Geometry", 3, "   The geometry needs to be rebuilt.\n";);
  }

  if (flagAutoUpdate){
  GateMessage("Geometry", 0,"The geometry is going to be updated.\n");
  UpdateGeometry();}
  }
*/
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
void GateDetectorConstruction::ClockHasChanged()
{
  GateMessage("Move", 5, "ClockHasChanged = " << GetFlagMove() << Gateendl; );

  if ( GetFlagMove()) {
    GateMessage("Move", 6, "moveFlag = 1\n");
    nGeometryStatus = geometry_needs_update;
  }
  else {
    GateMessage("Move", 6, "Geometry is uptodate.\n");
    nGeometryStatus = geometry_is_uptodate;
  }

  GateMessage("Move", 6, "  Geometry status = " << nGeometryStatus << Gateendl;);

  UpdateGeometry();
  GateMessage("Move", 6, "Clock has changed.\n");
}
//---------------------------------------------------------------------------------


//---------------------------------------------------------------------------------
void GateDetectorConstruction::insertARFSD( G4String aName , G4int stage )
{
  GateMessage("Geometry", 2, "GateDetectorConstruction::insertARFSD entered");

  if (m_ARFSD == 0) {
    m_ARFSD = new GateARFSD("/gate/arf", aName );
    G4SDManager* SDMan = G4SDManager::GetSDMpointer();
    SDMan->AddNewDetector( m_ARFSD );
  }
  m_ARFSD->SetStage( stage );
}
//---------------------------------------------------------------------------------

