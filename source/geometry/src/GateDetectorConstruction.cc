/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
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

#include "globals.hh"
#include "G4UniformMagField.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4SDManager.hh"
#include "G4Material.hh"
#include "G4Material.hh"

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
     m_magField(0), m_magFieldValue(0)
{

  GateMessage("Geometry", 1, "GateDetectorConstruction instantiating...\n");
  GateMessage("Geometry", 5, " GateDetectorConstruction constructor -- begin ");
  GateMessage("Geometry", 5, " nGeometryStatus = " << nGeometryStatus << G4endl;);

  pTheGateDetectorConstruction = this;

  pcreatorStore = GateObjectStore::GetInstance();
  psystemStore=GateSystemListManager::GetInstance();

  pdetectorMessenger = new GateDetectorMessenger(this);

  m_magFieldValue = G4ThreeVector(0.,0.,0. * tesla);

  G4double pworld_x = 50.*cm;
  G4double pworld_y = 50.*cm;
  G4double pworld_z = 50.*cm;

  //-------------------------------------------------------------------------
  // Create default material (air) for the world
  G4Element* N  = new G4Element("worldDefaultN","N" , 7., 14.01*g/mole );
  G4Element* O  = new G4Element("worldDefaultO"  ,"O" , 8., 16.00*g/mole);
  G4Material* Air = new G4Material("worldDefaultAir"  , 1.290*mg/cm3, 2);
  Air->AddElement(N, 0.7);
  Air->AddElement(O, 0.3);
  //-------------------------------------------------------------------------

  pworld = new GateBox("world", "worldDefaultAir",  pworld_x, pworld_y, pworld_z, true);
  pworld->SetMaterialName("worldDefaultAir");

  G4SDManager* SDman = G4SDManager::GetSDMpointer();

  if(!m_crystalSD) {
    G4String crystalSDname = "/gate/crystal";
    m_crystalSD = new GateCrystalSD(crystalSDname);
    SDman->AddNewDetector(m_crystalSD);
  }

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

  GateMessage("Physic", 1, " "<<G4endl);
  GateMessage("Physic", 1, "----------------------------------------------------------"<<G4endl);
  GateMessage("Physic", 1, "                    Ionization potential "<<G4endl);

  const G4MaterialTable * theTable = G4Material::GetMaterialTable();
  for(unsigned int i =0;i<(*theTable).size();i++){
    if(theListOfIonisationPotential[(*theTable)[i]->GetName()]){
      (*theTable)[i]->GetIonisation()->SetMeanExcitationEnergy(theListOfIonisationPotential[(*theTable)[i]->GetName()]);
      GateMessage("Physic", 1, " - " << (*theTable)[i]->GetName() << "\t defaut value: I = " <<
                  G4BestUnit((*theTable)[i]->GetIonisation()->GetMeanExcitationEnergy(),"Energy") <<
                  "\t-->  new value: I = " <<
                  G4BestUnit((*theTable)[i]->GetIonisation()->GetMeanExcitationEnergy(),"Energy") << G4endl);
    }
    else {
      GateMessage("Physic", 1, " - " << (*theTable)[i]->GetName() << "\t defaut value: I = " <<
                  G4BestUnit((*theTable)[i]->GetIonisation()->GetMeanExcitationEnergy(),"Energy") << G4endl);
    }
  }
  GateMessage("Physic", 1, "----------------------------------------------------------"<<G4endl);

  GateMessage("Geometry", 3, "Geometry has been constructed (status = " << nGeometryStatus << ").\n");

#ifdef GATE_USE_OPTICAL
  BuildSurfaces();
#endif
  BuildMagField();

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

//---------------------------------------------------------------------------------
void GateDetectorConstruction::SetMagField(G4ThreeVector fieldValue)
{
  m_magFieldValue = fieldValue;
}
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
void GateDetectorConstruction::BuildMagField()
{
  //apply a global uniform magnetic field along Z axis
  G4FieldManager* fieldMgr
    = G4TransportationManager::GetTransportationManager()->GetFieldManager();

  if(m_magField) delete m_magField;             //delete the existing magn field

  if(m_magFieldValue.mag()!=0.)                 // create a new one if non nul
    { m_magField = new G4UniformMagField(m_magFieldValue);
      fieldMgr->SetDetectorField(m_magField);
      fieldMgr->CreateChordFinder(m_magField);
    } else {
    m_magField = NULL;
    fieldMgr->SetDetectorField(m_magField);
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
  GateMessage("Geometry", 4,"Geometry is going to be destroyed. \n");

  pworld->DestroyGeometry();
  nGeometryStatus = geometry_needs_rebuild;

  GateMessage("Geometry", 4,"nGeometryStatus = geometry_needs_rebuild     \n");
  GateMessage("Geometry", 4,"Geometry has been destroyed.\n");
}
//---------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
/*
  void GateDetectorConstruction::GeometryHasChanged(GeometryStatus changeLevel)
  {

  GateMessage("Geometry", 3, "   nGeometryStatus = " << nGeometryStatus << " changeLevel = " << changeLevel << G4endl;);

  if (flagAutoUpdate == 0)
  GateMessage("Geometry", 3, "   flagAutoUpdate = " << flagAutoUpdate << G4endl;);

  if ( changeLevel > nGeometryStatus )
  nGeometryStatus = changeLevel;

  if (nGeometryStatus == 0){
  GateMessage("Geometry", 3, "   The geometry is uptodate." << G4endl;);
  }
  else if (nGeometryStatus == 1){
  GateMessage("Geometry", 3, "   The geometry needs to be uptodated." << G4endl;);
  }
  else if (nGeometryStatus == 2){
  GateMessage("Geometry", 3, "   The geometry needs to be rebuilt." << G4endl;);
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
  GateMessage("Move", 5, "ClockHasChanged = " << GetFlagMove() << G4endl; );

  if ( GetFlagMove()) {
    GateMessage("Move", 6, "moveFlag = 1\n");
    nGeometryStatus = geometry_needs_update;
  }
  else {
    GateMessage("Move", 6, "Geometry is uptodate.\n");
    nGeometryStatus = geometry_is_uptodate;
  }

  GateMessage("Move", 6, "  Geometry status = " << nGeometryStatus << G4endl;);

  UpdateGeometry();
  GateMessage("Move", 6, "Clock has changed.\n");
}
//---------------------------------------------------------------------------------
/*PY Descourt 08/09/2008 */
void GateDetectorConstruction::insertARFSD( G4String aName , G4int stage )
{
  G4cout << " GateDetectorConstruction::insertARFSD  entered " << G4endl;

  if ( m_ARFSD == 0 )
    {
      m_ARFSD = new GateARFSD("/gate/arf", aName );
      G4SDManager* SDMan = G4SDManager::GetSDMpointer();
      SDMan->AddNewDetector( m_ARFSD );
    }
  m_ARFSD->SetStage( stage );
}
/*PY Descourt 08/09/2008 */
