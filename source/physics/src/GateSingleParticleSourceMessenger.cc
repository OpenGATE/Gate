/*----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include <fstream>
#include <iomanip>
#include <sstream>

#include "G4PhysicalConstants.hh"
#include "G4Geantino.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithABool.hh"
#include "G4ios.hh"
#include "G4Tokenizer.hh"
#include "GateSPSEneDistribution.hh"

#include "GateSingleParticleSourceMessenger.hh"
#include "GateVSource.hh"
#include "GateMessageManager.hh"

//-------------------------------------------------------------------------------------------------
GateSingleParticleSourceMessenger::GateSingleParticleSourceMessenger
( GateVSource* fPtclGun)
  : GateMessenger( G4String("source/")+(fPtclGun->GetName()) + G4String("/gps")),
    fParticleGun(fPtclGun),fShootIon(false)
{
  histtype = "biasx";
  fNArbEHistPoints=0;
  fArbInterModeSet=false;
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  G4String cmdName;

  positionDirectory = 0;
  energyDirectory = 0;

  //    gpsDirectory = new G4UIdirectory("/gps/");
  GetDirectory()->SetGuidance("General Particle Source control commands.");
  GetDirectory()->SetGuidance(" The first 9 commands are the same as in G4ParticleGun ");

  // Additional command for attach to a volume
  cmdName = ComputeDirectoryName(G4String("source/")+(fPtclGun->GetName())) + "attachTo";
  relativePlacementCmd = new G4UIcmdWithAString(cmdName, this);
  relativePlacementCmd->SetGuidance("Set a volume name, the source will be placed (translation, rotation) according to this volume");

  // below we reproduce commands awailable in G4Particle Gun

  cmdName = GetDirectoryName() + "List";
  listCmd = new G4UIcmdWithoutParameter(cmdName,this);
  listCmd->SetGuidance("List available particles.");
  listCmd->SetGuidance(" Invoke G4ParticleTable.");

  cmdName = GetDirectoryName() + "particle";
  particleCmd = new G4UIcmdWithAString(cmdName,this);
  particleCmd->SetGuidance("Set particle to be generated.");
  particleCmd->SetGuidance(" (geantino is default)");
  particleCmd->SetGuidance(" (ion can be specified for shooting ions)");
  particleCmd->SetParameterName("particleName",true);
  particleCmd->SetDefaultValue("geantino");
  static G4String candidateList;
  static bool initialized = false;
  if (!initialized) {
    G4int nPtcl = particleTable->entries();
    for(G4int i=0;i<nPtcl;i++)
      {
        candidateList += particleTable->GetParticleName(i);
        candidateList += " ";
      }
    candidateList += "ion ";
    initialized = true;
  }
  particleCmd->SetCandidates(candidateList);



  cmdName = GetDirectoryName() + "direction";
  directionCmd = new G4UIcmdWith3Vector(cmdName,this);
  directionCmd->SetGuidance("Set momentum direction.");
  directionCmd->SetGuidance("Direction needs not to be a unit vector.");
  directionCmd->SetParameterName("Px","Py","Pz",true,true);
  directionCmd->SetRange("Px != 0 || Py != 0 || Pz != 0");

  cmdName = GetDirectoryName() + "energy";
  energyCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  energyCmd->SetGuidance("Set kinetic energy.");
  energyCmd->SetParameterName("Energy",true,true);
  energyCmd->SetDefaultUnit("GeV");
  //energyCmd->SetUnitCategory("Energy");
  //energyCmd->SetUnitCandidates("eV keV MeV GeV TeV");

  cmdName = GetDirectoryName() + "position";
  positionCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
  positionCmd->SetGuidance("Set starting position of the particle.");
  positionCmd->SetParameterName("X","Y","Z",true,true);
  positionCmd->SetDefaultUnit("cm");
  //positionCmd->SetUnitCategory("Length");
  //positionCmd->SetUnitCandidates("microm mm cm m km");

  // SR1.3
  //  ionCmd = new UIcmdWithNucleusAndUnit("/gps/ion",this);
  //ionCmd->SetGuidance("define the primary ion (a,z,e)");
  //ionCmd->SetParameterName("A","Z","E",true);
  //ionCmd->SetDefaultUnit("keV");
  //ionCmd->SetUnitCandidates("keV MeV");

  cmdName = GetDirectoryName() + "ion";
  ionCmd = new G4UIcommand(cmdName,this);
  ionCmd->SetGuidance("Set properties of ion to be generated.");
  ionCmd->SetGuidance("[usage] /gun/ion Z A Q E");
  ionCmd->SetGuidance("        Z:(int) AtomicNumber");
  ionCmd->SetGuidance("        A:(int) AtomicMass");
  ionCmd->SetGuidance("        Q:(int) Charge of Ion (in unit of e)");
  ionCmd->SetGuidance("        E:(double) Excitation energy (in keV)");

  G4UIparameter* param;
  param = new G4UIparameter("Z",'i',false);
  param->SetDefaultValue("1");
  ionCmd->SetParameter(param);
  param = new G4UIparameter("A",'i',false);
  param->SetDefaultValue("1");
  ionCmd->SetParameter(param);
  param = new G4UIparameter("Q",'i',true);
  param->SetDefaultValue("0");
  ionCmd->SetParameter(param);
  param = new G4UIparameter("E",'d',true);
  param->SetDefaultValue("0.0");
  ionCmd->SetParameter(param);


  cmdName = GetDirectoryName() + "time";
  timeCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  timeCmd->SetGuidance("Set initial time of the particle.");
  timeCmd->SetParameterName("t0",true,true);
  timeCmd->SetDefaultUnit("ns");
  //timeCmd->SetUnitCategory("Time");
  //timeCmd->SetUnitCandidates("ns ms s");

  cmdName = GetDirectoryName() + "polarization";
  polCmd = new G4UIcmdWith3Vector(cmdName,this);
  polCmd->SetGuidance("Set polarization.");
  polCmd->SetParameterName("Px","Py","Pz",true,true);
  polCmd->SetRange("Px>=-1.&&Px<=1.&&Py>=-1.&&Py<=1.&&Pz>=-1.&&Pz<=1.");

  cmdName = GetDirectoryName() + "number";
  numberCmd = new G4UIcmdWithAnInteger(cmdName,this);
  numberCmd->SetGuidance("Set number of particles to be generated.");
  numberCmd->SetParameterName("N",true,true);
  numberCmd->SetRange("N>0");

  // now extended commands
  // Positional ones:
  cmdName = GetDirectoryName() + "pos/type";
  typeCmd1 = new G4UIcmdWithAString(cmdName,this);
  typeCmd1->SetGuidance("Sets source distribution type.");
  typeCmd1->SetGuidance("Either Point, Beam, Plane, Surface, Volume or UserFluenceImage");
  typeCmd1->SetParameterName("DisType",true,true);
  typeCmd1->SetDefaultValue("Point");
  typeCmd1->SetCandidates("Point Beam Plane Surface Volume UserFluenceImage");

  cmdName = GetDirectoryName() + "pos/shape";
  shapeCmd1 = new G4UIcmdWithAString(cmdName,this);
  shapeCmd1->SetGuidance("Sets source shape for Plan, Surface or Volume type source.");
  shapeCmd1->SetParameterName("Shape",true,true);
  shapeCmd1->SetDefaultValue("NULL");
  shapeCmd1->SetCandidates("Circle Annulus Ellipse Square Rectangle Sphere Range Ellipsoid Cylinder Para");

  cmdName = GetDirectoryName() + "pos/centre";
  centreCmd1 = new G4UIcmdWith3VectorAndUnit(cmdName,this);
  centreCmd1->SetGuidance("Set centre coordinates of source.");
  centreCmd1->SetGuidance("   same effect as the /gps/position command");
  centreCmd1->SetParameterName("X","Y","Z",true,true);
  centreCmd1->SetDefaultUnit("cm");

  cmdName = GetDirectoryName() + "pos/rot1";
  posrot1Cmd1 = new G4UIcmdWith3Vector(cmdName,this);
  posrot1Cmd1->SetGuidance("Set the 1st vector defining the rotation matrix'.");
  posrot1Cmd1->SetGuidance("It does not need to be a unit vector.");
  posrot1Cmd1->SetParameterName("R1x","R1y","R1z",true,true);
  posrot1Cmd1->SetRange("R1x != 0 || R1y != 0 || R1z != 0");

  cmdName = GetDirectoryName() + "pos/rot2";
  posrot2Cmd1 = new G4UIcmdWith3Vector(cmdName,this);
  posrot2Cmd1->SetGuidance("Set the 2nd vector defining the rotation matrix'.");
  posrot2Cmd1->SetGuidance("It does not need to be a unit vector.");
  posrot2Cmd1->SetParameterName("R2x","R2y","R2z",true,true);
  posrot2Cmd1->SetRange("R2x != 0 || R2y != 0 || R2z != 0");

  cmdName = GetDirectoryName() + "pos/halfx";
  halfxCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  halfxCmd1->SetGuidance("Set x half length of source.");
  halfxCmd1->SetParameterName("Halfx",true,true);
  halfxCmd1->SetDefaultUnit("cm");

  cmdName = GetDirectoryName() + "pos/halfy";
  halfyCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  halfyCmd1->SetGuidance("Set y half length of source.");
  halfyCmd1->SetParameterName("Halfy",true,true);
  halfyCmd1->SetDefaultUnit("cm");

  cmdName = GetDirectoryName() + "pos/halfz";
  halfzCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  halfzCmd1->SetGuidance("Set z half length of source.");
  halfzCmd1->SetParameterName("Halfz",true,true);
  halfzCmd1->SetDefaultUnit("cm");

  cmdName = GetDirectoryName() + "pos/radius";
  radiusCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  radiusCmd1->SetGuidance("Set radius of source.");
  radiusCmd1->SetParameterName("Radius",true,true);
  radiusCmd1->SetDefaultUnit("cm");

  cmdName = GetDirectoryName() + "pos/inner_radius";
  radius0Cmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  radius0Cmd1->SetGuidance("Set inner radius of source when required.");
  radius0Cmd1->SetParameterName("Radius0",true,true);
  radius0Cmd1->SetDefaultUnit("cm");

  cmdName = GetDirectoryName() + "pos/sigma_r";
  possigmarCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  possigmarCmd1->SetGuidance("Set standard deviation in radial of the beam positional profile");
  possigmarCmd1->SetGuidance(" applicable to Beam type source only");
  possigmarCmd1->SetParameterName("Sigmar",true,true);
  possigmarCmd1->SetDefaultUnit("cm");

  cmdName = GetDirectoryName() + "pos/sigma_x";
  possigmaxCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  possigmaxCmd1->SetGuidance("Set standard deviation of beam positional profile in x-dir");
  possigmaxCmd1->SetGuidance(" applicable to Beam type source only");
  possigmaxCmd1->SetParameterName("Sigmax",true,true);
  possigmaxCmd1->SetDefaultUnit("cm");

  cmdName = GetDirectoryName() + "pos/sigma_y";
  possigmayCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  possigmayCmd1->SetGuidance("Set standard deviation of beam positional profile in y-dir");
  possigmayCmd1->SetGuidance(" applicable to Beam type source only");
  possigmayCmd1->SetParameterName("Sigmay",true,true);
  possigmayCmd1->SetDefaultUnit("cm");

  cmdName = GetDirectoryName() + "pos/paralp";
  paralpCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  paralpCmd1->SetGuidance("Angle from y-axis of y' in Para");
  paralpCmd1->SetParameterName("paralp",true,true);
  paralpCmd1->SetDefaultUnit("rad");

  cmdName = GetDirectoryName() + "pos/parthe";
  partheCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  partheCmd1->SetGuidance("Polar angle through centres of z faces");
  partheCmd1->SetParameterName("parthe",true,true);
  partheCmd1->SetDefaultUnit("rad");

  cmdName = GetDirectoryName() + "pos/parphi";
  parphiCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  parphiCmd1->SetGuidance("Azimuth angle through centres of z faces");
  parphiCmd1->SetParameterName("parphi",true,true);
  parphiCmd1->SetDefaultUnit("rad");

  cmdName = GetDirectoryName() + "pos/confine";
  confineCmd1 = new G4UIcmdWithAString(cmdName,this);
  confineCmd1->SetGuidance("Confine source to volume (NULL to unset).");
  confineCmd1->SetGuidance("usage: confine VolName");
  confineCmd1->SetParameterName("VolName",true,true);
  confineCmd1->SetDefaultValue("NULL");

  cmdName = GetDirectoryName() + "pos/setImage";
  setImageCmd1 = new G4UIcmdWithAString(cmdName,this);
  setImageCmd1->SetGuidance("Biased X and Y positions according to an image (UserFluenceImage source type only)");
  setImageCmd1->SetParameterName("Image",true,true);
  setImageCmd1->SetDefaultValue("");

  // old implementation
  cmdName = GetDirectoryName() + "type";
  typeCmd = new G4UIcmdWithAString(cmdName,this);
  typeCmd->SetGuidance("DEPRECATED: use 'pos/type' instead! Sets source distribution type.");
  typeCmd->SetParameterName("DisType",true,true);
  typeCmd->SetDefaultValue("Point");
  typeCmd->SetCandidates("Point Beam Plane Surface Volume");

  // old implementation
  cmdName = GetDirectoryName() + "shape";
  shapeCmd = new G4UIcmdWithAString(cmdName,this);
  shapeCmd->SetGuidance("DEPRECATED: use 'pos/shape' instead! Sets source shape type.");
  shapeCmd->SetParameterName("Shape",true,true);
  shapeCmd->SetDefaultValue("NULL");
  shapeCmd->SetCandidates("Circle Annulus Ellipse Square Rectangle Sphere Ellipsoid Cylinder Para");

  // this is NOT an old implementation (I don't see a replacement for it; DJB)
  cmdName = GetDirectoryName() + "positronRange";
  positronRangeCmd = new G4UIcmdWithAString(cmdName,this);
  positronRangeCmd->SetGuidance("Sets positron range.");
  positronRangeCmd->SetParameterName("positronrange",true,true);
  positronRangeCmd->SetDefaultValue("NULL");
  positronRangeCmd->SetCandidates("Fluor18 Carbon11 Oxygen15");

  // old implementation
  cmdName = GetDirectoryName() + "centre";
  centreCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
  centreCmd->SetGuidance("DEPRECATED: use 'pos/centre' instead! Set centre coordinates of source.");
  centreCmd->SetParameterName("X","Y","Z",true,true);
  centreCmd->SetDefaultUnit("cm");
  centreCmd->SetUnitCandidates("mum mm cm m km");

  cmdName = GetDirectoryName() + "posrot1";
  posrot1Cmd = new G4UIcmdWith3Vector(cmdName,this);
  posrot1Cmd->SetGuidance("DEPRECATED: use 'pos/rot1' instead! Set rotation matrix of x'.");
  posrot1Cmd->SetGuidance("Posrot1 does not need to be a unit vector.");
  posrot1Cmd->SetParameterName("R1x","R1y","R1z",true,true);
  posrot1Cmd->SetRange("R1x != 0 || R1y != 0 || R1z != 0");

  cmdName = GetDirectoryName() + "posrot2";
  posrot2Cmd = new G4UIcmdWith3Vector(cmdName,this);
  posrot2Cmd->SetGuidance("DEPRECATED: use 'pos/rot2' instead! Set rotation matrix of y'.");
  posrot2Cmd->SetGuidance("Posrot2 does not need to be a unit vector.");
  posrot2Cmd->SetParameterName("R2x","R2y","R2z",true,true);
  posrot2Cmd->SetRange("R2x != 0 || R2y != 0 || R2z != 0");

  cmdName = GetDirectoryName() + "halfx";
  halfxCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  halfxCmd->SetGuidance("DEPRECATED: use 'pos/halfx' instead! Set x half length of source.");
  halfxCmd->SetParameterName("Halfx",true,true);
  halfxCmd->SetDefaultUnit("cm");
  halfxCmd->SetUnitCandidates("mum mm cm m km");

  cmdName = GetDirectoryName() + "halfy";
  halfyCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  halfyCmd->SetGuidance("DEPRECATED: use 'pos/halfy' instead! Set y half length of source.");
  halfyCmd->SetParameterName("Halfy",true,true);
  halfyCmd->SetDefaultUnit("cm");
  halfyCmd->SetUnitCandidates("mum mm cm m km");

  cmdName = GetDirectoryName() + "halfz";
  halfzCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  halfzCmd->SetGuidance("DEPRECATED: use 'pos/halfz' instead! Set z half length of source.");
  halfzCmd->SetParameterName("Halfz",true,true);
  halfzCmd->SetDefaultUnit("cm");
  halfzCmd->SetUnitCandidates("mum mm cm m km");

  cmdName = GetDirectoryName() + "radius";
  radiusCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  radiusCmd->SetGuidance("DEPRECATED: use 'pos/radius' instead! Set radius of source.");
  radiusCmd->SetParameterName("Radius",true,true);
  radiusCmd->SetDefaultUnit("cm");
  radiusCmd->SetUnitCandidates("mum mm cm m km");

  cmdName = GetDirectoryName() + "radius0";
  radius0Cmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  radius0Cmd->SetGuidance("DEPRECATED: use 'pos/inner_radius' instead! Set inner radius of source.");
  radius0Cmd->SetParameterName("Radius0",true,true);
  radius0Cmd->SetDefaultUnit("cm");
  radius0Cmd->SetUnitCandidates("mum mm cm m km");

  cmdName = GetDirectoryName() + "sigmaposr";
  possigmarCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  possigmarCmd->SetGuidance("DEPRECATED: use 'pos/sigma_r' instead! Set standard deviation of beam position in radial");
  possigmarCmd->SetParameterName("Sigmar",true,true);
  possigmarCmd->SetDefaultUnit("cm");
  possigmarCmd->SetUnitCandidates("mum mm cm m km");

  cmdName = GetDirectoryName() + "sigmaposx";
  possigmaxCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  possigmaxCmd->SetGuidance("DEPRECATED: use 'pos/sigma_x' instead! Set standard deviation of beam position in x-dir");
  possigmaxCmd->SetParameterName("Sigmax",true,true);
  possigmaxCmd->SetDefaultUnit("cm");
  possigmaxCmd->SetUnitCandidates("mum mm cm m km");

  cmdName = GetDirectoryName() + "sigmaposy";
  possigmayCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  possigmayCmd->SetGuidance("DEPRECATED: use 'pos/sigma_y' instead! Set standard deviation of beam position in y-dir");
  possigmayCmd->SetParameterName("Sigmay",true,true);
  possigmayCmd->SetDefaultUnit("cm");
  possigmayCmd->SetUnitCandidates("mum mm cm m km");

  cmdName = GetDirectoryName() + "paralp";
  paralpCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  paralpCmd->SetGuidance("DEPRECATED: use 'pos/paralp' instead! Angle from y-axis of y' in Para");
  paralpCmd->SetParameterName("paralp",true,true);
  paralpCmd->SetDefaultUnit("rad");
  paralpCmd->SetUnitCandidates("rad deg");

  cmdName = GetDirectoryName() + "parthe";
  partheCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  partheCmd->SetGuidance("DEPRECATED: use 'pos/parthe' instead! Polar angle through centres of z faces");
  partheCmd->SetParameterName("parthe",true,true);
  partheCmd->SetDefaultUnit("rad");
  partheCmd->SetUnitCandidates("rad deg");

  cmdName = GetDirectoryName() + "parphi";
  parphiCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  parphiCmd->SetGuidance("DEPRECATED: use 'pos/parphi' instead! Azimuth angle through centres of z faces");
  parphiCmd->SetParameterName("parphi",true,true);
  parphiCmd->SetDefaultUnit("rad");
  parphiCmd->SetUnitCandidates("rad deg");

  cmdName = GetDirectoryName() + "confine";
  confineCmd = new G4UIcmdWithAString(cmdName,this);
  confineCmd->SetGuidance("DEPRECATED: use 'pos/confine' instead! Confine source to volume (NULL to unset).");
  confineCmd->SetGuidance("usage: confine VolName");
  confineCmd->SetParameterName("VolName",true,true);
  confineCmd->SetDefaultValue("NULL");

  cmdName = GetDirectoryName() + "Forbid";
  ForbidCmd = new G4UIcmdWithAString(cmdName,this);
  ForbidCmd->SetGuidance("Forbid source activity in volume (NULL to unset).");
  ForbidCmd->SetGuidance("usage: Forbid VolName");
  ForbidCmd->SetParameterName("VolName",true,true);
  ForbidCmd->SetDefaultValue("NULL");


  // Angular distribution commands
  cmdName = GetDirectoryName() + "ang/type";
  angtypeCmd1 = new G4UIcmdWithAString(cmdName,this);
  angtypeCmd1->SetGuidance("Sets angular source distribution type");
  angtypeCmd1->SetGuidance("Possible variables are: iso, cos, planar, beam1d, beam2d, focused, userFocused or user");
  angtypeCmd1->SetParameterName("AngDis",true,true);
  angtypeCmd1->SetDefaultValue("iso");
  angtypeCmd1->SetCandidates("iso cos planar beam1d beam2d focused userFocused user");

  cmdName = GetDirectoryName() + "ang/radius";
  angradiusCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  angradiusCmd1->SetGuidance("Set radius of aperture (userFocused angle distribution type only)");
  angradiusCmd1->SetParameterName("Radius",true,true);
  angradiusCmd1->SetDefaultUnit("cm");

  cmdName = GetDirectoryName() + "ang/centre";
  angcentreCmd1 = new G4UIcmdWith3VectorAndUnit(cmdName,this);
  angcentreCmd1->SetGuidance("Set centre coordinates of ang dist (userFocused angle distribution type only).");
  angcentreCmd1->SetParameterName("X","Y","Z",true,true);
  angcentreCmd1->SetDefaultUnit("cm");

  cmdName = GetDirectoryName() + "ang/rot1";
  angrot1Cmd1 = new G4UIcmdWith3Vector(cmdName,this);
  angrot1Cmd1->SetGuidance("Sets the 1st vector for angular distribution rotation matrix");
  angrot1Cmd1->SetGuidance("Need not be a unit vector");
  angrot1Cmd1->SetParameterName("AR1x","AR1y","AR1z",true,true);
  angrot1Cmd1->SetRange("AR1x != 0 || AR1y != 0 || AR1z != 0");

  cmdName = GetDirectoryName() + "ang/rot2";
  angrot2Cmd1 = new G4UIcmdWith3Vector(cmdName,this);
  angrot2Cmd1->SetGuidance("Sets the 2nd vector for angular distribution rotation matrix");
  angrot2Cmd1->SetGuidance("Need not be a unit vector");
  angrot2Cmd1->SetParameterName("AR2x","AR2y","AR2z",true,true);
  angrot2Cmd1->SetRange("AR2x != 0 || AR2y != 0 || AR2z != 0");

  cmdName = GetDirectoryName() + "ang/mintheta";
  minthetaCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  minthetaCmd1->SetGuidance("Set minimum theta");
  minthetaCmd1->SetParameterName("MinTheta",true,true);
  minthetaCmd1->SetDefaultValue(0.);
  minthetaCmd1->SetDefaultUnit("rad");

  cmdName = GetDirectoryName() + "ang/maxtheta";
  maxthetaCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  maxthetaCmd1->SetGuidance("Set maximum theta");
  maxthetaCmd1->SetParameterName("MaxTheta",true,true);
  maxthetaCmd1->SetDefaultValue(pi);
  maxthetaCmd1->SetDefaultUnit("rad");

  cmdName = GetDirectoryName() + "ang/minphi";
  minphiCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  minphiCmd1->SetGuidance("Set minimum phi");
  minphiCmd1->SetParameterName("MinPhi",true,true);
  minphiCmd1->SetDefaultUnit("rad");

  cmdName = GetDirectoryName() + "ang/maxphi";
  maxphiCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  maxphiCmd1->SetGuidance("Set maximum phi");
  maxphiCmd1->SetParameterName("MaxPhi",true,true);
  maxphiCmd1->SetDefaultValue(pi);
  maxphiCmd1->SetDefaultUnit("rad");

  cmdName = GetDirectoryName() + "ang/sigma_r";
  angsigmarCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  angsigmarCmd1->SetGuidance("Set standard deviation in direction for 1D beam.");
  angsigmarCmd1->SetParameterName("Sigmara",true,true);
  angsigmarCmd1->SetDefaultUnit("rad");

  cmdName = GetDirectoryName() + "ang/sigma_x";
  angsigmaxCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  angsigmaxCmd1->SetGuidance("Set standard deviation in direction in x-direc. for 2D beam");
  angsigmaxCmd1->SetParameterName("Sigmaxa",true,true);
  angsigmaxCmd1->SetDefaultUnit("rad");

  cmdName = GetDirectoryName() + "ang/sigma_y";
  angsigmayCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  angsigmayCmd1->SetGuidance("Set standard deviation in direction in y-direc. for 2D beam");
  angsigmayCmd1->SetParameterName("Sigmaya",true,true);
  angsigmayCmd1->SetDefaultUnit("rad");

  cmdName = GetDirectoryName() + "ang/focuspoint";
  angfocusCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
  angfocusCmd->SetGuidance("Set the focusing point for the beam");
  angfocusCmd->SetParameterName("x","y","z",true,true);
  angfocusCmd->SetDefaultUnit("cm");

  cmdName = GetDirectoryName() + "ang/user_coor";
  useuserangaxisCmd1 = new G4UIcmdWithABool(cmdName,this);
  useuserangaxisCmd1->SetGuidance("true for using user defined angular co-ordinates");
  useuserangaxisCmd1->SetGuidance("Default is false");
  useuserangaxisCmd1->SetParameterName("useuserangaxis",true);
  useuserangaxisCmd1->SetDefaultValue(false);

  cmdName = GetDirectoryName() + "ang/surfnorm";
  surfnormCmd1 = new G4UIcmdWithABool(cmdName,this);
  surfnormCmd1->SetGuidance("Makes a user-defined distribution with respect to surface normals rather than x,y,z axes.");
  surfnormCmd1->SetGuidance("Default is false");
  surfnormCmd1->SetParameterName("surfnorm",true);
  surfnormCmd1->SetDefaultValue(false);


  // old ones
  cmdName = GetDirectoryName() + "angtype";
  angtypeCmd = new G4UIcmdWithAString( cmdName,this) ;
  angtypeCmd->SetGuidance( "DEPRECATED: use 'ang/type' instead! Sets angular source distribution type") ;
  angtypeCmd->SetGuidance( "Possible variables are: iso, cos planar beam1d beam2d or user") ;
  angtypeCmd->SetParameterName( "AngDis",true,true) ;
  angtypeCmd->SetDefaultValue( "iso") ;
  angtypeCmd->SetCandidates( "iso cos planar beam1d beam2d user focused") ;

  cmdName = GetDirectoryName() + "angrot1";
  angrot1Cmd = new G4UIcmdWith3Vector(cmdName,this);
  angrot1Cmd->SetGuidance("DEPRECATED: use 'ang/rot1' instead! Sets the x' vector for angular distribution");
  angrot1Cmd->SetGuidance("Need not be a unit vector");
  angrot1Cmd->SetParameterName("AR1x","AR1y","AR1z",true,true);
  angrot1Cmd->SetRange("AR1x != 0 || AR1y != 0 || AR1z != 0");

  cmdName = GetDirectoryName() + "angrot2";
  angrot2Cmd = new G4UIcmdWith3Vector(cmdName,this);
  angrot2Cmd->SetGuidance("DEPRECATED: use 'ang/rot2' instead! Sets the y' vector for angular distribution");
  angrot2Cmd->SetGuidance("Need not be a unit vector");
  angrot2Cmd->SetParameterName("AR2x","AR2y","AR2z",true,true);
  angrot2Cmd->SetRange("AR2x != 0 || AR2y != 0 || AR2z != 0");

  cmdName = GetDirectoryName() + "mintheta";
  minthetaCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  minthetaCmd->SetGuidance("DEPRECATED: use 'ang/mintheta' instead! Set minimum theta");
  minthetaCmd->SetParameterName("MinTheta",true,true);
  minthetaCmd->SetDefaultUnit("rad");
  minthetaCmd->SetUnitCandidates("rad deg");

  cmdName = GetDirectoryName() + "maxtheta";
  maxthetaCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  maxthetaCmd->SetGuidance("DEPRECATED: use 'ang/maxtheta' instead! Set maximum theta");
  maxthetaCmd->SetParameterName("MaxTheta",true,true);
  maxthetaCmd->SetDefaultValue(pi);
  maxthetaCmd->SetDefaultUnit("rad");
  maxthetaCmd->SetUnitCandidates("rad deg");

  cmdName = GetDirectoryName() + "minphi";
  minphiCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  minphiCmd->SetGuidance("DEPRECATED: use 'ang/minphi' instead! Set minimum phi");
  minphiCmd->SetParameterName("MinPhi",true,true);
  minphiCmd->SetDefaultUnit("rad");
  minphiCmd->SetUnitCandidates("rad deg");

  cmdName = GetDirectoryName() + "maxphi";
  maxphiCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  maxphiCmd->SetGuidance("DEPRECATED: use 'ang/maxphi' instead! Set maximum phi");
  maxphiCmd->SetParameterName("MaxPhi",true,true);
  maxphiCmd->SetDefaultUnit("rad");
  maxphiCmd->SetUnitCandidates("rad deg");

  cmdName = GetDirectoryName() + "sigmaangr";
  angsigmarCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  angsigmarCmd->SetGuidance("DEPRECATED: use 'ang/sigma_r' instead! Set standard deviation of beam direction in radial.");
  angsigmarCmd->SetParameterName("Sigmara",true,true);
  angsigmarCmd->SetDefaultUnit("rad");
  angsigmarCmd->SetUnitCandidates("rad deg");

  cmdName = GetDirectoryName() + "sigmaangx";
  angsigmaxCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  angsigmaxCmd->SetGuidance("DEPRECATED: use 'ang/sigma_x' instead! Set standard deviation of beam direction in x-direc.");
  angsigmaxCmd->SetParameterName("Sigmaxa",true,true);
  angsigmaxCmd->SetDefaultUnit("rad");
  angsigmaxCmd->SetUnitCandidates("rad deg");

  cmdName = GetDirectoryName() + "sigmaangy";
  angsigmayCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  angsigmayCmd->SetGuidance("DEPRECATED: use 'ang/sigma_y' instead! Set standard deviation of beam direction in y-direc.");
  angsigmayCmd->SetParameterName("Sigmaya",true,true);
  angsigmayCmd->SetDefaultUnit("rad");
  angsigmayCmd->SetUnitCandidates("rad deg");

  cmdName = GetDirectoryName() + "useuserangaxis";
  useuserangaxisCmd = new G4UIcmdWithABool(cmdName,this);
  useuserangaxisCmd->SetGuidance("DEPRECATED: use 'ang/user_coor' instead! Set to true for using user defined angular co-ordinates");
  useuserangaxisCmd->SetGuidance("Default is false");
  useuserangaxisCmd->SetParameterName("useuserangaxis",true);
  useuserangaxisCmd->SetDefaultValue(false);

  cmdName = GetDirectoryName() + "surfnorm";
  surfnormCmd = new G4UIcmdWithABool(cmdName,this);
  surfnormCmd->SetGuidance("DEPRECATED: use 'ang/surfnorm' instead! Makes a user-defined distribution with respect to surface normals rather than x,y,z axes.");
  surfnormCmd->SetGuidance("Default is false");
  surfnormCmd->SetParameterName("surfnorm",true);
  surfnormCmd->SetDefaultValue(false);

  // Energy
  cmdName = GetDirectoryName() + "ene/type";
  energytypeCmd1 = new G4UIcmdWithAString(cmdName,this);
  energytypeCmd1->SetGuidance("Sets energy distribution type");
  energytypeCmd1->SetParameterName("EnergyDis",true,true);
  energytypeCmd1->SetDefaultValue("Mono");
  energytypeCmd1->SetCandidates("Mono Fluor18 Oxygen15 Carbon11 Lin Pow Exp Gauss Brem Bbody Range Cdg User Arb Epn UserSpectrum");

  cmdName = GetDirectoryName() + "ene/min";
  eminCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  eminCmd1->SetGuidance("Sets minimum energy");
  eminCmd1->SetParameterName("emin",true,true);
  eminCmd1->SetDefaultUnit("keV");

  cmdName = GetDirectoryName() + "ene/max";
  emaxCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  emaxCmd1->SetGuidance("Sets maximum energy");
  emaxCmd1->SetParameterName("emax",true,true);
  emaxCmd1->SetDefaultUnit("keV");

  cmdName = GetDirectoryName() + "ene/mono";
  monoenergyCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  monoenergyCmd1->SetGuidance("Sets a monocromatic energy (same as  gps/energy)");
  monoenergyCmd1->SetParameterName("monoenergy",true,true);
  monoenergyCmd1->SetDefaultUnit("keV");

  cmdName = GetDirectoryName() + "ene/sigma";
  engsigmaCmd1 = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  engsigmaCmd1->SetGuidance("Sets the standard deviation for Gaussian energy dist.");
  engsigmaCmd1->SetParameterName("Sigmae",true,true);
  engsigmaCmd1->SetDefaultUnit("keV");

  cmdName = GetDirectoryName() + "ene/alpha";
  alphaCmd1 = new G4UIcmdWithADouble(cmdName,this);
  alphaCmd1->SetGuidance("Sets Alpha (index) for power-law energy dist.");
  alphaCmd1->SetParameterName("alpha",true,true);

  cmdName = GetDirectoryName() + "ene/temp";
  tempCmd1 = new G4UIcmdWithADouble(cmdName,this);
  tempCmd1->SetGuidance("Sets the temperature for Brem and BBody distributions (in Kelvin)");
  tempCmd1->SetParameterName("temp",true,true);

  cmdName = GetDirectoryName() + "ene/ezero";
  ezeroCmd1 = new G4UIcmdWithADouble(cmdName,this);
  ezeroCmd1->SetGuidance("Sets E_0 for exponential distribution (in MeV)");
  ezeroCmd1->SetParameterName("ezero",true,true);

  cmdName = GetDirectoryName() + "ene/gradient";
  gradientCmd1 = new G4UIcmdWithADouble(cmdName,this);
  gradientCmd1->SetGuidance("Sets the gradient for Lin distribution (in 1/MeV)");
  gradientCmd1->SetParameterName("gradient",true,true);

  cmdName = GetDirectoryName() + "ene/intercept";
  interceptCmd1 = new G4UIcmdWithADouble(cmdName,this);
  interceptCmd1->SetGuidance("Sets the intercept for Lin distributions (in MeV)");
  interceptCmd1->SetParameterName("intercept",true,true);

  cmdName = GetDirectoryName() + "ene/calculate";
  calculateCmd1 = new G4UIcmdWithoutParameter(cmdName,this);
  calculateCmd1->SetGuidance("Calculates the distributions for Cdg and BBody");

  cmdName = GetDirectoryName() + "ene/emspec";
  energyspecCmd1 = new G4UIcmdWithABool(cmdName,this);
  energyspecCmd1->SetGuidance("True for energy and false for momentum spectra");
  energyspecCmd1->SetParameterName("energyspec",true);
  energyspecCmd1->SetDefaultValue(true);

  cmdName = GetDirectoryName() + "ene/diffspec";
  diffspecCmd1 = new G4UIcmdWithABool(cmdName,this);
  diffspecCmd1->SetGuidance("True for differential and flase for integral spectra");
  diffspecCmd1->SetParameterName("diffspec",true);
  diffspecCmd1->SetDefaultValue(true);


  cmdName = GetDirectoryName() + "energytype";
  energytypeCmd = new G4UIcmdWithAString(cmdName,this);
  energytypeCmd->SetGuidance("DEPRECATED: use 'ene/type' instead! Sets energy distribution type");
  energytypeCmd->SetParameterName("EnergyDis",true,true);
  energytypeCmd->SetDefaultValue("Mono");
  energytypeCmd->SetCandidates("Mono Fluor18 Oxygen15 Carbon11 Lin Pow Exp Gauss Brem Bbody Cdg User Arb Epn UserSpectrum");


  cmdName = GetDirectoryName() + "setSpectrumFile";
  setUserSpectrumCmd = new G4UIcmdWithAString(cmdName,this);
  setUserSpectrumCmd->SetGuidance("Sets the file to construct UserSpectrum");
  setUserSpectrumCmd->SetParameterName("FileName",true,true);


  cmdName = GetDirectoryName() + "emin";
  eminCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  eminCmd->SetGuidance("DEPRECATED: use 'ene/min' instead! Sets Emin");
  eminCmd->SetParameterName("emin",true,true);
  eminCmd->SetDefaultUnit("keV");
  eminCmd->SetUnitCandidates("eV keV MeV GeV TeV PeV");

  cmdName = GetDirectoryName() + "emax";
  emaxCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  emaxCmd->SetGuidance("DEPRECATED: use 'ene/max' instead! Sets Emax");
  emaxCmd->SetParameterName("emax",true,true);
  emaxCmd->SetDefaultUnit("keV");
  emaxCmd->SetUnitCandidates("eV keV MeV GeV TeV PeV");

  cmdName = GetDirectoryName() + "monoenergy";
  monoenergyCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  monoenergyCmd->SetGuidance("DEPRECATED: use 'ene/mono' instead! Sets Monoenergy");
  monoenergyCmd->SetParameterName("monoenergy",true,true);
  monoenergyCmd->SetDefaultUnit("keV");
  monoenergyCmd->SetUnitCandidates("eV keV MeV GeV TeV PeV");

  cmdName = GetDirectoryName() + "sigmae";
  engsigmaCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  engsigmaCmd->SetGuidance("DEPRECATED: use 'ene/sigma' instead! Sets the standard deviation for Gaussian energy dist.");
  engsigmaCmd->SetParameterName("Sigmae",true,true);
  engsigmaCmd->SetDefaultUnit("keV");
  engsigmaCmd->SetUnitCandidates("eV keV MeV GeV TeV PeV");

  cmdName = GetDirectoryName() + "alpha";
  alphaCmd = new G4UIcmdWithADouble(cmdName,this);
  alphaCmd->SetGuidance("DEPRECATED: use 'ene/alpha' instead! Sets Alpha (index) for power-law energy dist.");
  alphaCmd->SetParameterName("alpha",true,true);

  cmdName = GetDirectoryName() + "temp";
  tempCmd = new G4UIcmdWithADouble(cmdName,this);
  tempCmd->SetGuidance("DEPRECATED: use 'ene/temp' instead! Sets the temperature for Brem and BBody (in Kelvin)");
  tempCmd->SetParameterName("temp",true,true);

  cmdName = GetDirectoryName() + "ezero";
  ezeroCmd = new G4UIcmdWithADouble(cmdName,this);
  ezeroCmd->SetGuidance("DEPRECATED: use 'ene/ezero' instead! Sets ezero exponential distributions (in MeV)");
  ezeroCmd->SetParameterName("ezero",true,true);

  cmdName = GetDirectoryName() + "gradient";
  gradientCmd = new G4UIcmdWithADouble(cmdName,this);
  gradientCmd->SetGuidance("DEPRECATED: use 'ene/gradient' instead! Sets the gradient for Lin distributions (in 1/MeV)");
  gradientCmd->SetParameterName("gradient",true,true);

  cmdName = GetDirectoryName() + "intercept";
  interceptCmd = new G4UIcmdWithADouble(cmdName,this);
  interceptCmd->SetGuidance("DEPRECATED: use 'ene/intercept' instead! Sets the intercept for Lin distributions (in MeV)");
  interceptCmd->SetParameterName("intercept",true,true);

  cmdName = GetDirectoryName() + "calculate";
  calculateCmd = new G4UIcmdWithoutParameter(cmdName,this);
  calculateCmd->SetGuidance("DEPRECATED: use 'ene/calculate' instead! Calculates distributions for Cdg and BBody");

  cmdName = GetDirectoryName() + "energyspec";
  energyspecCmd = new G4UIcmdWithABool(cmdName,this);
  energyspecCmd->SetGuidance("DEPRECATED: use 'ene/emspec' instead! True for energy and false for momentum spectra");
  energyspecCmd->SetParameterName("energyspec",true);
  energyspecCmd->SetDefaultValue(true);

  cmdName = GetDirectoryName() + "diffspec";
  diffspecCmd = new G4UIcmdWithABool(cmdName,this);
  diffspecCmd->SetGuidance("DEPRECATED: use 'ene/diffspec' instead! True for differential and false for integral spectra");
  diffspecCmd->SetParameterName("diffspec",true);
  diffspecCmd->SetDefaultValue(true);



  // Biasing + histograms in general
  cmdName = GetDirectoryName() + "hist/type";
  histtypeCmd = new G4UIcmdWithAString(cmdName,this);
  histtypeCmd->SetGuidance("Sets histogram type. Should be set *before* providing histogram points with 'hist/point'.");
  histtypeCmd->SetParameterName("HistType",true,true);
  histtypeCmd->SetDefaultValue("biasx");
  histtypeCmd->SetCandidates("biasx biasy biasz biast biasp biase biaspt biaspp theta phi energy arb epn");

  cmdName = GetDirectoryName() + "hist/reset";
  resethistCmd1 = new G4UIcmdWithAString(cmdName,this);
  resethistCmd1->SetGuidance("Reset (clean) the histogram ");
  resethistCmd1->SetParameterName("HistType",true,true);
  resethistCmd1->SetDefaultValue("energy");
  resethistCmd1->SetCandidates("biasx biasy biasz biast biasp biase biaspt biaspp theta phi energy arb epn");

  cmdName = GetDirectoryName() + "hist/point";
  histpointCmd1 = new G4UIcmdWith3Vector(cmdName,this);
  histpointCmd1->SetGuidance("Allows user to define a histogram. Make sure to first set the type with 'hist/type'.");
  histpointCmd1->SetGuidance("Enter: Ehi Weight");
  histpointCmd1->SetParameterName("Ehi","Weight","Junk",true,true);
  histpointCmd1->SetRange("Ehi >= 0. && Weight >= 0.");

  cmdName = GetDirectoryName() + "hist/inter";
  arbintCmd1 = new G4UIcmdWithAString(cmdName,this);
  arbintCmd1->SetGuidance("Sets the interpolation method for arbitrary distribution.");
  arbintCmd1->SetParameterName("int",true,true);
  arbintCmd1->SetDefaultValue("Lin");
  arbintCmd1->SetCandidates("Lin Log Exp Spline");

  // old ones
  cmdName = GetDirectoryName() + "histname";
  histnameCmd = new G4UIcmdWithAString(cmdName,this);
  histnameCmd->SetGuidance("DEPRECATED, use 'hist/type' instead! Sets histogram *type*.");
  histnameCmd->SetParameterName("HistType",true,true);
  histnameCmd->SetDefaultValue("biasx");
  histnameCmd->SetCandidates("biasx biasy biasz biast biasp biase theta phi energy arb epn");

  cmdName = GetDirectoryName() + "resethist";
  resethistCmd = new G4UIcmdWithAString(cmdName,this);
  resethistCmd->SetGuidance("DEPRECATED, use 'hist/reset' instead! Re-Set the histogram.");
  resethistCmd->SetParameterName("HistType",true,true);
  resethistCmd->SetDefaultValue("energy");
  resethistCmd->SetCandidates("biasx biasy biasz biast biasp biase theta phi energy arb epn");

  cmdName = GetDirectoryName() + "histpoint";
  histpointCmd = new G4UIcmdWith3Vector(cmdName,this);
  histpointCmd->SetGuidance("DEPRECATED, use 'hist/point' instead! Allows user to define a histogram");
  histpointCmd->SetGuidance("Enter: Ehi Weight");
  histpointCmd->SetParameterName("Ehi","Weight","Junk",true,true);
  histpointCmd->SetRange("Ehi >= 0. && Weight >= 0.");

  cmdName = GetDirectoryName() + "arbint";
  arbintCmd = new G4UIcmdWithAString(cmdName,this);
  arbintCmd->SetGuidance("DEPRECATED, use 'hist/inter' instead! Sets Arbitrary Interpolation type.");
  arbintCmd->SetParameterName("int",true,true);
  arbintCmd->SetDefaultValue("NULL");
  arbintCmd->SetCandidates("Lin Log Exp Spline");

  // verbosity
  cmdName = GetDirectoryName() + "verbose";
  verbosityCmd = new G4UIcmdWithAnInteger(cmdName,this);
  verbosityCmd->SetGuidance("Set Verbose level for GPS");
  verbosityCmd->SetGuidance(" 0 : Silent");
  verbosityCmd->SetGuidance(" 1 : Limited information");
  verbosityCmd->SetGuidance(" 2 : Detailed information");
  verbosityCmd->SetParameterName("level",false);
  verbosityCmd->SetRange("level>=0 && level <=2");

}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateSingleParticleSourceMessenger::~GateSingleParticleSourceMessenger()
{
  delete relativePlacementCmd;
  if(positionDirectory) delete   positionDirectory;

  delete typeCmd;
  delete shapeCmd;
  delete centreCmd;
  delete posrot1Cmd;
  delete posrot2Cmd;
  delete halfxCmd;
  delete halfyCmd;
  delete halfzCmd;
  delete radiusCmd;
  delete radius0Cmd;
  delete possigmarCmd;
  delete possigmaxCmd;
  delete possigmayCmd;
  delete paralpCmd;
  delete partheCmd;
  delete parphiCmd;
  delete confineCmd;
  delete ForbidCmd;

  delete typeCmd1;
  delete shapeCmd1;
  delete centreCmd1;
  delete posrot1Cmd1;
  delete posrot2Cmd1;
  delete halfxCmd1;
  delete halfyCmd1;
  delete halfzCmd1;
  delete radiusCmd1;
  delete radius0Cmd1;
  delete possigmarCmd1;
  delete possigmaxCmd1;
  delete possigmayCmd1;
  delete paralpCmd1;
  delete partheCmd1;
  delete parphiCmd1;
  delete confineCmd1;
  delete setImageCmd1;

  delete angtypeCmd;
  delete angrot1Cmd;
  delete angrot2Cmd;
  delete minthetaCmd;
  delete maxthetaCmd;
  delete minphiCmd;
  delete maxphiCmd;
  delete angsigmarCmd;
  delete angsigmaxCmd;
  delete angsigmayCmd;
  delete useuserangaxisCmd;
  delete surfnormCmd;

  delete angtypeCmd1;
  delete angradiusCmd1;
  delete angcentreCmd1;
  delete angrot1Cmd1;
  delete angrot2Cmd1;
  delete minthetaCmd1;
  delete maxthetaCmd1;
  delete minphiCmd1;
  delete maxphiCmd1;
  delete angsigmarCmd1;
  delete angsigmaxCmd1;
  delete angsigmayCmd1;
  delete angfocusCmd;
  delete useuserangaxisCmd1;
  delete surfnormCmd1;


  delete energytypeCmd;
  delete eminCmd;
  delete emaxCmd;
  delete monoenergyCmd;
  delete engsigmaCmd;
  delete alphaCmd;
  delete tempCmd;
  delete ezeroCmd;
  delete gradientCmd;
  delete interceptCmd;
  delete calculateCmd;
  delete energyspecCmd;
  delete diffspecCmd;

  if(energyDirectory) delete energyDirectory;

  delete energytypeCmd1;
  delete eminCmd1;
  delete emaxCmd1;
  delete monoenergyCmd1;
  delete engsigmaCmd1;
  delete alphaCmd1;
  delete tempCmd1;
  delete ezeroCmd1;
  delete gradientCmd1;
  delete interceptCmd1;
  delete calculateCmd1;
  delete energyspecCmd1;
  delete diffspecCmd1;

  delete histnameCmd;
  delete resethistCmd;
  delete histpointCmd;
  delete arbintCmd;

  delete histtypeCmd;
  delete resethistCmd1;
  delete histpointCmd1;
  delete arbintCmd1;


  delete verbosityCmd;
  delete ionCmd;
  delete particleCmd;
  delete timeCmd;
  delete polCmd;
  delete numberCmd;

  delete positionCmd;
  delete directionCmd;
  delete energyCmd;
  delete listCmd;

  delete positronRangeCmd;
  delete setUserSpectrumCmd;

  //delete particleTable;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSingleParticleSourceMessenger::SetNewValue( G4UIcommand* command, G4String newValues)
{
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();

  if (command == relativePlacementCmd) {
    fParticleGun->SetRelativePlacementVolume(newValues);
  }
  else if (command == typeCmd) {
    GateWarning("The 'type' option is DEPRECATED, use 'pos/type' instead!");
    fParticleGun->GetPosDist()->SetPosDisType(newValues) ;
  }
  else if (command == shapeCmd) {
    GateWarning("The 'shape' option is DEPRECATED, use 'pos/shape' instead!");
    fParticleGun->GetPosDist()->SetPosDisShape(newValues) ;
  }
  else if (command == positronRangeCmd) {
    // this command actually does no seem to have a 'modern' replacement. No other "SetPositronRange" calls anywhere.
    fParticleGun->GetPosDist()->SetPositronRange(newValues) ;
  }
  else if (command == centreCmd) {
    GateWarning("The 'centre' option is DEPRECATED, use 'pos/centre' instead!");
    fParticleGun->GetPosDist()->SetCentreCoords( centreCmd->GetNew3VectorValue(newValues)) ;
    fParticleGun->SetCentreCoords( centreCmd->GetNew3VectorValue(newValues));
  }
  else if (command == posrot1Cmd) {
    GateWarning("The 'posrot1' option is DEPRECATED, use 'pos/rot1' instead!");
    fParticleGun->GetPosDist()->SetPosRot1( posrot1Cmd->GetNew3VectorValue(newValues)) ;
    fParticleGun->SetPosRot1(posrot1Cmd->GetNew3VectorValue(newValues));
  }
  else if (command == posrot2Cmd) {
    GateWarning("The 'posrot2' option is DEPRECATED, use 'pos/rot2' instead!");
    fParticleGun->GetPosDist()->SetPosRot2(posrot2Cmd->GetNew3VectorValue(newValues));
    fParticleGun->SetPosRot2(posrot2Cmd->GetNew3VectorValue(newValues));
  }
  else if (command == halfxCmd) {
    GateWarning("The 'halfx' option is DEPRECATED, use 'pos/halfx' instead!");
    fParticleGun->GetPosDist()->SetHalfX( halfxCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == halfyCmd) {
    GateWarning("The 'halfy' option is DEPRECATED, use 'pos/halfy' instead!");
    fParticleGun->GetPosDist()->SetHalfY( halfyCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == halfzCmd) {
    GateWarning("The 'halfz' option is DEPRECATED, use 'pos/halfz' instead!");
    fParticleGun->GetPosDist()->SetHalfZ( halfzCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == radiusCmd) {
    GateWarning("The 'radius' option is DEPRECATED, use 'pos/radius' instead!");
    fParticleGun->GetPosDist()->SetRadius( radiusCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == radius0Cmd) {
    GateWarning("The 'radius0' option is DEPRECATED, use 'pos/inner_radius' instead!");
    fParticleGun->GetPosDist()->SetRadius0( radius0Cmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == possigmarCmd) {
    GateWarning("The 'sigmaposr' option is DEPRECATED, use 'pos/sigma_r' instead!");
    fParticleGun->GetPosDist()->SetBeamSigmaInR( possigmarCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == possigmaxCmd) {
    GateWarning("The 'sigmaposx' option is DEPRECATED, use 'pos/sigma_x' instead!");
    fParticleGun->GetPosDist()->SetBeamSigmaInX( possigmaxCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == possigmayCmd) {
    GateWarning("The 'sigmaposy' option is DEPRECATED, use 'pos/sigma_y' instead!");
    fParticleGun->GetPosDist()->SetBeamSigmaInY( possigmayCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == paralpCmd) {
    GateWarning("The 'paralp' option is DEPRECATED, use 'pos/paralp' instead!");
    fParticleGun->GetPosDist()->SetParAlpha(paralpCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == partheCmd) {
    GateWarning("The 'parthe' option is DEPRECATED, use 'pos/parthe' instead!");
    fParticleGun->GetPosDist()->SetParTheta( partheCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == parphiCmd) {
    GateWarning("The 'parphi' option is DEPRECATED, use 'pos/parphi' instead!");
    fParticleGun->GetPosDist()->SetParPhi( parphiCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == confineCmd) {
    GateWarning("The 'confine' option is DEPRECATED, use 'pos/confine' instead!");
    // Modif DS/FS: for all names exept NULL, we add the tag "_phys" at the end
    // of the volume name when the user forgot to do it
    if ( newValues != "NULL")
      {
        bool test = false;
        if (newValues.length() < 5) { test = true; }
        else if (newValues.substr( newValues.length()-5) != "_phys") { test = true; }

        if (test)
          {
            newValues += "_phys";
            G4cout << "Confirming confinement to volume '" << newValues << "'...\n" ;
          }
      }
    fParticleGun->GetPosDist()->ConfineSourceToVolume(newValues) ;
  }
  else if (command == ForbidCmd) {
    // this command actually does no seem to have a 'modern' replacement. No other "ForbidSourceToVolume" calls anywhere.
    if ( newValues != "NULL") {
      if (newValues.substr( newValues.length()-5) != "_phys")
        newValues += "_phys";
      G4cout << "Confirming activity forbidden in volume '" << newValues << "'...\n";
    }
    fParticleGun->GetPosDist()->ForbidSourceToVolume(newValues);
  }
  else if (command == angtypeCmd) {
    GateWarning("The 'angtype' option is DEPRECATED, use 'ang/type' instead!");
    fParticleGun->GetAngDist()->SetAngDistType(newValues) ;
  }
  else if (command == angrot1Cmd) {
    GateWarning("The 'angrot1' option is DEPRECATED, use 'ang/rot1' instead!");
    G4String a = "angref1";
    fParticleGun->GetAngDist()->DefineAngRefAxes( a,angrot1Cmd->GetNew3VectorValue(newValues)) ;
  }
  else if (command == angrot2Cmd) {
    GateWarning("The 'angrot2' option is DEPRECATED, use 'ang/rot2' instead!");
    G4String a = "angref2";
    fParticleGun->GetAngDist()->DefineAngRefAxes( a,angrot2Cmd->GetNew3VectorValue(newValues)) ;
  }
  else if (command == minthetaCmd) {
    GateWarning("The 'mintheta' option is DEPRECATED, use 'ang/mintheta' instead!");
    fParticleGun->GetAngDist()->SetMinTheta( minthetaCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == minphiCmd) {
    GateWarning("The 'minphi' option is DEPRECATED, use 'ang/minphi' instead!");
    fParticleGun->GetAngDist()->SetMinPhi( minphiCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == maxthetaCmd) {
    GateWarning("The 'maxtheta' option is DEPRECATED, use 'ang/maxtheta' instead!");
    fParticleGun->GetAngDist()->SetMaxTheta( maxthetaCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == maxphiCmd) {
    GateWarning("The 'maxphi' option is DEPRECATED, use 'ang/maxphi' instead!");
    fParticleGun->GetAngDist()->SetMaxPhi( maxphiCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == angsigmarCmd) {
    GateWarning("The 'sigmaangr' option is DEPRECATED, use 'ang/sigma_r' instead!");
    fParticleGun->GetAngDist()->SetBeamSigmaInAngR( angsigmarCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == angsigmaxCmd) {
    GateWarning("The 'sigmaangx' option is DEPRECATED, use 'ang/sigma_x' instead!");
    fParticleGun->GetAngDist()->SetBeamSigmaInAngX( angsigmaxCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == angsigmayCmd) {
    GateWarning("The 'sigmaangy' option is DEPRECATED, use 'ang/sigma_y' instead!");
    fParticleGun->GetAngDist()->SetBeamSigmaInAngY( angsigmayCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == useuserangaxisCmd) {
    GateWarning("The 'useuserangaxis' option is DEPRECATED, use 'ang/user_coor' instead!");
    fParticleGun->GetAngDist()->SetUseUserAngAxis( useuserangaxisCmd->GetNewBoolValue(newValues)) ;
  }
  else if (command == surfnormCmd) {
    GateWarning("The 'surfnorm' option is DEPRECATED, use 'ang/surfnorm' instead!");
    fParticleGun->GetAngDist()->SetUserWRTSurface( surfnormCmd->GetNewBoolValue(newValues)) ;
  }
  else if (command == energytypeCmd) {
    GateWarning("The 'energytype' option is DEPRECATED, use 'ene/type' instead!");
    fParticleGun->GetEneDist()->SetEnergyDisType(newValues) ;
  }
  else if (command == eminCmd) {
    GateWarning("The 'emin' option is DEPRECATED, use 'ene/min' instead!");
    fParticleGun->GetEneDist()->SetEmin( eminCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == emaxCmd) {
    GateWarning("The 'emax' option is DEPRECATED, use 'ene/max' instead!");
    fParticleGun->GetEneDist()->SetEmax( emaxCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == monoenergyCmd) {
    GateWarning("The 'monoenergy' option is DEPRECATED, use 'ene/mono' instead!");
    fParticleGun->GetEneDist()->SetMonoEnergy( monoenergyCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == engsigmaCmd) {
    GateWarning("The 'sigmae' option is DEPRECATED, use 'ene/sigma' instead!");
    fParticleGun->GetEneDist()->SetBeamSigmaInE( engsigmaCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == alphaCmd) {
    GateWarning("The 'alpha' option is DEPRECATED, use 'ene/alpha' instead!");
    fParticleGun->GetEneDist()->SetAlpha( alphaCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == tempCmd) {
    GateWarning("The 'temp' option is DEPRECATED, use 'ene/temp' instead!");
    fParticleGun->GetEneDist()->SetTemp( tempCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == ezeroCmd) {
    GateWarning("The 'ezero' option is DEPRECATED, use 'ene/ezero' instead!");
    fParticleGun->GetEneDist()->SetEzero( ezeroCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == gradientCmd) {
    GateWarning("The 'gradient' option is DEPRECATED, use 'ene/gradient' instead!");
    fParticleGun->GetEneDist()->SetGradient( gradientCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == interceptCmd) {
    GateWarning("The 'intercept' option is DEPRECATED, use 'ene/intercept' instead!");
    fParticleGun->GetEneDist()->SetInterCept( interceptCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == calculateCmd) {
    GateWarning("The 'calculate' option is DEPRECATED, use 'ene/calculate' instead!");
    fParticleGun->GetEneDist()->Calculate() ;
  }
  else if (command == energyspecCmd) {
    GateWarning("The 'energyspec' option is DEPRECATED, use 'ene/emspec' instead!");
    fParticleGun->GetEneDist()->InputEnergySpectra( energyspecCmd->GetNewBoolValue(newValues)) ;
  }
  else if (command == diffspecCmd) {
    GateWarning("The 'diffspec' option is DEPRECATED, use 'ene/diffspec' instead!");
    fParticleGun->GetEneDist()->InputDifferentialSpectra( diffspecCmd->GetNewBoolValue(newValues)) ;
  }
  else if (command == histnameCmd) {
    GateWarning("The 'histname' option is DEPRECATED, use 'hist/type' instead!");
    histtype = newValues ;
  }
  else if (command == histpointCmd) {
    GateWarning("The 'histpoint' option is DEPRECATED, use 'hist/point' instead!");
    if( histtype == "biasx")
      fParticleGun->GetBiasRndm()->SetXBias( histpointCmd->GetNew3VectorValue(newValues)) ;
    if( histtype == "biasy")
      fParticleGun->GetBiasRndm()->SetYBias( histpointCmd->GetNew3VectorValue(newValues)) ;
    if( histtype == "biasz")
      fParticleGun->GetBiasRndm()->SetZBias( histpointCmd->GetNew3VectorValue(newValues)) ;
    if( histtype == "biast")
      fParticleGun->GetBiasRndm()->SetThetaBias( histpointCmd->GetNew3VectorValue(newValues)) ;
    if( histtype == "biasp")
      fParticleGun->GetBiasRndm()->SetPhiBias( histpointCmd->GetNew3VectorValue(newValues)) ;
    if( histtype == "biase")
      fParticleGun->GetBiasRndm()->SetEnergyBias( histpointCmd->GetNew3VectorValue(newValues)) ;
    if( histtype == "theta")
      fParticleGun->GetAngDist()->UserDefAngTheta( histpointCmd->GetNew3VectorValue(newValues)) ;
    if( histtype == "phi")
      fParticleGun->GetAngDist()->UserDefAngPhi( histpointCmd->GetNew3VectorValue(newValues)) ;
    if( histtype == "energy")
      fParticleGun->GetEneDist()->UserEnergyHisto( histpointCmd->GetNew3VectorValue(newValues)) ;
    if( histtype == "arb"){
      if(fArbInterModeSet){
	GateError("ERROR: After setting the interpolation mode, you cannot add any more points to the energy histogram.");
      }
      fParticleGun->GetEneDist()->ArbEnergyHisto( histpointCmd->GetNew3VectorValue(newValues)) ;
      ++fNArbEHistPoints;
    }
    if( histtype == "epn")
      fParticleGun->GetEneDist()->EpnEnergyHisto( histpointCmd->GetNew3VectorValue(newValues)) ;
  }
  else if (command == resethistCmd) {
    GateWarning("The 'histreset' option is DEPRECATED, use 'hist/reset' instead!");
    fParticleGun->GetAngDist()->ReSetHist(newValues) ;
  }
  else if (command == arbintCmd) {
    GateWarning("The 'arbint' option is DEPRECATED, use 'hist/inter' instead!");
    if (fNArbEHistPoints < 2){
      GateError("ERROR: Please set the interpolation mode AFTER providing at ALL (and least two) energy histogram points.");
    }
    fParticleGun->GetEneDist()->ArbInterpolate(newValues) ;
    fArbInterModeSet = true;
  }
  else if (command == verbosityCmd) {
    fParticleGun->SetVerbosity( verbosityCmd->GetNewIntValue(newValues)) ;
    fParticleGun->SetVerboseLevel( verbosityCmd->GetNewIntValue(newValues)) ;
  }
  else if (command == particleCmd) {
    if (newValues == "ion") {
      fShootIon = true ;
    } else {
      fShootIon = false ;
      G4ParticleDefinition* pd = particleTable->FindParticle(newValues) ;
      if( pd != NULL)
        { fParticleGun->SetParticleDefinition( pd) ; }
    }
  }
  //  else if(command == ionCmd)
  //    {
  //   fParticleGun->SetNucleus(ionCmd->GetNewNucleusValue(newValues));
  //  }
  else if (command == timeCmd) { fParticleGun->SetParticleTime( timeCmd->GetNewDoubleValue(newValues)) ; }
  else if (command == polCmd) { fParticleGun->SetParticlePolarization( polCmd->GetNew3VectorValue(newValues)) ; }
  else if (command == numberCmd) { fParticleGun->SetNumberOfParticles( numberCmd->GetNewIntValue(newValues)) ; }
  else if (command == ionCmd) { IonCommand(newValues) ; }
  else if (command == listCmd) { particleTable->DumpTable() ; }
  else if (command == directionCmd) {
    fParticleGun->GetAngDist()->SetAngDistType( "planar") ;
    fParticleGun->GetAngDist()->SetParticleMomentumDirection( directionCmd->GetNew3VectorValue(newValues)) ;
  }
  else if (command == energyCmd) {
    fParticleGun->GetEneDist()->SetEnergyDisType( "Mono") ;
    fParticleGun->GetEneDist()->SetMonoEnergy( energyCmd->GetNewDoubleValue(newValues)) ;
  }
  else if (command == positionCmd) {
    fParticleGun->GetPosDist()->SetPosDisType( "Point") ;
    fParticleGun->GetPosDist()->SetCentreCoords( positionCmd->GetNew3VectorValue(newValues)) ;
  }
	//
  // new implementations
  //
  //
  else if (command == posrot1Cmd1) {
    fParticleGun->GetPosDist()->SetPosRot1( posrot1Cmd1->GetNew3VectorValue(newValues)) ;
    fParticleGun->SetPosRot1(posrot1Cmd1->GetNew3VectorValue(newValues));
  }
  else if (command == posrot2Cmd1) {
    fParticleGun->GetPosDist()->SetPosRot2(posrot2Cmd1->GetNew3VectorValue(newValues));
    fParticleGun->SetPosRot2(posrot2Cmd1->GetNew3VectorValue(newValues));
  }
  else if(command == typeCmd1) {
    fParticleGun->GetPosDist()->SetPosDisType(newValues);
  }
  else if(command == shapeCmd1) {
    fParticleGun->GetPosDist()->SetPosDisShape(newValues);
  }
  else if(command == centreCmd1) {
    fParticleGun->GetPosDist()->SetCentreCoords(centreCmd1->GetNew3VectorValue(newValues));
    fParticleGun->SetCentreCoords( centreCmd1->GetNew3VectorValue(newValues));
  }
  /* DEAD CODE (posrot{1,2}Cmd1 are already checked above)
  else if(command == posrot1Cmd1) {
    fParticleGun->GetPosDist()->SetPosRot1(posrot1Cmd1->GetNew3VectorValue(newValues));
  }
  else if(command == posrot2Cmd1) {
    fParticleGun->GetPosDist()->SetPosRot2(posrot2Cmd1->GetNew3VectorValue(newValues));
  }
  */
  else if(command == halfxCmd1) {
    fParticleGun->GetPosDist()->SetHalfX(halfxCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == halfyCmd1) {
    fParticleGun->GetPosDist()->SetHalfY(halfyCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == halfzCmd1) {
    fParticleGun->GetPosDist()->SetHalfZ(halfzCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == radiusCmd1) {
    fParticleGun->GetPosDist()->SetRadius(radiusCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == radius0Cmd1) {
    fParticleGun->GetPosDist()->SetRadius0(radius0Cmd1->GetNewDoubleValue(newValues));
  }
  else if(command == possigmarCmd1) {
    fParticleGun->GetPosDist()->SetBeamSigmaInR(possigmarCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == possigmaxCmd1) {
    fParticleGun->GetPosDist()->SetBeamSigmaInX(possigmaxCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == possigmayCmd1) {
    fParticleGun->GetPosDist()->SetBeamSigmaInY(possigmayCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == paralpCmd1) {
    fParticleGun->GetPosDist()->SetParAlpha(paralpCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == partheCmd1) {
    fParticleGun->GetPosDist()->SetParTheta(partheCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == parphiCmd1) {
    fParticleGun->GetPosDist()->SetParPhi(parphiCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == confineCmd1) {
    // CORRECTION COPIED FROM OLD 'confine' COMMAND
    // Modif DS/FS: for all names exept NULL, we add the tag "_phys" at the end
    // of the volume name when the user forgot to do it
    if ( newValues != "NULL")
      {
        bool test = false;
        if (newValues.length() < 5) { test = true; }
        else if (newValues.substr( newValues.length()-5) != "_phys") { test = true; }

        if (test)
          {
            newValues += "_phys";
            G4cout << "Confirming confinement to volume '" << newValues << "'...\n" ;
          }
      }
    fParticleGun->GetPosDist()->ConfineSourceToVolume(newValues);
  }
  else if(command == setImageCmd1) {
    fParticleGun->SetUserFluenceFilename(newValues);
  }
  else if(command == angtypeCmd1) {
    if(newValues == "userFocused") {
      fParticleGun->SetUserFocalShapeFlag(true);
      fParticleGun->GetAngDist()->SetAngDistType("focused");
    }
    else {
      fParticleGun->GetAngDist()->SetAngDistType(newValues);
    }
  }
  else if(command == angradiusCmd1) {
    fParticleGun->GetUserFocalShape()->SetRadius(radiusCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == angcentreCmd1) {
    fParticleGun->GetUserFocalShape()->SetCentreCoords(centreCmd1->GetNew3VectorValue(newValues));
  }
  else if(command == angrot1Cmd1) {
    G4String a = "angref1";
    fParticleGun->GetAngDist()->DefineAngRefAxes(a,angrot1Cmd1->GetNew3VectorValue(newValues));
    fParticleGun->GetUserFocalShape()->SetPosRot1(angrot1Cmd1->GetNew3VectorValue(newValues));
  }
  else if(command == angrot2Cmd1) {
    G4String a = "angref2";
    fParticleGun->GetAngDist()->DefineAngRefAxes(a,angrot2Cmd1->GetNew3VectorValue(newValues));
    fParticleGun->GetUserFocalShape()->SetPosRot2(angrot2Cmd1->GetNew3VectorValue(newValues));
  }

  else if(command == minthetaCmd1) {
    fParticleGun->GetAngDist()->SetMinTheta(minthetaCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == minphiCmd1) {
    fParticleGun->GetAngDist()->SetMinPhi(minphiCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == maxthetaCmd1) {
    fParticleGun->GetAngDist()->SetMaxTheta(maxthetaCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == maxphiCmd1) {
    fParticleGun->GetAngDist()->SetMaxPhi(maxphiCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == angsigmarCmd1) {
    fParticleGun->GetAngDist()->SetBeamSigmaInAngR(angsigmarCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == angsigmaxCmd1) {
    fParticleGun->GetAngDist()->SetBeamSigmaInAngX(angsigmaxCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == angsigmayCmd1) {
    fParticleGun->GetAngDist()->SetBeamSigmaInAngY(angsigmayCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == angfocusCmd) {
    fParticleGun->GetAngDist()->SetFocusPointCopy(angfocusCmd->GetNew3VectorValue(newValues));
    fParticleGun->GetAngDist()->SetFocusPoint(angfocusCmd->GetNew3VectorValue(newValues));
  }
  else if(command == useuserangaxisCmd1) {
    fParticleGun->GetAngDist()->SetUseUserAngAxis(useuserangaxisCmd1->GetNewBoolValue(newValues));
  }
  else if(command == surfnormCmd1) {
    fParticleGun->GetAngDist()->SetUserWRTSurface(surfnormCmd1->GetNewBoolValue(newValues));
  }
  else if(command == energytypeCmd1) {
    fParticleGun->GetEneDist()->SetEnergyDisType(newValues);
  }
  else if(command == eminCmd1) {
    fParticleGun->GetEneDist()->SetEmin(eminCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == emaxCmd1) {
    fParticleGun->GetEneDist()->SetEmax(emaxCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == monoenergyCmd1) {
    fParticleGun->GetEneDist()->SetMonoEnergy(monoenergyCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == engsigmaCmd1) {
    fParticleGun->GetEneDist()->SetBeamSigmaInE(engsigmaCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == alphaCmd1) {
    fParticleGun->GetEneDist()->SetAlpha(alphaCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == tempCmd1) {
    fParticleGun->GetEneDist()->SetTemp(tempCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == ezeroCmd1) {
    fParticleGun->GetEneDist()->SetEzero(ezeroCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == gradientCmd1) {
    fParticleGun->GetEneDist()->SetGradient(gradientCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == interceptCmd1) {
    fParticleGun->GetEneDist()->SetInterCept(interceptCmd1->GetNewDoubleValue(newValues));
  }
  else if(command == calculateCmd1) {
    fParticleGun->GetEneDist()->Calculate();
  }
  else if(command == energyspecCmd1) {
    fParticleGun->GetEneDist()->InputEnergySpectra(energyspecCmd1->GetNewBoolValue(newValues));
  }
  else if(command == diffspecCmd1) {
    fParticleGun->GetEneDist()->InputDifferentialSpectra(diffspecCmd1->GetNewBoolValue(newValues));
  }
  else if(command == histtypeCmd) {
    histtype = newValues;
  }
  else if(command == histpointCmd1) {
    if(histtype == "biasx")
      fParticleGun->GetBiasRndm()->SetXBias(histpointCmd1->GetNew3VectorValue(newValues));
    if(histtype == "biasy")
      fParticleGun->GetBiasRndm()->SetYBias(histpointCmd1->GetNew3VectorValue(newValues));
    if(histtype == "biasz")
      fParticleGun->GetBiasRndm()->SetZBias(histpointCmd1->GetNew3VectorValue(newValues));
    if(histtype == "biast")
      fParticleGun->GetBiasRndm()->SetThetaBias(histpointCmd1->GetNew3VectorValue(newValues));
    if(histtype == "biasp")
      fParticleGun->GetBiasRndm()->SetPhiBias(histpointCmd1->GetNew3VectorValue(newValues));
    if(histtype == "biaspt")
      fParticleGun->GetBiasRndm()->SetPosThetaBias(histpointCmd1->GetNew3VectorValue(newValues));
    if(histtype == "biaspp")
      fParticleGun->GetBiasRndm()->SetPosPhiBias(histpointCmd1->GetNew3VectorValue(newValues));
    if(histtype == "biase")
      fParticleGun->GetBiasRndm()->SetEnergyBias(histpointCmd1->GetNew3VectorValue(newValues));
    if(histtype == "theta")
      fParticleGun->GetAngDist()->UserDefAngTheta(histpointCmd1->GetNew3VectorValue(newValues));
    if(histtype == "phi")
      fParticleGun->GetAngDist()->UserDefAngPhi(histpointCmd1->GetNew3VectorValue(newValues));
    if(histtype == "energy")
      fParticleGun->GetEneDist()->UserEnergyHisto(histpointCmd1->GetNew3VectorValue(newValues));
    if(histtype == "arb"){
      if(fArbInterModeSet){
	GateError("ERROR: After setting the interpolation mode, you cannot add any more points to the energy histogram.");
      }
      fParticleGun->GetEneDist()->ArbEnergyHisto(histpointCmd1->GetNew3VectorValue(newValues));
      ++fNArbEHistPoints;
    }
    if(histtype == "epn")
      fParticleGun->GetEneDist()->EpnEnergyHisto(histpointCmd1->GetNew3VectorValue(newValues));
  }
  else if(command == resethistCmd1) {
    if(newValues == "theta" || newValues == "phi") {
      fParticleGun->GetAngDist()->ReSetHist(newValues);
    } else if (newValues == "energy" || newValues == "arb" || newValues == "epn") {
      fParticleGun->GetEneDist()->ReSetHist(newValues);
    } else {
      fParticleGun->GetBiasRndm()->ReSetHist(newValues);
    }
  }
  else if(command == arbintCmd1) {
    if( (fParticleGun->GetEneDist()->GetEnergyDisType() != "Arb") || (histtype != "arb") ){
      GateWarning("'hist/inter' only works if the energy type is 'Arb' "
		  << "and the histograms type is 'arb', not on type '" << histtype
		  << "'. Unexpected behavior or crashes may happen...");
    }
    if (fNArbEHistPoints < 2){
      GateError("ERROR: Please set the interpolation mode AFTER providing at ALL (and least two) energy histogram points.");
    }
    fParticleGun->GetEneDist()->ArbInterpolate(newValues);
    fArbInterModeSet = true;
  }
  else if (command == setUserSpectrumCmd) {
    GateSPSEneDistribution* speEn =  fParticleGun->GetEneDist(); //->ContructUserSpectrum(newValues);
    speEn->BuildUserSpectrum(newValues);
    // fParticleGun->GetEneDist()->ContructUserSpectrum(newValues);
  }
  else {
    G4cout << "Error entering command\n";
  }



}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
G4String GateSingleParticleSourceMessenger::GetCurrentValue( G4UIcommand*)
{
  G4String cv ;

  //  if (command == directionCmd)
  //  { cv = directionCmd->ConvertToString( fParticleGun->GetParticleMomentumDirection()) ; }
  //  else if (command == energyCmd)
  //  { cv = energyCmd->ConvertToString( fParticleGun->GetParticleEnergy(),"GeV") ; }
  //  else if (command == positionCmd)
  //  { cv = positionCmd->ConvertToString( fParticleGun->GetParticlePosition(),"cm") ; }
  //  else if (command == timeCmd)
  //  { cv = timeCmd->ConvertToString( fParticleGun->GetParticleTime(),"ns") ; }
  //  else if (command == polCmd)
  //  { cv = polCmd->ConvertToString( fParticleGun->GetParticlePolarization()) ; }
  //  else if (command == numberCmd)
  //  { cv = numberCmd->ConvertToString( fParticleGun->GetNumberOfParticles()) ; }

  cv = "Not implemented yet" ;

  return cv ;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSingleParticleSourceMessenger::IonCommand( G4String newValues)
{

  // DEBUG SJAN G4 10.1
  //G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  G4IonTable* ionTable = G4IonTable::GetIonTable();
  if( fShootIon) {
    G4Tokenizer next(newValues) ;
    // check argument
    fAtomicNumber = StoI( next()) ;
    fAtomicMass = StoI( next()) ;
    G4String sQ = next() ;
    if( sQ.empty())
      {
        fIonCharge = fAtomicNumber ;
      }
    else
      {
        fIonCharge = StoI( sQ) ;
        sQ = next() ;
        if( sQ.empty())
          {
            fIonExciteEnergy = 0.0;
          }
        else
          {
            fIonExciteEnergy = StoD( sQ) * keV ;
          }
      }
    G4ParticleDefinition* ion ;
    ion =  ionTable->GetIon( fAtomicNumber, fAtomicMass, fIonExciteEnergy) ;
    if( ion==0)
      {
        G4cout << "Ion with Z=" << fAtomicNumber ;
        G4cout << " A=" << fAtomicMass << "is not be defined\n" ;
      }
    else
      {
        fParticleGun->SetParticleDefinition(ion) ;
        fParticleGun->SetParticleCharge( fIonCharge* eplus) ;
      }
  }
  else {
    G4cout << "Set /gps/particle to ion before using /gps/ion command" ;
    G4cout << Gateendl ;
  }
}
//-------------------------------------------------------------------------------------------------
// vim: ai sw=2 ts=2 et
